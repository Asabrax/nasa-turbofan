import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from load_data import load_test_data, load_test_rul, load_train_data
    from preprocessing import add_train_rul, create_test_targets
    from predictive_maintenance import (
        RESULTS_DIR,
        RUL_CAP,
        SEQUENCE_STRIDE,
        SEQUENCE_WINDOW,
        SUBSETS,
        build_maintenance_report,
        build_dashboard,
        maintenance_validation_score,
        nasa_score,
        make_sequence_test_set,
        make_sequence_training_set,
        plot_model_comparison,
        risk_metrics,
        sequence_feature_columns,
        sequence_window_array,
        split_engine_units,
        standardize_sequence_data,
    )
else:
    from .load_data import load_test_data, load_test_rul, load_train_data
    from .preprocessing import add_train_rul, create_test_targets
    from .predictive_maintenance import (
        RESULTS_DIR,
        RUL_CAP,
        SEQUENCE_STRIDE,
        SEQUENCE_WINDOW,
        SUBSETS,
        build_maintenance_report,
        build_dashboard,
        maintenance_validation_score,
        nasa_score,
        make_sequence_test_set,
        make_sequence_training_set,
        plot_model_comparison,
        risk_metrics,
        sequence_feature_columns,
        sequence_window_array,
        split_engine_units,
        standardize_sequence_data,
    )


TUNING_CONFIGS = (
    {
        "config": "baseline_mse",
        "hidden_size": 48,
        "dilations": (1, 2, 4, 8),
        "dropout": 0.08,
        "loss": "mse",
        "critical_weight": 1.0,
        "learning_rate": 0.002,
        "weight_decay": 0.001,
    },
    {
        "config": "compact_huber_critical",
        "hidden_size": 32,
        "dilations": (1, 2, 4, 8),
        "dropout": 0.08,
        "loss": "huber",
        "critical_weight": 2.0,
        "learning_rate": 0.002,
        "weight_decay": 0.001,
    },
    {
        "config": "wide_huber",
        "hidden_size": 64,
        "dilations": (1, 2, 4, 8),
        "dropout": 0.08,
        "loss": "huber",
        "critical_weight": 1.0,
        "learning_rate": 0.0015,
        "weight_decay": 0.001,
    },
    {
        "config": "short_receptive_field",
        "hidden_size": 48,
        "dilations": (1, 2, 4),
        "dropout": 0.10,
        "loss": "mse",
        "critical_weight": 1.0,
        "learning_rate": 0.002,
        "weight_decay": 0.001,
    },
    {
        "config": "regularized_huber",
        "hidden_size": 48,
        "dilations": (1, 2, 4, 8),
        "dropout": 0.18,
        "loss": "huber",
        "critical_weight": 1.0,
        "learning_rate": 0.002,
        "weight_decay": 0.003,
    },
    {
        "config": "critical_weighted_mse",
        "hidden_size": 48,
        "dilations": (1, 2, 4, 8),
        "dropout": 0.08,
        "loss": "mse",
        "critical_weight": 2.0,
        "learning_rate": 0.002,
        "weight_decay": 0.001,
    },
)


class TunableTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.projection = (
            nn.Conv1d(channels, hidden_channels, kernel_size=1)
            if channels != hidden_channels
            else nn.Identity()
        )
        self.padding = padding

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.projection(values)
        hidden = self.conv(values)
        if self.padding:
            hidden = hidden[:, :, :-self.padding]
        return self.dropout(self.activation(hidden)) + residual


class TunableTCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TunableTemporalBlock(
                    hidden_size,
                    hidden_size,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = values.transpose(1, 2)
        hidden = self.input_projection(hidden)
        hidden = self.blocks(hidden)
        return self.head(hidden[:, :, -1]).squeeze(-1)


def make_sequence_snapshot_set(
    data: pd.DataFrame, columns: list[str], stride_multiplier: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    stride = SEQUENCE_STRIDE * stride_multiplier

    for _, engine_rows in data.groupby("unit_number"):
        engine_rows = engine_rows.sort_values("time_in_cycles")
        max_cycle = int(engine_rows["time_in_cycles"].max())
        target_cycles = set(range(SEQUENCE_WINDOW, max_cycle + 1, stride))
        target_cycles.add(max_cycle)

        for cycle in sorted(target_cycles):
            features.append(sequence_window_array(engine_rows, cycle, columns))
            targets.append(min(max_cycle - cycle, RUL_CAP))

    return np.stack(features), np.array(targets, dtype=float)


def sequence_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    raw_targets: torch.Tensor,
    loss_name: str,
    critical_weight: float,
) -> torch.Tensor:
    if loss_name == "huber":
        per_sample_loss = nn.functional.smooth_l1_loss(
            predictions, targets, reduction="none"
        )
    else:
        per_sample_loss = (predictions - targets).pow(2)

    if critical_weight > 1.0:
        weights = torch.where(
            raw_targets <= 30,
            torch.full_like(raw_targets, critical_weight),
            torch.ones_like(raw_targets),
        )
        per_sample_loss = per_sample_loss * weights

    return per_sample_loss.mean()


def parse_dilations(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(part) for part in value.split("-"))
    return tuple(int(part) for part in value)


def build_model(config: dict[str, object], input_size: int) -> TunableTCN:
    return TunableTCN(
        input_size=input_size,
        hidden_size=int(config["hidden_size"]),
        dilations=parse_dilations(config["dilations"]),
        dropout=float(config["dropout"]),
    )


def fit_tcn_candidate(
    config: dict[str, object],
    train_x: np.ndarray,
    train_y: np.ndarray,
    max_epochs: int,
    patience: int,
) -> tuple[TunableTCN, float, float]:
    seed = 42
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    model = build_model(config, input_size=train_x.shape[-1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    target_mean = float(train_y.mean())
    target_std = float(train_y.std() or 1.0)
    scaled_y = (train_y - target_mean) / target_std

    x_tensor = torch.tensor(train_x, dtype=torch.float32)
    y_tensor = torch.tensor(scaled_y, dtype=torch.float32)
    raw_y_tensor = torch.tensor(train_y, dtype=torch.float32)

    sample_count = len(x_tensor)
    batch_size = min(128, sample_count)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    rng = np.random.default_rng(seed)

    for _ in range(max_epochs):
        model.train()
        indices = rng.permutation(sample_count)
        epoch_losses = []

        for start in range(0, sample_count, batch_size):
            batch_indices = indices[start : start + batch_size]
            predictions = model(x_tensor[batch_indices])
            loss = sequence_loss(
                predictions,
                y_tensor[batch_indices],
                raw_y_tensor[batch_indices],
                str(config["loss"]),
                float(config["critical_weight"]),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        epoch_loss = float(np.mean(epoch_losses))
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, target_mean, target_std


def predict_tcn(
    model: TunableTCN, values: np.ndarray, target_mean: float, target_std: float
) -> pd.Series:
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(values, dtype=torch.float32)).numpy()
    return pd.Series(predictions * target_std + target_mean).clip(lower=0)


def evaluate_predictions(actual: pd.Series, predictions: pd.Series) -> dict[str, float]:
    report = pd.DataFrame(
        {"actual_rul": actual.to_numpy(), "predicted_rul": predictions.to_numpy()}
    )
    metrics = risk_metrics(report)
    return {
        "mae": mean_absolute_error(actual, predictions),
        "rmse": mean_squared_error(actual, predictions) ** 0.5,
        "nasa_score": nasa_score(actual, predictions),
        "score": maintenance_validation_score(actual, predictions),
        "critical_recall": metrics["critical_recall"],
        "critical_false_negatives": metrics["critical_false_negatives"],
        "high_priority_recall": metrics["high_priority_recall"],
        "high_priority_false_negatives": metrics["high_priority_false_negatives"],
    }


def tune_subset(subset: str) -> tuple[pd.DataFrame, dict[str, object]]:
    train_data = add_train_rul(load_train_data(subset))
    training_units, validation_units = split_engine_units(train_data)
    model_data = train_data[train_data["unit_number"].isin(training_units)]
    validation_data = train_data[train_data["unit_number"].isin(validation_units)]
    columns = sequence_feature_columns(train_data)

    train_x, train_y = make_sequence_snapshot_set(model_data, columns)
    validation_x, validation_y = make_sequence_snapshot_set(validation_data, columns)
    train_x, validation_x = standardize_sequence_data(train_x, validation_x)
    actual = pd.Series(validation_y)

    rows = []
    for config in TUNING_CONFIGS:
        print(f"Tuning {subset} TCN: {config['config']}", flush=True)
        model, target_mean, target_std = fit_tcn_candidate(
            config, train_x, train_y, max_epochs=45, patience=7
        )
        predictions = predict_tcn(model, validation_x, target_mean, target_std)
        metrics = evaluate_predictions(actual, predictions)
        rows.append(
            {
                "subset": subset,
                **config,
                "dilations": "-".join(str(value) for value in config["dilations"]),
                "validation_score": metrics["score"],
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_critical_recall": metrics["critical_recall"],
                "validation_critical_misses": metrics["critical_false_negatives"],
                "validation_high_priority_recall": metrics["high_priority_recall"],
                "validation_high_priority_misses": metrics[
                    "high_priority_false_negatives"
                ],
                "training_windows": int(len(train_y)),
                "validation_windows": int(len(validation_y)),
                "window_cycles": SEQUENCE_WINDOW,
                "window_stride": SEQUENCE_STRIDE,
            }
        )

    results = pd.DataFrame(rows).sort_values("validation_score")
    return results, results.iloc[0].to_dict()


def evaluate_best_on_test(
    subset: str, best_config: dict[str, object]
) -> dict[str, object]:
    train_data = add_train_rul(load_train_data(subset))
    test_data = load_test_data(subset)
    test_rul = load_test_rul(subset)
    columns = sequence_feature_columns(train_data)

    train_x, train_y = make_sequence_training_set(train_data, columns)
    test_x, test_last_rows = make_sequence_test_set(test_data, columns)
    test_targets = create_test_targets(test_data, test_rul)
    train_x, test_x = standardize_sequence_data(train_x, test_x)

    model, target_mean, target_std = fit_tcn_candidate(
        best_config, train_x, train_y, max_epochs=80, patience=10
    )
    predictions = predict_tcn(model, test_x, target_mean, target_std)
    actual = test_targets["rul"]
    report = build_maintenance_report(subset, test_last_rows, actual, predictions)
    metrics = risk_metrics(report)

    return {
        "model": "Tuned TCN Sequence Model",
        "subset": subset,
        "selected_config": best_config["config"],
        "train_engines": int(train_data["unit_number"].nunique()),
        "test_engines": int(test_data["unit_number"].nunique()),
        "mae": mean_absolute_error(actual, predictions),
        "rmse": mean_squared_error(actual, predictions) ** 0.5,
        "nasa_score": nasa_score(actual, predictions),
        "critical_engines": int((report["risk_level"] == "critical").sum()),
        "risk_decision_accuracy": float(report["risk_match"].mean()),
        **metrics,
        "average_predicted_rul": float(report["predicted_rul"].mean()),
        "window_cycles": SEQUENCE_WINDOW,
        "window_stride": SEQUENCE_STRIDE,
        "training_windows": int(len(train_y)),
        "validation_score": float(best_config["validation_score"]),
        "validation_mae": float(best_config["validation_mae"]),
        "validation_critical_recall": float(best_config["validation_critical_recall"]),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    tuning_path = RESULTS_DIR / "tcn_tuning_results.csv"

    if tuning_path.exists():
        print(f"Loading saved tuning results from {tuning_path}", flush=True)
        tuning_results = pd.read_csv(tuning_path)
    else:
        tuning_frames = []
        for subset in SUBSETS:
            subset_results, _ = tune_subset(subset)
            tuning_frames.append(subset_results)

        tuning_results = pd.concat(tuning_frames, ignore_index=True)
        tuning_results.to_csv(tuning_path, index=False)

    selected_configs = [
        subset_results.sort_values("validation_score").iloc[0].to_dict()
        for _, subset_results in tuning_results.groupby("subset")
    ]

    test_path = RESULTS_DIR / "tcn_tuned_test_metrics.csv"
    if test_path.exists():
        print(f"Loading saved tuned test results from {test_path}", flush=True)
        test_results = pd.read_csv(test_path)

    required_test_columns = {
        "nasa_score",
        "actual_critical_engines",
        "critical_true_positives",
    }
    if not test_path.exists() or not required_test_columns.issubset(test_results.columns):
        test_metrics = []
        for best_config in selected_configs:
            subset = str(best_config["subset"])
            print(
                f"Evaluating tuned {subset} TCN on held-out test engines: "
                f"{best_config['config']}",
                flush=True,
            )
            test_metrics.append(evaluate_best_on_test(subset, best_config))

        test_results = pd.DataFrame(test_metrics)
        test_results.to_csv(test_path, index=False)

    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison[comparison["model"] != "Tuned TCN Sequence Model"]
        comparison = pd.concat([comparison, test_results], ignore_index=True, sort=False)
        comparison.to_csv(comparison_path, index=False)
        plot_model_comparison(comparison)

        report_path = RESULTS_DIR / "maintenance_report.csv"
        metrics_path = RESULTS_DIR / "subset_metrics.csv"
        if report_path.exists() and metrics_path.exists():
            build_dashboard(
                pd.read_csv(report_path),
                pd.read_csv(metrics_path),
                comparison,
            )

    print("\nSelected validation configs:")
    print(
        pd.DataFrame(selected_configs)[
            [
                "subset",
                "config",
                "validation_score",
                "validation_mae",
                "validation_critical_recall",
                "validation_critical_misses",
            ]
        ].to_string(index=False)
    )
    print("\nHeld-out test results:")
    print(
        test_results[
            [
                "subset",
                "selected_config",
                "mae",
                "rmse",
                "risk_decision_accuracy",
                "critical_recall",
                "critical_false_negatives",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
