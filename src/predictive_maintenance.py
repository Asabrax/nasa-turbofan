import sys
import json
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch import nn
from xgboost import XGBRegressor

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from load_data import load_test_data, load_test_rul, load_train_data
    from preprocessing import (
        add_engineered_features,
        add_train_rul,
        cap_rul,
        create_test_targets,
        feature_columns,
        get_last_cycle_rows,
        maintenance_action,
        risk_level,
    )
else:
    from .load_data import load_test_data, load_test_rul, load_train_data
    from .preprocessing import (
        add_engineered_features,
        add_train_rul,
        cap_rul,
        create_test_targets,
        feature_columns,
        get_last_cycle_rows,
        maintenance_action,
        risk_level,
    )

RESULTS_DIR = Path("results")
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
RUL_CAP = 125
SEQUENCE_WINDOW = 30
SEQUENCE_STRIDE = 10
ENGINE_DETAIL_SENSORS = ("sensor_2", "sensor_7", "sensor_11", "sensor_15")
MODEL_ORDER = (
    "Tuned XGBoost",
    "GRU Sequence Model",
    "TCN Sequence Model",
    "Tuned TCN Sequence Model",
)
DEFAULT_MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 700,
    "learning_rate": 0.035,
    "max_depth": 5,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "reg_alpha": 0.05,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}
BASE_SUBSET_MODEL_CONFIG = {
    "FD001": {
        "include_cycle": False,
        "model_params": {
            **DEFAULT_MODEL_PARAMS,
            "n_estimators": 250,
            "learning_rate": 0.05,
        },
    },
    "FD002": {
        "include_cycle": True,
        "model_params": {
            **DEFAULT_MODEL_PARAMS,
            "n_estimators": 900,
            "learning_rate": 0.025,
            "min_child_weight": 10,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 3.0,
            "reg_alpha": 0.1,
        },
    },
    "FD003": {"include_cycle": True, "model_params": DEFAULT_MODEL_PARAMS},
    "FD004": {"include_cycle": True, "model_params": DEFAULT_MODEL_PARAMS},
}
SUBSET_MODEL_CONFIG = BASE_SUBSET_MODEL_CONFIG.copy()
TUNING_CANDIDATES = (
    {
        "n_estimators": 500,
        "learning_rate": 0.045,
        "max_depth": 4,
        "min_child_weight": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.5,
        "reg_alpha": 0.05,
    },
    {
        "n_estimators": 900,
        "learning_rate": 0.028,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
    },
    {
        "n_estimators": 1100,
        "learning_rate": 0.022,
        "max_depth": 5,
        "min_child_weight": 12,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 4.0,
        "reg_alpha": 0.15,
    },
    {
        "n_estimators": 1400,
        "learning_rate": 0.018,
        "max_depth": 3,
        "min_child_weight": 6,
        "subsample": 0.95,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
    },
)


def tuning_candidate_configs(subset: str) -> list[dict[str, object]]:
    base_config = BASE_SUBSET_MODEL_CONFIG[subset]
    base_include_cycle = bool(base_config["include_cycle"])
    base_params = dict(base_config["model_params"])
    medium_params = {**DEFAULT_MODEL_PARAMS, **TUNING_CANDIDATES[1]}
    conservative_params = {**DEFAULT_MODEL_PARAMS, **TUNING_CANDIDATES[2]}

    return [
        {"include_cycle": base_include_cycle, "model_params": base_params},
        {"include_cycle": True, "model_params": medium_params},
        {"include_cycle": False, "model_params": medium_params},
    ]


def make_model(subset: str) -> XGBRegressor:
    return XGBRegressor(**SUBSET_MODEL_CONFIG[subset]["model_params"])


def subset_feature_columns(data: pd.DataFrame, subset: str) -> list[str]:
    features = feature_columns(data)
    if not SUBSET_MODEL_CONFIG[subset]["include_cycle"]:
        features = [feature for feature in features if feature != "time_in_cycles"]
    return features


def risk_metrics(report: pd.DataFrame) -> dict[str, float]:
    actual_critical = report["actual_rul"] <= 30
    predicted_critical = report["predicted_rul"] <= 30
    actual_high_priority = report["actual_rul"] <= 60
    predicted_high_priority = report["predicted_rul"] <= 60

    critical_count = int(actual_critical.sum())
    high_priority_count = int(actual_high_priority.sum())
    false_negative_count = int((actual_critical & ~predicted_critical).sum())
    critical_true_positive_count = int((actual_critical & predicted_critical).sum())
    high_priority_false_negative_count = int(
        (actual_high_priority & ~predicted_high_priority).sum()
    )

    return {
        "actual_critical_engines": critical_count,
        "critical_true_positives": critical_true_positive_count,
        "critical_recall": (
            float(critical_true_positive_count / critical_count)
            if critical_count
            else 0.0
        ),
        "critical_false_negatives": false_negative_count,
        "high_priority_recall": (
            float(
                (actual_high_priority & predicted_high_priority).sum()
                / high_priority_count
            )
            if high_priority_count
            else 0.0
        ),
        "high_priority_false_negatives": high_priority_false_negative_count,
    }


def nasa_score(actual: pd.Series, predictions: pd.Series) -> float:
    errors = predictions.to_numpy() - actual.to_numpy()
    penalties = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(penalties.sum())


def split_engine_units(data: pd.DataFrame, validation_fraction: float = 0.22) -> tuple[set[int], set[int]]:
    units = np.array(sorted(data["unit_number"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(units)
    validation_count = max(1, int(round(len(units) * validation_fraction)))
    validation_units = set(int(unit) for unit in units[:validation_count])
    training_units = set(int(unit) for unit in units[validation_count:])
    return training_units, validation_units


def make_validation_snapshots(data: pd.DataFrame) -> pd.DataFrame:
    snapshot_rows = []
    for _, engine_rows in data.groupby("unit_number"):
        engine_rows = engine_rows.sort_values("time_in_cycles")
        max_cycle = int(engine_rows["time_in_cycles"].max())
        target_cycles = {
            max(1, int(round(max_cycle * fraction)))
            for fraction in (0.45, 0.65, 0.8, 0.92, 1.0)
        }
        for cycle in sorted(target_cycles):
            candidates = engine_rows[engine_rows["time_in_cycles"] <= cycle]
            if candidates.empty:
                continue
            row = candidates.iloc[-1].copy()
            row["rul"] = max_cycle - int(row["time_in_cycles"])
            snapshot_rows.append(row)

    return pd.DataFrame(snapshot_rows).reset_index(drop=True)


def maintenance_validation_score(actual: pd.Series, predictions: pd.Series) -> float:
    validation_report = pd.DataFrame(
        {"actual_rul": actual.to_numpy(), "predicted_rul": predictions.to_numpy()}
    )
    metrics = risk_metrics(validation_report)
    mae = mean_absolute_error(actual, predictions)
    critical_fn_rate = 1.0 - metrics["critical_recall"]
    high_priority_fn_rate = 1.0 - metrics["high_priority_recall"]
    return float(mae + 12.0 * critical_fn_rate + 5.0 * high_priority_fn_rate)


def tune_subset_model(
    subset: str, engineered_data: pd.DataFrame
) -> tuple[dict[str, object], pd.DataFrame]:
    training_units, validation_units = split_engine_units(engineered_data)
    training_data = engineered_data[engineered_data["unit_number"].isin(training_units)]
    validation_data = engineered_data[engineered_data["unit_number"].isin(validation_units)]
    validation_snapshots = make_validation_snapshots(validation_data)

    candidates = []
    for candidate in tuning_candidate_configs(subset):
        include_cycle = bool(candidate["include_cycle"])
        candidate_params = dict(candidate["model_params"])
        features = feature_columns(training_data)
        if not include_cycle:
            features = [feature for feature in features if feature != "time_in_cycles"]

        print(
            f"Tuning {subset}: trees={candidate_params['n_estimators']}, "
            f"depth={candidate_params['max_depth']}, include_cycle={include_cycle}",
            flush=True,
        )
        model = XGBRegressor(**candidate_params)
        model.fit(training_data[features], training_data["rul"].clip(upper=RUL_CAP))
        predictions = pd.Series(model.predict(validation_snapshots[features])).clip(
            lower=0
        )
        actual = validation_snapshots["rul"]
        score = maintenance_validation_score(actual, predictions)

        candidate_report = pd.DataFrame(
            {
                "actual_rul": actual.to_numpy(),
                "predicted_rul": predictions.to_numpy(),
            }
        )
        candidate_risk_metrics = risk_metrics(candidate_report)
        candidates.append(
            {
                "subset": subset,
                "include_cycle": include_cycle,
                "validation_score": score,
                "validation_mae": mean_absolute_error(actual, predictions),
                "validation_rmse": mean_squared_error(actual, predictions) ** 0.5,
                "validation_critical_recall": candidate_risk_metrics[
                    "critical_recall"
                ],
                "validation_high_priority_recall": candidate_risk_metrics[
                    "high_priority_recall"
                ],
                "n_estimators": candidate_params["n_estimators"],
                "learning_rate": candidate_params["learning_rate"],
                "max_depth": candidate_params["max_depth"],
                "min_child_weight": candidate_params["min_child_weight"],
                "subsample": candidate_params["subsample"],
                "colsample_bytree": candidate_params["colsample_bytree"],
                "reg_lambda": candidate_params["reg_lambda"],
                "reg_alpha": candidate_params["reg_alpha"],
                "model_params": candidate_params,
            }
        )

    tuning_results = pd.DataFrame(candidates).sort_values("validation_score")
    best = tuning_results.iloc[0]
    config = {
        "include_cycle": bool(best["include_cycle"]),
        "model_params": best["model_params"],
    }
    return config, tuning_results.drop(columns=["model_params"])


def build_maintenance_report(
    subset: str, last_rows: pd.DataFrame, actual: pd.Series, predictions: pd.Series
) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "subset": subset,
            "unit_number": last_rows["unit_number"].to_numpy(),
            "last_observed_cycle": last_rows["time_in_cycles"].to_numpy(),
            "actual_rul": actual.to_numpy(),
            "predicted_rul": predictions.to_numpy(),
        }
    )
    report["engine_id"] = report["subset"] + "-" + report["unit_number"].astype(str)
    report["prediction_error"] = report["predicted_rul"] - report["actual_rul"]
    report["risk_level"] = report["predicted_rul"].apply(risk_level)
    report["actual_risk_level"] = report["actual_rul"].apply(risk_level)
    report["risk_match"] = report["risk_level"] == report["actual_risk_level"]
    report["recommended_action"] = report["predicted_rul"].apply(maintenance_action)

    return report.sort_values("predicted_rul").reset_index(drop=True)


def sequence_feature_columns(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in data.columns
        if column.startswith("operational_setting_") or column.startswith("sensor_")
    ]


def sequence_window_array(
    engine_rows: pd.DataFrame, end_cycle: int, columns: list[str]
) -> np.ndarray:
    history = engine_rows[engine_rows["time_in_cycles"] <= end_cycle].sort_values(
        "time_in_cycles"
    )
    window = history[columns].tail(SEQUENCE_WINDOW).to_numpy(dtype=float)

    if len(window) < SEQUENCE_WINDOW:
        pad_row = window[0] if len(window) else np.zeros(len(columns), dtype=float)
        padding = np.repeat(
            pad_row.reshape(1, -1), SEQUENCE_WINDOW - len(window), axis=0
        )
        window = np.vstack([padding, window])

    return window


def make_sequence_training_set(
    data: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []

    for _, engine_rows in data.groupby("unit_number"):
        engine_rows = engine_rows.sort_values("time_in_cycles")
        max_cycle = int(engine_rows["time_in_cycles"].max())
        target_cycles = set(range(SEQUENCE_WINDOW, max_cycle + 1, SEQUENCE_STRIDE))
        target_cycles.add(max_cycle)

        for cycle in sorted(target_cycles):
            features.append(sequence_window_array(engine_rows, cycle, columns))
            targets.append(min(max_cycle - cycle, RUL_CAP))

    return np.stack(features), np.array(targets, dtype=float)


def make_sequence_test_set(
    test_data: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    features = []
    last_rows = []

    for _, engine_rows in test_data.groupby("unit_number"):
        engine_rows = engine_rows.sort_values("time_in_cycles")
        end_cycle = int(engine_rows["time_in_cycles"].max())
        features.append(sequence_window_array(engine_rows, end_cycle, columns))
        last_rows.append(engine_rows.iloc[-1])

    return np.stack(features), pd.DataFrame(last_rows).reset_index(drop=True)


class GRURulModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(values)
        return self.head(hidden[-1]).squeeze(-1)


class TemporalBlock(nn.Module):
    def __init__(
        self, channels: int, hidden_channels: int, kernel_size: int, dilation: int
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
        self.dropout = nn.Dropout(0.08)
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
        hidden = self.dropout(self.activation(hidden))
        return hidden + residual


class TCNRulModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.Sequential(
            TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=1),
            TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=2),
            TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=4),
            TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=8),
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


def standardize_sequence_data(
    train_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    feature_mean = train_x.mean(axis=(0, 1), keepdims=True)
    feature_std = train_x.std(axis=(0, 1), keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    return (train_x - feature_mean) / feature_std, (test_x - feature_mean) / feature_std


def fit_torch_sequence_model(
    model: nn.Module, train_x: np.ndarray, train_y: np.ndarray
) -> tuple[nn.Module, float, float]:
    torch.manual_seed(42)
    torch.set_num_threads(2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    loss_fn = nn.MSELoss()

    target_mean = float(train_y.mean())
    target_std = float(train_y.std() or 1.0)
    scaled_y = (train_y - target_mean) / target_std

    x_tensor = torch.tensor(train_x, dtype=torch.float32)
    y_tensor = torch.tensor(scaled_y, dtype=torch.float32)
    sample_count = len(x_tensor)
    batch_size = min(128, sample_count)
    best_loss = float("inf")
    best_state = None
    patience = 12
    stale_epochs = 0

    rng = np.random.default_rng(42)
    for _ in range(100):
        model.train()
        indices = rng.permutation(sample_count)
        epoch_losses = []

        for start in range(0, sample_count, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_x = x_tensor[batch_indices]
            batch_y = y_tensor[batch_indices]

            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
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


def predict_torch_sequence_model(
    model: nn.Module, test_x: np.ndarray, target_mean: float, target_std: float
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(test_x, dtype=torch.float32)).numpy()
    return predictions * target_std + target_mean


def train_torch_sequence_subset(
    subset: str, model_name: str, model: nn.Module
) -> tuple[pd.DataFrame, dict[str, float]]:
    print(f"Training {subset}: {SEQUENCE_WINDOW}-cycle {model_name}", flush=True)
    train_data = add_train_rul(load_train_data(subset))
    test_data = load_test_data(subset)
    test_rul = load_test_rul(subset)
    columns = sequence_feature_columns(train_data)

    train_x, train_y = make_sequence_training_set(train_data, columns)
    test_x, test_last_rows = make_sequence_test_set(test_data, columns)
    test_targets = create_test_targets(test_data, test_rul)

    train_x, test_x = standardize_sequence_data(train_x, test_x)
    model, target_mean, target_std = fit_torch_sequence_model(model, train_x, train_y)

    predictions = pd.Series(
        predict_torch_sequence_model(model, test_x, target_mean, target_std)
    ).clip(lower=0)
    actual = test_targets["rul"]
    report = build_maintenance_report(subset, test_last_rows, actual, predictions)
    maintenance_metrics = risk_metrics(report)

    metrics = {
        "model": model_name,
        "subset": subset,
        "train_engines": int(train_data["unit_number"].nunique()),
        "test_engines": int(test_data["unit_number"].nunique()),
        "mae": mean_absolute_error(actual, predictions),
        "rmse": mean_squared_error(actual, predictions) ** 0.5,
        "nasa_score": nasa_score(actual, predictions),
        "critical_engines": int((report["risk_level"] == "critical").sum()),
        "risk_decision_accuracy": float(report["risk_match"].mean()),
        **maintenance_metrics,
        "average_predicted_rul": float(report["predicted_rul"].mean()),
        "window_cycles": SEQUENCE_WINDOW,
        "window_stride": SEQUENCE_STRIDE,
        "training_windows": int(len(train_y)),
    }

    return report, metrics


def train_gru_sequence_subset(subset: str) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = sequence_feature_columns(load_train_data(subset))
    model = GRURulModel(input_size=len(columns))
    return train_torch_sequence_subset(subset, "GRU Sequence Model", model)


def train_tcn_sequence_subset(subset: str) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = sequence_feature_columns(load_train_data(subset))
    model = TCNRulModel(input_size=len(columns))
    return train_torch_sequence_subset(subset, "TCN Sequence Model", model)


def train_subset(
    subset: str,
) -> tuple[pd.DataFrame, dict[str, float], pd.Series, pd.DataFrame, pd.DataFrame]:
    train_data = add_train_rul(load_train_data(subset))
    test_data = load_test_data(subset)
    test_rul = load_test_rul(subset)

    train_features_data = add_engineered_features(train_data)
    test_features_data = add_engineered_features(test_data)

    best_config, tuning_results = tune_subset_model(subset, train_features_data)
    SUBSET_MODEL_CONFIG[subset] = best_config
    train_features_data = cap_rul(train_features_data, cap=RUL_CAP)

    features = subset_feature_columns(train_features_data, subset)
    model = make_model(subset)
    model.fit(train_features_data[features], train_features_data["rul"])

    test_last_rows = get_last_cycle_rows(test_features_data)
    test_targets = create_test_targets(test_data, test_rul)

    predictions = pd.Series(model.predict(test_last_rows[features])).clip(lower=0)
    actual = test_targets["rul"]

    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5
    report = build_maintenance_report(subset, test_last_rows, actual, predictions)
    maintenance_metrics = risk_metrics(report)

    metrics = {
        "subset": subset,
        "train_engines": int(train_data["unit_number"].nunique()),
        "test_engines": int(test_data["unit_number"].nunique()),
        "mae": mae,
        "rmse": rmse,
        "nasa_score": nasa_score(actual, predictions),
        "critical_engines": int((report["risk_level"] == "critical").sum()),
        "risk_decision_accuracy": float(report["risk_match"].mean()),
        **maintenance_metrics,
        "average_predicted_rul": float(report["predicted_rul"].mean()),
        "uses_cycle_feature": bool(SUBSET_MODEL_CONFIG[subset]["include_cycle"]),
        "model_trees": int(SUBSET_MODEL_CONFIG[subset]["model_params"]["n_estimators"]),
        "validation_mae": float(tuning_results.iloc[0]["validation_mae"]),
        "validation_score": float(tuning_results.iloc[0]["validation_score"]),
    }
    importance = pd.Series(model.feature_importances_, index=features, name=subset)
    train_data["subset"] = subset

    return report, metrics, importance, train_data, tuning_results


def save_metrics(metrics: pd.DataFrame) -> None:
    with (RESULTS_DIR / "model_metrics.txt").open("w", encoding="utf-8") as file:
        file.write(f"Subsets: {', '.join(metrics['subset'])}\n")
        file.write(f"RUL cap used for training: {RUL_CAP} cycles\n\n")

        for _, row in metrics.iterrows():
            file.write(f"{row['subset']}\n")
            file.write(f"Train engines: {int(row['train_engines'])}\n")
            file.write(f"Test engines: {int(row['test_engines'])}\n")
            file.write(f"MAE: {row['mae']:.2f} cycles\n")
            file.write(f"RMSE: {row['rmse']:.2f} cycles\n")
            file.write(f"NASA score: {row['nasa_score']:.1f}\n")
            file.write(f"Engines needing urgent maintenance: {int(row['critical_engines'])}\n")
            file.write(f"Risk decision accuracy: {row['risk_decision_accuracy']:.1%}\n")
            file.write(f"Critical recall: {row['critical_recall']:.1%}\n")
            file.write(
                f"Critical false negatives: {int(row['critical_false_negatives'])}\n"
            )
            file.write(f"High-priority recall: {row['high_priority_recall']:.1%}\n")
            file.write(
                "High-priority false negatives: "
                f"{int(row['high_priority_false_negatives'])}\n"
            )
            file.write(f"Average predicted RUL: {row['average_predicted_rul']:.2f} cycles\n")
            file.write(f"Validation MAE: {row['validation_mae']:.2f} cycles\n")
            file.write(f"Uses cycle feature: {bool(row['uses_cycle_feature'])}\n")
            file.write(f"XGBoost trees: {int(row['model_trees'])}\n\n")


def plot_rul_distribution(train_data: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    for subset in SUBSETS:
        subset_data = train_data[train_data["subset"] == subset]
        subset_data["rul"].hist(bins=30, alpha=0.55, label=subset)

    plt.title("Capped Training RUL Target Distribution")
    plt.xlabel("capped remaining useful life (cycles)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rul_distribution.png")
    plt.close()


def plot_predicted_vs_actual(report: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 7))

    for subset in SUBSETS:
        subset_report = report[report["subset"] == subset]
        plt.scatter(
            subset_report["actual_rul"],
            subset_report["predicted_rul"],
            alpha=0.7,
            label=subset,
        )

    max_rul = max(report["actual_rul"].max(), report["predicted_rul"].max())
    plt.plot([0, max_rul], [0, max_rul], color="black", linestyle="--")
    plt.axhline(RUL_CAP, color="gray", linestyle=":", linewidth=1)
    plt.title("Predicted vs Actual RUL (Training Target Capped)")
    plt.xlabel("true cycles left")
    plt.ylabel("predicted cycles left")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "predicted_vs_actual.png")
    plt.close()


def plot_feature_importance(importances: pd.DataFrame) -> None:
    mean_importance = importances.mean(axis=1)
    top_features = mean_importance.sort_values(ascending=False).head(15).sort_values()

    plt.figure(figsize=(9, 6))
    top_features.plot(kind="barh")
    plt.title("Top Predictive Maintenance Features")
    plt.xlabel("mean feature importance across subsets")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png")
    plt.close()


def plot_risk_summary(report: pd.DataFrame) -> None:
    order = ["critical", "high", "medium", "low"]
    counts = pd.crosstab(report["risk_level"], report["subset"]).reindex(
        order, fill_value=0
    )
    percentages = counts.div(counts.sum(axis=0), axis=1) * 100

    plt.figure(figsize=(8, 5))
    percentages.plot(
        kind="bar",
        ax=plt.gca(),
        color=["#2563eb", "#ea580c", "#16a34a", "#7c3aed"],
    )
    plt.title("Fleet Risk Share By Dataset")
    plt.xlabel("risk level")
    plt.ylabel("percent of test engines")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fleet_risk_summary.png")
    plt.close()


def plot_prediction_error(report: pd.DataFrame) -> None:
    data = [
        report.loc[report["subset"] == subset, "prediction_error"]
        for subset in SUBSETS
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=SUBSETS, showfliers=False)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Prediction Error By Dataset")
    plt.xlabel("dataset")
    plt.ylabel("predicted cycles left minus true cycles left")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "prediction_error_by_dataset.png")
    plt.close()


def save_engine_timeseries() -> None:
    engine_data = {}

    for subset in SUBSETS:
        test_data = load_test_data(subset)
        columns = ["unit_number", "time_in_cycles", *ENGINE_DETAIL_SENSORS]
        for unit_number, engine_rows in test_data[columns].groupby("unit_number"):
            engine_rows = engine_rows.sort_values("time_in_cycles")
            engine_id = f"{subset}-{int(unit_number)}"
            engine_data[engine_id] = {
                "cycle": engine_rows["time_in_cycles"].astype(int).tolist(),
                **{
                    sensor: engine_rows[sensor].round(4).tolist()
                    for sensor in ENGINE_DETAIL_SENSORS
                },
            }

    (RESULTS_DIR / "engine_timeseries.json").write_text(
        json.dumps(engine_data, separators=(",", ":")),
        encoding="utf-8",
    )


def make_dashboard_table(report: pd.DataFrame, rows: int = 14) -> str:
    table_rows = []
    for _, row in report.head(rows).iterrows():
        table_rows.append(
            "<tr>"
            f"<td>{escape(row['engine_id'])}</td>"
            f"<td>{int(row['last_observed_cycle'])}</td>"
            f"<td>{row['predicted_rul']:.1f}</td>"
            f"<td>{int(row['actual_rul'])}</td>"
            f"<td><span class=\"risk {escape(row['risk_level'])}\">"
            f"{escape(row['risk_level'])}</span></td>"
            f"<td>{'yes' if bool(row['risk_match']) else 'no'}</td>"
            f"<td>{escape(row['recommended_action'])}</td>"
            "</tr>"
        )
    return "\n".join(table_rows)


def make_metrics_table(metrics: pd.DataFrame) -> str:
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(row['subset'])}</td>"
            f"<td>{int(row['train_engines'])}</td>"
            f"<td>{int(row['test_engines'])}</td>"
            f"<td>{row['mae']:.2f}</td>"
            f"<td>{row['rmse']:.2f}</td>"
            f"<td>{row['risk_decision_accuracy']:.1%}</td>"
            f"<td>{row['critical_recall']:.1%}</td>"
            f"<td>{int(row['critical_false_negatives'])}</td>"
            f"<td>{int(row['critical_engines'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def weighted_model_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        model_rows = comparison[comparison["model"] == model_name]
        if model_rows.empty:
            continue
        weights = model_rows["test_engines"]
        rows.append(
            {
                "model": model_name,
                "subset": "Overall",
                "test_engines": int(weights.sum()),
                "mae": float((model_rows["mae"] * weights).sum() / weights.sum()),
                "risk_decision_accuracy": float(
                    (model_rows["risk_decision_accuracy"] * weights).sum()
                    / weights.sum()
                ),
                "rmse": float(
                    np.sqrt((model_rows["rmse"].pow(2) * weights).sum() / weights.sum())
                ),
                "nasa_score": float(model_rows["nasa_score"].sum()),
                "critical_recall": float(
                    model_rows["critical_true_positives"].sum()
                    / model_rows["actual_critical_engines"].sum()
                ),
                "critical_false_negatives": int(
                    model_rows["critical_false_negatives"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def make_model_comparison_table(comparison: pd.DataFrame) -> str:
    ordered = comparison.copy()
    ordered["model"] = pd.Categorical(
        ordered["model"], categories=MODEL_ORDER, ordered=True
    )
    ordered = ordered.sort_values(["model", "subset"])
    ordered["model"] = ordered["model"].astype(str)
    display_rows = pd.concat(
        [weighted_model_summary(comparison), ordered], ignore_index=True
    )
    rows = []
    for _, row in display_rows.iterrows():
        row_class = " class=\"summary-row\"" if row["subset"] == "Overall" else ""
        rows.append(
            f"<tr{row_class}>"
            f"<td>{escape(row['model'])}</td>"
            f"<td>{escape(row['subset'])}</td>"
            f"<td>{int(row['test_engines'])}</td>"
            f"<td>{row['mae']:.2f}</td>"
            f"<td>{row['rmse']:.2f}</td>"
            f"<td>{row['nasa_score']:.1f}</td>"
            f"<td>{row['risk_decision_accuracy']:.1%}</td>"
            f"<td>{row['critical_recall']:.1%}</td>"
            f"<td>{int(row['critical_false_negatives'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def plot_model_comparison(comparison: pd.DataFrame) -> None:
    summary = weighted_model_summary(comparison)
    x = np.arange(len(summary))
    width = 0.35

    fig, left_axis = plt.subplots(figsize=(8, 5))
    right_axis = left_axis.twinx()

    left_axis.bar(
        x - width / 2,
        summary["mae"],
        width,
        label="MAE",
        color="#2563eb",
    )
    right_axis.bar(
        x + width / 2,
        summary["critical_recall"] * 100,
        width,
        label="Critical recall",
        color="#16a34a",
    )

    left_axis.set_title("Model Comparison: Error vs Urgent-Engine Recall")
    left_axis.set_ylabel("MAE, cycles")
    right_axis.set_ylabel("critical recall, percent")
    left_axis.set_xticks(x)
    short_labels = {
        "Tuned XGBoost": "XGBoost",
        "GRU Sequence Model": "GRU",
        "TCN Sequence Model": "TCN",
        "Tuned TCN Sequence Model": "Tuned TCN",
    }
    left_axis.set_xticklabels(
        [short_labels.get(model, model) for model in summary["model"]], rotation=0
    )
    left_axis.set_ylim(bottom=0)
    right_axis.set_ylim(0, 100)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2563eb"),
        plt.Rectangle((0, 0), 1, 1, color="#16a34a"),
    ]
    left_axis.legend(handles, ["MAE", "Critical recall"], loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png")
    plt.close(fig)


def build_dashboard(
    report: pd.DataFrame, metrics: pd.DataFrame, model_comparison: pd.DataFrame | None = None
) -> None:
    urgent_count = int((report["risk_level"] == "critical").sum())
    high_count = int((report["risk_level"] == "high").sum())
    medium_count = int((report["risk_level"] == "medium").sum())
    low_count = int((report["risk_level"] == "low").sum())
    metric_weights = metrics["test_engines"]
    mean_mae = (metrics["mae"] * metric_weights).sum() / metric_weights.sum()
    mean_rmse = np.sqrt(
        (metrics["rmse"].pow(2) * metric_weights).sum() / metric_weights.sum()
    )
    mean_risk_accuracy = (
        metrics["risk_decision_accuracy"] * metric_weights
    ).sum() / metric_weights.sum()
    mean_critical_recall = (
        metrics["critical_true_positives"].sum()
        / metrics["actual_critical_engines"].sum()
    )
    critical_false_negatives = int(metrics["critical_false_negatives"].sum())
    report_records = report.sort_values("predicted_rul").to_dict(orient="records")
    report_json = json.dumps(report_records, separators=(",", ":"))
    default_engine_id = escape(str(report_records[0]["engine_id"]))
    model_comparison_section = ""

    if model_comparison is not None and not model_comparison.empty:
        model_comparison_section = f"""
    <section>
      <h2>Model Comparison</h2>
      <p class="section-note">This compares the selected XGBoost model against real GRU and TCN sequence models. The sequence models read the last {SEQUENCE_WINDOW} cycles and are trained with sliding windows every {SEQUENCE_STRIDE} cycles, so they learn from many in-service points per engine instead of only the final state. All models are evaluated on the same test engines and maintenance metrics.</p>
      <img src="model_comparison.png" alt="Model comparison chart">
      <div class="table-wrap">
        <table class="comparison-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Dataset</th>
              <th>Test Engines</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>NASA Score</th>
              <th>Risk Match</th>
              <th>Crit. Recall</th>
              <th>Crit. Misses</th>
            </tr>
          </thead>
          <tbody>
            {make_model_comparison_table(model_comparison)}
          </tbody>
        </table>
      </div>
      <div class="metric-note">
        <p><strong>Interpretation:</strong> lower MAE, RMSE, and NASA score are better. The NASA score penalizes late RUL predictions more strongly because overestimating remaining life is more dangerous. Critical recall and critical misses show whether urgent engines were caught.</p>
      </div>
    </section>
"""

    dashboard = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NASA Turbofan Predictive Maintenance Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #eef2f7;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #5b6472;
      --line: #d8dee8;
      --soft: #f8fafc;
      --accent: #2563eb;
      --critical: #b91c1c;
      --high: #ea580c;
      --medium: #b38b00;
      --low: #15803d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
    }}
    header {{
      padding: 30px 32px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.5; }}
    main {{ padding: 24px 32px 36px; max-width: 1280px; margin: 0 auto; }}
    .eyebrow {{
      margin-bottom: 8px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .intro {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 20px;
    }}
    .intro h2 {{ margin-bottom: 8px; }}
    .intro-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .explain {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .explain strong {{ display: block; margin-bottom: 6px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric strong {{ font-size: 24px; }}
    .metric p {{ margin-top: 8px; font-size: 13px; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
      gap: 16px;
      align-items: start;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: var(--soft); }}
    tr:last-child td {{ border-bottom: 0; }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .table-wrap table {{ border: 0; }}
    .table-wrap th, .table-wrap td {{ white-space: nowrap; }}
    .priority-table {{ min-width: 880px; }}
    .dataset-table {{ min-width: 820px; }}
    .comparison-table {{ min-width: 880px; }}
    .summary-row td {{
      background: #eef6ff;
      font-weight: 700;
    }}
    .metric-note {{
      border-left: 4px solid var(--accent);
      background: var(--soft);
      border-radius: 8px;
      margin-top: 12px;
      padding: 12px 14px;
    }}
    .section-note {{ margin: 0 0 12px; font-size: 14px; }}
    .risk {{
      display: inline-block;
      min-width: 68px;
      border-radius: 999px;
      padding: 4px 8px;
      color: white;
      text-align: center;
      font-size: 12px;
    }}
    .critical {{ background: var(--critical); }}
    .high {{ background: var(--high); }}
    .medium {{ background: var(--medium); }}
    .low {{ background: var(--low); }}
    .split {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .risk-guide {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .risk-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--soft);
    }}
    .risk-card strong {{ display: block; margin-bottom: 6px; }}
    .lookup-controls {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 180px 130px;
      gap: 10px;
      margin-bottom: 14px;
    }}
    input, select, button {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: white;
      color: var(--ink);
      font: inherit;
    }}
    button {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      font-weight: 700;
      cursor: pointer;
    }}
    .engine-summary {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0 14px;
    }}
    .engine-card {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
    }}
    .engine-card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .engine-card strong {{ font-size: 18px; }}
    .chart-wrap {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 10px;
    }}
    #engineChart {{
      display: block;
      width: 100%;
      height: 280px;
    }}
    .empty-state {{
      padding: 20px;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .metrics, .grid, .split, .intro-grid, .risk-guide, .lookup-controls, .engine-summary {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Predictive maintenance demo</div>
    <h1>NASA Turbofan Maintenance Evaluation</h1>
    <p>A plain-language evaluation of which simulated jet engines are closest to failure and what maintenance action the model recommends.</p>
  </header>
  <main>
    <section class="intro">
      <h2>What This Dashboard Shows</h2>
      <p>
        Each engine has sensor readings over time. The model uses those readings to estimate
        Remaining Useful Life, or RUL: roughly how many operating cycles the engine has left.
        Lower predicted RUL means higher maintenance priority.
      </p>
      <div class="intro-grid">
        <div class="explain">
          <strong>1. Predict engine life</strong>
          <p>The model estimates cycles left before failure for every test engine.</p>
        </div>
        <div class="explain">
          <strong>2. Convert to risk</strong>
          <p>Predictions are grouped into critical, high, medium, and low risk bands.</p>
        </div>
        <div class="explain">
          <strong>3. Recommend action</strong>
          <p>The highest-risk engines are listed first so maintenance teams know where to start.</p>
        </div>
      </div>
    </section>

    <div class="metrics">
      <div class="metric">
        <span>Datasets Modeled</span><strong>{len(metrics)}</strong>
        <p>FD001-FD004 cover different operating conditions and fault modes.</p>
      </div>
      <div class="metric">
        <span>Average Error (MAE)</span><strong>{mean_mae:.2f}</strong>
        <p>Typical prediction miss, measured in engine cycles. Lower is better.</p>
      </div>
      <div class="metric">
        <span>Risk Bucket Match</span><strong>{mean_risk_accuracy:.1%}</strong>
        <p>Exact match between predicted and true risk bands.</p>
      </div>
      <div class="metric">
        <span>Critical Recall</span><strong>{mean_critical_recall:.1%}</strong>
        <p>Share of truly urgent engines caught by the critical-risk rule.</p>
      </div>
    </div>

    <section>
      <h2>Engine Lookup</h2>
      <p class="section-note">Search an engine ID to inspect its prediction, risk level, and recent sensor history. Example: <strong>{default_engine_id}</strong>.</p>
      <div class="lookup-controls">
        <input id="engineSearch" list="engineOptions" placeholder="Search engine ID, e.g. FD004-12" value="{default_engine_id}">
        <datalist id="engineOptions"></datalist>
        <select id="sensorSelect">
          <option value="sensor_2">Sensor 2</option>
          <option value="sensor_7">Sensor 7</option>
          <option value="sensor_11">Sensor 11</option>
          <option value="sensor_15">Sensor 15</option>
        </select>
        <button id="engineButton" type="button">Show Engine</button>
      </div>
      <div id="engineSummary" class="engine-summary"></div>
      <div class="chart-wrap">
        <svg id="engineChart" viewBox="0 0 900 280" role="img" aria-label="Selected engine sensor history"></svg>
      </div>
      <p class="section-note">The line chart shows the selected sensor across the engine's observed test cycles. It is not the prediction by itself; it is context for the model's RUL estimate.</p>
    </section>

    <section>
      <h2>Highest Priority Engines</h2>
      <p class="section-note">Read this table first. These engines have the lowest predicted life remaining and should be checked before the rest of the fleet. Risk match compares the predicted risk bucket with the bucket implied by the true RUL.</p>
      <div class="table-wrap">
        <table class="priority-table">
          <thead>
            <tr>
              <th>Engine ID</th>
              <th>Current Cycle</th>
              <th>Predicted Cycles Left</th>
              <th>True Cycles Left</th>
              <th>Risk</th>
              <th>Risk Match</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {make_dashboard_table(report)}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Dataset Performance</h2>
      <p class="section-note">Each dataset is a different test scenario. FD002 changes operating conditions, FD003 changes fault modes, and FD004 combines both.</p>
      <div class="metric-note">
        <p><strong>How to read this:</strong> Risk bucket match is strict because the model must land in the same risk band as the true RUL. Critical recall is the safety metric: it asks whether truly urgent engines were caught.</p>
      </div>
      <div class="table-wrap">
        <table class="dataset-table">
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Train Engines</th>
              <th>Test Engines</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>Risk Match</th>
              <th>Crit. Recall</th>
              <th>Crit. Misses</th>
              <th>Pred. Critical</th>
            </tr>
          </thead>
          <tbody>
            {make_metrics_table(metrics)}
          </tbody>
        </table>
      </div>
    </section>

    {model_comparison_section}

    <section>
      <h2>Fleet Risk Summary</h2>
      <p class="section-note">This chart shows the percentage of test engines in each risk band for each dataset, so datasets with different fleet sizes are easier to compare.</p>
      <img src="fleet_risk_summary.png" alt="Fleet risk summary chart">
      <p>Total engines by action level: Critical {urgent_count} | High {high_count} | Medium {medium_count} | Low {low_count}. Truly critical engines missed by the critical-risk rule: {critical_false_negatives}.</p>
      <div class="risk-guide">
        <div class="risk-card"><strong><span class="risk critical">critical</span></strong><p>0-30 predicted cycles left. Act first.</p></div>
        <div class="risk-card"><strong><span class="risk high">high</span></strong><p>31-60 cycles left. Schedule maintenance.</p></div>
        <div class="risk-card"><strong><span class="risk medium">medium</span></strong><p>61-90 cycles left. Inspect soon.</p></div>
        <div class="risk-card"><strong><span class="risk low">low</span></strong><p>91+ cycles left. Continue monitoring.</p></div>
      </div>
    </section>

    <div class="split">
      <section>
        <h2>Predicted vs Actual RUL</h2>
        <p class="section-note">Dots close to the dashed diagonal line are accurate predictions. The model was trained with RUL capped at 125 cycles, so high-life engines naturally flatten near that level. For maintenance, the risk bucket is often more important than the exact cycle count.</p>
        <img src="predicted_vs_actual.png" alt="Predicted versus actual RUL chart">
      </section>
      <section>
        <h2>Most Important Features</h2>
        <p class="section-note">These are the sensor-derived signals the model used most. They help explain what the model paid attention to.</p>
        <img src="feature_importance.png" alt="Feature importance chart">
      </section>
    </div>

    <section>
      <h2>Prediction Error By Dataset</h2>
      <p class="section-note">Values near zero mean the model was close. Positive values mean the model predicted too many cycles left; negative values mean it was conservative.</p>
      <img src="prediction_error_by_dataset.png" alt="Prediction error by dataset chart">
    </section>

    <section>
      <h2>Source And Method</h2>
      <p class="section-note">Source: NASA C-MAPSS FD001-FD004 train, test, and true-RUL files in <strong>data/raw/</strong>. Grain: one prediction per test engine at its latest observed cycle. Model: separate tuned XGBoost regressor per dataset, trained on capped RUL at {RUL_CAP} cycles. Validation: training engines are split by engine ID, then evaluated using simulated in-service snapshots. Maintenance metrics prioritize critical recall because a missed urgent engine is worse than a conservative early warning.</p>
    </section>
  </main>
  <script>
    const REPORT_DATA = {report_json};
    let ENGINE_SERIES = {{}};

    const reportByEngine = new Map(REPORT_DATA.map((row) => [row.engine_id, row]));
    const searchInput = document.getElementById("engineSearch");
    const sensorSelect = document.getElementById("sensorSelect");
    const button = document.getElementById("engineButton");
    const summary = document.getElementById("engineSummary");
    const chart = document.getElementById("engineChart");
    const options = document.getElementById("engineOptions");

    for (const row of REPORT_DATA) {{
      const option = document.createElement("option");
      option.value = row.engine_id;
      options.appendChild(option);
    }}

    function formatNumber(value, digits = 1) {{
      return Number(value).toFixed(digits);
    }}

    function renderSummary(row) {{
      summary.innerHTML = `
        <div class="engine-card"><span>Engine ID</span><strong>${{row.engine_id}}</strong></div>
        <div class="engine-card"><span>Predicted Cycles Left</span><strong>${{formatNumber(row.predicted_rul)}}</strong></div>
        <div class="engine-card"><span>True Cycles Left</span><strong>${{Math.round(row.actual_rul)}}</strong></div>
        <div class="engine-card"><span>Risk</span><strong><span class="risk ${{row.risk_level}}">${{row.risk_level}}</span></strong></div>
        <div class="engine-card"><span>Risk Match</span><strong>${{row.risk_match ? "yes" : "no"}}</strong></div>
        <div class="engine-card"><span>Action</span><strong>${{row.recommended_action}}</strong></div>
      `;
    }}

    function scale(value, min, max, start, end) {{
      if (max === min) return (start + end) / 2;
      return start + ((value - min) / (max - min)) * (end - start);
    }}

    function renderChart(engineId, sensor) {{
      const series = ENGINE_SERIES[engineId];
      if (!series || !series[sensor]) {{
        chart.innerHTML = `<text x="450" y="140" text-anchor="middle" class="empty-state">No sensor history found for ${{engineId}}</text>`;
        return;
      }}

      const cycles = series.cycle;
      const values = series[sensor];
      const width = 900;
      const height = 280;
      const left = 58;
      const right = 22;
      const top = 24;
      const bottom = 42;
      const minCycle = Math.min(...cycles);
      const maxCycle = Math.max(...cycles);
      const minValue = Math.min(...values);
      const maxValue = Math.max(...values);
      const pad = (maxValue - minValue || 1) * 0.08;
      const yMin = minValue - pad;
      const yMax = maxValue + pad;

      const points = cycles.map((cycle, index) => {{
        const x = scale(cycle, minCycle, maxCycle, left, width - right);
        const y = scale(values[index], yMin, yMax, height - bottom, top);
        return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
      }}).join(" ");

      chart.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="white"></rect>
        <line x1="${{left}}" y1="${{height - bottom}}" x2="${{width - right}}" y2="${{height - bottom}}" stroke="#94a3b8"></line>
        <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{height - bottom}}" stroke="#94a3b8"></line>
        <text x="${{left}}" y="18" fill="#475569" font-size="13">${{sensor}} reading</text>
        <text x="${{width / 2}}" y="${{height - 8}}" text-anchor="middle" fill="#475569" font-size="13">cycle</text>
        <text x="${{left - 8}}" y="${{top + 5}}" text-anchor="end" fill="#475569" font-size="12">${{yMax.toFixed(1)}}</text>
        <text x="${{left - 8}}" y="${{height - bottom}}" text-anchor="end" fill="#475569" font-size="12">${{yMin.toFixed(1)}}</text>
        <text x="${{left}}" y="${{height - 24}}" fill="#475569" font-size="12">${{minCycle}}</text>
        <text x="${{width - right}}" y="${{height - 24}}" text-anchor="end" fill="#475569" font-size="12">${{maxCycle}}</text>
        <polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="${{points}}"></polyline>
      `;
    }}

    function renderEngine() {{
      const engineId = searchInput.value.trim();
      const row = reportByEngine.get(engineId);
      if (!row) {{
        summary.innerHTML = `<div class="engine-card"><span>No match</span><strong>Try an ID like FD001-81</strong></div>`;
        chart.innerHTML = "";
        return;
      }}
      renderSummary(row);
      renderChart(engineId, sensorSelect.value);
    }}

    button.addEventListener("click", renderEngine);
    sensorSelect.addEventListener("change", renderEngine);
    searchInput.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") renderEngine();
    }});

    fetch("engine_timeseries.json")
      .then((response) => response.json())
      .then((data) => {{
        ENGINE_SERIES = data;
        renderEngine();
      }})
      .catch(() => {{
        renderEngine();
        chart.innerHTML = `<text x="450" y="140" text-anchor="middle" class="empty-state">Open through the local server to load engine sensor history.</text>`;
      }});
  </script>
</body>
</html>
"""
    (RESULTS_DIR / "dashboard.html").write_text(dashboard, encoding="utf-8")


def train_predictive_maintenance_model() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    metrics = []
    importances = []
    train_frames = []
    tuning_frames = []

    for subset in SUBSETS:
        report, subset_metrics, importance, train_data, tuning_results = train_subset(
            subset
        )
        reports.append(report)
        metrics.append(subset_metrics)
        importances.append(importance)
        train_frames.append(train_data)
        tuning_frames.append(tuning_results)

    full_report = pd.concat(reports, ignore_index=True).sort_values("predicted_rul")
    metrics_data = pd.DataFrame(metrics)
    importance_data = pd.concat(importances, axis=1)
    train_data = pd.concat(train_frames, ignore_index=True)
    tuning_data = pd.concat(tuning_frames, ignore_index=True)

    sequence_reports = []
    sequence_metrics = []
    for sequence_trainer in (train_gru_sequence_subset, train_tcn_sequence_subset):
        for subset in SUBSETS:
            sequence_report, sequence_subset_metrics = sequence_trainer(subset)
            sequence_reports.append(sequence_report)
            sequence_metrics.append(sequence_subset_metrics)

    xgboost_comparison = metrics_data.copy()
    xgboost_comparison.insert(0, "model", "Tuned XGBoost")
    sequence_comparison = pd.DataFrame(sequence_metrics)
    model_comparison = pd.concat(
        [xgboost_comparison, sequence_comparison], ignore_index=True
    )

    full_report.to_csv(RESULTS_DIR / "maintenance_report.csv", index=False)
    metrics_data.to_csv(RESULTS_DIR / "subset_metrics.csv", index=False)
    tuning_data.to_csv(RESULTS_DIR / "model_tuning_results.csv", index=False)
    model_comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    full_report[["subset", "unit_number", "actual_rul", "predicted_rul"]].to_csv(
        RESULTS_DIR / "predicted_vs_actual.csv", index=False
    )

    save_metrics(metrics_data)
    plot_rul_distribution(train_data)
    plot_predicted_vs_actual(full_report)
    plot_feature_importance(importance_data)
    plot_risk_summary(full_report)
    plot_prediction_error(full_report)
    plot_model_comparison(model_comparison)
    save_engine_timeseries()
    build_dashboard(full_report, metrics_data, model_comparison)

    print("Predictive maintenance pipeline finished.")
    for _, row in metrics_data.iterrows():
        print(
            f"{row['subset']} - MAE: {row['mae']:.2f} cycles, "
            f"RMSE: {row['rmse']:.2f} cycles, "
            f"critical recall: {row['critical_recall']:.1%}, "
            f"critical misses: {int(row['critical_false_negatives'])}"
        )
    print(f"Maintenance report saved to {RESULTS_DIR / 'maintenance_report.csv'}")
    print(f"Model comparison saved to {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"Dashboard saved to {RESULTS_DIR / 'dashboard.html'}")


if __name__ == "__main__":
    train_predictive_maintenance_model()
