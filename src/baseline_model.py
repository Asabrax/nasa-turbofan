import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from load_data import load_test_data, load_test_rul, load_train_data
    from preprocessing import (
        add_train_rul,
        create_test_targets,
        feature_columns,
        get_last_cycle_rows,
    )
else:
    from .load_data import load_test_data, load_test_rul, load_train_data
    from .preprocessing import (
        add_train_rul,
        create_test_targets,
        feature_columns,
        get_last_cycle_rows,
    )

RESULTS_DIR = Path("results")


def plot_sensor_trends(train_data: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_units = [1, 2, 3]
    sensors = ["sensor_2", "sensor_7", "sensor_11", "sensor_15"]

    fig, axes = plt.subplots(len(sensors), 1, figsize=(10, 8), sharex=True)

    for axis, sensor in zip(axes, sensors):
        for unit in sample_units:
            unit_data = train_data[train_data["unit_number"] == unit]
            axis.plot(
                unit_data["time_in_cycles"],
                unit_data[sensor],
                label=f"engine {unit}",
            )
        axis.set_ylabel(sensor)
        axis.legend()

    axes[-1].set_xlabel("cycle")
    fig.suptitle("Selected Sensor Trends For Sample Engines")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sensor_trends.png")
    plt.close(fig)


def plot_rul_distribution(train_data: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    train_data["rul"].hist(bins=30)
    plt.title("Training RUL Distribution")
    plt.xlabel("remaining useful life")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rul_distribution.png")
    plt.close()


def save_metrics(mae: float, rmse: float) -> None:
    with (RESULTS_DIR / "model_metrics.txt").open("w", encoding="utf-8") as file:
        file.write(f"MAE: {mae:.2f} cycles\n")
        file.write(f"RMSE: {rmse:.2f} cycles\n")


def train_baseline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_data = add_train_rul(load_train_data())
    test_data = load_test_data()
    test_rul = load_test_rul()

    plot_sensor_trends(train_data)
    plot_rul_distribution(train_data)

    features = feature_columns(train_data)
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    model.fit(train_data[features], train_data["rul"])

    test_last_rows = get_last_cycle_rows(test_data)
    test_targets = create_test_targets(test_data, test_rul)

    predictions = model.predict(test_last_rows[features])
    actual = test_targets["rul"]

    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5

    comparison = pd.DataFrame(
        {
            "unit_number": test_last_rows["unit_number"].to_numpy(),
            "actual_rul": actual.to_numpy(),
            "predicted_rul": predictions,
        }
    )
    comparison.to_csv(RESULTS_DIR / "predicted_vs_actual.csv", index=False)

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predictions, alpha=0.75)
    plt.plot([0, max(actual)], [0, max(actual)], color="black", linestyle="--")
    plt.title("Predicted vs Actual RUL")
    plt.xlabel("actual RUL")
    plt.ylabel("predicted RUL")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "predicted_vs_actual.png")
    plt.close()

    save_metrics(mae, rmse)

    print(f"MAE: {mae:.2f} cycles")
    print(f"RMSE: {rmse:.2f} cycles")


if __name__ == "__main__":
    train_baseline()
