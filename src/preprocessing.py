import pandas as pd


def add_train_rul(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    max_cycles = data.groupby("unit_number")["time_in_cycles"].transform("max")
    data["rul"] = max_cycles - data["time_in_cycles"]
    return data


def create_test_targets(test_data: pd.DataFrame, rul_data: pd.DataFrame) -> pd.DataFrame:
    last_cycles = test_data.groupby("unit_number")["time_in_cycles"].max().reset_index()
    targets = last_cycles.copy()
    targets["rul"] = rul_data["true_rul"].to_numpy()
    return targets


def get_last_cycle_rows(data: pd.DataFrame) -> pd.DataFrame:
    last_cycle_index = data.groupby("unit_number")["time_in_cycles"].idxmax()
    return data.loc[last_cycle_index].copy()


def feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {"unit_number", "rul"}
    return [column for column in data.columns if column not in excluded]


def sensor_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column.startswith("sensor_")]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["unit_number", "time_in_cycles"]).copy()
    sensors = sensor_columns(data)
    grouped = data.groupby("unit_number", group_keys=False)
    engineered = {}

    for sensor in sensors:
        grouped_sensor = grouped[sensor]
        engineered[f"{sensor}_delta"] = grouped_sensor.diff().fillna(0)

        for window in (5, 15):
            rolling_mean = grouped_sensor.rolling(window=window, min_periods=1).mean()
            engineered[f"{sensor}_rolling_mean_{window}"] = rolling_mean.reset_index(
                level=0, drop=True
            )

            rolling_std = grouped_sensor.rolling(window=window, min_periods=2).std()
            engineered[f"{sensor}_rolling_std_{window}"] = (
                rolling_std.reset_index(level=0, drop=True).fillna(0)
            )

            engineered[f"{sensor}_slope_{window}"] = (
                grouped_sensor.diff(periods=window).fillna(0) / window
            )

        early_baseline = grouped_sensor.transform(
            lambda values: values.expanding(min_periods=1).mean().shift().bfill()
        )
        engineered[f"{sensor}_from_engine_baseline"] = data[sensor] - early_baseline

    return pd.concat([data, pd.DataFrame(engineered, index=data.index)], axis=1)


def cap_rul(data: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    data = data.copy()
    data["rul"] = data["rul"].clip(upper=cap)
    return data


def maintenance_action(predicted_rul: float) -> str:
    if predicted_rul <= 30:
        return "urgent maintenance"
    if predicted_rul <= 60:
        return "schedule maintenance"
    if predicted_rul <= 90:
        return "inspect soon"
    return "normal monitoring"


def risk_level(predicted_rul: float) -> str:
    if predicted_rul <= 30:
        return "critical"
    if predicted_rul <= 60:
        return "high"
    if predicted_rul <= 90:
        return "medium"
    return "low"
