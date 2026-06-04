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
    excluded = {"unit_number", "time_in_cycles", "rul"}
    return [column for column in data.columns if column not in excluded]
