"""Provenance checks for reusable TCN experiment results."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_fingerprint() -> str:
    source_dir = Path(__file__).resolve().parent
    paths = sorted(source_dir.glob("*.py"))
    paths += [
        Path("data/raw") / f"{kind}_{subset}.txt"
        for subset in ("FD001", "FD002", "FD003", "FD004")
        for kind in ("train", "test", "RUL")
    ]
    payload = {
        "files": {path.name: file_digest(path) for path in paths},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "torch", "scikit-learn", "xgboost")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_cached_results(
    path: Path, fingerprint: str, required_columns: set[str]
) -> pd.DataFrame | None:
    """Legacy, incomplete, edited, or incompatible results are cache misses."""
    try:
        metadata = json.loads(path.with_suffix(".meta.json").read_text())
        if metadata.get("fingerprint") != fingerprint:
            return None
        if metadata.get("csv_sha256") != file_digest(path):
            return None
        frame = pd.read_csv(path)
        if not required_columns.issubset(frame.columns):
            return None
        if set(frame["subset"]) != {"FD001", "FD002", "FD003", "FD004"}:
            return None
        if frame[list(required_columns)].isna().any().any():
            return None
        return frame
    except (OSError, ValueError, AttributeError, KeyError, pd.errors.ParserError):
        return None


def save_cached_results(frame: pd.DataFrame, path: Path, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(
        json.dumps({"fingerprint": fingerprint, "csv_sha256": file_digest(path)}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def tuned_test_fingerprint(fingerprint: str, tuning_path: Path) -> str:
    return fingerprint + ":" + file_digest(tuning_path)


def merge_tuned_comparison(
    comparison: pd.DataFrame, results_dir: Path, fingerprint: str
) -> pd.DataFrame:
    """Keep compatible tuned results when the main pipeline is rerun."""
    tuning_path = results_dir / "tcn_tuning_results.csv"
    if not tuning_path.exists():
        return comparison
    tuned = load_cached_results(
        results_dir / "tcn_tuned_test_metrics.csv",
        tuned_test_fingerprint(fingerprint, tuning_path),
        {
            "model", "subset", "test_engines", "mae", "rmse", "nasa_score",
            "risk_decision_accuracy", "critical_true_positives",
            "actual_critical_engines", "critical_false_negatives",
            "critical_recall", "window_cycles", "window_stride", "training_windows",
        },
    )
    if tuned is None:
        print("Tuned TCN results are stale or unverified; run src/tcn_tuning_experiment.py.", flush=True)
        return comparison
    if len(tuned) != 4 or set(tuned["model"]) != {"Tuned TCN Sequence Model"}:
        return comparison
    comparison = comparison[comparison["model"] != "Tuned TCN Sequence Model"]
    return pd.concat([comparison, tuned], ignore_index=True, sort=False)
