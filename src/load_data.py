from pathlib import Path
import shutil
import subprocess
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

DATA_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
RAW_DATA_DIR = Path("data/raw")
ZIP_PATH = RAW_DATA_DIR / "CMAPSSData.zip"

COLUMNS = (
    ["unit_number", "time_in_cycles"]
    + [f"operational_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def download_data() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"Found existing file: {ZIP_PATH}")
        return

    print("Downloading NASA C-MAPSS data...")

    if shutil.which("curl"):
        subprocess.run(
            ["curl", "-L", "--fail", "--show-error", "-o", str(ZIP_PATH), DATA_URL],
            check=True,
        )
    else:
        request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            ZIP_PATH.write_bytes(response.read())

    print(f"Saved: {ZIP_PATH}")


def extract_data() -> None:
    expected_file = RAW_DATA_DIR / "train_FD001.txt"

    if expected_file.exists():
        print("FD001 files already extracted.")
        return

    with ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(RAW_DATA_DIR)

    print(f"Extracted files to: {RAW_DATA_DIR}")


def read_fd001_file(filename: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run `python src/load_data.py` first."
        )

    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_train_data() -> pd.DataFrame:
    return read_fd001_file("train_FD001.txt")


def load_test_data() -> pd.DataFrame:
    return read_fd001_file("test_FD001.txt")


def load_test_rul() -> pd.DataFrame:
    path = RAW_DATA_DIR / "RUL_FD001.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run `python src/load_data.py` first."
        )

    return pd.read_csv(path, sep=r"\s+", header=None, names=["true_rul"])


def main() -> None:
    download_data()
    extract_data()

    train = load_train_data()
    test = load_test_data()
    rul = load_test_rul()

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"RUL shape: {rul.shape}")
    print("Missing values in train:", int(train.isna().sum().sum()))


if __name__ == "__main__":
    main()
