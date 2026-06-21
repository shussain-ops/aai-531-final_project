"""
Download the Kaggle Credit Card Fraud Detection dataset.
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Do not hard-code or commit your Kaggle API key.
"""

import os
import sys
import subprocess
from pathlib import Path
from getpass import getpass


def download_dataset(output_dir: str = "data/raw") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("KAGGLE_USERNAME"):
        os.environ["KAGGLE_USERNAME"] = input("Kaggle username: ").strip()

    if not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = getpass("Kaggle API key: ")

    try:
        import kaggle  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        "mlg-ulb/creditcardfraud",
        "-p",
        str(output_path),
        "--unzip",
    ]

    print("Downloading dataset from Kaggle...")
    subprocess.check_call(cmd)

    csv_path = output_path / "creditcard.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"creditcard.csv not found at {csv_path}")

    print("Dataset ready at:", csv_path)
    return csv_path


if __name__ == "__main__":
    download_dataset()
