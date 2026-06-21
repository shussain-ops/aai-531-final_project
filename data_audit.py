"""Basic data audit utilities."""

import pandas as pd


def audit_dataframe(df: pd.DataFrame) -> None:
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print()
    print("Column data types:")
    print(df.dtypes.value_counts())

    print()
    print("Missing values by column:")
    missing = df.isna().sum().sort_values(ascending=False)
    if (missing > 0).any():
        print(missing[missing > 0])
    else:
        print("No missing values")

    print()
    print("Duplicate row count:", df.duplicated().sum())

    if "Class" in df.columns:
        print()
        print("Target distribution:")
        print(df["Class"].value_counts().rename(index={0: "Genuine", 1: "Fraud"}))

        fraud_rate = df["Class"].mean() * 100
        print()
        print(f"Fraud rate: {fraud_rate:.4f}%")
