"""Feature engineering utilities."""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()
    model_df["LogAmount"] = np.log1p(model_df["Amount"])
    model_df["Hour"] = ((model_df["Time"] // 3600) % 24).astype(int)
    return model_df
