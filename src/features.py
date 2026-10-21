"""Rekayasa fitur dan utilitas pemuatan data."""
import numpy as np
import pandas as pd

from .config import DATA_CSV, FEATURES


def load_alerts(path=DATA_CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ndvi_drop"] = out["ndvi_before"] - out["ndvi_after"]
    out["log_area"] = np.log1p(out["area_ha"])
    out["dry_season"] = out["date"].dt.month.isin([7, 8, 9, 10]).astype(int)
    out["in_protected"] = out["in_protected"].astype(int)
    return out


def get_xy(df: pd.DataFrame):
    d = add_features(df)
    return d[FEATURES], d["driver"]
