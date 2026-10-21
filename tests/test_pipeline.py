import numpy as np

from src.config import DRIVERS, FEATURES
from src.features import add_features, get_xy
from src.generate_data import generate
from src.gis_analysis import grid_hotspots, kabupaten_summary
from src.parallel_benchmark import process_tile


def _df(n=600):
    df = generate(n=n, seed=1)
    df["date"] = __import__("pandas").to_datetime(df["date"])
    return df


def test_generate_schema_and_classes():
    df = _df()
    assert len(df) == 600 and df["alert_id"].is_unique
    assert set(df["driver"]) <= set(DRIVERS)
    assert df["lat"].between(-5.5, -2.0).all() and df["lon"].between(100.5, 104.5).all()


def test_features_no_nan_and_shape():
    X, y = get_xy(_df())
    assert list(X.columns) == FEATURES and not X.isna().any().any() and len(X) == len(y)
    assert (add_features(_df())["ndvi_drop"] > 0).all()


def test_reproducible():
    assert generate(n=200, seed=7).equals(generate(n=200, seed=7))


def test_gis_aggregations():
    df = _df()
    assert grid_hotspots(df)["alerts"].sum() == len(df)
    assert kabupaten_summary(df)["alerts"].sum() == len(df)


def test_tile_fraction_in_unit_interval():
    v = process_tile(0)
    assert 0.0 <= v <= 1.0 and process_tile(0) == v
