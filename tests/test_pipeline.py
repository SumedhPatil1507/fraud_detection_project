import pandas as pd
import numpy as np
import pytest
from src.pipeline import preprocess, feature_engineering


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "transaction_amount": [100.0, 500.0, 2000.0],
        "distance_from_home_km": [5.0, 50.0, 300.0],
        "hour": [10, 2, 23],
        "avg_amount_30d": [120.0, 400.0, 800.0],
        "label": [0, 0, 1],
        "device_id": ["d1", "d2", "d3"],
        "fraud_type": ["legit", "legit", "identity_theft"],
        "channel": ["pos", "online", "online"],
    })


def test_preprocess_drops_ids(sample_df):
    result = preprocess(sample_df)
    assert "transaction_id" not in result.columns
    assert "device_id" not in result.columns


def test_preprocess_no_duplicates(sample_df):
    df_dup = pd.concat([sample_df, sample_df])
    result = preprocess(df_dup)
    assert len(result) == len(sample_df)


def test_feature_engineering_amount_log(sample_df):
    result = feature_engineering(sample_df)
    assert "amount_log" in result.columns
    assert np.allclose(result["amount_log"], np.log1p(sample_df["transaction_amount"]))


def test_feature_engineering_hour_cyclical(sample_df):
    result = feature_engineering(sample_df)
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns


def test_feature_engineering_amount_vs_avg(sample_df):
    result = feature_engineering(sample_df)
    assert "amount_vs_avg" in result.columns
