"""Point-in-time guarantees for Feast historical feature retrieval."""
from datetime import datetime, timezone
import shutil
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

feast = pytest.importorskip("feast")


def test_feature_views_share_point_in_time_source_and_expected_schema():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "feature_definitions_under_test", repo / "feature_definitions.py"
    )
    definitions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definitions)

    source = definitions.transaction_features_source
    assert source.timestamp_field == "event_timestamp"
    assert source.created_timestamp_column == "created_timestamp"
    assert definitions.customer_transaction_features.source == source
    assert definitions.merchant_transaction_features.source == source
    assert {field.name for field in definitions.customer_transaction_features.schema} == {
        "customer_txn_count_5m", "customer_txn_count_1h", "customer_txn_count_24h",
        "customer_amount_sum_1h", "customer_amount_sum_24h",
        "customer_cross_channel_count_1h",
    }
    assert {field.name for field in definitions.merchant_transaction_features.schema} == {
        "merchant_txn_count_5m", "merchant_txn_count_1h",
        "merchant_amount_sum_1h", "merchant_cross_channel_count_1h",
    }


def test_historical_join_excludes_future_and_late_arriving_features(tmp_path, monkeypatch):
    """A row sees only feature values available by its event timestamp."""
    repo = Path(__file__).resolve().parents[1]
    shutil.copy(repo / "feature_store.yaml", tmp_path / "feature_store.yaml")
    shutil.copy(repo / "feature_definitions.py", tmp_path / "feature_definitions.py")
    (tmp_path / "data").mkdir()

    history = pd.DataFrame([
        {
            "customer_id": "c-1", "merchant_id": "m-1",
            "event_timestamp": pd.Timestamp("2025-01-01T09:00:00Z"),
            "created_timestamp": pd.Timestamp("2025-01-01T09:00:01Z"),
            "customer_txn_count_5m": 2, "customer_txn_count_1h": 4,
            "customer_txn_count_24h": 8, "customer_amount_sum_1h": 125.0,
            "customer_amount_sum_24h": 300.0, "customer_cross_channel_count_1h": 2,
            "merchant_txn_count_5m": 1, "merchant_txn_count_1h": 3,
            "merchant_amount_sum_1h": 50.0, "merchant_cross_channel_count_1h": 1,
        },
        {
            "customer_id": "c-1", "merchant_id": "m-1",
            "event_timestamp": pd.Timestamp("2025-01-01T11:00:00Z"),
            "created_timestamp": pd.Timestamp("2025-01-01T11:00:01Z"),
            "customer_txn_count_5m": 99, "customer_txn_count_1h": 99,
            "customer_txn_count_24h": 99, "customer_amount_sum_1h": 9999.0,
            "customer_amount_sum_24h": 9999.0, "customer_cross_channel_count_1h": 9,
            "merchant_txn_count_5m": 99, "merchant_txn_count_1h": 99,
            "merchant_amount_sum_1h": 9999.0, "merchant_cross_channel_count_1h": 9,
        },
        {
            # Event occurred earlier but only arrived after the entity timestamp.
            "customer_id": "c-1", "merchant_id": "m-1",
            "event_timestamp": pd.Timestamp("2025-01-01T09:30:00Z"),
            "created_timestamp": pd.Timestamp("2025-01-01T10:30:00Z"),
            "customer_txn_count_5m": 77, "customer_txn_count_1h": 77,
            "customer_txn_count_24h": 77, "customer_amount_sum_1h": 7777.0,
            "customer_amount_sum_24h": 7777.0, "customer_cross_channel_count_1h": 7,
            "merchant_txn_count_5m": 77, "merchant_txn_count_1h": 77,
            "merchant_amount_sum_1h": 7777.0, "merchant_cross_channel_count_1h": 7,
        },
    ])
    history.to_parquet(tmp_path / "data" / "transaction_features.parquet", index=False)

    monkeypatch.chdir(tmp_path)
    store = feast.FeatureStore(repo_path=str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "temporary_feature_definitions", tmp_path / "feature_definitions.py"
    )
    definitions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definitions)
    store.apply([
        definitions.customer, definitions.merchant,
        definitions.customer_transaction_features,
        definitions.merchant_transaction_features,
    ])

    entities = pd.DataFrame({
        "customer_id": ["c-1"], "merchant_id": ["m-1"],
        "event_timestamp": [datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)],
    })
    result = store.get_historical_features(
        entity_df=entities,
        features=["customer_transaction_features:customer_txn_count_1h",
                  "customer_transaction_features:customer_amount_sum_1h",
                  "merchant_transaction_features:merchant_txn_count_1h"],
    ).to_df()

    assert result.loc[0, "customer_txn_count_1h"] == 4
    assert result.loc[0, "customer_amount_sum_1h"] == 125.0
    assert result.loc[0, "merchant_txn_count_1h"] == 3
