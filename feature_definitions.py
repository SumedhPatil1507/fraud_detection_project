"""Feast entities and event-time feature views for FraudGuard."""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

customer = Entity(name="customer_id", join_keys=["customer_id"])
merchant = Entity(name="merchant_id", join_keys=["merchant_id"])

# One immutable event-time source supports point-in-time joins across views.
transaction_features_source = FileSource(
    name="transaction_features_source",
    path="data/transaction_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

customer_transaction_features = FeatureView(
    name="customer_transaction_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="customer_txn_count_5m", dtype=Int64),
        Field(name="customer_txn_count_1h", dtype=Int64),
        Field(name="customer_txn_count_24h", dtype=Int64),
        Field(name="customer_amount_sum_1h", dtype=Float64),
        Field(name="customer_amount_sum_24h", dtype=Float64),
        Field(name="customer_cross_channel_count_1h", dtype=Int64),
    ],
    source=transaction_features_source,
    online=True,
    tags={"domain": "fraud", "temporal_semantics": "event_time"},
)

merchant_transaction_features = FeatureView(
    name="merchant_transaction_features",
    entities=[merchant],
    ttl=timedelta(days=365),
    schema=[
        Field(name="merchant_txn_count_5m", dtype=Int64),
        Field(name="merchant_txn_count_1h", dtype=Int64),
        Field(name="merchant_amount_sum_1h", dtype=Float64),
        Field(name="merchant_cross_channel_count_1h", dtype=Int64),
    ],
    source=transaction_features_source,
    online=True,
    tags={"domain": "fraud", "temporal_semantics": "event_time"},
)
