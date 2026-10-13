"""PyFlink event-time feature processor for ``financial.transactions.v1``.

Each customer receives a trailing 10-second snapshot every second. Snapshots
are written to ``fraudguard:features:customer:<customer_id>`` and guarded by
event time so late windows cannot replace a newer feature vector.

Run with PyFlink installed and the matching Flink Kafka connector JAR available
to the cluster (see README.md, "Flink streaming job").
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

TOPIC = "financial.transactions.v1"
WINDOW_SECONDS = 10
SLIDE_SECONDS = 1
FEATURE_PREFIX = "fraudguard:features:customer:"


def _event_time_ms(value: Any) -> int:
    """Normalize supported event timestamp values to epoch milliseconds."""
    from datetime import datetime, timezone

    if isinstance(value, (int, float)):
        # The event contract uses epoch seconds for numeric timestamps.
        return int(value * 1000)
    if not isinstance(value, str):
        raise ValueError("event_timestamp is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("event_timestamp must include a timezone")
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def parse_transaction(raw: str | bytes) -> dict[str, Any]:
    """Validate and normalize the fields used by the streaming feature job."""
    payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    for field in ("transaction_id", "customer_id"):
        if payload.get(field) is None or not str(payload[field]).strip():
            raise ValueError(f"{field} is required")
    amount = float(payload["transaction_amount"])
    if amount < 0:
        raise ValueError("transaction_amount cannot be negative")
    device = payload.get("device_id")
    return {
        "transaction_id": str(payload["transaction_id"]),
        "customer_id": str(payload["customer_id"]),
        "transaction_amount": amount,
        "device_id": str(device) if device is not None and str(device).strip() else None,
        "event_time_ms": _event_time_ms(payload.get("event_timestamp")),
    }


def compute_window_features(events: Iterable[dict[str, Any]], window_end_ms: int) -> dict[str, Any]:
    """Compute deterministic trailing-window velocity, amount, and device changes."""
    ordered = sorted(events, key=lambda event: (event["event_time_ms"], event["transaction_id"]))
    devices = [event["device_id"] for event in ordered if event.get("device_id") is not None]
    switches = sum(left != right for left, right in zip(devices, devices[1:]))
    return {
        "transaction_velocity_10s": len(ordered),
        "rolling_amount_10s": round(sum(event["transaction_amount"] for event in ordered), 2),
        "device_switching_frequency_10s": switches,
        "window_end_ms": int(window_end_ms),
    }


class _CustomerWindowFeatures:
    """Window function wrapper kept importable without a local PyFlink install."""

    def __new__(cls, *args, **kwargs):
        from pyflink.datastream.functions import WindowFunction

        class WindowFunctionImpl(WindowFunction):
            def apply(self, key, window, values):
                events = list(values)
                if not events:
                    return
                yield {
                    "customer_id": key,
                    **compute_window_features(events, window.end),
                }

        return WindowFunctionImpl()


class _RedisFeatureSink:
    """PyFlink sink function with a monotonic event-time Redis update."""

    def __new__(cls, redis_url: str):
        from pyflink.datastream.functions import RichSinkFunction

        class RedisSinkImpl(RichSinkFunction):
            _UPDATE_IF_NEWER = """
            local old = tonumber(redis.call('HGET', KEYS[1], 'flink_window_end_ms') or '-1')
            local incoming = tonumber(ARGV[1])
            if incoming < old then return 0 end
            for i = 2, #ARGV, 2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1]) end
            redis.call('EXPIRE', KEYS[1], 172800)
            return 1
            """

            def __init__(self):
                self.redis_url = redis_url
                self.client = None
                self.update = None

            def open(self, runtime_context):
                import redis

                self.client = redis.Redis.from_url(self.redis_url, decode_responses=True)
                self.client.ping()
                self.update = self.client.register_script(self._UPDATE_IF_NEWER)

            def invoke(self, value, context):
                key = FEATURE_PREFIX + value["customer_id"]
                fields = {
                    "transaction_velocity_10s": value["transaction_velocity_10s"],
                    "rolling_amount_10s": value["rolling_amount_10s"],
                    "device_switching_frequency_10s": value["device_switching_frequency_10s"],
                    "flink_window_end_ms": value["window_end_ms"],
                }
                args = [value["window_end_ms"]]
                for field, field_value in fields.items():
                    args.extend((field, field_value))
                self.update(keys=[key], args=args)

            def close(self):
                if self.client is not None:
                    self.client.close()

        return RedisSinkImpl()


def build_job(env=None):
    """Build the Flink graph; dependencies are loaded only when invoked."""
    from pyflink.common import Duration, Time, WatermarkStrategy
    from pyflink.common.serialization import SimpleStringSchema
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
    from pyflink.datastream.functions import MapFunction
    from pyflink.datastream.window import SlidingEventTimeWindows

    env = env or StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(int(os.environ.get("FLINK_PARALLELISM", "2")))
    env.enable_checkpointing(int(os.environ.get("FLINK_CHECKPOINT_INTERVAL_MS", "30000")))

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    source = (KafkaSource.builder()
              .set_bootstrap_servers(bootstrap)
              .set_topics(os.environ.get("KAFKA_TOPIC", TOPIC))
              .set_group_id(os.environ.get("KAFKA_GROUP_ID", "fraudguard-flink-features"))
              .set_starting_offsets(KafkaOffsetsInitializer.earliest())
              .set_value_only_deserializer(SimpleStringSchema())
              .build())

    class ParseTransaction(MapFunction):
        def map(self, raw):
            return parse_transaction(raw)

    stream = env.from_source(source, WatermarkStrategy.no_watermarks(), "kafka-transactions")
    # Assign watermarks after parsing so the strategy can use the event timestamp.
    stream = stream.map(ParseTransaction())
    stream = stream.assign_timestamps_and_watermarks(
        WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(
            int(os.environ.get("FLINK_MAX_OUT_OF_ORDERNESS_SECONDS", "5"))))
        .with_timestamp_assigner(lambda event, _timestamp: event["event_time_ms"])
    )
    features = (stream.key_by(lambda event: event["customer_id"])
                .window(SlidingEventTimeWindows.of(
                    Time.seconds(WINDOW_SECONDS), Time.seconds(SLIDE_SECONDS)))
                .apply(_CustomerWindowFeatures()))
    features.add_sink(_RedisFeatureSink(os.environ.get("REDIS_URL", "redis://localhost:6379/0")))
    return env


def main() -> None:
    build_job().execute("FraudGuard Flink transaction features")


if __name__ == "__main__":
    main()
