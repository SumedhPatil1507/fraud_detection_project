"""Async Kafka ingestion with audit/cache dual writes before offset commits.

Delivery is at least once: PostgreSQL writes are idempotent on transaction_id,
and Redis feature state is keyed by entity. Configure DATABASE_URL,
REDIS_URL, and KAFKA_BOOTSTRAP_SERVERS in the runtime environment.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

TOPIC = "financial.transactions.v1"


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (int, float)):
        result = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("event_timestamp is required")
    if result.tzinfo is None:
        raise ValueError("event_timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def validate_transaction(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the event contract; reject ambiguous event time."""
    txn = dict(payload)
    for field in ("transaction_id", "customer_id", "merchant_id", "channel"):
        if txn.get(field) is None or str(txn[field]).strip() == "":
            raise ValueError(f"{field} is required")
    txn["event_timestamp"] = _parse_timestamp(txn.get("event_timestamp"))
    txn["transaction_amount"] = float(txn["transaction_amount"])
    if txn["transaction_amount"] < 0:
        raise ValueError("transaction_amount cannot be negative")
    return txn


async def write_audit_postgres(txn: dict[str, Any], pool: Any) -> None:
    """Persist an idempotent audit record; database errors propagate."""
    await pool.execute(
        """INSERT INTO transaction_audit
           (transaction_id, customer_id, merchant_id, channel,
            transaction_amount, event_timestamp, payload)
           VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
           ON CONFLICT (transaction_id) DO NOTHING""",
        str(txn["transaction_id"]), str(txn["customer_id"]),
        str(txn["merchant_id"]), str(txn["channel"]),
        txn["transaction_amount"], txn["event_timestamp"],
        json.dumps(txn, default=lambda value: value.isoformat()),
    )


async def ensure_audit_table(pool: Any) -> None:
    """Create the audit table once before consuming events."""
    await pool.execute(
        """CREATE TABLE IF NOT EXISTS transaction_audit (
             transaction_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
             merchant_id TEXT NOT NULL, channel TEXT NOT NULL,
             transaction_amount DOUBLE PRECISION NOT NULL,
             event_timestamp TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL,
             ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"""
    )


async def write_online_features(txn: dict[str, Any], redis: Any) -> None:
    """Update rolling entity features and cache using Redis sorted-set windows.

    The Lua script makes event insertion and window aggregation atomic per
    entity. Duplicate transaction IDs are ignored to tolerate Kafka redelivery.
    """
    script = redis.register_script(_FEATURE_UPDATE_LUA)
    event_ms = int(txn["event_timestamp"].timestamp() * 1000)
    fields = (str(txn["transaction_id"]), event_ms,
              txn["transaction_amount"], str(txn["channel"]))
    for entity_type, entity_id in (("customer", txn["customer_id"]),
                                   ("merchant", txn["merchant_id"])):
        await script(
            keys=[f"fraudguard:events:{entity_type}:{entity_id}",
                  f"fraudguard:features:{entity_type}:{entity_id}",
                  f"fraudguard:event_payloads:{entity_type}:{entity_id}"],
            args=[*fields, entity_type],
        )


_FEATURE_UPDATE_LUA = r"""
local events, cache, payloads = KEYS[1], KEYS[2], KEYS[3]
local id, ts, amount, channel, entity = ARGV[1], tonumber(ARGV[2]),
      tonumber(ARGV[3]), ARGV[4], ARGV[5]
local cutoff = ts - 86400000
redis.call('ZREMRANGEBYSCORE', events, '-inf', cutoff)
if redis.call('ZSCORE', events, id) ~= false then return 1 end
-- Equal-time events have no reliable ordering; exclude them from this snapshot.
local rows = redis.call('ZRANGEBYSCORE', events, cutoff, ts - 1)
local n5, n60, n24, sum60, sum24, channels60 = 0, 0, #rows, 0, 0, {}
for _, event_id in ipairs(rows) do
  local raw = redis.call('HGET', payloads, event_id)
  local e = cjson.decode(raw)
  local age = ts - e.ts
  sum24 = sum24 + e.amount
  if age <= 3600000 then
    n60 = n60 + 1; sum60 = sum60 + e.amount; channels60[e.channel] = true
  end
  if age <= 300000 then n5 = n5 + 1 end
end
local channel_count = 0
for _ in pairs(channels60) do channel_count = channel_count + 1 end
if entity == 'customer' then
  redis.call('HSET', cache, 'customer_txn_count_5m', n5, 'customer_txn_count_1h', n60,
    'customer_txn_count_24h', n24, 'customer_amount_sum_1h', sum60,
    'customer_amount_sum_24h', sum24, 'customer_cross_channel_count_1h', channel_count,
    'event_timestamp_ms', ts)
else
  redis.call('HSET', cache, 'merchant_txn_count_5m', n5, 'merchant_txn_count_1h', n60,
    'merchant_amount_sum_1h', sum60, 'merchant_cross_channel_count_1h', channel_count,
    'event_timestamp_ms', ts)
end
-- Save the pre-transaction snapshot, then expose this event to later transactions.
redis.call('ZADD', events, ts, id)
redis.call('HSET', payloads, id, cjson.encode({ts=ts, amount=amount, channel=channel}))
redis.call('EXPIRE', events, 172800)
redis.call('EXPIRE', cache, 172800)
redis.call('EXPIRE', payloads, 172800)
return 1
"""


async def process_message(msg: Any, *, pool: Any, redis: Any,
                          audit_writer: Callable = write_audit_postgres,
                          feature_writer: Callable = write_online_features) -> None:
    raw = msg.value.decode("utf-8") if isinstance(msg.value, bytes) else msg.value
    txn = validate_transaction(json.loads(raw) if isinstance(raw, str) else raw)
    # Independent destinations run concurrently; Kafka is committed only after
    # both settle successfully. Idempotent keys make retries safe on partials.
    await asyncio.gather(audit_writer(txn, pool), feature_writer(txn, redis))


async def consume(*, stop_event: asyncio.Event | None = None) -> None:
    """Consume transactions and commit each offset only after both durable writes."""
    from aiokafka import AIOKafkaConsumer
    import asyncpg
    import redis.asyncio as redis_async

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    consumer = AIOKafkaConsumer(
        TOPIC, bootstrap_servers=bootstrap,
        group_id=os.environ.get("KAFKA_GROUP_ID", "fraudguard-feature-pipeline"),
        enable_auto_commit=False, auto_offset_reset="earliest",
        value_deserializer=None,
    )
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=10)
    redis = redis_async.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                                 decode_responses=True)
    await consumer.start()
    try:
        await ensure_audit_table(pool)
        async for msg in consumer:
            if stop_event and stop_event.is_set():
                break
            await process_message(msg, pool=pool, redis=redis)
            await consumer.commit({msg.topic_partition: msg.offset + 1})
    finally:
        await consumer.stop()
        await redis.aclose()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(consume())
