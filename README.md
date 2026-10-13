# FraudGuard AI

Interactive fraud detection and investigation workbench built with Streamlit. It combines transaction scoring, graph intelligence, event-time features, streaming ingestion, and analyst-reviewed SAR drafting in one dashboard.

[![Open the Streamlit app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)
[![CI](https://github.com/SumedhPatil1507/fraud_detection_project/actions/workflows/ci.yml/badge.svg)](https://github.com/SumedhPatil1507/fraud_detection_project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Run the Streamlit dashboard

```bash
git clone https://github.com/SumedhPatil1507/fraud_detection_project.git
cd fraud_detection_project
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. It starts with the built-in sample data; use the sidebar to upload a CSV, Excel, JSON, or Parquet dataset with a binary `label` column. PII masking is enabled by default. The Explorer tab has class and hour filters that update its charts and raw-data preview. All dashboard plots use Plotly with hover details, zoom, pan, selection, and legend toggles. A shared dark theme is configured in `.streamlit/config.toml`.

### Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app from this repository and select `main` and `app.py`.
3. Select Python 3.12 in Streamlit's Advanced settings. Add optional credentials such as `GROQ_API_KEY` through the app's Secrets settings to enable integrations; the dashboard and built-in sample data run without them.

If an existing Community Cloud app was created with Python 3.14, changing this repository will not change that app's interpreter. Streamlit requires deleting and redeploying the app to change Python versions; select 3.12 during redeployment and re-enter any secrets. See [Streamlit's Python version guidance](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).

For local Docker use, build from the repository root with `docker build -f Dockerfile.streamlit -t fraudguard-streamlit .` and run `docker run --rm -p 8501:8501 fraudguard-streamlit`.

### Dashboard areas

| Area | What you can explore |
|---|---|
| Explorer | Filter by fraud class and hour; inspect distributions, transaction amounts, temporal/channel patterns, correlation heatmap, and matching raw rows |
| Train and Metrics | Train the ensemble, review ROC/PR, confusion matrix, threshold and cost curves |
| Explainability and Drift | SHAP views, model drivers, and feature drift |
| Predict and Live Stream | Score individual or simulated transactions and inspect changing risk |
| HITL, Audit, and SAR | Review queue, prediction history, and SAR records |
| Graph Intel and Streaming & GNN | Fraud rings, heterogeneous entity/relationship schema, service configuration, and evaluation report |
| Vault, Savings, Observability, and Compliance | Key-management tools, business impact, runtime metrics, and control checks |
| Copilot | Retrieve precedents, inspect graph/SHAP signals, draft a guarded SAR, and route it for review |

The **Streaming & GNN** tab shows configured service settings and available model artifacts. It does not probe remote services or start a Kafka consumer. Its graph schema and RAGAS charts are interactive Plotly figures.

## Feature pipeline

### Feast event-time features

`feature_store.yaml` configures the local Feast registry, offline file source, and Redis online store. `feature_definitions.py` defines `customer_id` and `merchant_id` entities and customer/merchant rolling transaction and cross-channel feature views. Feature source rows include `event_timestamp` and `created_timestamp` for point-in-time historical retrieval.

```bash
feast -c . apply
```

The offline source defaults to `data/transaction_features.parquet`; populate that file with event-time feature rows before running historical retrieval. `tests/test_feature_store.py` covers point-in-time behavior.

### Kafka streaming consumer

The asynchronous consumer reads `financial.transactions.v1`. It validates transaction identity and event time, writes the audit record to PostgreSQL and rolling features to Redis, and manually commits the Kafka offset only after both writes succeed. Idempotent transaction IDs support safe redelivery.

Configure `KAFKA_BOOTSTRAP_SERVERS`, `DATABASE_URL`, and `REDIS_URL`, then start it with:

```bash
python -m src.streaming.kafka_consumer
```

### Flink streaming job

The optional PyFlink job consumes `financial.transactions.v1` and emits a
customer feature snapshot every second from a trailing 10-second event-time
window. It calculates transaction velocity, total amount, and device changes,
then updates `fraudguard:features:customer:<customer_id>` in Redis. Configure
`KAFKA_BOOTSTRAP_SERVERS`, `REDIS_URL`, and optionally `KAFKA_GROUP_ID`,
`FLINK_PARALLELISM`, and `FLINK_MAX_OUT_OF_ORDERNESS_SECONDS`.

Install the Flink dependencies with `pip install -r requirements-flink.txt`.
The Flink cluster must also have the Kafka SQL/DataStream connector JAR matching
its Flink version on the classpath. Submit with:

```bash
python -m src.streaming.flink_fraud_processor
```

The default topic is `financial.transactions.v1`; the default consumer group is `fraudguard-feature-pipeline`. PostgreSQL connectivity and Redis are required for the consumer.

## Graph fraud model

The optional two-layer heterogeneous R-GCN uses customer, merchant, device, and IP nodes with transaction, shared-device, and associated-IP relations. It predicts transaction fraud, return-to-origin (RTO) probability, and node collusion risk. The training CLI writes learned embeddings and scores to Neo4j when a Neo4j backend is configured. Include a `fraud_type` column with RTO examples labeled `rto`, `rto_fraud`, `return_to_origin`, or `return_to_origin_fraud` to supervise its RTO head.

```bash
python scripts/train_gnn_fraud.py data/transactions.csv --epochs 40
```

Provide a labeled transaction dataset with customer/merchant IDs and optional device/IP columns. The script saves model weights under `outputs/gnn_fraud.pt` by default. GNN scores are consumed as a primary collusion signal by the Copilot when present in Neo4j.

## Fraud Investigation Copilot

The LangGraph pipeline retrieves historical vector chunks, gathers graph/SHAP signals, and drafts a SAR narrative. Guardrails checks require citations to retrieved chunk IDs and lexical support from cited context. Unverified output is withheld and routed to mandatory analyst review. Drafts are never auto-filed. Production writer validation requires `initial_detection_at` for FinCEN deadline checks, and `suspicion_concluded_at`, a holiday calendar, and `principal_officer_reason` for RBI checks. Set `RBI_HOLIDAY_CALENDAR` to a newline-delimited list of `YYYY-MM-DD` holidays. The FinCEN narrative limit is 20,000 characters. The optional Lerner cap applies when a `pricing` object with `proposed_price` and `marginal_cost` is supplied; `LERNER_INDEX_MAX` defaults to `0.30` and can be overridden per pricing object.

OpenTelemetry spans wrap LangGraph nodes, vector searches, graph/SHAP tools, GNN lookups, and evaluation steps. Configure `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) to export traces. Spans intentionally avoid transaction text, customer identifiers, and retrieved document content.

Useful Copilot settings:

```text
GROQ_API_KEY                 Writer LLM key
COPILOT_LLM_MODEL            Writer model (default configured in the app)
CHROMA_PERSIST_DIR           ChromaDB path (default: outputs/chroma_db)
COPILOT_TOP_K                Number of retrieved precedent chunks
SAR_JURISDICTION             Default SAR jurisdiction (US or RBI/India)
RBI_HOLIDAY_CALENDAR         Newline-delimited holiday dates for RBI due-date checks
LERNER_INDEX_MAX             Maximum allowed Lerner index (default 0.30)
```

## RAGAS audit evaluation

The evaluation suite creates 50 repeatable synthetic SAR investigation scenarios and reports Context Precision, Context Recall, Faithfulness, and Answer Relevancy. It needs `OPENAI_API_KEY` for evaluation and embeddings and `GROQ_API_KEY` for Copilot answers.

```bash
python -m eval.ragas_suite --output outputs/ragas_audit_report.json
```

The Streaming & GNN dashboard tab reads this JSON report and plots aggregate and per-scenario metric scores interactively. `outputs/` is git-ignored because it contains generated artifacts and audit data.

## Optional API and local services

Run the FastAPI service separately:

```bash
uvicorn api:app --reload --port 8000
```

The API docs are at `http://localhost:8000/docs`. Docker assets are provided for Streamlit and the API. External integrations such as Neo4j, Kafka, PostgreSQL, Redis, object storage, and LLM APIs are optional for the dashboard's sample-data experience; configure the relevant service URLs and credentials to enable them.

## Testing

```bash
python -m pytest tests/ -q
```

Key additions include point-in-time feature-store tests and Copilot guardrail/fallback tests. Optional integrations require their corresponding services and credentials for end-to-end operation.

## Repository map

```text
app.py                         Streamlit dashboard
feature_store.yaml             Feast project and store configuration
feature_definitions.py         Feast entities and event-time feature views
src/streaming/kafka_consumer.py Kafka audit/cache dual-write consumer
src/models/gnn_fraud.py        Heterogeneous two-layer R-GCN
scripts/train_gnn_fraud.py     GNN training and Neo4j output CLI
src/agent_graph.py             LangGraph fraud investigation Copilot
src/copilot_guardrails.py      SAR grounding, citation, and jurisdiction checks
src/telemetry.py               OpenTelemetry SDK/span helpers
eval/ragas_suite.py            50-scenario RAGAS evaluation and JSON export
tests/                         Unit and integration tests
```

## License

MIT. See [LICENSE](LICENSE).
