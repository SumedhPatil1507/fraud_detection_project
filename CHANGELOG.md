# Changelog — Enterprise Fraud Detection

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-04-28 — Enterprise Edition

### 🔒 Security & Compliance

#### Added
- **Data Encryption at Rest** (`src/encryption.py`)
  - Fernet (AES-128-CBC) symmetric encryption
  - Automatic encryption of PII fields (customer_id, merchant_id, device_id)
  - Key management via ENCRYPTION_KEY environment variable
  - Encrypt/decrypt helpers for DataFrames

- **Role-Based Access Control** (`src/rbac.py`)
  - Three roles: Admin, Analyst, Viewer
  - Granular permission system (predict, train, audit, etc.)
  - FastAPI dependency injection for endpoint protection
  - Streamlit session-based role management

- **Data Validation Layer** (`src/validation.py`)
  - Schema validation with type checking
  - Range validation (min/max bounds)
  - Business rule validation (cross-field checks)
  - Validation warnings for suspicious patterns

- **API Rate Limiting** (`src/rate_limit.py`)
  - Per-role rate limits (admin: 1000/min, analyst: 200/min, viewer: 30/min)
  - slowapi integration with graceful fallback
  - Configurable via environment variables

### 🏦 Financial Intelligence

#### Added
- **Automated SAR Generation** (`src/sar.py`)
  - FinCEN-style Suspicious Activity Report generation
  - ML-driven risk assessment with SHAP factors
  - Structured JSON output with narrative generation
  - SAR status management (DRAFT → FILED)
  - Automatic risk indicator detection

- **Dynamic Cost-Benefit Optimizer** (`src/optimizer.py`)
  - Threshold optimization by maximizing net financial impact
  - Sweep analysis across threshold range
  - ROI projection calculator
  - Monthly/annual savings forecasting

- **Fraud Savings Tracker** (`src/savings_tracker.py`)
  - Real-time accumulation of prevented losses
  - Confirmed vs. estimated savings tracking
  - Persistent CSV storage
  - Summary statistics (total caught, avg amount, highest catch)

### 🚀 MLOps & Deployment

#### Added
- **Shadow Mode Deployment** (`src/shadow.py`)
  - Run challenger models alongside champion
  - Silent divergence logging
  - Probability delta tracking
  - Divergence rate statistics

- **Automated Retraining Pipeline** (`src/retrain.py`)
  - Drift-triggered retraining
  - Model versioning with timestamps
  - Retraining history log (JSON)
  - Webhook notifications for retraining events
  - Success/failure tracking with metrics

### 🔧 API Enhancements

#### Changed
- **api.py** — Complete rewrite with enterprise features
  - RBAC middleware on all endpoints
  - Data validation before inference
  - Shadow model integration
  - SAR generation endpoint
  - Savings summary endpoint
  - Retraining log endpoint
  - Audit log endpoint
  - CORS middleware
  - Rate limiting (optional)
  - Enhanced error handling

#### Added Endpoints
- `GET /audit` — Recent predictions (analyst+)
- `GET /shadow/stats` — Shadow model divergence (admin)
- `POST /sar/generate` — Generate SAR report (analyst+)
- `GET /sar/list` — List all SAR reports (analyst+)
- `GET /savings` — Fraud savings summary (analyst+)
- `GET /retrain/log` — Retraining history (admin)

### 🧪 Testing

#### Added
- `tests/test_validation.py` — Data validation tests
- `tests/test_rbac.py` — RBAC permission tests
- `tests/test_optimizer.py` — Cost optimizer tests

### 📚 Documentation

#### Added
- **DEPLOYMENT.md** — Comprehensive deployment guide
  - Local development setup
  - Docker deployment
  - Streamlit Cloud configuration
  - Production deployment (AWS, DigitalOcean, K8s)
  - Environment variables reference
  - Security checklist
  - Troubleshooting guide

- **GIT_COMMANDS.md** — Git workflow guide
  - Quick push commands
  - Conventional commits examples
  - Branch strategy
  - Release tagging
  - Troubleshooting

- **CHANGELOG.md** — This file

#### Changed
- **README.md** — Complete rewrite
  - Premium features section
  - Comprehensive API documentation
  - Environment variable guide
  - Performance benchmarks
  - "Built by a College Student" section
  - Professional formatting

### 📦 Dependencies

#### Added
- `cryptography` — Encryption at rest
- `slowapi` — API rate limiting
- `groq` — LLM explanations (already present)
- `fastapi` — API framework (already present)
- `pydantic` — Data validation (already present)

### 🔄 Breaking Changes

- **API Authentication:** Old `API_KEY` env var replaced with role-specific keys:
  - `API_KEY_ADMIN`
  - `API_KEY_ANALYST`
  - `API_KEY_VIEWER`

- **API Responses:** `/predict` now returns additional fields:
  - `validation_warnings` (list)
  - `shadow` (dict, optional)

### 🐛 Bug Fixes

- Fixed duplicate function definition in `src/simulator.py`
- Improved error handling in all new modules
- Added graceful fallbacks for optional dependencies

### ⚡ Performance

- Optimized validation layer (< 1ms overhead)
- Efficient encryption (only sensitive fields)
- Shadow mode runs async (no latency impact)
- Rate limiting with minimal overhead

---

## [1.0.0] - 2026-04-10 — Initial Release

### Added
- XGBoost + LightGBM ensemble model
- SHAP explainability
- Drift detection (PSI-based)
- Streamlit dashboard with 9 tabs
- FastAPI prediction service
- Supabase integration
- Docker support
- CI/CD with GitHub Actions
- HITL review queue
- Live transaction simulator
- PII masking
- Audit logging

---

## Migration Guide (1.0 → 2.0)

### Environment Variables

**Old:**
```bash
API_KEY="fraud-dev-key"
```

**New:**
```bash
ENCRYPTION_KEY="your-fernet-key"
API_KEY_ADMIN="admin-key"
API_KEY_ANALYST="analyst-key"
API_KEY_VIEWER="viewer-key"
```

### API Calls

**Old:**
```bash
curl -H "X-API-Key: fraud-dev-key" ...
```

**New:**
```bash
curl -H "X-API-Key: analyst-key" ...
```

### Code Changes

**Old:**
```python
from src.audit import log_prediction
```

**New:**
```python
from src.database import log_prediction  # Moved to database.py
from src.savings_tracker import record_catch  # New
```

---

## Roadmap

### v2.1.0 (Planned)
- [ ] Real-time model monitoring dashboard
- [ ] A/B testing framework
- [ ] Multi-model ensemble voting
- [ ] GraphQL API
- [ ] Webhook integrations (Slack, PagerDuty)

### v2.2.0 (Planned)
- [ ] Federated learning support
- [ ] Explainable AI report generation (PDF)
- [ ] Advanced anomaly detection (Isolation Forest++)
- [ ] Time-series fraud patterns
- [ ] Customer risk scoring

### v3.0.0 (Future)
- [ ] Multi-tenant support
- [ ] Real-time streaming (Kafka integration)
- [ ] Advanced RBAC (custom roles)
- [ ] Compliance reporting (SOC 2, PCI-DSS)
- [ ] Mobile app (React Native)

---

## Contributors

- **Sumedh Patil** ([@SumedhPatil1507](https://github.com/SumedhPatil1507)) — Creator & Maintainer

---

## License

MIT License — See [LICENSE](LICENSE) for details.
