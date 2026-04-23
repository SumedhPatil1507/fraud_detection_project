# Contributing

## Setup

```bash
git clone https://github.com/SumedhPatil1507/fraud_detection_project
cd fraud_detection_project
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
src/          — all ML and app logic
tests/        — pytest unit tests
app.py        — Streamlit dashboard
api.py        — FastAPI backend
```

## Adding Features

- New plots go in `src/plots.py` — return a Plotly figure
- New ML features go in `src/pipeline.py` → `feature_engineering()`
- New API endpoints go in `api.py`
- All DB operations go through `src/database.py`

## Code Style

- Keep functions small and single-purpose
- All plots must return interactive Plotly figures
- Wrap optional dependencies in `try/except` with graceful fallback
