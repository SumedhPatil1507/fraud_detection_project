"""Run a 50-scenario SAR RAG evaluation and write a JSON audit report.

Requires OPENAI_API_KEY for RAGAS judge/embedding calls and GROQ_API_KEY for
the Copilot writer. Run from the repository root:
    python -m eval.ragas_suite --output outputs/ragas_audit_report.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent_graph import _candidate_narrative
from src.copilot_guardrails import validate_sar_narrative
from src.telemetry import span
from src.vector_store import search_all


SCENARIO_THEMES = [
    ("shared device", "multiple customer accounts used the same device"),
    ("transaction velocity", "rapid repeated transfers exceeded normal activity"),
    ("unusual amount", "transaction amount departed from the customer's history"),
    ("account takeover", "new device access preceded an unusual transaction"),
    ("cross-border activity", "cross-border activity differed from the account profile"),
]


def build_scenarios() -> list[dict[str, Any]]:
    """Create a fixed, repeatable suite of 50 synthetic SAR investigation queries."""
    scenarios = []
    for theme_index, (theme, evidence) in enumerate(SCENARIO_THEMES, start=1):
        for case_number in range(1, 11):
            amount = float(100 + theme_index * 375 + case_number * 83)
            transaction = {
                "transaction_id": f"EVAL-{theme_index:02d}-{case_number:02d}",
                "customer_id": f"EVAL-C-{theme_index:02d}-{case_number:02d}",
                "merchant_id": f"EVAL-M-{theme_index:02d}",
                "transaction_amount": amount,
                "timestamp": f"2026-01-{case_number:02d}T12:00:00Z",
                "hour": 12,
                "is_foreign": theme_index == 5,
                "is_new_device": theme_index == 4,
                "vpn_detected": False,
                "jurisdiction": "US",
            }
            query = (f"Investigate a synthetic SAR scenario involving {theme}: {evidence}. "
                     f"Case {case_number:02d}; transaction amount ${amount:,.2f}.")
            reference = (f"The investigation concerns {theme}. Relevant evidence describes {evidence}. "
                         "The activity requires review against the customer profile and source records.")
            scenarios.append({"id": transaction["transaction_id"], "query": query,
                              "transaction": transaction, "reference": reference})
    if len(scenarios) != 50:
        raise RuntimeError("RAGAS SAR scenario suite must contain exactly 50 cases")
    return scenarios


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return None
    return value


def collect_samples() -> tuple[list[dict], list[dict]]:
    rows, guardrail_audit = [], []
    for scenario in build_scenarios():
        with span("eval.ragas.scenario", {"eval.case_id": scenario["id"]}):
            cases = search_all(scenario["query"], top_k=5)
            state = {
                "transaction": scenario["transaction"],
                "fraud_probability": 0.75,
                "retrieved_cases": cases,
                "graph_intel": {}, "fraud_rings": [], "shap_factors": [],
            }
            answer = _candidate_narrative(state)
            validation = validate_sar_narrative(answer, cases, scenario["transaction"])
            guardrail_audit.append({"id": scenario["id"], "valid": validation["valid"],
                                    "errors": validation["errors"], "checks": validation["checks"]})
            if not validation["valid"]:
                answer = ("Automated SAR narrative withheld because generated assertions did not pass "
                          "citation, grounding, or statutory validation. Analyst review is required "
                          "before the SAR narrative can be completed.")
            rows.append({
                "user_input": scenario["query"],
                "retrieved_contexts": [hit.get("text", "") for hit in cases],
                "response": answer,
                "reference": scenario["reference"],
            })
    return rows, guardrail_audit


def run_evaluation(output_path: str = "outputs/ragas_audit_report.json") -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY for RAGAS evaluator and embedding calls")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY to generate Copilot SAR answers")
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    rows, guardrail_audit = collect_samples()
    dataset = Dataset.from_list(rows)
    judge = ChatOpenAI(model=os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini"), temperature=0)
    embeddings = OpenAIEmbeddings(model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"))
    with span("eval.ragas.metrics", {"eval.sample_count": len(rows)}):
        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            llm=judge,
            embeddings=embeddings,
        )
    scored = result.to_pandas()
    metric_names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    aggregate = scored[metric_names].mean(numeric_only=True).to_dict()
    per_sample = scored.to_dict(orient="records")
    report = {
        "suite": "fraudguard_sar_ragas",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "metrics": _json_safe(aggregate),
        "per_sample": _json_safe(per_sample),
        "writer_guardrail_audit": guardrail_audit,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/ragas_audit_report.json")
    args = parser.parse_args()
    report = run_evaluation(args.output)
    print(f"Wrote {report['sample_count']} SAR scenario results to {args.output}")


if __name__ == "__main__":
    main()
