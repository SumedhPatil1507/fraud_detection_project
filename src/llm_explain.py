"""
LLM-powered fraud explanation using Groq (llama-3).
Takes SHAP values + transaction details and returns a plain-English explanation.
"""
import os
import numpy as np
import pandas as pd


def _get_client():
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY") or \
                  _get_streamlit_secret()
        if not api_key:
            return None
        return Groq(api_key=api_key)
    except Exception:
        return None


def _get_streamlit_secret():
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        return None


def get_top_shap_factors(model, input_df: pd.DataFrame,
                          feature_names: list, top_n: int = 6) -> list[dict]:
    """Extract top SHAP contributors for a single prediction."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        vals = shap_values[0] if shap_values.ndim > 1 else shap_values
        indices = np.argsort(np.abs(vals))[-top_n:][::-1]
        return [
            {
                "feature": feature_names[i],
                "value": round(float(input_df.iloc[0, i]), 4),
                "shap_impact": round(float(vals[i]), 4),
                "direction": "increases" if vals[i] > 0 else "decreases",
            }
            for i in indices
        ]
    except Exception:
        return []


def explain_prediction(
    fraud_probability: float,
    transaction: dict,
    shap_factors: list[dict],
    threshold: float = 0.3,
) -> str:
    """
    Call Groq LLM to generate a plain-English fraud explanation.
    Falls back to a rule-based explanation if LLM is unavailable.
    """
    client = _get_client()

    # Build context string
    factors_text = "\n".join([
        f"- {f['feature']} = {f['value']} "
        f"({f['direction']} fraud risk by {abs(f['shap_impact']):.3f})"
        for f in shap_factors
    ]) if shap_factors else "No SHAP data available."

    verdict = "FRAUD" if fraud_probability >= threshold else "LEGITIMATE"

    prompt = f"""You are a fraud analyst AI. Explain the following transaction decision in 3-4 clear sentences for a non-technical risk officer.

Transaction verdict: {verdict} (fraud probability: {fraud_probability:.1%})
Transaction amount: ${transaction.get('transaction_amount', 'N/A')}
Distance from home: {transaction.get('distance_from_home_km', 'N/A')} km
Hour of day: {transaction.get('hour', 'N/A')}
Foreign transaction: {'Yes' if transaction.get('is_foreign') else 'No'}
New device: {'Yes' if transaction.get('is_new_device') else 'No'}
VPN detected: {'Yes' if transaction.get('vpn_detected') else 'No'}

Top factors driving this decision (SHAP analysis):
{factors_text}

Write a concise, professional explanation. Start with the verdict. Mention the 2-3 most important risk factors. End with a recommended action."""

    if client:
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return _fallback_explanation(fraud_probability, transaction,
                                         shap_factors, threshold, str(e))
    else:
        return _fallback_explanation(fraud_probability, transaction,
                                     shap_factors, threshold)


def _fallback_explanation(fraud_probability, transaction,
                           shap_factors, threshold, error=None) -> str:
    """Rule-based fallback when LLM is unavailable."""
    verdict = "flagged as HIGH RISK" if fraud_probability >= threshold else "classified as LEGITIMATE"
    top = shap_factors[:2] if shap_factors else []
    factors = ", ".join([f"{f['feature']} ({f['direction']} risk)" for f in top])
    note = f" (LLM unavailable: {error})" if error else " (Set GROQ_API_KEY to enable AI explanations)"
    return (
        f"This transaction was {verdict} with a fraud probability of "
        f"{fraud_probability:.1%}. "
        f"{'Key risk drivers: ' + factors + '.' if factors else ''}"
        f"{note}"
    )


def chat_with_analyst(question: str, model_metrics: dict) -> str:
    """
    Simple analyst chatbot — answers questions about model performance.
    """
    client = _get_client()
    if not client:
        return "LLM unavailable. Set GROQ_API_KEY in Streamlit secrets."

    context = "\n".join([f"{k}: {v}" for k, v in model_metrics.items()])
    prompt = f"""You are a data science assistant. Answer the following question about a fraud detection model.

Model metrics:
{context}

Question: {question}

Give a concise, accurate answer in 2-3 sentences."""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"
