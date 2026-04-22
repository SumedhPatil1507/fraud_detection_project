"""Human-in-the-Loop (HITL) review queue using session state."""
import pandas as pd
import streamlit as st
from datetime import datetime


def add_to_review_queue(txn: dict, prob: float, reason: str = ""):
    if "review_queue" not in st.session_state:
        st.session_state.review_queue = []
    st.session_state.review_queue.append({
        "timestamp": datetime.utcnow().isoformat(),
        "transaction_amount": txn.get("transaction_amount"),
        "distance_from_home_km": txn.get("distance_from_home_km"),
        "hour": txn.get("hour"),
        "fraud_probability": round(prob, 4),
        "reason": reason,
        "status": "Pending",
        "analyst_decision": None,
    })


def get_review_queue() -> pd.DataFrame:
    if "review_queue" not in st.session_state or not st.session_state.review_queue:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.review_queue)


def render_hitl_tab():
    st.subheader("👤 Human-in-the-Loop Review Queue")
    st.markdown("Transactions flagged for manual analyst review.")
    queue_df = get_review_queue()
    if queue_df.empty:
        st.info("No transactions pending review. High-risk predictions are auto-queued.")
        return

    st.metric("Pending Reviews", len(queue_df[queue_df['status'] == 'Pending']))

    for i, row in queue_df.iterrows():
        with st.expander(
            f"TXN #{i+1} | ${row['transaction_amount']:.2f} | "
            f"Prob: {row['fraud_probability']:.1%} | {row['status']}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Amount", f"${row['transaction_amount']:.2f}")
            c2.metric("Fraud Prob", f"{row['fraud_probability']:.1%}")
            c3.metric("Hour", row['hour'])
            st.caption(f"Reason: {row['reason']}")

            col1, col2 = st.columns(2)
            if col1.button("✅ Confirm Fraud", key=f"confirm_{i}"):
                st.session_state.review_queue[i]['status'] = 'Reviewed'
                st.session_state.review_queue[i]['analyst_decision'] = 'Fraud'
                st.rerun()
            if col2.button("❌ Mark Legitimate", key=f"legit_{i}"):
                st.session_state.review_queue[i]['status'] = 'Reviewed'
                st.session_state.review_queue[i]['analyst_decision'] = 'Legitimate'
                st.rerun()
