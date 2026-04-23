"""
Simple password-based auth for Streamlit.
Password is stored in Streamlit secrets as APP_PASSWORD.
"""
import streamlit as st
import os
import hashlib


def _get_password():
    try:
        val = st.secrets.get("APP_PASSWORD")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get("APP_PASSWORD", "admin123")


def check_auth() -> bool:
    """Returns True if user is authenticated. Shows login form if not."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown("## 🔐 Login")
    st.caption("Enter the app password to continue.")

    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        expected = _get_password()
        if password == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password.")

    return False


def logout():
    st.session_state.authenticated = False
    st.rerun()
