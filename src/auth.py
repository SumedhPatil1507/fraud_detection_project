"""
Auth for Streamlit.
Set APP_PASSWORD in Streamlit secrets to enable password protection.
If not set, the app is open to everyone (good for demos/portfolios).
"""
import streamlit as st
import os


def _get_password():
    try:
        val = st.secrets.get("APP_PASSWORD")
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.environ.get("APP_PASSWORD", "")  # empty = no password required


def check_auth() -> bool:
    if st.session_state.get("authenticated"):
        return True

    expected = _get_password()

    # No password configured — let everyone in
    if not expected:
        st.session_state.authenticated = True
        return True

    st.markdown("## 🔐 Login")
    st.caption("Enter the app password to continue.")

    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        if password.strip() == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password.")

    return False


def logout():
    st.session_state.authenticated = False
    st.rerun()
