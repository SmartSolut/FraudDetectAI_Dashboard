"""
Authentication System
====================
Simple authentication system for admin and regular users
"""

import streamlit as st
try:
    from database import DEFAULT_ADMIN
except ImportError:
    from .database import DEFAULT_ADMIN

def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'username' not in st.session_state:
        st.session_state.username = None


def login(username: str, password: str) -> bool:
    """Login user"""
    # Check admin credentials
    if username == DEFAULT_ADMIN["username"] and password == DEFAULT_ADMIN["password"]:
        st.session_state.authenticated = True
        st.session_state.user_role = "admin"
        st.session_state.username = username
        return True
    
    # In production, check against user database
    # For now, allow any user (non-admin)
    if username and password:
        st.session_state.authenticated = True
        st.session_state.user_role = "user"
        st.session_state.username = username
        return True
    
    return False


def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.username = None


def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)


def is_admin() -> bool:
    """Check if user is admin"""
    return st.session_state.get("user_role") == "admin"


def get_username() -> str:
    """Get current username"""
    return st.session_state.get("username", "")


def require_auth():
    """Require authentication - redirect to login if not authenticated"""
    if not is_authenticated():
        st.error("⚠️ يرجى تسجيل الدخول أولاً")
        st.stop()


def require_admin():
    """Require admin role"""
    require_auth()
    if not is_admin():
        st.error("⚠️ يجب أن تكون مسؤولاً للوصول إلى هذه الصفحة")
        st.stop()

