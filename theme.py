"""
Shared theme utilities for all pages using Native Streamlit Themes.
"""

import streamlit as st

# Accent colors designed to look good on both Light and Dark backgrounds.
ACCENTS = {
    "primary":    "#3A86FF",
    "success":    "#10B981",
    "danger":     "#EF4444",
    "warning":    "#F59E0B",
    "purple":     "#8B5CF6",
    "pink":       "#EC4899",
    "teal":       "#00D2FF",
    "card":       "var(--secondary-background-color)",
    "bg":         "var(--background-color)",
    "text":       "var(--text-color)",
    "border":     "rgba(128, 128, 128, 0.2)",
    "muted":      "gray",
}

def init_theme():
    pass

def get_theme():
    return ACCENTS

def render_theme_toggle():
    pass

def inject_global_css(c):
    """Clean CSS that relies entirely on Streamlit's native Light/Dark engine."""
    st.markdown("""
    <style>
    /* Remove global overrides to let DataFrames, Inputs, and text adapt automatically */
    </style>
    """, unsafe_allow_html=True)

def dash_bar(title: str, subtitle: str, accent: str):
    st.markdown(f"""
    <div style="
        background-color: var(--secondary-background-color);
        padding: 1.3rem 1.8rem;
        border-radius: 14px;
        border-left: 6px solid {accent};
        border: 1px solid rgba(128,128,128,0.2);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    ">
        <h1 style="
            font-family:'Helvetica Neue',Arial,sans-serif;
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--text-color);
            margin: 0 0 0.3rem 0;
            line-height: 1.2;
        ">{title}</h1>
        <p style="
            font-size: 0.98rem;
            color: gray;
            margin: 0;
        ">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def section_header(text: str, accent: str = None):
    color = accent or ACCENTS['primary']
    st.markdown(f"""
    <div style="
        color: {color};
        font-size: 1.2rem;
        font-weight: 800;
        border-bottom: 2px solid {color};
        padding-bottom: 4px;
        margin: 1.5rem 0 1rem 0;
        display: inline-block;
    ">{text}</div>
    """, unsafe_allow_html=True)

def insight_card(title: str, body: str, accent: str = None, action: str = None):
    color = accent or ACCENTS['primary']
    action_html = f"""
    <div style="
        background: rgba(16,185,129,0.12);
        border-left: 3px solid {ACCENTS['success']};
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-top: 0.75rem;
        font-size: 0.93rem;
        color: var(--text-color);
    "><strong>Business Action:</strong> {action}</div>
    """ if action else ""

    st.markdown(f"""
    <div style="
        background-color: var(--secondary-background-color);
        border-left: 5px solid {color};
        border: 1px solid rgba(128,128,128,0.2);
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    ">
        <div style="font-weight:700; color:{color}; margin-bottom:0.5rem; font-size:1rem;">{title}</div>
        <div style="color:var(--text-color); font-size:0.95rem; line-height:1.65;">{body}</div>
        {action_html}
    </div>
    """, unsafe_allow_html=True)

def mini_card(label: str, value: str, accent: str):
    st.markdown(f"""
    <div style="
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-top: 4px solid {accent};
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    ">
        <div style="font-size:1.7rem; font-weight:800; color:{accent};">{value}</div>
        <div style="font-size:0.82rem; font-weight:600; color:gray; text-transform:uppercase; letter-spacing:0.05em; margin-top:4px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)
