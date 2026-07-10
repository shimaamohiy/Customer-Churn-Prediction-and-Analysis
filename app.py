"""
KKBox Customer Churn Prediction — Home Page
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from theme import init_theme, get_theme, inject_global_css, render_theme_toggle, dash_bar, section_header, mini_card

st.set_page_config(
    page_title="KKBox Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_theme()
c = get_theme()
inject_global_css(c)
render_theme_toggle()

dash_bar(
    "KKBox Customer Churn Prediction Engine",
    "Enterprise Machine Learning Platform for Proactive Retention",
    c["teal"]
)

col1, col2, col3, col4 = st.columns(4)
with col1: mini_card("ROC-AUC Score",   "0.9926",    c["teal"])
with col2: mini_card("Churn Recall",    "95.6%",     c["success"])
with col3: mini_card("Champion Model",  "LightGBM",  c["primary"])
with col4: mini_card("Baseline Churn",  "6.39%",     c["warning"])

section_header("Business Context & ROI", c["primary"])
st.markdown(f"""
<div style="
    background:{c['card']};
    border-left: 6px solid {c['purple']};
    border: 1px solid {c['border']};
    border-radius: 12px;
    padding: 1.5rem 2rem;
    line-height: 1.8;
    color: {c['text']};
    margin-bottom: 1.5rem;
">
<b>Business Objective:</b> Predict whether a KKBox subscriber will cancel their membership <b>30 days before expiration</b>, enabling a proactive retention window.
<br><br>
<b>Financial Impact:</b> Each retained subscriber saves <b style="color:{c['teal']};">$10–$15 / month</b> in recurring revenue. The primary goal is to reduce overall churn by <b>20%</b>, and the model was intentionally tuned to maximize <b>Recall</b> over Precision because failing to flag a churner (False Negative) costs real money, while a false alarm only costs a small retention offer.
<br><br>
<b>Key Constraints from Phase 1:</b>
<ul>
  <li>Prediction must happen <b>30 days before</b> subscription expiry to allow marketing to act.</li>
  <li>Data includes 30 GB+ of raw logs — required chunked processing via Dask.</li>
  <li>Severe class imbalance: only <b>6.39%</b> of users actually churned.</li>
</ul>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    section_header("Feature Glossary", c["success"])
    data = {
        "Feature": [
            "total_secs", "latest_auto_renew", "num_100", "skip_ratio",
            "days_since_last_transaction", "days_until_expiration",
            "total_cancellations", "average_under_overpayment"
        ],
        "Description": [
            "Total seconds of music played (engagement proxy)",
            "Whether subscription auto-renews (strongest retention signal)",
            "Songs completed 100% — indicates deep engagement",
            "Fraction of songs skipped early — indicates dissatisfaction",
            "Days since last payment — high = critical churn signal",
            "Days remaining on current plan",
            "Count of prior manual cancellations",
            "Billing gap: negative means payment failed"
        ]
    }
    import pandas as pd
    st.table(pd.DataFrame(data))

with col_b:
    section_header("CRISP-DM Methodology", c["purple"])
    for i, step in enumerate([
        ("1. Business Understanding", "Aligned KPIs: Recall > Accuracy. Defined 30-day churn window."),
        ("2. Data Understanding",     "Explored 400M+ daily log rows, identified class imbalance."),
        ("3. Data Preparation",       "Aggregated logs → user-level features. Temporal train/test split."),
        ("4. Modeling",               "Benchmarked: LogReg, RF, GBM, LightGBM (tuned)."),
        ("5. Evaluation",             "LightGBM won with AUC=0.9926 and Recall=95.6%."),
        ("6. Deployment",             "This interactive Streamlit dashboard with Dark/Light mode."),
    ], 1):
        title, desc = step
        st.markdown(f"""
        <div style="background:{c['card']}; border:1px solid {c['border']}; border-radius:8px;
                    padding:0.75rem 1rem; margin-bottom:0.5rem;">
            <b style="color:{c['primary']};">{title}</b><br>
            <span style="color:{c['muted']}; font-size:0.93rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

section_header("Platform Navigation", c["warning"])
nav_items = [
    ("Single Prediction",   "Enter one subscriber's data → real-time risk gauge + AI-driven retention recommendations.", c["primary"]),
    ("Batch Prediction",    "Upload a CSV → bulk churn scores + downloadable report.",                                  c["purple"]),
    ("Model Info",          "LightGBM hyperparameters, feature importance chart, trade-off analysis.",                  c["pink"]),
    ("Project Dashboards",  "4 interactive Plotly charts: Auto-Renew impact, Engagement, Recency, Heatmap.",           c["success"]),
    ("Model Comparison",    "Benchmark all 5 algorithms by ROC-AUC, Recall, Accuracy & Training Time.",                c["warning"]),
]
cols = st.columns(len(nav_items))
for col, (name, desc, accent) in zip(cols, nav_items):
    with col:
        st.markdown(f"""
        <div style="background:{c['card']}; border-top:4px solid {accent};
                    border:1px solid {c['border']}; border-radius:10px;
                    padding:1rem; height:140px;">
            <b style="color:{accent}; font-size:1rem;">{name}</b><br>
            <span style="color:{c['muted']}; font-size:0.88rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)
