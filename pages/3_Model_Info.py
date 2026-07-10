"""
Page 3: Model Technical Deep-Dive
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from theme import init_theme, get_theme, inject_global_css, render_theme_toggle, dash_bar, section_header, insight_card

st.set_page_config(page_title="Model Info", layout="wide")

init_theme()
c = get_theme()
inject_global_css(c)
render_theme_toggle()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_churn_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

dash_bar(
    "Model Technical Deep-Dive",
    "LightGBM architecture, hyperparameter rationale, feature importance, and trade-off analysis.",
    c["pink"]
)

section_header("Performance Metrics: Champion vs. Targets", c["pink"])
col1, col2 = st.columns(2)
with col1:
    metrics = pd.DataFrame({
        "Metric":   ["ROC-AUC", "Recall", "Accuracy", "Precision", "F1-Score"],
        "Achieved": ["0.9926",  "0.9561", "0.9445",   "0.5368",   "0.6876"],
        "Target":   [">= 0.90", ">= 0.82","N/A",      ">= 0.80",  ">= 0.80"],
        "Status":   ["Pass",    "Pass",    "Pass",     "Fail",     "Fail"]
    })
    st.table(metrics)

with col2:
    insight_card(
        "Strategic Decision: Why accept low Precision?",
        "The model was intentionally tuned to maximize Recall (95.6%) over Precision (53.7%).<br><br>"
        "In subscription streaming, a <b>False Negative</b> (missing a churner) = lost $10-15/month forever.<br>"
        "A <b>False Positive</b> (flagging a loyal user as churner) = they receive a discount offer → usually increases satisfaction.<br><br>"
        "Therefore, maximizing Recall is the correct business decision.",
        c["pink"]
    )

model = load_model()
if model:
    section_header("Live Feature Importance (from saved model)", c["primary"])
    st.markdown(f"<p style='color:{c['muted']}'>Dynamically extracted from the saved LightGBM model weights — these are the exact features driving predictions.</p>", unsafe_allow_html=True)

    if hasattr(model, "feature_importances_") and hasattr(model, "feature_name_"):
        feat_df = pd.DataFrame({
            "Feature": model.feature_name_,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(12)

        fig = px.bar(
            feat_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            text_auto=True
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

section_header("Hyperparameter Tuning (RandomizedSearchCV)", c["success"])
col_a, col_b = st.columns([1, 2])
with col_a:
    st.json({
        "n_estimators": 300, "learning_rate": 0.05,
        "max_depth": 8, "num_leaves": 63,
        "is_unbalance": True, "random_state": 42
    })
with col_b:
    for param, reason in [
        ("`n_estimators = 300`",   "Builds 300 sequential trees, each correcting errors from the previous."),
        ("`learning_rate = 0.05`", "Slow rate prevents overfitting — the model learns conservatively."),
        ("`is_unbalance = True`",  "SECRET WEAPON: Heavily penalizes misclassifying the rare 6% churn class, boosting Recall to 95%."),
        ("`num_leaves = 63`",      "Allows complex, high-order feature interactions — key for capturing behavioral patterns."),
        ("`max_depth = 8`",        "Caps tree depth to prevent memorizing training noise (overfitting)."),
    ]:
        st.markdown(f"""
        <div style='background:{c["card"]}; border:1px solid {c["border"]}; border-radius:8px;
                    padding:0.6rem 1rem; margin-bottom:0.4rem;'>
            <code style='color:{c["primary"]};'>{param}</code> —
            <span style='color:{c["muted"]}; font-size:0.92rem;'>{reason}</span>
        </div>
        """, unsafe_allow_html=True)
