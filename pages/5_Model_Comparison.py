"""
Page 5: Model Comparison Dashboard
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from theme import init_theme, get_theme, inject_global_css, render_theme_toggle, dash_bar, section_header, insight_card

st.set_page_config(page_title="Model Comparison", layout="wide")

init_theme()
c = get_theme()
inject_global_css(c)
render_theme_toggle()

dash_bar(
    "Algorithm Benchmark & Comparison",
    "Interactive evaluation of all machine learning architectures tested in Phase 4.",
    c["warning"]
)

data = {
    "Algorithm": ["LightGBM (Tuned)", "LightGBM (Base)", "Random Forest", "Gradient Boosting", "Logistic Regression"],
    "ROC-AUC":   [0.9926, 0.9920, 0.9899, 0.9893, 0.9140],
    "Recall":    [0.9561, 0.9420, 0.9200, 0.9150, 0.7800],
    "Accuracy":  [0.9445, 0.9407, 0.9844, 0.9843, 0.9160],
    "Training Time (s)": [45.2, 38.5, 312.4, 256.0, 15.3]
}
df = pd.DataFrame(data)

section_header("Interactive Metric Comparison", c["warning"])
metric = st.selectbox("Select metric:", ["ROC-AUC","Recall","Accuracy","Training Time (s)"])

fig = px.bar(
    df, x="Algorithm", y=metric, color="Algorithm",
    text_auto=".4f" if metric != "Training Time (s)" else ".1f",
    color_discrete_sequence=[c["teal"], c["primary"], c["success"], c["warning"], c["danger"]]
)
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    margin=dict(l=20,r=20,t=40,b=20)
)
st.plotly_chart(fig, use_container_width=True)

INSIGHTS = {
    "ROC-AUC": (
        "LightGBM (Tuned) leads at 0.9926 vs Logistic Regression at 0.9140.",
        "Tree-based gradient boosting captures the complex, non-linear interactions between behavioral features (engagement × billing × recency) that linear models physically cannot."
    ),
    "Recall": (
        "LightGBM (Tuned) achieves 95.6% Recall — catching 19 out of 20 churners.",
        "This is the primary business metric. Using is_unbalance=True forces the model to heavily penalize missed churners, perfectly aligning model behavior with revenue protection goals."
    ),
    "Accuracy": (
        "Random Forest shows 98.4% accuracy — but this is misleading.",
        "With only 6.39% churn, a model that always says 'Not Churning' gets 93.6% accuracy automatically. Random Forest exploits this — its Recall is only 92% vs LightGBM's 95.6%."
    ),
    "Training Time (s)": (
        "LightGBM trained in 45s vs Random Forest's 312s — 7x faster.",
        "LightGBM's histogram-based tree building algorithm is specifically designed for large-scale datasets. Given the 400M+ row KKBox dataset, speed is a critical operational requirement."
    ),
}

title, body = INSIGHTS[metric]
insight_card(f"Why does this metric matter? — {metric}", f"<b>{title}</b><br><br>{body}", c["warning"])

st.divider()
section_header("Radar Chart: Holistic Multi-Metric View", c["purple"])
st.markdown(f"<p style='color:{c['muted']}'>Three-way comparison across ROC-AUC, Recall, and Accuracy for the top 3 algorithms.</p>", unsafe_allow_html=True)

radar_df = pd.DataFrame({
    "r":     [0.9926,0.9561,0.9445, 0.9899,0.9200,0.9844, 0.9140,0.7800,0.9160],
    "theta": ["ROC-AUC","Recall","Accuracy"]*3,
    "Model": ["LightGBM (Tuned)"]*3 + ["Random Forest"]*3 + ["Logistic Regression"]*3
})
fig_r = px.line_polar(radar_df, r="r", theta="theta", color="Model",
                      line_close=True,
                      color_discrete_map={
                          "LightGBM (Tuned)": c["teal"],
                          "Random Forest":    c["success"],
                          "Logistic Regression": c["danger"]
                      })
fig_r.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0.7,1.0]), bgcolor="rgba(0,0,0,0)"),
    paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_r, use_container_width=True)

st.divider()
section_header("Full Leaderboard", c["primary"])
st.table(df)
