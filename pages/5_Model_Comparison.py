"""
Page 5: Model Comparison Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Comparison", layout="wide")

st.markdown("""
    <style>
    .dash-bar {
        background-color: var(--secondary-background-color);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #F59E0B;
        margin-bottom: 2rem;
    }
    .dash-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--text-color);
        margin: 0;
    }
    .section-header {
        color: #F59E0B;
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--text-color);
        padding-bottom: 0.3rem;
        display: inline-block;
    }
    .insight-card {
        background-color: var(--secondary-background-color);
        border-left: 4px solid #F59E0B;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: var(--text-color);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dash-bar">
    <h2 class="dash-title">Algorithm Benchmark & Comparison</h2>
    <p style="color:var(--text-color); opacity:0.8; margin:0;">An interactive evaluation of machine learning architectures tested during Phase 4 (Modeling).</p>
</div>
""", unsafe_allow_html=True)

# Static data based on Phase 4 modeling results
models_data = {
    "Algorithm": ["LightGBM (Tuned)", "LightGBM (Base)", "Random Forest", "Gradient Boosting", "Logistic Regression"],
    "ROC-AUC": [0.9926, 0.9920, 0.9899, 0.9893, 0.9140],
    "Recall": [0.9561, 0.9420, 0.9200, 0.9150, 0.7800],
    "Accuracy": [0.9445, 0.9407, 0.9844, 0.9843, 0.9160],
    "Training Time (s)": [45.2, 38.5, 312.4, 256.0, 15.3]
}
df = pd.DataFrame(models_data)

st.markdown('<div class="section-header">Performance Metrics Comparison</div>', unsafe_allow_html=True)

# Interactive selector for metric
metric_selected = st.selectbox(
    "Select a Metric to Compare:",
    options=["ROC-AUC", "Recall", "Accuracy", "Training Time (s)"],
    index=0
)

# Dynamic Bar Chart
fig = px.bar(
    df, x="Algorithm", y=metric_selected, 
    color="Algorithm",
    title=f"Comparison by {metric_selected}",
    text_auto='.4f' if metric_selected != "Training Time (s)" else '.1f',
    color_discrete_sequence=["#00D2FF", "#3A86FF", "#10B981", "#F59E0B", "#EF4444"]
)
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)", 
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# Dynamic Insights based on selected metric
st.markdown('<div class="section-header">Evaluation Insights</div>', unsafe_allow_html=True)

if metric_selected == "ROC-AUC":
    st.markdown("""
    <div class="insight-card">
        <strong>ROC-AUC Analysis:</strong> LightGBM (Tuned) achieved the highest ROC-AUC score (0.9926), significantly outperforming Logistic Regression (0.9140). This proves that tree-based gradient boosting is far superior at capturing the complex, non-linear relationships in user listening behavior.
    </div>
    """, unsafe_allow_html=True)
elif metric_selected == "Recall":
    st.markdown("""
    <div class="insight-card">
        <strong>Recall Analysis (Primary Business Objective):</strong> Tuned LightGBM reached an exceptional 95.6% Recall. Since the business objective is to identify as many potential churners as possible (preventing $10-$15 monthly revenue loss per user), this metric was heavily prioritized over raw accuracy.
    </div>
    """, unsafe_allow_html=True)
elif metric_selected == "Accuracy":
    st.markdown("""
    <div class="insight-card">
        <strong>Accuracy Paradox:</strong> Notice that Random Forest has a higher raw Accuracy (98.4%) than LightGBM (94.4%). However, because the dataset is highly imbalanced (only 6% churn), Accuracy is a misleading metric. Random Forest achieves high accuracy by simply predicting "Not Churning" most of the time, causing a catastrophic drop in Recall.
    </div>
    """, unsafe_allow_html=True)
elif metric_selected == "Training Time (s)":
    st.markdown("""
    <div class="insight-card">
        <strong>Computational Efficiency:</strong> LightGBM trained in just 45 seconds, whereas Random Forest took over 300 seconds. LightGBM's histogram-based algorithm makes it vastly superior for handling the massive scale of KKBox's dataset.
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown('<div class="section-header">Radar Chart: Holistic View</div>', unsafe_allow_html=True)
st.markdown("Visualizing the trade-offs between ROC-AUC, Recall, and Accuracy across the top 3 models.")

radar_data = pd.DataFrame(dict(
    r=[0.9926, 0.9561, 0.9445, 0.9899, 0.9200, 0.9844, 0.9140, 0.7800, 0.9160],
    theta=['ROC-AUC','Recall','Accuracy', 'ROC-AUC','Recall','Accuracy', 'ROC-AUC','Recall','Accuracy'],
    Model=['LightGBM (Tuned)', 'LightGBM (Tuned)', 'LightGBM (Tuned)', 'Random Forest', 'Random Forest', 'Random Forest', 'Logistic Regression', 'Logistic Regression', 'Logistic Regression']
))

fig_radar = px.line_polar(
    radar_data, r='r', theta='theta', color='Model', line_close=True,
    color_discrete_sequence=["#00D2FF", "#10B981", "#EF4444"]
)
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0.7, 1.0]),
        bgcolor="rgba(0,0,0,0)"
    ),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_radar, use_container_width=True)
