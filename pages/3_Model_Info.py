"""
Page 3: Model Information
"""
import streamlit as st
import joblib
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Information", layout="wide")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_churn_model.pkl")

st.markdown("""
    <style>
    .dash-bar {
        background-color: var(--secondary-background-color);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #EC4899;
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
        color: #EC4899;
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--text-color);
        padding-bottom: 0.3rem;
        display: inline-block;
    }
    .info-box {
        background-color: var(--secondary-background-color);
        border-left: 5px solid #EC4899;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: var(--text-color);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

st.markdown("""
<div class="dash-bar">
    <h2 class="dash-title">Model Technical Deep-Dive</h2>
    <p style="color:var(--text-color); opacity:0.8; margin:0;">Detailed breakdown of the LightGBM champion architecture and feature engineering.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">Champion Architecture: LightGBM (Tuned)</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Core Performance Metrics**")
    metrics = pd.DataFrame({
        "Metric": ["ROC-AUC", "Recall", "Accuracy", "Precision", "F1-Score"],
        "Achieved Value": ["0.9926", "0.9561", "0.9445", "0.5368", "0.6876"],
        "Target Baseline": [">= 0.90", ">= 0.82", "N/A", ">= 0.80", ">= 0.80"],
        "Validation": ["Pass", "Pass", "Pass", "Fail", "Fail"]
    })
    st.dataframe(metrics, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**Business Trade-off Analysis**")
    st.markdown("""
    <div class="info-box">
    <b>Strategic Decision:</b> Why did we accept a low Precision (53%)?<br><br>
    The algorithm was intentionally optimized to maximize <b>Recall (95.6%)</b>. 
    In the subscription music streaming industry, <b>False Negatives</b> result in a direct loss of recurring revenue. 
    Conversely, <b>False Positives</b> simply result in a loyal customer receiving a retention offer, which typically increases customer satisfaction at a minor marketing cost. Prioritizing Recall perfectly aligns with KKBox's financial objectives.
    </div>
    """, unsafe_allow_html=True)

model = load_model()
if model:
    st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)
    st.markdown("The chart below dynamically extracts the exact internal feature weights from the saved LightGBM model. It illustrates which data points most heavily influence the algorithm's decision-making process.")
    
    if hasattr(model, "feature_importances_") and hasattr(model, "feature_name_"):
        importances = model.feature_importances_
        features = list(model.feature_name_)
        
        feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=False).head(10)
        
        fig = px.bar(
            feat_df, x="Importance", y="Feature", orientation='h',
            color="Importance", color_continuous_scale="Pinkyl",
            title="Top 10 Most Predictive Features for Churn"
        )
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'}, 
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-header">Hyperparameter Tuning (RandomizedSearchCV)</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([1, 2])
with col_a:
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 63,
        "is_unbalance": True,
        "random_state": 42
    }
    st.json(params)
with col_b:
    st.markdown("""
    **Algorithm Mechanics:**
    - `n_estimators = 300`: The model builds 300 sequential decision trees to correct the errors of the previous ones.
    - `learning_rate = 0.05`: A slow learning rate prevents the model from overfitting to the training data.
    - `is_unbalance = True`: This is the secret weapon. Because only 6% of users churned, this parameter forces the model to heavily penalize errors made on the minority class (the churners), ensuring our Recall hits 95%.
    - `num_leaves = 63`: Allows for highly complex interactions between features.
    """)
