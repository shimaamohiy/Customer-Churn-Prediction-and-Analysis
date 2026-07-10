"""
KKBox Customer Churn Prediction - Streamlit App
Main entry point
"""
import streamlit as st

st.set_page_config(
    page_title="KKBox Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adaptive Light/Dark CSS using Native Variables
st.markdown("""
    <style>
    .dash-bar {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid var(--primary-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .dash-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-color);
        margin: 0;
    }
    .dash-subtitle {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.8;
        margin: 0;
    }
    .section-header {
        color: var(--text-color);
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-color);
        display: inline-block;
    }
    .story-text {
        font-size: 1.1rem;
        line-height: 1.7;
        color: var(--text-color);
        background-color: var(--secondary-background-color);
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid var(--primary-color);
        margin-bottom: 2rem;
    }
    .highlight {
        color: var(--primary-color);
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dash-bar">
    <h1 class="dash-title">KKBox Customer Churn Prediction Engine</h1>
    <p class="dash-subtitle">Enterprise Machine Learning Platform for Proactive Retention</p>
</div>
""", unsafe_allow_html=True)

# Using native Streamlit Metrics for seamless Light/Dark mode support
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("ROC-AUC Score", "0.9926", delta="Excellent", delta_color="normal")
with col2: st.metric("Churn Recall", "95.6%", delta="High Priority", delta_color="normal")
with col3: st.metric("Champion Model", "LightGBM")
with col4: st.metric("Baseline Churn", "6.39%", delta="Imbalanced", delta_color="inverse")

st.markdown('<div class="section-header">Business Context & Cost-Benefit Analysis</div>', unsafe_allow_html=True)
st.markdown("""
<div class="story-text">
<b>Business Objective:</b> To predict whether a KKBox user will cancel their membership after the current subscription period ends. Prediction must be made 30 days prior to expiry to allow for proactive retention campaigns.
<br><br>
<b>Financial Impact (ROI):</b> Each retained subscriber is worth approximately <span class="highlight">$10-$15 per month</span> in recurring revenue. By accurately predicting churn 30 days early, we can significantly reduce subscriber acquisition costs. A primary success criterion of this project is to reduce overall subscription churn by 20%.
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("### Feature Glossary")
    st.markdown("""
    The dataset was engineered from millions of rows of user activity. Here is what the key features mean:
    - `total_secs`: Total seconds of music listened per day. A strong indicator of engagement.
    - `payment_plan_days`: Number of days in the subscription plan.
    - `is_auto_renew`: Whether the subscription renews automatically (crucial for retention).
    - `num_100`: Number of songs played to 100% completion.
    - `skip_ratio`: Calculated ratio of songs skipped before 25% duration.
    - `bd`: User age (filtered strictly between 10-80).
    """)

with col_b:
    st.markdown("### The CRISP-DM Methodology")
    st.markdown("""
    1. **Business Understanding:** Defined ROI, costs, constraints (e.g., memory limits), and success criteria.
    2. **Data Understanding:** Addressed the massive scale of `user_logs.csv` and severe class imbalance (6% churn rate).
    3. **Data Preparation:** Aggregated daily logs into user-level features (mean, sum, count) to prevent data leakage.
    4. **Modeling:** Evaluated Logistic Regression, Random Forest, Gradient Boosting, and LightGBM.
    5. **Evaluation:** Prioritized AUC-ROC and Recall. LightGBM achieved exceptional performance due to `is_unbalance=True`.
    6. **Deployment:** This theme-adaptive interactive dashboard serving predictions and generating actionable insights.
    """)
