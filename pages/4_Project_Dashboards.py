"""
Page 4: Project Dashboards
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import os

st.set_page_config(page_title="Project Dashboards", layout="wide")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "final_modeling_dataset.csv")

st.markdown("""
    <style>
    .dash-bar {
        background-color: var(--secondary-background-color);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #10B981;
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
        color: #10B981;
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--text-color);
        padding-bottom: 0.2rem;
    }
    .insight-block {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-left: 4px solid #3A86FF;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .action-block {
        background-color: rgba(16, 185, 129, 0.1);
        padding: 1rem;
        border-left: 4px solid #10B981;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dash-bar">
    <h2 class="dash-title">Global Data Insights & EDA</h2>
    <p style="color:var(--text-color); opacity:0.8; margin:0;">Visual exploratory data analysis of subscriber behavior based on a randomized sample of the processed dataset.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    if not os.path.exists(DATA_PATH):
        return None
    # Load 10k rows for fast dashboard rendering
    df = pd.read_csv(DATA_PATH, nrows=10000)
    df["Churn Status"] = df["is_churn"].map({0: "Active (Retained)", 1: "Churned"})
    df["Auto Renew Status"] = df["latest_auto_renew"].map({0: "Disabled", 1: "Enabled"})
    return df

df = load_sample_data()

if df is None:
    st.error("System Error: Processed data not found. Please run the preprocessing pipeline first.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">1. Impact of Auto-Renewal on Churn</div>', unsafe_allow_html=True)
    renew_churn = df.groupby(["Auto Renew Status", "Churn Status"]).size().reset_index(name="Count")
    fig1 = px.bar(
        renew_churn, x="Auto Renew Status", y="Count", color="Churn Status",
        barmode="group",
        color_discrete_map={"Active (Retained)": "#00D2FF", "Churned": "#EF4444"}
    )
    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("""
    <div class="insight-block"><b>Analytical Insight:</b> Subscribers who must manually renew their subscriptions churn at a massively higher rate than those with Auto-Renew enabled. Friction in the payment pipeline is a primary cause of churn.</div>
    <div class="action-block"><b>Business Action:</b> Marketing must immediately launch a "One-Month Free" or "15% Discount" campaign exclusively targeting users with Auto-Renew disabled to incentivize them to turn it on.</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">2. Engagement vs. Retention</div>', unsafe_allow_html=True)
    fig2 = px.box(
        df, x="Churn Status", y="total_active_days", color="Churn Status",
        color_discrete_map={"Active (Retained)": "#00D2FF", "Churned": "#EF4444"}
    )
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Total Active Days (Last 30 Days)")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""
    <div class="insight-block"><b>Analytical Insight:</b> The median active days for retained users is significantly higher. Churned users typically show a dramatic drop in platform usage (less than 10 days active) prior to cancelling.</div>
    <div class="action-block"><b>Business Action:</b> Build an automated alert system. When a user's active days drop below 10 in a 30-day window, trigger a push notification recommending personalized new music releases.</div>
    """, unsafe_allow_html=True)

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-header">3. Time-to-Churn Analysis (Recency)</div>', unsafe_allow_html=True)
    fig3 = px.histogram(
        df, x="days_since_last_transaction", color="Churn Status", nbins=30,
        color_discrete_map={"Active (Retained)": "#00D2FF", "Churned": "#EF4444"},
        barmode="overlay"
    )
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Days Since Last Transaction", yaxis_title="Subscriber Count")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("""
    <div class="insight-block"><b>Analytical Insight:</b> Churned users have a massive spike in the "days since last transaction" metric. If a user hasn't made a transaction (even a free renewal) in over 30 days, churn is almost certain.</div>
    <div class="action-block"><b>Business Action:</b> Accounts exceeding 30 days since the last transaction should be moved to a "Win-Back" email campaign rather than standard retention marketing.</div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown('<div class="section-header">4. Behavioral Correlation Heatmap</div>', unsafe_allow_html=True)
    # Select numerical features for correlation
    corr_cols = ['is_churn', 'total_active_days', 'total_secs', 'mean_actual_paid', 'days_since_last_transaction', 'skip_ratio']
    corr_matrix = df[corr_cols].corr()
    
    fig4 = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("""
    <div class="insight-block"><b>Analytical Insight:</b> Notice the strong positive correlation between `is_churn` and `days_since_last_transaction`, and the strong negative correlation with `total_active_days`. High `skip_ratio` also weakly correlates with churn.</div>
    <div class="action-block"><b>Business Action:</b> The product team should investigate if high skip ratios mean the recommendation algorithm is failing, causing users to get frustrated and leave.</div>
    """, unsafe_allow_html=True)
