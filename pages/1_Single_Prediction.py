"""
Page 1: Single User Prediction
"""
import streamlit as st
import joblib
import pandas as pd
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Single Prediction", layout="wide")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_churn_model.pkl")

# Adaptive Light/Dark CSS using Native Variables
st.markdown("""
    <style>
    .dash-bar {
        background-color: var(--secondary-background-color);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 6px solid var(--primary-color);
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
        color: var(--primary-color);
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--text-color);
        padding-bottom: 0.2rem;
    }
    .insight-card {
        background-color: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: var(--text-color);
    }
    .insight-title {
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .solution-box {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #EF4444;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        color: var(--text-color);
    }
    .solution-box.positive {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10B981;
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
    <h2 class="dash-title">Dynamic Churn Assessment & Insights</h2>
    <p style="color:var(--text-color); opacity:0.8; margin:0;">Modify the inputs below. The risk gauge and business insights will update in real-time.</p>
</div>
""", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.error("System Error: Predictive model not found. Please ensure the pipeline has been executed.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-header">Demographics & Account</div>', unsafe_allow_html=True)
    bd = st.slider("Age (bd)", min_value=10, max_value=80, value=30)
    city = st.number_input("City ID", min_value=1, max_value=22, value=1)
    gender = st.selectbox("Gender", options=["Male", "Female", "Unknown"])
    registered_via = st.selectbox("Registration Method", options=[3, 4, 7, 9, 13], index=2)
    account_age_days = st.number_input("Account Age (Days)", min_value=1, value=365)
    reg_year = st.number_input("Registration Year", min_value=2000, value=2015)
    reg_month = st.number_input("Registration Month", min_value=1, max_value=12, value=1)
    reg_day = st.number_input("Registration Day", min_value=1, max_value=31, value=1)

with col2:
    st.markdown('<div class="section-header">Transaction History</div>', unsafe_allow_html=True)
    total_transactions = st.slider("Total Transactions", min_value=1, max_value=50, value=12)
    total_actual_paid = st.number_input("Total Actual Paid", min_value=0, value=1500)
    mean_actual_paid = st.number_input("Mean Actual Paid", min_value=0.0, value=149.0)
    mean_plan_days = st.number_input("Mean Plan Days", min_value=0.0, value=30.0)
    average_under_overpayment = st.number_input("Avg Under/Overpayment", value=0.0, help="Negative indicates underpayment (failed billing), Positive indicates overpayment.")
    total_cancellations = st.slider("Total Cancellations", min_value=0, max_value=10, value=0)
    latest_auto_renew = st.selectbox("Latest Auto Renew", options=[0, 1], index=1)
    latest_payment_method = st.number_input("Latest Payment Method ID", min_value=1, max_value=41, value=41)
    days_since_last_transaction = st.slider("Days Since Last Transaction", min_value=0, max_value=365, value=15)
    days_until_expiration = st.slider("Days Until Expiration", min_value=-30, max_value=365, value=15)

with col3:
    st.markdown('<div class="section-header">Platform Engagement Logs</div>', unsafe_allow_html=True)
    total_active_days = st.slider("Total Active Days (Last 30 Days)", min_value=0, max_value=30, value=20)
    total_secs = st.number_input("Total Seconds Played", min_value=0.0, value=50000.0)
    num_100 = st.number_input("Songs Played 100%", min_value=0.0, value=200.0)
    num_unq = st.number_input("Unique Songs Played", min_value=0.0, value=150.0)
    skip_ratio = st.slider("Skip Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
    loop_ratio = st.slider("Loop Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.05)
    secs_per_active_day = st.number_input("Seconds per Active Day", min_value=0.0, value=2500.0)

# AUTOMATIC REAL-TIME PREDICTION
gender_male = 1 if gender == "Male" else 0
gender_unknown = 1 if gender == "Unknown" else 0

input_data = {
    'total_transactions': float(total_transactions),
    'total_actual_paid': float(total_actual_paid),
    'mean_actual_paid': float(mean_actual_paid),
    'mean_plan_days': float(mean_plan_days),
    'total_cancellations': float(total_cancellations),
    'latest_auto_renew': float(latest_auto_renew),
    'latest_payment_method': float(latest_payment_method),
    'average_under_overpayment': float(average_under_overpayment),
    'city': float(city),
    'bd': float(bd),
    'registered_via': float(registered_via),
    'reg_year': float(reg_year),
    'reg_month': float(reg_month),
    'reg_day': float(reg_day),
    'account_age_days': float(account_age_days),
    'num_100': float(num_100),
    'num_unq': float(num_unq),
    'total_secs': float(total_secs),
    'total_active_days': float(total_active_days),
    'skip_ratio': float(skip_ratio),
    'loop_ratio': float(loop_ratio),
    'secs_per_active_day': float(secs_per_active_day),
    'days_since_last_transaction': float(days_since_last_transaction),
    'days_until_expiration': float(days_until_expiration),
    'gender_male': float(gender_male),
    'gender_unknown': float(gender_unknown)
}

input_df = pd.DataFrame([input_data])
expected_columns = [
    'total_transactions', 'total_actual_paid', 'mean_actual_paid', 'mean_plan_days', 
    'total_cancellations', 'latest_auto_renew', 'latest_payment_method', 
    'average_under_overpayment', 'city', 'bd', 'registered_via', 'reg_year', 
    'reg_month', 'reg_day', 'account_age_days', 'num_100', 'num_unq', 'total_secs', 
    'total_active_days', 'skip_ratio', 'loop_ratio', 'secs_per_active_day', 
    'days_since_last_transaction', 'days_until_expiration', 'gender_male', 'gender_unknown'
]
input_df = input_df[expected_columns]

try:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]
    
    st.divider()
    
    res_col1, res_col2 = st.columns([1, 1.5])
    
    with res_col1:
        st.markdown('<div class="section-header">Live Risk Assessment</div>', unsafe_allow_html=True)
        # Transparent Gauge Chart (works in both Light & Dark modes)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability %", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#3A86FF"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 4},
                    'thickness': 0.75,
                    'value': 60}
            }
        ))
        fig.update_layout(
            height=350, 
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with res_col2:
        st.markdown('<div class="section-header">Strategic Business Insights</div>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            <div class="insight-card" style="border-left-color: #EF4444;">
                <div class="insight-title" style="color: #EF4444;">Critical: Impending Revenue Loss</div>
                <div>This subscriber is modeled to churn. Based on a $10-$15/month recurring value, immediate action is required to prevent portfolio decay.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card" style="border-left-color: #10B981;">
                <div class="insight-title" style="color: #10B981;">Stable: Revenue Retained</div>
                <div>Subscriber behavior aligns with long-term retention patterns. No aggressive intervention needed.</div>
            </div>
            """, unsafe_allow_html=True)

        # Dynamic Business Rules based on Phase 1 logic
        if latest_auto_renew == 0:
            st.markdown("""
            **Risk Factor: Manual Renewal Required**
            <div class="solution-box">
                <strong>Business Solution:</strong> Payment failures and manual friction indicate high churn risk. Dispatch an automated campaign offering a 1-month 50% discount to re-enable Auto-Renew.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            **Retention Driver: Auto-Renew Enabled**
            <div class="solution-box positive">
                <strong>Status:</strong> The user has auto-renewal active, a primary driver of retention. Monitor payment method expiration dates closely.
            </div>
            """, unsafe_allow_html=True)

        if total_active_days < 10:
            st.markdown("""
            <br>**Risk Factor: Declining Engagement**
            <div class="solution-box">
                <strong>Business Solution:</strong> User engagement drops typically precede churn. Implement a re-engagement strategy utilizing personalized push notifications (e.g., "Discover Weekly") to pull them back into the platform.
            </div>
            """, unsafe_allow_html=True)
            
        if average_under_overpayment < -10:
            st.markdown("""
            <br>**Risk Factor: Billing Discrepancy (Underpayment)**
            <div class="solution-box">
                <strong>Business Solution:</strong> The user's actual paid amount is less than the plan price, indicating a potential payment failure or card decline. Escalate to the Billing Support team to capture lost revenue.
            </div>
            """, unsafe_allow_html=True)
        
        if registered_via in [4, 7]:
            st.markdown("""
            <br>**Demographic Insight: High-Risk Registration Channel**
            <div class="solution-box">
                <strong>Business Solution:</strong> Users from this registration channel historically exhibit lower loyalty. Ensure they receive the specialized "Premium Onboarding" email sequence to build platform habituation.
            </div>
            """, unsafe_allow_html=True)
        
        if total_cancellations > 0:
            st.markdown("""
            <br>**Risk Factor: Historical Cancellation**
            <div class="solution-box">
                <strong>Business Solution:</strong> This user has cancelled before. They are highly sensitive to value propositions. Consider migrating them to a long-term discounted annual plan to lock them in.
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Prediction Error: {e}")
