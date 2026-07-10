"""
Page 1: Single Subscriber Prediction
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from theme import init_theme, get_theme, inject_global_css, render_theme_toggle, dash_bar, section_header, insight_card

st.set_page_config(page_title="Single Prediction", layout="wide")

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
    "Dynamic Churn Assessment & Insights",
    "Adjust inputs — the risk gauge and business insights update in real-time.",
    c["primary"]
)

model = load_model()
if model is None:
    st.error("Model not found. Please run the training pipeline first.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    section_header("Demographics & Account", c["teal"])
    bd                   = st.slider("Age",                     10, 80,  30)
    city                 = st.number_input("City ID",           1,  22,  1)
    gender               = st.selectbox("Gender",               ["Male","Female","Unknown"])
    registered_via       = st.selectbox("Registration Method",  [3, 4, 7, 9, 13], index=2)
    account_age_days     = st.number_input("Account Age (Days)",1,  None, 365)
    reg_year             = st.number_input("Registration Year",  2000, 2030, 2015)
    reg_month            = st.number_input("Registration Month", 1, 12, 1)
    reg_day              = st.number_input("Registration Day",   1, 31, 1)

with col2:
    section_header("Transaction History", c["warning"])
    total_transactions          = st.slider("Total Transactions",        1, 50, 12)
    total_actual_paid           = st.number_input("Total Actual Paid",   0, None, 1500)
    mean_actual_paid            = st.number_input("Mean Actual Paid",    0.0, None, 149.0)
    mean_plan_days              = st.number_input("Mean Plan Days",       0.0, None, 30.0)
    average_under_overpayment   = st.number_input("Avg Under/Overpayment", value=0.0,
                                                   help="Negative = underpayment / card decline")
    total_cancellations         = st.slider("Total Cancellations",       0, 10, 0)
    latest_auto_renew           = st.selectbox("Latest Auto Renew",      [0, 1], index=1)
    latest_payment_method       = st.number_input("Payment Method ID",   1, 41, 41)
    days_since_last_transaction = st.slider("Days Since Last Transaction", 0, 365, 15)
    days_until_expiration       = st.slider("Days Until Expiration",    -30, 365, 15)

with col3:
    section_header("Platform Engagement", c["success"])
    total_active_days  = st.slider("Active Days (Last 30)",   0, 30, 20)
    total_secs         = st.number_input("Total Seconds Played", 0.0, None, 50000.0)
    num_100            = st.number_input("Songs Played 100%",    0.0, None, 200.0)
    num_unq            = st.number_input("Unique Songs Played",  0.0, None, 150.0)
    skip_ratio         = st.slider("Skip Ratio",  0.0, 1.0, 0.1)
    loop_ratio         = st.slider("Loop Ratio",  0.0, 1.0, 0.05)
    secs_per_active_day = st.number_input("Seconds/Active Day", 0.0, None, 2500.0)

# ─── Build Feature Vector ─────────────────────────────────
gender_male    = 1 if gender == "Male"    else 0
gender_unknown = 1 if gender == "Unknown" else 0

features = pd.DataFrame([{
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
    'gender_unknown': float(gender_unknown),
}])[[ 
    'total_transactions','total_actual_paid','mean_actual_paid','mean_plan_days',
    'total_cancellations','latest_auto_renew','latest_payment_method',
    'average_under_overpayment','city','bd','registered_via','reg_year',
    'reg_month','reg_day','account_age_days','num_100','num_unq','total_secs',
    'total_active_days','skip_ratio','loop_ratio','secs_per_active_day',
    'days_since_last_transaction','days_until_expiration','gender_male','gender_unknown'
]]

try:
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]

    st.divider()
    g_col, i_col = st.columns([1, 1.5])

    with g_col:
        section_header("Live Risk Gauge", c["primary"])
        bar_color = c["danger"] if probability > 0.6 else c["warning"] if probability > 0.3 else c["success"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability %", 'font': {'size': 18}},
            number={'font': {'color': bar_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': c['muted']},
                'bar': {'color': bar_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30],   'color': 'rgba(16,185,129,0.15)'},
                    {'range': [30, 60],  'color': 'rgba(245,158,11,0.15)'},
                    {'range': [60, 100], 'color': 'rgba(239,68,68,0.15)'},
                ],
                'threshold': {'line': {'color': c['danger'], 'width': 3}, 'thickness': 0.75, 'value': 60}
            }
        ))
        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary verdict
        if prediction == 1:
            st.error(f"**Verdict: HIGH CHURN RISK** — Probability: {probability:.1%}")
        else:
            st.success(f"**Verdict: LIKELY RETAINED** — Probability: {probability:.1%}")

    with i_col:
        section_header("Executive Summary & Actions", c["primary"])
        
        if prediction == 1:
            insight_card(
                " This Customer is Highly Likely to Churn",
                "Based on the prediction model, this customer exhibits strong churn signals and is projected to cancel their subscription. Immediate intervention is required to save the $10-$15/month recurring revenue.",
                c["danger"],
                "Review the specific risk factors below and execute the recommended marketing interventions."
            )
        else:
            insight_card(
                " This Customer is Likely to Stay",
                "This customer shows healthy engagement and billing patterns. They are projected to renew their subscription without issue.",
                c["success"],
                "No aggressive intervention needed. Continue standard marketing engagement."
            )
        
        section_header("Detailed Risk Factors", c["warning"])

        if latest_auto_renew == 0:
            insight_card(
                "Risk: Auto-Renewal Disabled",
                "Manual renewal creates billing friction — the single strongest predictor of churn in this dataset.",
                c["danger"],
                "Launch a targeted campaign offering a 15% discount or 1 free month to re-enable Auto-Renew."
            )
        else:
            insight_card(
                "Strength: Auto-Renewal Active",
                "Auto-renew is the top retention driver. This user is locked into frictionless billing.",
                c["success"],
                "Monitor payment method expiration to ensure the auto-renewal actually processes successfully."
            )

        if total_active_days < 10:
            insight_card(
                "Risk: Low Platform Engagement",
                f"Only {total_active_days} active days in the last 30 days. Engagement drops are the earliest visible churn signal.",
                c["danger"],
                "Trigger a 'Discover Weekly' personalized push notification to re-engage the user with fresh content."
            )

        if skip_ratio > 0.5:
            insight_card(
                "Risk: High Skip Ratio",
                f"The user skips {skip_ratio:.0%} of tracks. This indicates the recommendation engine is poorly aligned.",
                c["warning"],
                "Prompt the user with a genre preference survey to re-calibrate the recommendation algorithm."
            )

        if average_under_overpayment < -10:
            insight_card(
                "Risk: Billing Discrepancy",
                "Actual payment is less than the plan price — likely a card decline or partial payment failure.",
                c["danger"],
                "Escalate to Billing Support immediately. Failed payments that go unresolved lead to involuntary churn."
            )

        if registered_via in [4, 7]:
            insight_card(
                "Demographic: High-Risk Registration Channel",
                "Users from this registration channel historically have below-average loyalty scores.",
                c["warning"],
                "Enroll this user in the 'Premium Onboarding' email sequence (5 touchpoints over 14 days)."
            )

        if total_cancellations > 0:
            insight_card(
                "Risk: Repeat Cancellation History",
                f"This user has cancelled {total_cancellations} time(s) before — a strong predictor of future cancellation.",
                c["danger"],
                "Offer a long-term annual plan at a steep discount to increase switching costs and lock them in."
            )

        if days_since_last_transaction > 30:
            insight_card(
                "Critical: No Recent Transactions",
                f"{days_since_last_transaction} days since last payment. Churn is almost certain without intervention.",
                c["danger"],
                "Move to Win-Back campaign immediately — standard retention marketing is too weak at this stage."
            )

except Exception as e:
    st.error(f"Prediction Error: {e}")
