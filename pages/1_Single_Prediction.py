"""
Page 1: Single Subscriber Prediction
Accepts RAW features → Cleaning → Feature Engineering → Prediction
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from theme import init_theme, get_theme, inject_global_css, render_theme_toggle, dash_bar, section_header, insight_card
from src.preprocess import preprocess_single_record

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
    "Enter raw subscriber data — the system cleans, engineers features, and predicts churn in real-time.",
    c["primary"]
)

model = load_model()
if model is None:
    st.error("Model not found. Please run the training pipeline first.")
    st.stop()

# ─── RAW INPUT FIELDS ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    section_header("Demographics & Account", c["teal"])
    bd_raw           = st.number_input("Age (Raw)", min_value=0, max_value=150, value=30,
                                       help="Any integer — invalid values (outside 10-80) will be cleaned automatically.")
    city             = st.number_input("City ID", min_value=1, max_value=22, value=1)
    gender_raw       = st.selectbox("Gender", ["male", "female", "unknown"],
                                    help="Will be one-hot encoded during preprocessing.")
    registered_via   = st.selectbox("Registration Method", [3, 4, 7, 9, 13], index=2)
    registration_date = st.date_input("Registration Date", value=date(2015, 1, 1),
                                       help="reg_year, reg_month, reg_day and account_age_days will be derived automatically.")

with col2:
    section_header("Transaction History", c["warning"])
    total_transactions          = st.slider("Total Transactions",        1, 50, 12)
    total_actual_paid           = st.number_input("Total Actual Paid",   min_value=0, value=1500)
    mean_actual_paid            = st.number_input("Mean Actual Paid",    min_value=0.0, value=149.0)
    mean_plan_days              = st.number_input("Mean Plan Days",       min_value=0.0, value=30.0)
    plan_list_price_mean        = st.number_input("Mean Plan List Price", min_value=0.0, value=149.0,
                                                   help="Used to compute average under/overpayment = mean_actual_paid − mean_plan_list_price.")
    total_cancellations         = st.slider("Total Cancellations",       0, 10, 0)
    latest_auto_renew           = st.selectbox("Latest Auto Renew",      [0, 1], index=1)
    latest_payment_method       = st.number_input("Payment Method ID",   min_value=1, max_value=41, value=41)
    last_transaction_date       = st.date_input("Last Transaction Date", value=date(2017, 4, 15),
                                                 help="days_since_last_transaction will be computed from snapshot date 2017-04-30.")
    membership_expire_date      = st.date_input("Membership Expire Date", value=date(2017, 5, 15),
                                                 help="days_until_expiration will be computed from snapshot date 2017-04-30.")

with col3:
    section_header("Platform Engagement", c["success"])
    total_active_days  = st.slider("Active Days (Last 30)",   0, 30, 20)
    total_secs         = st.number_input("Total Seconds Played", min_value=0.0, value=50000.0)
    num_25             = st.number_input("Songs Played 25%  (num_25)",  min_value=0.0, value=50.0,
                                         help="عدد الأغاني اللي اتسمعت 25% منها فأقل — بتدخل في حساب skip_ratio")
    num_50             = st.number_input("Songs Played 50%  (num_50)",  min_value=0.0, value=30.0,
                                         help="عدد الأغاني اللي اتسمعت 50% منها — بتدخل في حساب skip_ratio")
    num_75             = st.number_input("Songs Played 75%  (num_75)",  min_value=0.0, value=20.0,
                                         help="عدد الأغاني اللي اتسمعت 75% منها")
    num_985            = st.number_input("Songs Played 98.5% (num_985)", min_value=0.0, value=10.0,
                                         help="عدد الأغاني اللي اتسمعت 98.5% منها")
    num_100            = st.number_input("Songs Played 100% (num_100)", min_value=0.0, value=200.0,
                                         help="عدد الأغاني اللي اتسمعت كاملة — بتدخل في حساب loop_ratio")
    num_unq            = st.number_input("Unique Songs Played (num_unq)", min_value=0.0, value=150.0,
                                         help="عدد الأغاني الفريدة — بتدخل في حساب loop_ratio")
    secs_per_active_day = st.number_input("Seconds/Active Day", min_value=0.0, value=2500.0)


# ─── CLEANING & FEATURE ENGINEERING (via preprocess.py) ───────
features, meta = preprocess_single_record(
    bd_raw=bd_raw,
    city=city,
    gender_raw=gender_raw,
    registered_via=registered_via,
    registration_date=registration_date,
    total_transactions=total_transactions,
    total_actual_paid=total_actual_paid,
    mean_actual_paid=mean_actual_paid,
    mean_plan_days=mean_plan_days,
    plan_list_price_mean=plan_list_price_mean,
    total_cancellations=total_cancellations,
    latest_auto_renew=latest_auto_renew,
    latest_payment_method=latest_payment_method,
    last_transaction_date=last_transaction_date,
    membership_expire_date=membership_expire_date,
    total_active_days=total_active_days,
    total_secs=total_secs,
    num_25=num_25,
    num_50=num_50,
    num_75=num_75,
    num_985=num_985,
    num_100=num_100,
    num_unq=num_unq,
    secs_per_active_day=secs_per_active_day,
)

# Unpack meta for use in insight cards below
bd_clean                    = meta['bd_clean']
age_was_cleaned             = meta['age_was_cleaned']
skip_ratio                  = meta['skip_ratio']
loop_ratio                  = meta['loop_ratio']
account_age_days            = meta['account_age_days']
days_since_last_transaction = meta['days_since_last_transaction']
days_until_expiration       = meta['days_until_expiration']
average_under_overpayment   = meta['average_under_overpayment']

# ─── Preview engineered features ──────────────────────────────
with st.expander(" Preview: Engineered Features After Cleaning & Preprocessing"):
    display_df = features.T.rename(columns={0: "Value"})
    display_df.index.name = "Feature"
    st.dataframe(display_df, use_container_width=True)

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
                f"{days_since_last_transaction:.0f} days since last payment. Churn is almost certain without intervention.",
                c["danger"],
                "Move to Win-Back campaign immediately — standard retention marketing is too weak at this stage."
            )

        if age_was_cleaned:
            insight_card(
                "Data Quality: Invalid Age Detected",
                f"Age value {bd_raw} is outside the valid range (10–80) and was automatically replaced with the training median ({bd_clean:.0f}).",
                c["warning"],
                "Ensure data collection captures accurate age values to improve prediction accuracy."
            )

except Exception as e:
    st.error(f"Prediction Error: {e}")
