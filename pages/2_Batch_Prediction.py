"""
Page 2: Batch Prediction
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import joblib
import pandas as pd
from theme import init_theme, get_theme, inject_global_css, render_theme_toggle, dash_bar, section_header, mini_card

st.set_page_config(page_title="Batch Prediction", layout="wide")

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
    "Batch Churn Assessment",
    "Upload a CSV of subscribers to generate bulk risk scores and a downloadable report.",
    c["purple"]
)

model = load_model()
if model is None:
    st.error("Model not found. Please run the training pipeline first.")
    st.stop()

EXPECTED = [
    'total_transactions','total_actual_paid','mean_actual_paid','mean_plan_days',
    'total_cancellations','latest_auto_renew','latest_payment_method',
    'average_under_overpayment','city','bd','registered_via','reg_year',
    'reg_month','reg_day','account_age_days','num_100','num_unq','total_secs',
    'total_active_days','skip_ratio','loop_ratio','secs_per_active_day',
    'days_since_last_transaction','days_until_expiration','gender_male','gender_unknown'
]

st.info(f"Required columns: {len(EXPECTED)} features matching the final modeling dataset schema.")
uploaded_file = st.file_uploader("Upload Subscriber Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"**Loaded:** {len(df):,} records")
    st.table(df.head())

    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
        user_ids = df["msno"].copy() if "msno" in df.columns else pd.Series(range(len(df)))
        missing = [col for col in EXPECTED if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            try:
                preds = model.predict(df[EXPECTED].astype(float))
                probs = model.predict_proba(df[EXPECTED].astype(float))[:, 1]
                results = pd.DataFrame({
                    "Subscriber_ID":    user_ids,
                    "Churn_Prediction": preds,
                    "Churn_Probability": probs.round(4),
                    "Risk_Level": pd.cut(probs, bins=[0,0.3,0.6,1.0], labels=["Low","Medium","High"])
                })
                st.divider()
                section_header("Execution Summary", c["success"])
                m1, m2, m3 = st.columns(3)
                with m1: mini_card("Total Processed",    f"{len(results):,}", c["primary"])
                with m2: mini_card("Predicted Churners", f"{int((results.Churn_Prediction==1).sum()):,}", c["danger"])
                with m3: mini_card("Projected Churn Rate", f"{(results.Churn_Prediction==1).mean():.1%}", c["warning"])
                st.markdown(f"<p style='color:gray; font-size:0.9rem;'>Previewing top 50 records. Download the CSV for the full report.</p>", unsafe_allow_html=True)
                st.table(results.head(50))
                st.download_button("Download Report (CSV)", results.to_csv(index=False).encode(), "batch_predictions.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
