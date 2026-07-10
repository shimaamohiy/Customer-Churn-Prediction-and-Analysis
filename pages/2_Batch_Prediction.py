"""
Page 2: Batch Prediction
"""
import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(page_title="Batch Prediction", layout="wide")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_churn_model.pkl")

st.markdown("""
    <style>
    .dash-bar {
        background-color: var(--secondary-background-color);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #8B5CF6;
        margin-bottom: 2rem;
    }
    .dash-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--text-color);
        margin: 0;
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
    <h2 class="dash-title">Batch Churn Assessment</h2>
    <p style="color:var(--text-color); opacity:0.8; margin:0;">Upload a CSV file containing pre-processed subscriber features to execute bulk predictions.</p>
</div>
""", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.error("System Error: Predictive model not found. Please ensure the pipeline has been executed.")
    st.stop()

st.info("The uploaded CSV must contain the 26 structured features matching the final modeling dataset schema.")

uploaded_file = st.file_uploader("Upload Subscriber Data (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown(f"**Dataset Loaded:** {len(df):,} records")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Execute Bulk Predictions", type="primary", use_container_width=True):
        user_ids = df["msno"].copy() if "msno" in df.columns else pd.Series(range(len(df)))
        
        expected_columns = [
            'total_transactions', 'total_actual_paid', 'mean_actual_paid', 'mean_plan_days', 
            'total_cancellations', 'latest_auto_renew', 'latest_payment_method', 
            'average_under_overpayment', 'city', 'bd', 'registered_via', 'reg_year', 
            'reg_month', 'reg_day', 'account_age_days', 'num_100', 'num_unq', 'total_secs', 
            'total_active_days', 'skip_ratio', 'loop_ratio', 'secs_per_active_day', 
            'days_since_last_transaction', 'days_until_expiration', 'gender_male', 'gender_unknown'
        ]
        
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            st.error(f"Prediction Failed: The uploaded dataset is missing the following required columns: {', '.join(missing)}")
        else:
            features = df[expected_columns].astype(float)
            
            try:
                predictions = model.predict(features)
                probabilities = model.predict_proba(features)[:, 1]

                results = pd.DataFrame({
                    "Subscriber_ID": user_ids,
                    "Churn_Prediction": predictions,
                    "Churn_Probability": probabilities.round(4),
                    "Risk_Level": pd.cut(probabilities, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
                })

                st.divider()
                st.markdown("### Execution Summary")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Processed", f"{len(results):,}")
                with c2:
                    churners_count = int((results["Churn_Prediction"] == 1).sum())
                    st.metric("Predicted Churners", f"{churners_count:,}", delta="High Risk", delta_color="inverse")
                with c3:
                    churn_rate = (results['Churn_Prediction'] == 1).mean()
                    st.metric("Projected Churn Rate", f"{churn_rate:.1%}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(results, use_container_width=True)

                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button("Download Output Report (CSV)", csv, "batch_predictions.csv", "text/csv", use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
