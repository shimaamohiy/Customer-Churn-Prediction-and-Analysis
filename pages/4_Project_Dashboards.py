"""
Page 4: Project Dashboards
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import plotly.express as px
from theme import init_theme, get_theme, inject_global_css, render_theme_toggle, dash_bar, section_header

st.set_page_config(page_title="Project Dashboards", layout="wide")

init_theme()
c = get_theme()
inject_global_css(c)
render_theme_toggle()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "dataset_sample.csv")

dash_bar(
    "Global Data Insights & EDA",
    "Interactive Plotly dashboards on a 20,000-row representative sample of the processed dataset.",
    c["success"]
)


@st.cache_data
def load_sample():
    if not os.path.exists(DATA_PATH):
        st.error("Sample data file not found. Please ensure dataset_sample.csv is in data/processed/.")
        return None
    df = pd.read_csv(DATA_PATH)
    df["Churn Status"] = df["is_churn"].map({0: "Retained", 1: "Churned"})
    df["Auto Renew Status"] = df["latest_auto_renew"].map({0: "Disabled", 1: "Enabled"})
    return df

df = load_sample()
if df is None:
    st.error("Processed data not found. Please run the preprocessing pipeline first.")
    st.stop()

COLORS = {"Retained": c["primary"], "Churned": c["danger"]}

def plotly_defaults():
    return dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20,r=20,t=40,b=20))

# ─── Row 1 ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    section_header("1. Auto-Renewal Impact on Churn", c["success"])
    st.markdown(f"<p style='color:{c['muted']};font-size:0.9rem;'>Does enabling auto-renew really prevent churn?</p>", unsafe_allow_html=True)
    agg = df.groupby(["Auto Renew Status","Churn Status"]).size().reset_index(name="Count")
    fig = px.bar(agg, x="Auto Renew Status", y="Count", color="Churn Status",
                 barmode="group", color_discrete_map=COLORS)
    fig.update_layout(**plotly_defaults())
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
    <div style='background:{c['card']};border-left:4px solid {c['primary']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;margin-bottom:0.4rem;font-size:0.9rem;color:{c['text']};'>
        <b>Insight:</b> Subscribers with auto-renew <i>disabled</i> churn at a dramatically higher rate.
        Manual renewal introduces billing friction — the single strongest controllable churn driver.
    </div>
    <div style='background:rgba(16,185,129,0.1);border-left:4px solid {c['success']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;font-size:0.9rem;color:{c['text']};'>
        <b>Action:</b> Offer a 15% discount or 1 free month to every user with auto-renew disabled.
        This single campaign could eliminate the largest driver of voluntary churn.
    </div>
    """, unsafe_allow_html=True)

with col2:
    section_header("2. Engagement vs. Retention", c["teal"])
    st.markdown(f"<p style='color:{c['muted']};font-size:0.9rem;'>How many days was the user active in the last 30 days?</p>", unsafe_allow_html=True)
    fig2 = px.box(df, x="Churn Status", y="total_active_days", color="Churn Status",
                  color_discrete_map=COLORS)
    fig2.update_layout(**plotly_defaults(), yaxis_title="Active Days (Last 30)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"""
    <div style='background:{c['card']};border-left:4px solid {c['teal']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;margin-bottom:0.4rem;font-size:0.9rem;color:{c['text']};'>
        <b>Insight:</b> Retained users have a significantly higher median active days.
        Churned users typically drop below 10 active days weeks before cancelling — this is detectable early.
    </div>
    <div style='background:rgba(16,185,129,0.1);border-left:4px solid {c['success']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;font-size:0.9rem;color:{c['text']};'>
        <b>Action:</b> Build an alert: when active days drop below 10 in any 30-day window → trigger 
        personalized "Discover Weekly" push notifications to re-engage the user before they leave.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─── Row 2 ────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    section_header("3. Recency Analysis (Time-to-Churn)", c["warning"])
    st.markdown(f"<p style='color:{c['muted']};font-size:0.9rem;'>Days since last transaction — how does recency predict churn?</p>", unsafe_allow_html=True)
    fig3 = px.histogram(df, x="days_since_last_transaction", color="Churn Status",
                        nbins=30, barmode="overlay", color_discrete_map=COLORS)
    fig3.update_layout(**plotly_defaults(), xaxis_title="Days Since Last Transaction")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(f"""
    <div style='background:{c['card']};border-left:4px solid {c['warning']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;margin-bottom:0.4rem;font-size:0.9rem;color:{c['text']};'>
        <b>Insight:</b> Churned users have a massive spike in the 30+ day bucket.
        If a user hasn't transacted in 30 days, churn is near-certain without intervention.
    </div>
    <div style='background:rgba(16,185,129,0.1);border-left:4px solid {c['success']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;font-size:0.9rem;color:{c['text']};'>
        <b>Action:</b> At 25 days of no transaction, move the user to a "Win-Back" campaign —
        standard retention marketing is too weak after 30+ days of inactivity.
    </div>
    """, unsafe_allow_html=True)

with col4:
    section_header("4. Behavioral Correlation Heatmap", c["purple"])
    st.markdown(f"<p style='color:{c['muted']};font-size:0.9rem;'>Which features are most correlated with churn?</p>", unsafe_allow_html=True)
    corr_cols = ['is_churn','total_active_days','total_secs','mean_actual_paid','days_since_last_transaction','skip_ratio','total_cancellations']
    corr = df[corr_cols].corr()
    fig4 = px.imshow(corr, text_auto=".2f", aspect="auto",
                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig4.update_layout(**plotly_defaults())
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(f"""
    <div style='background:{c['card']};border-left:4px solid {c['purple']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;margin-bottom:0.4rem;font-size:0.9rem;color:{c['text']};'>
        <b>Insight:</b> Strong <b>positive</b> correlation between <code>is_churn</code> and
        <code>days_since_last_transaction</code>. Strong <b>negative</b> correlation with <code>total_active_days</code>.
        <code>skip_ratio</code> weakly correlates with churn — suggesting content dissatisfaction.
    </div>
    <div style='background:rgba(16,185,129,0.1);border-left:4px solid {c['success']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;font-size:0.9rem;color:{c['text']};'>
        <b>Action:</b> The Product team should investigate whether high skip ratios indicate
        a failing recommendation algorithm — fixing it could reduce churn without any marketing spend.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─── Row 3 ────────────────────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    section_header("5. Payment Amount vs. Plan Duration", c["pink"])
    st.markdown(f"<p style='color:{c['muted']};font-size:0.9rem;'>Do high-paying subscribers on long plans stay longer?</p>", unsafe_allow_html=True)
    fig5 = px.scatter(df, x="mean_plan_days", y="mean_actual_paid", color="Churn Status",
                      opacity=0.5, color_discrete_map=COLORS)
    fig5.update_layout(**plotly_defaults(),
                       xaxis_title="Avg Plan Duration (Days)",
                       yaxis_title="Avg Paid Amount")
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(f"""
    <div style='background:{c['card']};border-left:4px solid {c['pink']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;font-size:0.9rem;color:{c['text']};'>
        <b>Insight:</b> Higher-value subscribers on longer plans tend to be retained more.
        Short-plan, low-spend users are a disproportionately large fraction of churners.
        <br><b>Action:</b> Upsell monthly subscribers to annual plans with a 2-month discount — annual plans
        dramatically increase CLV and reduce churn probability.
    </div>
    """, unsafe_allow_html=True)

with col6:
    section_header("6. Age Distribution of Churners vs. Retained", c["warning"])
    st.markdown(f"<p style='color:{c['muted']};font-size:0.9rem;'>Are certain age groups more likely to churn?</p>", unsafe_allow_html=True)
    fig6 = px.histogram(df, x="bd", color="Churn Status", nbins=20,
                        barmode="overlay", color_discrete_map=COLORS)
    fig6.update_layout(**plotly_defaults(), xaxis_title="Age (Years)")
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown(f"""
    <div style='background:{c['card']};border-left:4px solid {c['warning']};border:1px solid {c['border']};
                padding:0.8rem 1rem;border-radius:8px;font-size:0.9rem;color:{c['text']};'>
        <b>Insight:</b> Churn is concentrated in younger user segments (18-30), who tend to be
        more price-sensitive and platform-hopping.
        <br><b>Action:</b> Design a Student/Young Adult plan with a lower price point and social sharing
        features to increase stickiness in this high-churn demographic.
    </div>
    """, unsafe_allow_html=True)
