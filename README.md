# KKBox Customer Churn Prediction 🎵

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=Docker&logoColor=white)](https://www.docker.com/)

An End-to-End CRISP-DM Machine Learning Platform designed to predict subscription cancellations for KKBox (Asia's leading music streaming service) 30 days in advance, preventing $10-$15 in recurring monthly revenue loss per user.

---

## 📑 Table of Contents
1. [Business Objective & ROI](#-business-objective--roi)
2. [Platform Features](#-platform-features)
3. [Project Architecture](#-project-architecture)
4. [Data Dictionary](#-data-dictionary)
5. [Installation & Deployment](#-installation--deployment)
6. [Machine Learning Methodology](#-machine-learning-methodology)

---

## 📈 Business Objective & ROI
Retaining an existing customer is vastly cheaper than acquiring a new one. This project leverages user engagement and transaction history to predict churn.
- **Financial Impact**: Retaining a single user saves **$10-$15 per month** in recurring revenue.
- **Success Criteria**: Reduce overall subscription churn by 20% and achieve an AUC-ROC > 0.90.
- **Champion Model**: LightGBM (Tuned), achieving **95.6% Recall** and **0.9926 ROC-AUC**.

---

## 🚀 Platform Features
- **Adaptive Light/Dark Mode**: The UI automatically switches to match your OS preferences natively.
- **Dynamic Risk Assessment**: Real-time business insights and actionable solutions generated instantly as you manipulate user features (e.g., offering discounts if Auto-Renew is disabled).
- **Interactive Dashboards**: Plotly-powered exploratory data analysis featuring Correlation Heatmaps and Time-to-Churn analysis.
- **Model Comparison Engine**: Visual benchmarking of LightGBM vs. Random Forest vs. Gradient Boosting.
- **Batch Processing**: Upload CSVs for bulk predictions.

---

## 🏗️ Project Architecture
```text
customer_churn_prediction/
├── app.py                      # Main Streamlit Application (Home)
├── theme.py                    # CSS styling and Native Light/Dark engine config
├── pages/                      # Multi-page interactive modules
│   ├── 1_Single_Prediction.py  # Real-time risk gauge & business insights
│   ├── 2_Batch_Prediction.py   # Bulk CSV processing
│   ├── 3_Model_Info.py         # LightGBM architecture & hyperparameter dive
│   ├── 4_Project_Dashboards.py # Interactive EDA & Correlation Heatmaps
│   └── 5_Model_Comparison.py   # Model evaluation dashboard
├── src/
│   ├── pipeline.py             # Data preparation & model training logic
│   └── config.py               # Global ML paths and parameters
├── Notebook/                   # CRISP-DM Jupyter Notebooks (Phases 1-6)
├── Dockerfile                  # Containerization setup for Streamlit
└── requirements.txt            # Python dependencies
```

---

## 📖 Data Dictionary
Below are the core features engineered from the raw user logs and transactions:
- `total_secs`: Total seconds of music listened per day. (High = Better retention).
- `is_auto_renew`: Whether the subscription renews automatically.
- `num_100`: Number of songs played to 100% completion.
- `skip_ratio`: Calculated ratio of songs skipped before 25% duration.
- `days_since_last_transaction`: Days elapsed since the user last paid for a subscription. (High = Extremely high churn risk).

---

## 💻 Installation & Deployment

### Option 1: Local Environment
Ensure you have Python 3.11+ installed.
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer_churn_prediction.git
cd customer_churn_prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the platform
streamlit run app.py
```

### Option 2: Docker Container
Ideal for cloud deployment (AWS, GCP, Azure).
```bash
# 1. Build the Docker image
docker build -t kkbox-churn-app .

# 2. Run the container
docker run -p 8501:8501 kkbox-churn-app
```
Navigate to `http://localhost:8501` in your browser.

---

## 🧠 Machine Learning Methodology
This project rigorously followed the CRISP-DM methodology:
1. **Business Understanding:** Aligning metrics with revenue retention (Recall > Accuracy).
2. **Data Understanding:** Processing 30GB+ of raw logs.
3. **Data Preparation:** Feature engineering, temporal splits, and median imputation.
4. **Modeling:** Training Logistic Regression, Random Forest, Gradient Boosting, and LightGBM.
5. **Evaluation:** Tuning LightGBM with `is_unbalance=True` to drastically maximize Recall.
6. **Deployment:** The dynamic Streamlit dashboard.
