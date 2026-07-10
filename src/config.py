"""
Central Configuration for KKBox Customer Churn Prediction Project.
All paths, model parameters, and feature settings.
"""

import os

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
SRC_DIR = os.path.join(PROJECT_DIR, "src")

# ============================================================
# DATA FILES
# ============================================================
FINAL_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "final_modeling_dataset.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_churn_model.pkl")

# ============================================================
# MODEL TRAINING PARAMETERS
# ============================================================
RANDOM_SEED = 42
TEST_SIZE = 0.20

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 63,
    "is_unbalance": True,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ============================================================
# FEATURE CONFIGURATION
# ============================================================
ID_COLUMN = "msno"
TARGET_COLUMN = "is_churn"
DROP_COLUMNS = [ID_COLUMN, TARGET_COLUMN]

# Risk level bins for churn probability
RISK_BINS = [0, 0.3, 0.6, 1.0]
RISK_LABELS = ["Low", "Medium", "High"]

# ============================================================
# ENSURE DIRECTORIES EXIST
# ============================================================
def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
              PREDICTIONS_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

ensure_dirs()
