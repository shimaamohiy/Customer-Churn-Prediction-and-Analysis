"""
Full Pipeline for KKBox Customer Churn Prediction.
Replicates the exact workflow from Phase 3 (Data Preparation)
and Phase 4 (Modeling) notebooks.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import gdown
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, log_loss, classification_report
)

try:
    from src.config import (
        PROJECT_DIR, PROCESSED_DATA_DIR, MODEL_DIR,
        RANDOM_SEED, TEST_SIZE, LIGHTGBM_PARAMS
    )
except ImportError:
    from config import (
        PROJECT_DIR, PROCESSED_DATA_DIR, MODEL_DIR,
        RANDOM_SEED, TEST_SIZE, LIGHTGBM_PARAMS
    )


# ============================================================
# STEP 1: DATA LOADING (same as Phase 4 notebook)
# ============================================================
def load_data(filepath=None):
    """Load the final modeling dataset."""

    if filepath is None:
        filepath = os.path.join(PROCESSED_DATA_DIR, "final_modeling_dataset.csv")

   
    if not os.path.exists(filepath):
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        gdown.download(
            id="1JyuT3FocnybqJc2z9feED2gFtxU1WNC9",
            output=filepath,
            quiet=False
        )

    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['is_churn'].mean() * 100:.2f}%")

    return df

# ============================================================
# STEP 2: FEATURE PREPARATION (same as Phase 4 notebook)
# ============================================================
def prepare_features(df):
    """
    Separate features and target — exact same as Phase 4 notebook:
    - Drop 'msno' (identifier) and 'is_churn' (target)
    - Convert object columns to category type
    """
    X = df.drop(columns=["msno", "is_churn"])
    y = df["is_churn"]

    # Convert object/categorical columns (LightGBM native support)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    print(f"Features: {X.shape[1]} | Target classes: {y.nunique()}")
    return X, y


# ============================================================
# STEP 3: DATA SPLITTING (same as Phase 4 notebook)
# ============================================================
def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """Stratified train/test split (80/20) — same as Phase 4 notebook."""
    print("Splitting data into stratified train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nData Splitting Complete!")
    print(f"X_train Shape (Features): {X_train.shape} | y_train Shape (Target): {y_train.shape}")
    print(f"X_test Shape (Features):  {X_test.shape}  | y_test Shape (Target):  {y_test.shape}")
    print(f"\nTrain Target Churn Rate: {y_train.mean() * 100:.2f}%")
    print(f"Test Target Churn Rate:  {y_test.mean() * 100:.2f}%")
    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 4: MODEL TRAINING (same as Phase 4 notebook)
# ============================================================
def train_all_models(X_train, y_train):
    """
    Train all 4 models — exact same as Phase 4 notebook:
    1. Logistic Regression (baseline)
    2. Random Forest
    3. Gradient Boosting (sklearn)
    4. LightGBM Classifier
    """
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    trained_models = {}

    # Model 1: Logistic Regression
    print("Training Model 1: Logistic Regression...")
    model_1 = LogisticRegression(
        random_state=RANDOM_SEED, max_iter=1000, class_weight="balanced"
    )
    model_1.fit(X_train, y_train)
    trained_models["Logistic Regression"] = model_1
    print("Logistic Regression trained successfully.\n")

    # Model 2: Random Forest
    print("Training Model 2: Random Forest...")
    model_2 = RandomForestClassifier(
        random_state=RANDOM_SEED, n_estimators=50, max_depth=10, n_jobs=-1
    )
    model_2.fit(X_train, y_train)
    trained_models["Random Forest"] = model_2
    print("Random Forest trained successfully.\n")

    # Model 3: Gradient Boosting
    print("Training Model 3: Gradient Boosting (sklearn)...")
    model_3 = GradientBoostingClassifier(
        random_state=RANDOM_SEED, n_estimators=50, max_depth=4, subsample=0.8
    )
    model_3.fit(X_train, y_train)
    trained_models["Gradient Boosting"] = model_3
    print("Gradient Boosting trained successfully.\n")

    # Model 4: LightGBM
    print("Training Model 4: LightGBM Classifier...")
    model_4 = LGBMClassifier(random_state=RANDOM_SEED, is_unbalance=True, n_jobs=-1)
    model_4.fit(X_train, y_train)
    trained_models["LightGBM Classifier"] = model_4
    print("LightGBM Classifier trained successfully.\n")

    print(f"Successfully trained all {len(trained_models)} models!")
    return trained_models


def tune_lightgbm(X_train, y_train):
    """
    Hyperparameter tuning for LightGBM — exact same as Phase 4 notebook:
    RandomizedSearchCV with 5 iterations, 3-fold CV, scoring='roc_auc'
    """
    from lightgbm import LGBMClassifier

    print("Starting Optimized LightGBM Tuning...")

    param_dist = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [6, 8, 10],
        'num_leaves': [31, 63]
    }

    random_search = RandomizedSearchCV(
        estimator=LGBMClassifier(random_state=RANDOM_SEED, is_unbalance=True, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=5, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1, random_state=RANDOM_SEED
    )

    random_search.fit(X_train, y_train)
    print(f"Tuning complete. Best CV ROC-AUC: {random_search.best_score_:.4f}")
    return random_search.best_estimator_


# ============================================================
# STEP 5: MODEL EVALUATION (same as Phase 4 notebook)
# ============================================================
def evaluate_models(trained_models, X_test, y_test):
    """Evaluate all models — exact same leaderboard as Phase 4 notebook."""
    results = []

    for name, model in trained_models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)

        results.append({"Model": name, "ROC-AUC": auc, "Accuracy": acc})

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    print("Final Model Leaderboard:")
    print(results_df.to_string(index=False))
    return results_df


# ============================================================
# STEP 6: MODEL SAVING
# ============================================================
def save_model(model, filename="best_churn_model.pkl"):
    """Save the trained model to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
    return filepath


# ============================================================
# FULL PIPELINE
# ============================================================
def run_pipeline(data_path=None, save=True):
    """
    Run the complete ML pipeline (replicating Phase 4 notebook):
    1. Load final modeling dataset
    2. Prepare features
    3. Split data (stratified 80/20)
    4. Train all 4 models + tune LightGBM
    5. Evaluate and compare
    6. Save best model
    """
    print("=" * 60)
    print("  KKBox Churn Prediction Pipeline")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Steps 1-3
    df = load_data(data_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Train all models + tune
    trained_models = train_all_models(X_train, y_train)
    tuned_lgbm = tune_lightgbm(X_train, y_train)
    trained_models['LightGBM (Tuned)'] = tuned_lgbm

    # Step 5: Evaluate
    results_df = evaluate_models(trained_models, X_test, y_test)

    # Step 6: Save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    if save:
        save_model(best_model)

    print(f"\nPipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return trained_models, results_df


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(data_path)
