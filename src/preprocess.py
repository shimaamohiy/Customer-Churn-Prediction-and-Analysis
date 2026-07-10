"""
Data Preprocessing Module for KKBox Customer Churn Prediction.
Replicates the exact cleaning and feature engineering steps
from Phase 3 (Data Preparation) notebook.
"""

import os
import pandas as pd
import numpy as np

try:
    from src.config import (
        PROJECT_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        ID_COLUMN, TARGET_COLUMN, DROP_COLUMNS
    )
except ImportError:
    from config import (
        PROJECT_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        ID_COLUMN, TARGET_COLUMN, DROP_COLUMNS
    )


# ============================================================
# DATA LOADING (exact same as notebook Phase 3)
# ============================================================
def load_logs(filepath):
    """Load user logs in chunks with date parsing (same as notebook)."""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=1_000_000):
        chunk['date'] = pd.to_datetime(chunk['date'], format='%Y%m%d')
        chunks.append(chunk)
    return pd.concat(chunks)


def load_all_user_logs():
    """Load and combine both user log files (same as notebook)."""
    logs = pd.concat([
        load_logs(os.path.join(RAW_DATA_DIR, 'User_logs.csv')),
        load_logs(os.path.join(RAW_DATA_DIR, 'User_logs_v2.csv'))
    ], ignore_index=True)
    return logs


def load_all_transactions():
    """Load and combine both transaction files (same as notebook)."""
    trans = pd.concat([
        pd.read_csv(os.path.join(RAW_DATA_DIR, 'transactions_v2.csv')),
        pd.read_csv(os.path.join(RAW_DATA_DIR, 'transactions.csv'))
    ], ignore_index=True)
    return trans


def load_members():
    """Load members dataset (same as notebook)."""
    return pd.read_csv(os.path.join(RAW_DATA_DIR, 'members_v3.csv'))


def load_train():
    """Load train labels (same as notebook)."""
    return pd.read_csv(os.path.join(RAW_DATA_DIR, 'train.csv'))


def load_final_dataset(filepath=None):
    """Load the final modeling dataset."""
    if filepath is None:
        filepath = os.path.join(PROCESSED_DATA_DIR, "final_modeling_dataset.csv")
    df = pd.read_csv(filepath)
    print(f"Final dataset loaded: {df.shape}")
    return df


# ============================================================
# CLEANING (exact same logic as notebook Phase 3)
# ============================================================
def clean_transactions(trans):
    """
    Clean transactions — exact same steps as notebook:
    1. Drop duplicates
    2. Convert date columns from int64 to datetime
    3. Filter invalid expiry dates (keep 2015-2020)
    4. Impute NaN dates with median
    """
    print("Step 1: Running Data Cleaning on Transactions...")

    trans_cleaned = trans.copy()

    # Drop duplicate rows
    trans_cleaned.drop_duplicates(inplace=True)

    # Convert date columns
    trans_cleaned["transaction_date"] = pd.to_datetime(
        trans_cleaned["transaction_date"], format="%Y%m%d", errors="coerce"
    )
    trans_cleaned["membership_expire_date"] = pd.to_datetime(
        trans_cleaned["membership_expire_date"], format="%Y%m%d", errors="coerce"
    )

    # Handle invalid expiration dates (keep 2015-2020)
    valid_expire_mask = (trans_cleaned["membership_expire_date"].dt.year >= 2015) & (
        trans_cleaned["membership_expire_date"].dt.year <= 2020
    )
    trans_cleaned.loc[~valid_expire_mask, "membership_expire_date"] = np.nan

    # Impute NaN dates with median
    median_expire_date = trans_cleaned["membership_expire_date"].median()
    trans_cleaned["membership_expire_date"] = trans_cleaned[
        "membership_expire_date"
    ].fillna(median_expire_date)

    median_trans_date = trans_cleaned["transaction_date"].median()
    trans_cleaned["transaction_date"] = trans_cleaned["transaction_date"].fillna(
        median_trans_date
    )

    print("Data Cleaning Complete! Checking for any remaining missing values:")
    print(trans_cleaned.isnull().sum())
    print(f"Cleaned Transactions Shape: {trans_cleaned.shape}")
    return trans_cleaned


def clean_members(members):
    """
    Clean members — exact same steps as notebook:
    1. Clean age (bd): keep 10-80, else NaN
    2. Impute missing ages with city-level median, then global median
    3. Fill missing gender with 'unknown'
    4. Convert registration_init_time to datetime
    5. Impute invalid dates with mode
    """
    print("Step 1: Running Data Cleaning on Members...")

    members_cleaned = members.copy()

    # Clean age: keep valid 10-80
    members_cleaned["bd"] = members_cleaned["bd"].apply(
        lambda x: x if (10 <= x <= 80) else np.nan
    )

    # Impute missing ages with city-level median
    members_cleaned["bd"] = members_cleaned.groupby("city")["bd"].transform(
        lambda x: x.fillna(x.median())
    )
    # Fill remaining NaNs with global median
    members_cleaned["bd"] = members_cleaned["bd"].fillna(members_cleaned["bd"].median())

    # Fill missing gender with 'unknown'
    members_cleaned["gender"] = members_cleaned["gender"].fillna("unknown")

    # Convert registration date
    members_cleaned["registration_init_time"] = pd.to_datetime(
        members_cleaned["registration_init_time"], format="%Y%m%d", errors="coerce"
    )
    # Impute invalid dates with mode
    members_cleaned["registration_init_time"] = members_cleaned[
        "registration_init_time"
    ].fillna(members_cleaned["registration_init_time"].mode()[0])

    print("Data Cleaning Complete! Checking if any nulls remain:")
    print(members_cleaned.isnull().sum())
    return members_cleaned


def clean_train(train):
    """Clean train labels — drop duplicates (same as notebook)."""
    train.drop_duplicates(inplace=True)
    print(f"Train Shape: {train.shape}")
    print(f"Churn rate: {train['is_churn'].mean() * 100:.2f}%")
    return train


# ============================================================
# FEATURE ENGINEERING (exact same as notebook Phase 3)
# ============================================================
def engineer_transaction_features(trans_cleaned):
    """
    Aggregate transactions to user-level features — exact same as notebook:
    - total_transactions, total_actual_paid, mean_actual_paid
    - mean_plan_days, total_cancellations, latest_auto_renew
    - latest_payment_method, latest_transaction_date
    - latest_membership_expire_date, average_under_overpayment
    """
    print("Step 2: Aggregating Transactions to User-Level Features...")

    trans_features = (
        trans_cleaned.groupby("msno")
        .agg(
            total_transactions=("transaction_date", "count"),
            total_actual_paid=("actual_amount_paid", "sum"),
            mean_actual_paid=("actual_amount_paid", "mean"),
            mean_plan_days=("payment_plan_days", "mean"),
            total_cancellations=("is_cancel", "sum"),
            latest_auto_renew=("is_auto_renew", "last"),
            latest_payment_method=("payment_method_id", "last"),
            latest_transaction_date=("transaction_date", "max"),
            latest_membership_expire_date=("membership_expire_date", "max"),
        )
        .reset_index()
    )

    # Average price variance (overpayment/underpayment delta)
    mean_paid = trans_cleaned.groupby("msno")["actual_amount_paid"].mean().values
    mean_price = trans_cleaned.groupby("msno")["plan_list_price"].mean().values
    trans_features["average_under_overpayment"] = mean_paid - mean_price

    print(f"Transaction features shape: {trans_features.shape}")
    return trans_features


def engineer_member_features(members_cleaned):
    """
    Engineer member features — exact same as notebook:
    - Extract reg_year, reg_month, reg_day
    - Calculate account_age_days (snapshot: 2017-04-30)
    - Drop registration_init_time
    """
    print("Step 2: Running Feature Engineering on Cleaned Members...")

    members_features = members_cleaned.copy()
    members_features["reg_year"] = members_features["registration_init_time"].dt.year
    members_features["reg_month"] = members_features["registration_init_time"].dt.month
    members_features["reg_day"] = members_features["registration_init_time"].dt.day

    snapshot_date = pd.to_datetime("2017-04-30")
    members_features["account_age_days"] = (
        snapshot_date - members_features["registration_init_time"]
    ).dt.days

    members_features.drop(columns=["registration_init_time"], inplace=True)

    print(f"Members features shape: {members_features.shape}")
    return members_features


# ============================================================
# MERGE & FORMAT (exact same as notebook Phase 3)
# ============================================================
def merge_all_features(train, trans_features, members_features, logs_features):
    """
    Merge all feature tables — exact same as notebook:
    1. Start with train labels
    2. Left join transactions, members, user logs on msno
    """
    integrated_df = train.copy()
    print(f"Initial Train dataset shape: {integrated_df.shape}")

    print("Merging Transactions features...")
    integrated_df = pd.merge(integrated_df, trans_features, on="msno", how="left")

    print("Merging Members features...")
    integrated_df = pd.merge(integrated_df, members_features, on="msno", how="left")

    print("Merging User Logs features...")
    integrated_df = pd.merge(integrated_df, logs_features, on="msno", how="left")

    print(f"\nAll data sources unified. Shape: {integrated_df.shape}")
    return integrated_df


def format_final_dataset(integrated_df):
    """
    Format the final dataset — exact same as notebook:
    1. Convert date columns to numeric day deltas from snapshot
    2. Handle missing values with median imputation
    3. Drop date columns that tree models can't use
    """
    print("Converting raw date columns into numeric day deltas...")

    snapshot_date = pd.to_datetime("2017-04-30")

    # Convert dates to day deltas
    if "latest_transaction_date" in integrated_df.columns:
        integrated_df["latest_transaction_date"] = pd.to_datetime(
            integrated_df["latest_transaction_date"], errors="coerce"
        )
        integrated_df["days_since_last_transaction"] = (
            snapshot_date - integrated_df["latest_transaction_date"]
        ).dt.days

    if "latest_membership_expire_date" in integrated_df.columns:
        integrated_df["latest_membership_expire_date"] = pd.to_datetime(
            integrated_df["latest_membership_expire_date"], errors="coerce"
        )
        integrated_df["days_until_expiration"] = (
            integrated_df["latest_membership_expire_date"] - snapshot_date
        ).dt.days

    # Drop the raw date columns
    date_cols_to_drop = [
        "latest_transaction_date",
        "latest_membership_expire_date",
        "latest_date",
    ]
    cols_present = [c for c in date_cols_to_drop if c in integrated_df.columns]
    integrated_df.drop(columns=cols_present, inplace=True)

    # Handle missing values with median
    print("Handling missing values created during merge...")
    for col in integrated_df.select_dtypes(include=[np.number]).columns:
        if integrated_df[col].isna().any():
            integrated_df[col] = integrated_df[col].fillna(integrated_df[col].median())

    return integrated_df


def prepare_for_model(df):
    """
    Final preparation for model training/prediction:
    - Drop msno and is_churn
    - Convert object columns to category (LightGBM native support)
    Same approach as Phase 4 notebook.
    """
    df = df.copy()
    user_ids = df[ID_COLUMN].copy() if ID_COLUMN in df.columns else None

    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    print(f"Model-ready features: {X.shape}")
    return X, y, user_ids


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  KKBox Churn Prediction - Preprocessing Module")
    print("=" * 60)
    print("\nThis module replicates the exact preprocessing from Phase 3 notebook.")
    print("Import and use in your pipeline or notebook.")
