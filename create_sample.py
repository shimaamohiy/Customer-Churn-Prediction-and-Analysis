"""
Script to create a 20,000-row sample from the full modeling dataset.
Run this ONCE locally before pushing to GitHub.
"""
import pandas as pd
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_PATH   = os.path.join(PROJECT_DIR, "data", "processed", "final_modeling_dataset.csv")
SAMPLE_PATH = os.path.join(PROJECT_DIR, "data", "processed", "dataset_sample.csv")

print(f"Reading from: {FULL_PATH}")
df = pd.read_csv(FULL_PATH)
print(f"Full dataset shape: {df.shape}")

# Stratified sample: keep churn ratio representative
n_total   = 20000
n_churned = int(n_total * df["is_churn"].mean())
n_retained = n_total - n_churned

churned  = df[df["is_churn"] == 1].sample(n=n_churned,  random_state=42)
retained = df[df["is_churn"] == 0].sample(n=n_retained, random_state=42)
sample   = pd.concat([churned, retained]).sample(frac=1, random_state=42).reset_index(drop=True)

sample.to_csv(SAMPLE_PATH, index=False)
print(f"Sample saved -> {SAMPLE_PATH}  (shape: {sample.shape})")
print(f"Churn rate in sample: {sample['is_churn'].mean():.3%}")
