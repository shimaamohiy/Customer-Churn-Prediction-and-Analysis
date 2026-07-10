"""
Prediction Script for KKBox Customer Churn Prediction.
"""

import joblib
import pandas as pd
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_churn_model.pkl")

def predict(input_data: dict) -> dict:
    """
    Takes a dictionary of feature values,
    returns the model's prediction.
    """
    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([input_data])
    
    # Convert object columns to category (LightGBM native support)
    for col in input_df.select_dtypes(include=["object"]).columns:
        input_df[col] = input_df[col].astype("category")
        
    prediction = model.predict(input_df)
    return {"prediction": prediction.tolist()}
