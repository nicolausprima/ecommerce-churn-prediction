"""
export_predictions.py
---------------------
Generates predictions using the best model and exports a CSV optimized for Power BI.
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db_connector import fetch_data

def assign_risk(prob):
    if prob >= 0.70:
        return 'High Risk'
    elif prob >= 0.40:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def export_for_powerbi():
    # 1. Fetch RAW data
    print("Fetching raw data from database...")
    raw_df = fetch_data("SELECT * FROM e_commerce_dataset_e_comm")

    raw_df.dropna(subset=['Churn'], inplace=True)
    raw_df['Tenure'] = pd.to_numeric(raw_df['Tenure'], errors='coerce')
    raw_df['Tenure'].fillna(raw_df['Tenure'].median(), inplace=True)

    # 2. Split exactly like in preprocessing to get the exact test customers
    X_raw = raw_df.drop('Churn', axis=1)
    y_raw = raw_df['Churn']
    _, X_test_raw, _, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # 3. Load preprocessed test data for predictions
    print("Loading preprocessed test data...")
    X_test_scaled = pd.read_csv('../data/processed/X_test_preprocessed.csv')

    # 4. Load Best Model (assuming XGBoost is best based on tuning)
    model_path = '../models/tuned_xgboost.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train models first.")
        return

    print(f"Loading best model from {model_path}...")
    best_model = joblib.load(model_path)

    # 5. Generate predictions
    print("Generating predictions...")
    churn_probabilities = best_model.predict_proba(X_test_scaled)[:, 1]

    # 6. Combine predictions with unscaled raw data
    df = X_test_raw.copy()
    df['Churn'] = y_test_raw
    df['Churn_Probability'] = churn_probabilities
    df['Risk_Level'] = df['Churn_Probability'].apply(assign_risk)

    # 7. Select columns for Power BI
    cols_for_powerbi = [
        'CustomerID',
        'Tenure',
        'Complain',
        'CityTier',
        'Gender',
        'MaritalStatus',
        'PreferredPaymentMode',
        'OrderAmountHikeFromlastYear', 
        'DaySinceLastOrder',
        'Churn',
        'Churn_Probability',
        'Risk_Level'
    ]
    
    # Ensure all required columns exist
    available_cols = [col for col in cols_for_powerbi if col in df.columns]
    powerbi_df = df[available_cols].copy()

    # Clean nulls
    for col in powerbi_df.columns:
        if powerbi_df[col].dtype == 'object':
            powerbi_df[col].fillna('Unknown', inplace=True)
        else:
            powerbi_df[col].fillna(0, inplace=True)

    # 8. Export
    export_dir = '../data/powerbi_exports'
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, 'churn_predictions_powerbi.csv')
    
    powerbi_df.to_csv(export_path, index=False)

    print("EXPORT SUCCESSFUL")
    print("The data is clean and fully optimized for the Power BI Dashboard!")
    print(f"File saved at: {export_path}")

if __name__ == "__main__":
    export_for_powerbi()
