"""
evaluate.py
-----------
Evaluates trained models on the test set and prints performance metrics.
"""
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

def load_test_data(data_dir='../data/processed'):
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_preprocessed.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test_preprocessed.csv')).squeeze()
    return X_test, y_test

def evaluate_models(models_dir='../models', data_dir='../data/processed'):
    X_test, y_test = load_test_data(data_dir)
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print(f"No models found in {models_dir}")
        return

    print("ROC-AUC Scores Comparison (After Tuning)")
    
    for file in model_files:
        model_path = os.path.join(models_dir, file)
        model_name = file.replace('tuned_', '').replace('.pkl', '').replace('_', ' ').title()
        
        try:
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred_prob)
            print(f"{model_name:20} : {auc_score:.4f}")
            
            print(f"\nClassification Report for {model_name}")
            print(classification_report(y_test, y_pred))
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

if __name__ == "__main__":
    evaluate_models()
