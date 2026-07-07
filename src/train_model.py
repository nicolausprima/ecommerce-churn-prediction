"""
train_model.py
--------------
Trains machine learning models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM),
performs hyperparameter tuning, and saves the best models.
"""
import os
import joblib
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

def load_data(data_dir='../data/processed'):
    print(f"Loading data from {data_dir}...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_preprocessed.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train_preprocessed.csv')).squeeze()
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_preprocessed.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test_preprocessed.csv')).squeeze()
    return X_train, y_train, X_test, y_test

def train_and_tune_models(X_train, y_train, X_test, y_test):
    # 1. Define base models
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
    }

    # 2. Define hyperparameter grids
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        },
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, -1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    }

    tuned_models = {}
    best_scores = {}

    # 3. Tuning Loop
    for model_name in models.keys():
        print(f"Starting Hyperparameter Tuning for {model_name}...")
        
        random_search = RandomizedSearchCV(
            estimator=models[model_name],
            param_distributions=param_grids[model_name],
            n_iter=10,
            scoring='roc_auc',
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        tuned_models[model_name] = best_model
        
        y_pred_prob_test = best_model.predict_proba(X_test)[:, 1]
        final_test_auc = roc_auc_score(y_test, y_pred_prob_test)
        best_scores[model_name] = final_test_auc
        
        print(f"Best Parameters: {random_search.best_params_}")
        print(f"Final Test ROC-AUC: {final_test_auc:.4f}\n")

    print("Final Model Ranking by Test ROC-AUC:")
    ranking_df = pd.DataFrame(list(best_scores.items()), columns=['Model', 'Test ROC-AUC'])
    ranking_df = ranking_df.sort_values(by='Test ROC-AUC', ascending=False).reset_index(drop=True)
    print(ranking_df)

    return tuned_models

def save_models(tuned_models, output_dir='../models'):
    os.makedirs(output_dir, exist_ok=True)
    print("Exporting Tuned Models...")
    for model_name, model_object in tuned_models.items():
        safe_name = model_name.replace(' ', '_').lower()
        model_filepath = os.path.join(output_dir, f'tuned_{safe_name}.pkl')
        joblib.dump(model_object, model_filepath)
        print(f"[{model_name}] successfully saved to {model_filepath}")
    print("All models have been successfully exported!")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    tuned_models = train_and_tune_models(X_train, y_train, X_test, y_test)
    save_models(tuned_models)
