# 🛒 E-Commerce Customer Churn Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn%20%7C%20XGBoost-orange.svg)
![Data Viz](https://img.shields.io/badge/Data%20Visualization-Power%20BI-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end Machine Learning pipeline designed to predict customer churn for an E-Commerce platform. This project goes beyond just building a model; it translates raw data into actionable business insights through rigorous preprocessing, hyperparameter tuning, and an interactive Power BI Dashboard.

---

## 📑 Table of Contents
- [Business Value](#-business-value)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Machine Learning Results](#-machine-learning-results)
- [Power BI Dashboard](#-power-bi-dashboard)
- [Getting Started](#-getting-started)

---

## 💡 Business Value
Customer retention is often much cheaper than customer acquisition. By accurately predicting which customers are at high risk of churning, the business can:
1. **Target Retention Campaigns:** Offer proactive discounts or perks to high-risk customers.
2. **Calculate Revenue at Risk:** Understand the direct financial impact of churn.
3. **Identify Churn Drivers:** Use A.I. to discover *why* customers are leaving (e.g., long warehouse-to-home distances or app usage patterns).

---

## 📊 Dataset
The dataset is publicly sourced from Kaggle:  
**[E-commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)**  
*Credits to Ankit Verma for providing this comprehensive dataset.*

---

## 🏗️ Project Architecture
```text
📦 ecommerce-churn-prediction
 ┣ 📂 dashboard/       # Power BI dashboard file (.pbix) and final screenshots
 ┣ 📂 data/            # Contains processed data ready for modeling and BI tools
 ┣ 📂 images/          # Auto-generated plots for evaluation metrics
 ┣ 📂 models/          # Saved model artifacts (.pkl) ready for production
 ┣ 📂 notebooks/       # Jupyter Notebooks (EDA, Preprocessing, Modelling)
 ┣ 📂 src/             # Modularized, production-ready Python scripts
 ┣ 📜 config.py        # Database connection settings
 ┣ 📜 README.md        # Project documentation
 ┗ 📜 requirements.txt # Python dependencies
```

---

## 🤖 Machine Learning Results
We trained and evaluated multiple classifiers: Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM. All models were optimized using `RandomizedSearchCV` on a SMOTE-balanced training set.

### 1. ROC-AUC Curve
The ROC-AUC metric measures the model's ability to distinguish between retained and churned customers. **XGBoost** and **LightGBM** demonstrated the strongest predictive power.
<p align="center">
  <img src="images/roc_auc_curve.png" alt="ROC-AUC Curve" width="700">
</p>

### 2. Precision-Recall Curve
Due to class imbalance, the Precision-Recall curve is a critical metric. It highlights the trade-off between accurately identifying churners (Precision) without missing too many of them (Recall).
<p align="center">
  <img src="images/pr_curve.png" alt="Precision-Recall Curve" width="700">
</p>

### 3. Feature Importance
Understanding *why* a customer leaves is crucial. The chart below (extracted from the tuned XGBoost model) highlights the top drivers of customer churn (e.g., Tenure, Complain status, etc.).
<p align="center">
  <img src="images/feature_importance.png" alt="Feature Importance" width="700">
</p>

---

## 📈 Power BI Dashboard
The machine learning predictions (Churn Probability & Risk Level) are seamlessly exported into an interactive **Power BI Dashboard**. This empowers non-technical stakeholders to drill down into the data and execute targeted marketing strategies.

<p align="center">
  <img src="dashboard/Customer%20Churn.png" alt="Power BI Dashboard" width="800">
</p>

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Power BI Desktop (to view the dashboard)
- MySQL (for the raw data connection)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/nicolausprima/ecommerce-churn-prediction.git
   cd ecommerce-churn-prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Database:**
   - Rename `config.example.py` to `config.py`.
   - Update `config.py` with your local MySQL credentials.

4. **Run the Pipeline:**
   Execute the modular scripts in `src/` or explore the step-by-step process in the `notebooks/` directory.

---
*Created by [Nicolaus Prima](https://github.com/nicolausprima)*
