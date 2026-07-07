"""
preprocessing.py
----------------
Handles all data cleaning, imputation, encoding, scaling, and SMOTE resampling.
Mirrors the exact steps performed in notebooks/02_Preprocessing.ipynb.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE


# Columns that need to be coerced to numeric (may arrive as object dtype from MySQL)
NUMERIC_COLS = [
    'Tenure',
    'WarehouseToHome',
    'HourSpendOnApp',
    'OrderAmountHikeFromlastYear',
    'CouponUsed',
    'OrderCount',
    'DaySinceLastOrder'
]

# Columns filled with 0 (new customers / no prior activity)
ZERO_FILL_COLS = ['Tenure', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount']

# Columns to apply RobustScaler
SCALE_COLS = [
    'Tenure',
    'WarehouseToHome',
    'HourSpendOnApp',
    'NumberOfDeviceRegistered',
    'NumberOfAddress',
    'OrderAmountHikeFromlastYear',
    'CouponUsed',
    'OrderCount',
    'DaySinceLastOrder',
    'CashbackAmount'
]

# Categorical columns to one-hot encode
CATEGORICAL_COLS = [
    'PreferredLoginDevice',
    'PreferredPaymentMode',
    'Gender',
    'PreferedOrderCat',
    'MaritalStatus'
]


def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns that may have been read as object dtype to numeric."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def impute(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Impute missing values following the strategy decided in 02_Preprocessing.ipynb:
    - Zero fill: Tenure, OrderAmountHikeFromlastYear, CouponUsed, OrderCount
    - Sentinel (-1): DaySinceLastOrder (indicating 'never ordered')
    - CityTier-grouped median: WarehouseToHome
    - Overall training median: HourSpendOnApp
    """
    # 1. Zero fill
    X_train[ZERO_FILL_COLS] = X_train[ZERO_FILL_COLS].fillna(0)
    X_test[ZERO_FILL_COLS] = X_test[ZERO_FILL_COLS].fillna(0)

    # 2. Sentinel for DaySinceLastOrder
    X_train['DaySinceLastOrder'] = X_train['DaySinceLastOrder'].fillna(-1)
    X_test['DaySinceLastOrder'] = X_test['DaySinceLastOrder'].fillna(-1)

    # 3. CityTier-grouped median for WarehouseToHome (fit on train only)
    train_city_median = X_train.groupby('CityTier')['WarehouseToHome'].transform('median')
    X_train['WarehouseToHome'] = X_train['WarehouseToHome'].fillna(train_city_median)

    city_median_map = X_train.groupby('CityTier')['WarehouseToHome'].median()
    overall_dist_median = X_train['WarehouseToHome'].median()
    test_city_median = X_test['CityTier'].map(city_median_map)
    X_test['WarehouseToHome'] = (
        X_test['WarehouseToHome'].fillna(test_city_median).fillna(overall_dist_median)
    )

    # 4. Overall median for HourSpendOnApp (fit on train only)
    hour_median = X_train['HourSpendOnApp'].median()
    X_train['HourSpendOnApp'] = X_train['HourSpendOnApp'].fillna(hour_median)
    X_test['HourSpendOnApp'] = X_test['HourSpendOnApp'].fillna(hour_median)

    return X_train, X_test


def encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot encode categorical columns (drop_first=True to avoid multicollinearity)."""
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, drop_first=True)

    # Convert any remaining boolean columns to int
    bool_cols = X_train.select_dtypes(include=['bool']).columns
    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols] = X_test[bool_cols].astype(int)

    return X_train, X_test


def scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Apply RobustScaler to numerical columns (fit on train only to prevent data leakage)."""
    scaler = RobustScaler()
    cols = [c for c in SCALE_COLS if c in X_train.columns]
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_test, scaler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """Apply SMOTE oversampling strictly to the training set to prevent data leakage."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("Class Distribution After SMOTE")
    print(pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled


def run_preprocessing(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                      save_path: str = None):
    """
    Full preprocessing pipeline:
    1. Drop CustomerID
    2. Cast numeric columns
    3. Train/test split (stratified)
    4. Impute missing values
    5. One-hot encode categoricals
    6. Apply SMOTE to training set
    7. Scale numerical columns
    8. (Optional) Save to CSV

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Drop CustomerID if present (not a predictive feature)
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # Cast columns that may arrive as object dtype
    df = cast_numeric(df)

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Impute
    X_train, X_test = impute(X_train, X_test)

    # Encode
    X_train, X_test = encode(X_train, X_test)

    # SMOTE (training set only)
    X_train, y_train = apply_smote(X_train, y_train, random_state=random_state)

    # Scale
    X_train, X_test, _ = scale(X_train, X_test)

    print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

    # Save to disk if a path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        X_train.to_csv(os.path.join(save_path, 'X_train_preprocessed.csv'), index=False)
        X_test.to_csv(os.path.join(save_path, 'X_test_preprocessed.csv'), index=False)
        pd.Series(y_train, name='Churn').to_csv(
            os.path.join(save_path, 'y_train_preprocessed.csv'), index=False
        )
        y_test.to_csv(os.path.join(save_path, 'y_test_preprocessed.csv'), index=False)
        print(f"Preprocessed data saved to '{save_path}'")

    return X_train, X_test, y_train, y_test
