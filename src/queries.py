"""
queries.py
----------
Centralized SQL query strings used across the project.
"""

# Main table name used across the project
TABLE_NAME = "e_commerce_dataset_e_comm"

# Fetch the entire dataset including all columns
FETCH_ALL = f"SELECT * FROM {TABLE_NAME}"

# Fetch only feature columns (excluding CustomerID which is dropped during preprocessing)
FETCH_FEATURES = f"""
    SELECT
        Tenure,
        PreferredLoginDevice,
        CityTier,
        WarehouseToHome,
        PreferredPaymentMode,
        Gender,
        HourSpendOnApp,
        NumberOfDeviceRegistered,
        PreferedOrderCat,
        SatisfactionScore,
        MaritalStatus,
        NumberOfAddress,
        Complain,
        OrderAmountHikeFromlastYear,
        CouponUsed,
        OrderCount,
        DaySinceLastOrder,
        CashbackAmount,
        Churn
    FROM {TABLE_NAME}
"""

# Fetch raw data with CustomerID (used in Business Insight notebook for Power BI export)
FETCH_WITH_CUSTOMER_ID = f"""
    SELECT
        CustomerID,
        Tenure,
        PreferredLoginDevice,
        CityTier,
        WarehouseToHome,
        PreferredPaymentMode,
        Gender,
        HourSpendOnApp,
        NumberOfDeviceRegistered,
        PreferedOrderCat,
        SatisfactionScore,
        MaritalStatus,
        NumberOfAddress,
        Complain,
        OrderAmountHikeFromlastYear,
        CouponUsed,
        OrderCount,
        DaySinceLastOrder,
        CashbackAmount,
        Churn
    FROM {TABLE_NAME}
"""
