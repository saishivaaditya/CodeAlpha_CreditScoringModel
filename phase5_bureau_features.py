import pandas as pd

# Load data
app = pd.read_csv("home-credit-default-risk/application_train.csv")
bureau = pd.read_csv("home-credit-default-risk/bureau.csv")

print("Application Shape:", app.shape)
print("Bureau Shape:", bureau.shape)

print("\nUnique Customers in Bureau:")
print(bureau["SK_ID_CURR"].nunique())

# Customer-level aggregation
bureau_features = bureau.groupby("SK_ID_CURR").agg({
    "SK_ID_BUREAU": "count",
    "AMT_CREDIT_SUM": ["mean", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
    "AMT_CREDIT_MAX_OVERDUE": "max"
})

# Flatten column names
bureau_features.columns = [
    "BUREAU_LOAN_COUNT",
    "AVG_CREDIT_SUM",
    "TOTAL_CREDIT_SUM",
    "AVG_DEBT",
    "TOTAL_DEBT",
    "MAX_OVERDUE"
]

bureau_features = bureau_features.reset_index()

print("\nBureau Features Shape:")
print(bureau_features.shape)

print("\nSample Bureau Features:")
print(bureau_features.head())