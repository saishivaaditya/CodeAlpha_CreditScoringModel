import pandas as pd

# Load files
app = pd.read_csv("home-credit-default-risk/application_train.csv")
bureau = pd.read_csv("home-credit-default-risk/bureau.csv")

# Bureau Aggregation
bureau_features = bureau.groupby("SK_ID_CURR").agg({
    "SK_ID_BUREAU": "count",
    "AMT_CREDIT_SUM": ["mean", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
    "AMT_CREDIT_MAX_OVERDUE": "max"
})

bureau_features.columns = [
    "BUREAU_LOAN_COUNT",
    "AVG_CREDIT_SUM",
    "TOTAL_CREDIT_SUM",
    "AVG_DEBT",
    "TOTAL_DEBT",
    "MAX_OVERDUE"
]

bureau_features = bureau_features.reset_index()

# Merge
app_merged = app.merge(
    bureau_features,
    on="SK_ID_CURR",
    how="left"
)

print("Merged Shape:")
print(app_merged.shape)

print("\nMissing Values After Merge:")
print(
    app_merged[
        [
            "BUREAU_LOAN_COUNT",
            "AVG_CREDIT_SUM",
            "TOTAL_DEBT",
            "MAX_OVERDUE"
        ]
    ].isnull().sum()
)