import pandas as pd

prev = pd.read_csv("home-credit-default-risk/previous_application.csv")

print("Shape:")
print(prev.shape)

print("\nColumns:")
print(prev.columns.tolist())

print("\nApplication Status Counts:")
print(prev["NAME_CONTRACT_STATUS"].value_counts())

print("\nUnique Customers:")
print(prev["SK_ID_CURR"].nunique())
# ======================================
# BASIC AGGREGATIONS
# ======================================

prev_features = prev.groupby(
    "SK_ID_CURR"
).agg(
    {
        "SK_ID_PREV": "count",
        "AMT_APPLICATION": "mean",
        "AMT_CREDIT": "mean"
    }
)

prev_features.columns = [
    "PREV_APP_COUNT",
    "AVG_PREV_APPLICATION",
    "AVG_PREV_CREDIT"
]

prev_features.reset_index(inplace=True)

print(prev_features.head())
approved = prev[
    prev["NAME_CONTRACT_STATUS"] == "Approved"
]

approved_count = approved.groupby(
    "SK_ID_CURR"
)["SK_ID_PREV"].count()

approved_count = approved_count.reset_index()

approved_count.columns = [
    "SK_ID_CURR",
    "APPROVED_COUNT"
]
refused = prev[
    prev["NAME_CONTRACT_STATUS"] == "Refused"
]

refused_count = refused.groupby(
    "SK_ID_CURR"
)["SK_ID_PREV"].count()

refused_count = refused_count.reset_index()

refused_count.columns = [
    "SK_ID_CURR",
    "REFUSED_COUNT"
]
prev_features = prev_features.merge(
    approved_count,
    on="SK_ID_CURR",
    how="left"
)

prev_features = prev_features.merge(
    refused_count,
    on="SK_ID_CURR",
    how="left"
)

prev_features[
    ["APPROVED_COUNT", "REFUSED_COUNT"]
] = prev_features[
    ["APPROVED_COUNT", "REFUSED_COUNT"]
].fillna(0)
prev_features["APPROVAL_RATE"] = (
    prev_features["APPROVED_COUNT"]
    /
    prev_features["PREV_APP_COUNT"]
)
prev_features.to_csv(
    "home-credit-default-risk/previous_application_features.csv",
    index=False
)

print("\nSaved Successfully")
print(prev_features.shape)
