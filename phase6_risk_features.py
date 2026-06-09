import pandas as pd
import numpy as np

# Load merged dataset
df = pd.read_csv("home-credit-default-risk/application_train.csv")
bureau = pd.read_csv("home-credit-default-risk/bureau.csv")

# -------------------------
# Bureau Aggregation
# -------------------------

bureau_features = bureau.groupby("SK_ID_CURR").agg({
    "SK_ID_BUREAU":"count",
    "AMT_CREDIT_SUM":"sum",
    "AMT_CREDIT_SUM_DEBT":"sum",
    "AMT_CREDIT_MAX_OVERDUE":"max"
})

bureau_features.columns = [
    "BUREAU_LOAN_COUNT",
    "TOTAL_CREDIT_SUM",
    "TOTAL_DEBT",
    "MAX_OVERDUE"
]

bureau_features = bureau_features.reset_index()

# -------------------------
# Merge
# -------------------------

df = df.merge(
    bureau_features,
    on="SK_ID_CURR",
    how="left"
)

# -------------------------
# Fill Missing Values
# -------------------------

risk_cols = [
    "BUREAU_LOAN_COUNT",
    "TOTAL_CREDIT_SUM",
    "TOTAL_DEBT",
    "MAX_OVERDUE"
]

df[risk_cols] = df[risk_cols].fillna(0)

# -------------------------
# Risk Features
# -------------------------

df["DEBT_INCOME_RATIO"] = (
    df["TOTAL_DEBT"] /
    df["AMT_INCOME_TOTAL"]
)

df["CREDIT_UTILIZATION"] = (
    df["TOTAL_DEBT"] /
    (
        df["TOTAL_CREDIT_SUM"] + 1
    )
)

df["OVERDUE_INCOME_RATIO"] = (
    df["MAX_OVERDUE"] /
    df["AMT_INCOME_TOTAL"]
)

print("\nNew Risk Features")

print(
    df[
        [
            "DEBT_INCOME_RATIO",
            "CREDIT_UTILIZATION",
            "OVERDUE_INCOME_RATIO"
        ]
    ].describe()
)
# Remove impossible values

df["DEBT_INCOME_RATIO"] = np.where(
    df["DEBT_INCOME_RATIO"] < 0,
    np.nan,
    df["DEBT_INCOME_RATIO"]
)

df["CREDIT_UTILIZATION"] = np.where(
    df["CREDIT_UTILIZATION"] < 0,
    np.nan,
    df["CREDIT_UTILIZATION"]
)

df["OVERDUE_INCOME_RATIO"] = np.where(
    df["OVERDUE_INCOME_RATIO"] < 0,
    np.nan,
    df["OVERDUE_INCOME_RATIO"]
)
print(
    df[
        [
            "DEBT_INCOME_RATIO",
            "CREDIT_UTILIZATION",
            "OVERDUE_INCOME_RATIO"
        ]
    ].describe()
)



print("\nNegative Debt Records")
print((df["TOTAL_DEBT"] < 0).sum())

print("\nNegative Credit Sum Records")
print((df["TOTAL_CREDIT_SUM"] < 0).sum())

print("\nTop 10 Credit Utilization")
print(
    df["CREDIT_UTILIZATION"]
    .sort_values(ascending=False)
    .head(10)
)


df["CREDIT_UTILIZATION"] = np.where(
    df["TOTAL_CREDIT_SUM"] > 0,
    df["TOTAL_DEBT"] / df["TOTAL_CREDIT_SUM"],
    np.nan
)
df["DEBT_INCOME_RATIO"] = np.where(
    df["AMT_INCOME_TOTAL"] > 0,
    df["TOTAL_DEBT"] / df["AMT_INCOME_TOTAL"],
    np.nan
)

# Remove negatives

df.loc[df["DEBT_INCOME_RATIO"] < 0,
       "DEBT_INCOME_RATIO"] = np.nan

df.loc[df["CREDIT_UTILIZATION"] < 0,
       "CREDIT_UTILIZATION"] = np.nan

# Cap extreme values

for col in [
    "DEBT_INCOME_RATIO",
    "CREDIT_UTILIZATION",
    "OVERDUE_INCOME_RATIO"
]:

    p99 = df[col].quantile(0.99)

    df[col] = df[col].clip(
        upper=p99
    )

print(df[
    [
        "DEBT_INCOME_RATIO",
        "CREDIT_UTILIZATION",
        "OVERDUE_INCOME_RATIO"
    ]
].describe())