import pandas as pd
import numpy as np

# ==========================================
# LOAD DATA
# ==========================================

print("Loading installments_payments.csv...")

inst = pd.read_csv("home-credit-default-risk/installments_payments.csv")

print("\nDataset Shape:")
print(inst.shape)

print("\nColumns:")
print(inst.columns.tolist())

print("\nUnique Customers:")
print(inst["SK_ID_CURR"].nunique())

# ==========================================
# FEATURE ENGINEERING
# ==========================================

print("\nCreating payment behavior features...")

# Payment delay
# Positive = Late Payment
# Negative = Early Payment

inst["PAYMENT_DELAY"] = (
    inst["DAYS_ENTRY_PAYMENT"]
    - inst["DAYS_INSTALMENT"]
)

# Late Payment Flag

inst["LATE_PAYMENT"] = (
    inst["PAYMENT_DELAY"] > 0
).astype(int)

# Underpayment Flag

inst["UNDERPAID"] = (
    inst["AMT_PAYMENT"]
    < inst["AMT_INSTALMENT"]
).astype(int)

# Payment Difference

inst["PAYMENT_DIFF"] = (
    inst["AMT_PAYMENT"]
    - inst["AMT_INSTALMENT"]
)

# ==========================================
# CUSTOMER LEVEL AGGREGATION
# ==========================================

print("\nAggregating customer features...")

inst_features = inst.groupby(
    "SK_ID_CURR"
).agg(
    {
        "SK_ID_PREV": "count",
        "PAYMENT_DELAY": ["mean", "max"],
        "LATE_PAYMENT": "mean",
        "UNDERPAID": "mean",
        "AMT_PAYMENT": ["mean", "sum"],
        "AMT_INSTALMENT": ["mean", "sum"],
        "PAYMENT_DIFF": "mean"
    }
)

# ==========================================
# FLATTEN COLUMN NAMES
# ==========================================

inst_features.columns = [
    "INSTALLMENT_COUNT",
    "AVG_PAYMENT_DELAY",
    "MAX_PAYMENT_DELAY",
    "LATE_PAYMENT_RATE",
    "UNDERPAYMENT_RATE",
    "AVG_PAYMENT",
    "TOTAL_PAYMENT",
    "AVG_INSTALLMENT",
    "TOTAL_INSTALLMENT",
    "AVG_PAYMENT_DIFF"
]

inst_features.reset_index(
    inplace=True
)

# ==========================================
# PAYMENT RATIO
# ==========================================

inst_features["PAYMENT_RATIO"] = (
    inst_features["AVG_PAYMENT"]
    /
    (
        inst_features["AVG_INSTALLMENT"]
        + 1
    )
)

# ==========================================
# CHECK OUTPUT
# ==========================================

print("\nFeature Dataset Shape:")
print(inst_features.shape)

print("\nFirst 5 Rows:")
print(inst_features.head())

print("\nFeature Statistics:")
print(
    inst_features[
        [
            "AVG_PAYMENT_DELAY",
            "LATE_PAYMENT_RATE",
            "UNDERPAYMENT_RATE",
            "PAYMENT_RATIO"
        ]
    ].describe()
)

# ==========================================
# SAVE FILE
# ==========================================

inst_features.to_csv(
    "installment_features.csv",
    index=False
)

print("\n====================================")
print("installment_features.csv SAVED")
print("====================================")