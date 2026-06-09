import pandas as pd
import numpy as np

df = pd.read_csv("home-credit-default-risk/application_train.csv")

# ------------------------------
# Age in Years
# ------------------------------

df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365

# ------------------------------
# Employment Years
# ------------------------------

df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"]) / 365

# ------------------------------
# Credit to Income Ratio
# ------------------------------

df["CREDIT_INCOME_RATIO"] = (
    df["AMT_CREDIT"] /
    df["AMT_INCOME_TOTAL"]
)

# ------------------------------
# Annuity to Income Ratio
# ------------------------------

df["ANNUITY_INCOME_RATIO"] = (
    df["AMT_ANNUITY"] /
    df["AMT_INCOME_TOTAL"]
)

# ------------------------------
# Goods Price to Credit Ratio
# ------------------------------

df["GOODS_CREDIT_RATIO"] = (
    df["AMT_GOODS_PRICE"] /
    df["AMT_CREDIT"]
)

print("\nNew Features Created\n")

print(
    df[
        [
            "AGE_YEARS",
            "EMPLOYMENT_YEARS",
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "GOODS_CREDIT_RATIO"
        ]
    ].head()
)
print("\nAGE STATISTICS")
print(df["AGE_YEARS"].describe())

print("\nEMPLOYMENT STATISTICS")
print(df["EMPLOYMENT_YEARS"].describe())

print("\nCREDIT-INCOME RATIO")
print(df["CREDIT_INCOME_RATIO"].describe())