import pandas as pd
import numpy as np

df = pd.read_csv("home-credit-default-risk/application_train.csv")

# Replace anomaly

df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
    365243,
    np.nan
)

# Convert to years

df["EMPLOYMENT_YEARS"] = (
    -df["DAYS_EMPLOYED"]
) / 365

print(df["EMPLOYMENT_YEARS"].describe())