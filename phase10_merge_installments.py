import pandas as pd

# Dataset after previous applications
df = pd.read_csv(
    "home-credit-default-risk/application_train_prev.csv"
)

# Installment features
inst = pd.read_csv(
    "installment_features.csv"
)

merged = df.merge(
    inst,
    on="SK_ID_CURR",
    how="left"
)

print("Merged Shape:")
print(merged.shape)

cols = [
    "INSTALLMENT_COUNT",
    "AVG_PAYMENT_DELAY",
    "LATE_PAYMENT_RATE",
    "UNDERPAYMENT_RATE",
    "PAYMENT_RATIO"
]

print("\nMissing Values:")
print(
    merged[cols]
    .isnull()
    .sum()
)

# Fill missing values

merged[cols] = merged[
    cols
].fillna(0)

merged.to_csv(
    "home-credit-default-risk/application_train_prev_inst.csv",
    index=False
)

print("\nSaved Successfully")