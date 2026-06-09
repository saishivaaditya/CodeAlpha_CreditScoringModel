import pandas as pd

# Main dataset
app = pd.read_csv("home-credit-default-risk/application_train.csv")

# Previous application features
prev_features = pd.read_csv(
    "home-credit-default-risk/previous_application_features.csv"
)

# Merge
merged = app.merge(
    prev_features,
    on="SK_ID_CURR",
    how="left"
)

print("Merged Shape:")
print(merged.shape)

print("\nMissing Values:")
print(
    merged[
        [
            "PREV_APP_COUNT",
            "APPROVED_COUNT",
            "REFUSED_COUNT",
            "APPROVAL_RATE"
        ]
    ].isnull().sum()
)

# Fill missing values
prev_cols = [
    "PREV_APP_COUNT",
    "AVG_PREV_APPLICATION",
    "AVG_PREV_CREDIT",
    "APPROVED_COUNT",
    "REFUSED_COUNT",
    "APPROVAL_RATE"
]

merged[prev_cols] = merged[
    prev_cols
].fillna(0)

# Save merged dataset
merged.to_csv(
    "home-credit-default-risk/application_train_prev.csv",
    index=False
)

print("\nSaved Successfully")