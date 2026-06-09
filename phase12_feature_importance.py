import pandas as pd
import joblib

# Load model
model = joblib.load("credit_scoring_xgboost.pkl")

# Load dataset
df = pd.read_csv("home-credit-default-risk/application_train_prev_inst.csv")

X = df.drop(
    columns=[
        "TARGET",
        "SK_ID_CURR"
    ]
)

# Get transformed feature names
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get XGBoost model
xgb_model = model.named_steps[
    "classifier"
]

# Feature importance
importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": xgb_model.feature_importances_
    }
)

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 25 Important Features\n")

print(
    importance_df.head(25)
)

# Save
importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nSaved: feature_importance.csv")