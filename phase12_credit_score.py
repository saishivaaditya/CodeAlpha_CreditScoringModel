import pandas as pd
import joblib

# Load model
model = joblib.load(
    "credit_scoring_xgboost.pkl"
)

# Load dataset
df = pd.read_csv(
    "home-credit-default-risk/application_train_prev_inst.csv"
)

# Sample customers
sample = df.sample(
    10,
    random_state=42
)

customer_ids = sample["SK_ID_CURR"]

X = sample.drop(
    columns=[
        "TARGET",
        "SK_ID_CURR"
    ]
)

# Predict probability
pd_values = model.predict_proba(X)[:,1]

# Credit Score
credit_scores = (
    850 - (pd_values * 550)
).astype(int)

results = pd.DataFrame(
    {
        "Customer_ID": customer_ids.values,
        "Probability_of_Default": pd_values,
        "Credit_Score": credit_scores
    }
)

print(results)

def risk_category(score):

    if score >= 800:
        return "Very Low Risk"

    elif score >= 700:
        return "Low Risk"

    elif score >= 600:
        return "Medium Risk"

    elif score >= 500:
        return "High Risk"

    else:
        return "Very High Risk"


results["Risk_Category"] = (
    results["Credit_Score"]
    .apply(risk_category)
)

def risk_category(score):

    if score >= 800:
        return "Very Low Risk"

    elif score >= 700:
        return "Low Risk"

    elif score >= 600:
        return "Medium Risk"

    elif score >= 500:
        return "High Risk"

    else:
        return "Very High Risk"

results["Risk_Category"] = (
    results["Credit_Score"]
    .apply(risk_category)
)

print(results)


import pandas as pd
import joblib

model = joblib.load("credit_scoring_xgboost.pkl")

xgb_model = model.named_steps["classifier"]

importance = pd.DataFrame({
    "Importance": xgb_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(20))