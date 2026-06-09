import streamlit as st
import pandas as pd
import joblib

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Scoring System")

st.markdown(
"""
Predict customer default risk and generate
a credit score using the trained XGBoost model.
"""
)

# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():
    return joblib.load("credit_scoring_xgboost.pkl")

model = load_model()

# ==========================
# LOAD SAMPLE DATA
# ==========================

@st.cache_data
def load_data():
    return pd.read_csv(
        "home-credit-default-risk/application_train_prev_inst.csv"
    )

df = load_data()

# ==========================
# CUSTOMER SELECTOR
# ==========================

customer_id = st.selectbox(
    "Select Customer ID",
    df["SK_ID_CURR"].head(5000)
)

# ==========================
# PREDICT
# ==========================

if st.button("Generate Credit Score"):

    customer = df[
        df["SK_ID_CURR"] == customer_id
    ]

    X = customer.drop(
        columns=[
            "TARGET",
            "SK_ID_CURR"
        ]
    )

    probability = model.predict_proba(X)[0][1]

    score = int(
        850 - probability * 550
    )

    if score >= 800:
        risk = "Very Low Risk"

    elif score >= 700:
        risk = "Low Risk"

    elif score >= 600:
        risk = "Medium Risk"

    elif score >= 500:
        risk = "High Risk"

    else:
        risk = "Very High Risk"

    st.subheader("Results")

    st.metric(
        "Credit Score",
        score
    )

    st.metric(
        "Probability of Default",
        f"{probability:.2%}"
    )

    st.metric(
        "Risk Category",
        risk
    )