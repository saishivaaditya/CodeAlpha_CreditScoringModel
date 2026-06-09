import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# =====================================
# LOAD DATA
# =====================================

print("Loading Dataset...")

df = pd.read_csv("home-credit-default-risk/application_train_prev_inst.csv")

print("\nDataset Shape:")
print(df.shape)

# =====================================
# TARGET
# =====================================

y = df["TARGET"]

X = df.drop(
    columns=[
        "TARGET",
        "SK_ID_CURR"
    ]
)

# =====================================
# FEATURE TYPES
# =====================================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns

print("\nCategorical Features:", len(categorical_cols))
print("Numerical Features:", len(numerical_cols))

# =====================================
# PREPROCESSING
# =====================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_cols
        ),
        (
            "cat",
            categorical_transformer,
            categorical_cols
        )
    ]
)

# =====================================
# SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# CLASS IMBALANCE
# =====================================

scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

print("\nScale Pos Weight:", round(scale_pos_weight, 2))

# =====================================
# XGBOOST
# =====================================

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            xgb
        )
    ]
)

# =====================================
# TRAIN
# =====================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

# =====================================
# PREDICT
# =====================================

pred_probs = model.predict_proba(
    X_test
)[:, 1]

preds = model.predict(
    X_test
)

# =====================================
# EVALUATE
# =====================================

auc = roc_auc_score(
    y_test,
    pred_probs
)

print("\n=================================")
print("XGBOOST RESULTS")
print("=================================")

print("\nROC-AUC:")
print(round(auc, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        preds
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        preds
    )
)

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    model,
    "credit_scoring_xgboost.pkl"
)

print("\nModel Saved Successfully")
print("credit_scoring_xgboost.pkl")