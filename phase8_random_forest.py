import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv("home-credit-default-risk/application_train.csv")

print("Dataset Shape:", df.shape)

# ==================================================
# TARGET & FEATURES
# ==================================================

y = df["TARGET"]

X = df.drop(
    columns=[
        "TARGET",
        "SK_ID_CURR"
    ]
)

# ==================================================
# FEATURE TYPES
# ==================================================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns

print("Categorical Features:", len(categorical_cols))
print("Numerical Features:", len(numerical_cols))

# ==================================================
# NUMERICAL PIPELINE
# ==================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# ==================================================
# CATEGORICAL PIPELINE
# ==================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# ==================================================
# PREPROCESSOR
# ==================================================

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

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==================================================
# RANDOM FOREST
# ==================================================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

# ==================================================
# PIPELINE
# ==================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ]
)

# ==================================================
# TRAIN
# ==================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

# ==================================================
# PREDICTIONS
# ==================================================

pred_probs = model.predict_proba(
    X_test
)[:, 1]

preds = model.predict(
    X_test
)

# ==================================================
# EVALUATION
# ==================================================

auc = roc_auc_score(
    y_test,
    pred_probs
)

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

print("\nProbability Statistics:")
print(
    pd.Series(pred_probs).describe()
)