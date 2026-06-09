import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# =====================================
# LOAD NEW DATASET
# =====================================

df = pd.read_csv(
    "home-credit-default-risk/application_train_prev.csv"
)

print("Dataset Shape:", df.shape)

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

print("Categorical:", len(categorical_cols))
print("Numerical:", len(numerical_cols))

# =====================================
# PREPROCESSING
# =====================================

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
# MODEL
# =====================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

print("\nTraining...")

model.fit(
    X_train,
    y_train
)

# =====================================
# PREDICT
# =====================================

pred_probs = model.predict_proba(
    X_test
)[:,1]

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

print("\nROC-AUC:")
print(round(auc,4))

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