import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    DATA_PATH,
)

def load_data(path=DATA_PATH):
    """Load the IBM Telco churn CSV."""
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """Clean raw Telco data without using the target to create features."""
    data = df.copy()

    # TotalCharges is stored as text in the original dataset.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Remove the identifier; it has no useful predictive meaning.
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    # Keep rows with valid target values.
    data = data[data[TARGET].isin(["Yes", "No"])].copy()

    return data

def split_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"No": 0, "Yes": 1})
    return X, y

def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
