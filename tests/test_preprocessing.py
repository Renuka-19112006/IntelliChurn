import pandas as pd

from src.data_preprocessing import (
    clean_data,
    split_features_target,
    build_preprocessor,
)


def test_clean_data():
    df = pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "tenure": [1, 12, 24],
        "TotalCharges": ["100", "", "500"],
        "Churn": ["Yes", "No", "Unknown"],
    })

    cleaned = clean_data(df)

    assert "customerID" not in cleaned.columns
    assert len(cleaned) == 2
    assert cleaned["TotalCharges"].isna().sum() == 1
    assert set(cleaned["Churn"]) == {"Yes", "No"}


def test_split_features_target():
    df = pd.DataFrame({
        "tenure": [1, 12, 24],
        "Churn": ["Yes", "No", "Yes"],
    })

    X, y = split_features_target(df)

    assert "Churn" not in X.columns
    assert y.tolist() == [1, 0, 1]


def test_build_preprocessor():
    preprocessor = build_preprocessor()

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2
