import pandas as pd

from src.predict import prepare_customer
from src.config import MODEL_INPUT_FEATURES


def test_prepare_customer():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0,
    }

    result = prepare_customer(customer)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == MODEL_INPUT_FEATURES
    assert result.shape == (1, len(MODEL_INPUT_FEATURES))


def test_prepare_customer_numeric_values():
    customer = {
        "SeniorCitizen": "0",
        "tenure": "12",
        "MonthlyCharges": "75.50",
        "TotalCharges": "906.00",
    }

    result = prepare_customer(customer)

    assert result.loc[0, "tenure"] == 12
    assert result.loc[0, "MonthlyCharges"] == 75.50
    assert result.loc[0, "TotalCharges"] == 906.00


def test_prepare_customer_invalid_numeric_values():
    customer = {
        "tenure": "invalid",
        "MonthlyCharges": "unknown",
    }

    result = prepare_customer(customer)

    assert pd.isna(result.loc[0, "tenure"])
    assert pd.isna(result.loc[0, "MonthlyCharges"])
