import numpy as np
import pandas as pd
import pytest

from src import batch_prediction


class MockModel:
    feature_names_in_ = np.array(["tenure", "MonthlyCharges", "TotalCharges"])

    def predict(self, features):
        return np.array([0, 1, 1])

    def predict_proba(self, features):
        return np.array([
            [0.80, 0.20],
            [0.45, 0.55],
            [0.10, 0.90],
        ])


def test_generate_batch_predictions(tmp_path, monkeypatch):
    input_file = tmp_path / "customers.csv"
    output_file = tmp_path / "predictions.csv"

    customers = pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "tenure": [12, 2, 5],
        "MonthlyCharges": [50, 85, 95],
        "TotalCharges": ["600", "170", "475"],
    })
    customers.to_csv(input_file, index=False)

    monkeypatch.setattr(
        batch_prediction.joblib,
        "load",
        lambda path: MockModel()
    )

    result = batch_prediction.generate_batch_predictions(
        input_file, output_file
    )

    assert len(result) == 3
    assert result["Predicted_Churn"].tolist() == [0, 1, 1]
    assert result["Risk_Level"].astype(str).tolist() == [
        "Low", "Medium", "High"
    ]
    assert result["Churn_Probability_Percent"].tolist() == [
        20.0, 55.0, 90.0
    ]
    assert output_file.exists()


def test_batch_prediction_missing_columns(tmp_path, monkeypatch):
    input_file = tmp_path / "incomplete.csv"

    pd.DataFrame({
        "tenure": [12],
        "MonthlyCharges": [50],
    }).to_csv(input_file, index=False)

    monkeypatch.setattr(
        batch_prediction.joblib,
        "load",
        lambda path: MockModel()
    )

    with pytest.raises(ValueError, match="missing required columns"):
        batch_prediction.generate_batch_predictions(input_file)


def test_batch_prediction_missing_model(tmp_path, monkeypatch):
    missing_dir = tmp_path / "missing"

    monkeypatch.setattr(
        batch_prediction,
        "ARTIFACTS_DIR",
        missing_dir
    )

    with pytest.raises(FileNotFoundError, match="Train the model first"):
        batch_prediction.generate_batch_predictions("customers.csv")
