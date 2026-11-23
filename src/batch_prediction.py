
from pathlib import Path

import joblib
import pandas as pd

from .config import ARTIFACTS_DIR, REPORTS_DIR


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def generate_batch_predictions(input_csv, output_csv=None):
    model_path = ARTIFACTS_DIR / "best_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError("Train the model first using: python -m src.train_model")

    print("Loading trained model...")
    model = joblib.load(model_path)

    print("Reading customer CSV...")
    customers = pd.read_csv(input_csv)

    # Remove target and customer ID columns if present.
    features = customers.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    ).copy()

    # Convert TotalCharges into numeric values.
    if "TotalCharges" in features.columns:
        features["TotalCharges"] = pd.to_numeric(
            features["TotalCharges"],
            errors="coerce"
        )

    # Keep exactly the feature columns used during model training.
    expected_columns = list(model.feature_names_in_)
    missing_columns = [
        column for column in expected_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {missing_columns}"
        )

    features = features[expected_columns]

    print("Predicting customer churn...")
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    results = customers.copy()
    results["Predicted_Churn"] = predictions
    results["Churn_Probability_Percent"] = (
        probabilities * 100
    ).round(2)

    results["Risk_Level"] = pd.cut(
        probabilities,
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    if output_csv is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_csv = REPORTS_DIR / "batch_churn_predictions.csv"

    results.to_csv(output_csv, index=False)

    print("\nBATCH PREDICTION SUMMARY")
    print("-----------------------")
    print("Total customers:", len(results))
    print("Predicted churn:", int((predictions == 1).sum()))
    print("Low risk:", int((results["Risk_Level"] == "Low").sum()))
    print("Medium risk:", int((results["Risk_Level"] == "Medium").sum()))
    print("High risk:", int((results["Risk_Level"] == "High").sum()))

    print("\nOutput saved to:", output_csv)
    print("\nBATCH PREDICTION COMPLETED!")

    return results


if __name__ == "__main__":
    generate_batch_predictions(DATASET_PATH)
