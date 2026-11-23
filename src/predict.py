
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from .config import ARTIFACTS_DIR, MODEL_INPUT_FEATURES
from .recommendations import get_retention_recommendations


def load_best_model():
    metadata_path = ARTIFACTS_DIR / "best_model_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. "
            "Run: python -m src.train_model"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    model_type = metadata["model_type"]

    if model_type == "sklearn":
        model = joblib.load(
            ARTIFACTS_DIR / "best_model.pkl"
        )

    elif model_type == "ann":
        # TensorFlow is imported only when ANN is selected.
        try:
            from tensorflow.keras.models import load_model
        except ImportError as e:
            raise RuntimeError(
                "TensorFlow ANN cannot be loaded. "
                "Windows Application Control may be blocking "
                "a TensorFlow DLL. Use an sklearn model instead."
            ) from e

        model = load_model(
            ARTIFACTS_DIR / "ann_model.keras"
        )

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}"
        )

    return model, metadata


def prepare_customer(customer):
    row = {
        feature: customer.get(feature)
        for feature in MODEL_INPUT_FEATURES
    }

    df = pd.DataFrame([row])

    for col in [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


def predict_customer(customer):
    model, metadata = load_best_model()

    df = prepare_customer(customer)

    if metadata["model_type"] == "sklearn":

        probability = float(
            model.predict_proba(df)[0, 1]
        )

        prediction = int(probability >= 0.50)

    else:

        import joblib

        preprocessor = joblib.load(
            ARTIFACTS_DIR / "ann_preprocessor.pkl"
        )

        X = preprocessor.transform(df)

        probability = float(
            model.predict(
                X, verbose=0
            ).ravel()[0]
        )

        prediction = int(probability >= 0.50)

    risk, recommendations = get_retention_recommendations(
        customer,
        probability
    )

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": probability,
        "risk": risk,
        "recommendations": recommendations,
        "model_name": metadata["model_name"],
    }


if __name__ == "__main__":

    example = {
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

    print(predict_customer(example))
