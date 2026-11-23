
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from .config import (
    ARTIFACTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

from .data_preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    build_preprocessor,
)


THRESHOLD = 0.50
VALIDATION_SIZE = 0.20


def evaluate_model(model, X, y):
    """Calculate classification performance metrics."""

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y, probabilities),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def build_models():
    """Create the machine-learning models."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def create_pipeline(estimator):
    """Combine preprocessing and model into one pipeline."""

    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])


def main():

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INTELLICHURN - MODEL TRAINING")
    print("=" * 60)

    # 1. Load and prepare dataset
    print("\n[1/6] Loading dataset...")

    df = clean_data(load_data())
    X, y = split_features_target(df)

    print("Dataset shape:", df.shape)
    print("Total customers:", len(df))
    print("Churn customers:", int(y.sum()))

    # 2. Split into training and testing data
    print("\n[2/6] Splitting dataset...")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Create a separate validation set from the remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))

    # 3. Train and compare models using validation data
    print("\n[3/6] Training machine-learning models...")

    models = build_models()
    validation_results = []
    validation_scores = {}

    for name, estimator in models.items():

        print(f"\nTraining {name}...")

        pipeline = create_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_val, y_val)

        validation_scores[name] = metrics["F1"]

        result = {
            "Model": name,
            **{
                f"Validation_{key}": value
                for key, value in metrics.items()
            },
        }

        validation_results.append(result)

        print(
            f"Validation F1: {metrics['F1']:.4f} | "
            f"Recall: {metrics['Recall']:.4f} | "
            f"ROC-AUC: {metrics['ROC_AUC']:.4f}"
        )

    # 4. Select the best model using validation F1
    print("\n[4/6] Selecting the best model...")

    best_name = max(
        validation_scores,
        key=validation_scores.get,
    )

    print("Selected model:", best_name)
    print(
        "Validation F1:",
        f"{validation_scores[best_name]:.4f}",
    )

    # 5. Retrain models using training + validation data
    print("\n[5/6] Final model training and test evaluation...")

    X_final_train = pd.concat(
        [X_train, X_val],
        axis=0,
    )

    y_final_train = pd.concat(
        [y_train, y_val],
        axis=0,
    )

    final_models = {}
    comparison_results = []

    for name, estimator in models.items():

        print(f"\nFinal training: {name}")

        pipeline = create_pipeline(estimator)

        pipeline.fit(X_final_train, y_final_train)

        test_metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        final_models[name] = pipeline

        validation_row = next(
            item for item in validation_results
            if item["Model"] == name
        )

        comparison_results.append({
            **validation_row,
            **{
                f"Test_{key}": value
                for key, value in test_metrics.items()
            },
        })

        print(
            f"Test F1: {test_metrics['F1']:.4f} | "
            f"Recall: {test_metrics['Recall']:.4f} | "
            f"ROC-AUC: {test_metrics['ROC_AUC']:.4f}"
        )

    # 6. Save models and reports
    print("\n[6/6] Saving models and reports...")

    results_df = pd.DataFrame(comparison_results)

    results_df = results_df.sort_values(
        by="Validation_F1",
        ascending=False,
    ).reset_index(drop=True)

    # Save comparison report
    report_path = REPORTS_DIR / "model_comparison.csv"

    results_df.to_csv(
        report_path,
        index=False,
    )

    # Save all trained models
    for name, model in final_models.items():

        safe_name = name.lower().replace(" ", "_")

        joblib.dump(
            model,
            ARTIFACTS_DIR / f"{safe_name}.pkl",
        )

    # Save selected best model
    best_model = final_models[best_name]

    joblib.dump(
        best_model,
        ARTIFACTS_DIR / "best_model.pkl",
    )

    best_test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    metadata = {
        "model_name": best_name,
        "model_type": "sklearn",
        "selection_metric": "Validation F1",
        "validation_f1": float(validation_scores[best_name]),
        "test_metrics": {
            key: float(value)
            for key, value in best_test_metrics.items()
        },
        "threshold": THRESHOLD,
        "training_records": int(len(X_final_train)),
        "testing_records": int(len(X_test)),
    }

    metadata_path = (
        ARTIFACTS_DIR / "best_model_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    # Display final comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    print(
        results_df[
            [
                "Model",
                "Validation_F1",
                "Test_Accuracy",
                "Test_Precision",
                "Test_Recall",
                "Test_F1",
                "Test_ROC_AUC",
            ]
        ].to_string(index=False)
    )

    print("\nBest model:", best_name)
    print("Saved model:", ARTIFACTS_DIR / "best_model.pkl")
    print("Saved metadata:", metadata_path)
    print("Comparison report:", report_path)

    print("\nINTELLICHURN MODEL TRAINING COMPLETED!")


if __name__ == "__main__":
    main()
