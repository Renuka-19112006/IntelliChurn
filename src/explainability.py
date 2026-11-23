
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from functools import lru_cache

from .config import (
    ARTIFACTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR
)

from .data_preprocessing import (
    load_data,
    clean_data,
    split_features_target
)

from .predict import prepare_customer


@lru_cache(maxsize=1)
def load_explainer_context():

    model_path = ARTIFACTS_DIR / "best_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "Best model not found. Run model training first."
        )

    print("Loading trained model...")

    pipeline = joblib.load(model_path)

    if not hasattr(pipeline, "named_steps"):
        raise ValueError(
            "The best model is not a preprocessing pipeline."
        )

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]

    print("Loading customer dataset...")

    df = clean_data(load_data())
    X, y = split_features_target(df)

    print("Preparing SHAP background data...")

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    background = X_transformed.sample(
        n=min(100, len(X_transformed)),
        random_state=42
    )

    explainer = shap.LinearExplainer(
        classifier,
        background
    )

    return (
        preprocessor,
        classifier,
        explainer,
        X_transformed,
        feature_names
    )


def get_shap_array(explanation, feature_count):

    values = np.asarray(explanation.values)

    if values.ndim == 3:

        if values.shape[1] == feature_count:
            values = values[:, :, 1]

        elif values.shape[2] == feature_count:
            values = values[:, 1, :]

        else:
            raise ValueError(
                "Unexpected SHAP output dimensions."
            )

    if values.ndim == 1:
        values = values.reshape(1, -1)

    return values


def explain_customer(customer):

    (
        preprocessor,
        classifier,
        explainer,
        X_transformed,
        feature_names
    ) = load_explainer_context()

    # Prepare individual customer
    customer_df = prepare_customer(customer)

    transformed = preprocessor.transform(customer_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = pd.DataFrame(
        transformed,
        columns=feature_names
    )

    # Calculate individual SHAP values
    explanation = explainer(transformed)

    shap_values = get_shap_array(
        explanation,
        len(feature_names)
    )

    result = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })

    # Convert technical feature names into readable names
    result["feature"] = (
        result["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    result = result.reindex(
        result["shap_value"].abs().sort_values(
            ascending=False
        ).index
    ).reset_index(drop=True)

    return result


def main():

    print("=" * 55)
    print("INTELLICHURN - SHAP EXPLAINABLE AI")
    print("=" * 55)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading SHAP explainer...")

    (
        preprocessor,
        classifier,
        explainer,
        X_transformed,
        feature_names
    ) = load_explainer_context()

    explain_data = X_transformed.sample(
        n=min(500, len(X_transformed)),
        random_state=42
    )

    print("Calculating global SHAP values...")

    shap_explanation = explainer(explain_data)

    shap_values = get_shap_array(
        shap_explanation,
        len(feature_names)
    )

    # Global feature importance
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": np.abs(
            shap_values
        ).mean(axis=0)
    })

    importance = importance.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )

    importance.to_csv(
        REPORTS_DIR / "shap_feature_importance.csv",
        index=False
    )

    # Feature importance chart
    print("Generating feature importance chart...")

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        explain_data,
        plot_type="bar",
        show=False,
        max_display=15
    )

    plt.title(
        "Top Features Influencing Customer Churn"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # SHAP summary chart
    print("Generating SHAP summary plot...")

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        explain_data,
        show=False,
        max_display=15
    )

    plt.title(
        "SHAP Summary - Customer Churn"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nTop 10 churn influencing features:")

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    print("\nSHAP analysis completed successfully!")

    print("Charts saved in:", FIGURES_DIR)

    print(
        "Importance report saved in:",
        REPORTS_DIR
    )


if __name__ == "__main__":
    main()
