
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import ARTIFACTS_DIR, REPORTS_DIR, FIGURES_DIR
from .data_preprocessing import (
    load_data,
    clean_data,
    split_features_target,
)


N_CLUSTERS = 4


def main():

    print("=" * 55)
    print("INTELLICHURN - CUSTOMER SEGMENTATION")
    print("=" * 55)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and trained preprocessing pipeline
    print("\n[1/5] Loading customer data...")

    df = clean_data(load_data())
    X, y = split_features_target(df)

    best_pipeline = joblib.load(
        ARTIFACTS_DIR / "best_model.pkl"
    )

    preprocessor = best_pipeline.named_steps["preprocessor"]

    # 2. Transform customer data
    print("\n[2/5] Preparing customer features...")

    transformed = preprocessor.transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    # 3. Apply K-Means clustering
    print("\n[3/5] Applying K-Means clustering...")

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10,
    )

    clusters = kmeans.fit_predict(transformed)

    # Save clustering model
    joblib.dump(
        kmeans,
        ARTIFACTS_DIR / "customer_segmentation_model.pkl"
    )

    # 4. Create customer-level segment report
    print("\n[4/5] Creating customer segment reports...")

    assignments = X.copy().reset_index(drop=True)

    assignments.insert(
        0,
        "Record_ID",
        [f"CUST{i:05d}" for i in range(1, len(X) + 1)]
    )

    assignments["Churn"] = y.reset_index(drop=True).values
    assignments["Segment"] = clusters + 1

    assignments.to_csv(
        REPORTS_DIR / "customer_segment_assignments.csv",
        index=False,
    )

    # Build segment profiles
    profile_data = assignments.copy()

    profiles = []

    for segment in sorted(profile_data["Segment"].unique()):

        group = profile_data[
            profile_data["Segment"] == segment
        ]

        profile = {
            "Segment": f"Segment {segment}",
            "Customer_Count": len(group),
            "Churn_Rate_Percent": round(
                group["Churn"].mean() * 100, 2
            ),
        }

        for column in [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]:
            if column in group.columns:
                profile[f"Average_{column}"] = round(
                    pd.to_numeric(
                        group[column], errors="coerce"
                    ).mean(),
                    2,
                )

        for column in ["Contract", "InternetService", "PaymentMethod"]:
            if column in group.columns:
                modes = group[column].mode()

                profile[f"Common_{column}"] = (
                    modes.iloc[0] if not modes.empty else "Unknown"
                )

        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)

    profiles_df.to_csv(
        REPORTS_DIR / "customer_segment_profiles.csv",
        index=False,
    )

    # Calculate silhouette score on a sample
    score = silhouette_score(
        transformed,
        clusters,
        sample_size=min(1000, len(X)),
        random_state=42,
    )

    metadata = {
        "algorithm": "K-Means",
        "number_of_clusters": N_CLUSTERS,
        "silhouette_score": float(score),
        "total_customers": int(len(X)),
    }

    (ARTIFACTS_DIR / "segmentation_metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    # 5. Visualize clusters using PCA
    print("\n[5/5] Generating customer segmentation chart...")

    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(transformed)

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=clusters,
        cmap="viridis",
        alpha=0.6,
        s=18,
    )

    plt.colorbar(scatter, label="Customer Segment")
    plt.title("IntelliChurn - Customer Segmentation using K-Means")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "customer_segments.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nCUSTOMER SEGMENT PROFILES")
    print(profiles_df.to_string(index=False))

    print("\nSilhouette Score:", round(score, 4))
    print("Total customer segments:", N_CLUSTERS)
    print("\nSaved files:")
    print("- customer_segmentation_model.pkl")
    print("- segmentation_metadata.json")
    print("- customer_segment_assignments.csv")
    print("- customer_segment_profiles.csv")
    print("- customer_segments.png")

    print("\nCUSTOMER SEGMENTATION COMPLETED!")


if __name__ == "__main__":
    main()
