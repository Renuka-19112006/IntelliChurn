import matplotlib.pyplot as plt
import seaborn as sns

from .config import DATA_PATH, FIGURES_DIR
from .data_preprocessing import load_data, clean_data

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_data(load_data())

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Churn by Contract Type")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_by_contract.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges by Churn")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_charges_by_churn.png", dpi=160)
    plt.close()

    print(f"EDA charts saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main()
