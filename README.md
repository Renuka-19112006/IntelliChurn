# IntelliChurn - AI-Powered Customer Churn Prediction and Retention Analytics
## Project Overview

IntelliChurn is an Artificial Intelligence and Data Science project that predicts customer churn in the telecommunications industry and generates actionable retention recommendations.

The system analyzes customer information, estimates the probability of churn, classifies customers into risk levels, and helps businesses identify customers who may need retention support.

## Key Features

- Customer churn prediction using Machine Learning.
- Comparison of Logistic Regression, Random Forest and XGBoost.
- Customer risk classification into Low, Medium and High.
- Personalized, rule-based retention recommendations.
- Batch prediction for multiple customers.
- Customer segmentation and analytics.
- Interactive Streamlit dashboard.
- Model performance evaluation and visualizations.
- SHAP-based feature importance report.

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Deep Learning:** TensorFlow / Keras (supporting artifacts included; ANN execution not verified)
- **Dashboard:** Streamlit
- **Model Saving:** Joblib
- **Explainability:** SHAP

## Dataset

This project uses the IBM Telco Customer Churn dataset.

**Dataset file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Download source:**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the dataset inside:

`dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Dataset details:
- Total records: 7,043
- Total columns: 21
- Target column: `Churn`

## Project Structure

```text
IntelliChurn/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   ├── predict.py
│   ├── batch_prediction.py
│   ├── customer_segmentation.py
│   ├── recommendations.py
│   └── explainability.py
│
├── artifacts/
│   ├── best_model.pkl
│   ├── best_model_metadata.json
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── customer_segmentation_model.pkl
│
├── reports/
│   ├── model_comparison.csv
│   ├── batch_churn_predictions.csv
│   ├── customer_segment_assignments.csv
│   ├── customer_segment_profiles.csv
│   ├── shap_feature_importance.csv
│   └── figures/
│
└── tests/
    ├── test_batch_prediction.py
    ├── test_prediction.py
    ├── test_preprocessing.py
    └── test_recommendations.py
```

## Windows PowerShell Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Project

### Run Exploratory Data Analysis

```powershell
python -m src.eda
```

### Train Machine Learning Models

```powershell
python -m src.train_model
```

The model comparison report is saved in:

`reports/model_comparison.csv`

### Run Customer Prediction

```powershell
python -m src.predict
```

### Run the Streamlit Dashboard

```powershell
python -m streamlit run app.py
```

Open the application at:

`http://localhost:8501`

### Run Batch Prediction

```powershell
python -c "from src.batch_prediction import generate_batch_predictions, DATASET_PATH; generate_batch_predictions(DATASET_PATH)"
```

The generated predictions are saved in:

`reports/batch_churn_predictions.csv`

### Run Unit Tests

```powershell
python -m pytest -v
```

## Model Comparison Results

The project compares three verified models using validation metrics.

| Model | Validation Accuracy | Validation F1 | Validation ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 75.42% | 63.31% | 84.83% |
| Random Forest | 76.84% | 62.01% | 84.05% |
| XGBoost | 80.57% | 57.14% | 85.06% |

### Selected Model: Logistic Regression

Logistic Regression was selected based on the highest validation F1-score.

**Test performance:**

| Metric | Result |
|---|---:|
| Accuracy | 73.81% |
| Precision | 50.43% |
| Recall | 78.34% |
| F1-score | 61.36% |
| ROC-AUC | 84.13% |

The model was evaluated on 1,409 test records.

## Batch Prediction Results

The batch prediction module was executed on all 7,043 customer records.

| Result | Number of Customers |
|---|---:|
| Predicted Churn | 2,887 |
| Predicted No Churn | 4,156 |
| Low Risk | 2,972 |
| Medium Risk | 1,722 |
| High Risk | 2,349 |

## Streamlit Dashboard Modules

The application contains seven dashboard pages:

1. Home
2. Dataset Overview
3. Customer Analytics
4. Churn Analysis
5. Model Performance
6. Customer Prediction
7. Risk & Recommendations

## Testing

The project includes unit tests using Pytest.

**Latest test result: 15 tests passed.**

Test modules cover:
- Data preprocessing
- Customer input preparation
- Risk classification
- Retention recommendations
- Batch prediction logic and error handling

Note: The batch prediction unit tests use a mock model. Actual batch prediction was also separately executed using the saved model on 7,043 customer records.

## Project Output

The system produces:
- Customer churn predictions.
- Churn probability scores.
- Customer risk classifications.
- Retention recommendations.
- Customer segmentation reports.
- Model comparison metrics.
- Data visualizations and feature importance reports.

## Conclusion

IntelliChurn demonstrates how Machine Learning can be used to identify customers who are likely to leave a telecommunications service. By combining churn prediction, risk classification, customer segmentation and retention recommendations, the system provides useful insights to support customer retention decisions.