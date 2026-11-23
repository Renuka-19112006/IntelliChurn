import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import DATA_PATH, ARTIFACTS_DIR, REPORTS_DIR
from src.data_preprocessing import load_data, clean_data
from src.predict import predict_customer

st.set_page_config(
    page_title="IntelliChurn",
    page_icon="📊",
    layout="wide",
)

st.title("📊 IntelliChurn")
st.caption("AI-Powered Customer Churn Prediction and Retention Analytics")

@st.cache_data
def get_data():
    return clean_data(load_data())

def model_results():
    path = REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place "
        "WA_Fn-UseC_-Telco-Customer-Churn.csv inside the dataset folder."
    )
    st.stop()

df = get_data()
results = model_results()

pages = [
    "Home",
    "Dataset Overview",
    "Customer Analytics",
    "Churn Analysis",
    "Model Performance",
    "Customer Prediction",
    "Risk & Recommendations",
]
page = st.sidebar.radio("Navigation", pages)

if page == "Home":
    st.subheader("Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", len(df))
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Churn Rate", f"{(df['Churn'].eq('Yes').mean() * 100):.1f}%")
    c4.metric("Target", "Churn")

    st.markdown(
        """
        **Goal:** predict whether a telecom customer is likely to churn and
        support retention decisions with transparent risk categories and
        practical recommendations.

        **Models:** Logistic Regression, Random Forest, XGBoost and ANN.
        """
    )

elif page == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write(df.head(20))
    st.write("Shape:", df.shape)
    st.write(df.dtypes.astype(str).to_frame("Data Type"))

elif page == "Customer Analytics":
    st.subheader("Customer Analytics")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Contract distribution")
        st.bar_chart(df["Contract"].value_counts())

    with col2:
        st.write("Internet service distribution")
        st.bar_chart(df["InternetService"].value_counts())

    st.subheader("Monthly Charges")
    st.line_chart(df["MonthlyCharges"].sort_values().reset_index(drop=True))

elif page == "Churn Analysis":
    st.subheader("Churn Analysis")
    churn_rate = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values()
    st.write("Churn rate by contract")
    st.bar_chart(churn_rate)

    st.write("Churn count by payment method")
    table = pd.crosstab(df["PaymentMethod"], df["Churn"])
    st.dataframe(table)

elif page == "Model Performance":
    st.subheader("Model Performance Comparison")
    if results is None:
        st.warning("Run the training script first.")
    else:
        st.dataframe(results, use_container_width=True)
        chart = results.set_index("Model")[["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC"]]
        st.bar_chart(chart)

elif page == "Customer Prediction":
    st.subheader("Individual Customer Churn Prediction")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
            total = st.number_input("Total Charges", min_value=0.0, value=monthly * max(tenure, 1))

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        try:
            result = predict_customer(customer)
            st.success(f"Prediction: {result['prediction']}")
            st.metric("Churn Probability", f"{result['probability'] * 100:.2f}%")
            st.info(f"Risk Category: {result['risk']}")
            st.write(f"Model used: **{result['model_name']}**")
            st.subheader("Recommended Retention Actions")
            for item in result["recommendations"]:
                st.write("•", item)
        except FileNotFoundError as exc:
            st.error(str(exc))

elif page == "Risk & Recommendations":
    st.subheader("Risk Classification")
    st.write(
        """
        The application uses the predicted churn probability:
        - **Low:** probability < 30%
        - **Medium:** 30% to < 60%
        - **High:** ≥ 60%

        These thresholds are transparent business rules and can be adjusted
        after discussing the company's intervention capacity and cost.
        """
    )

    st.subheader("Example retention logic")
    st.write("High risk → proactive outreach and targeted retention offer.")
    st.write("Medium risk → personalized offer and closer monitoring.")
    st.write("Low risk → normal engagement and loyalty communication.")
