import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

FASTAPI_PREDICT_URL = "http://127.0.0.1:8000/predict/"
FASTAPI_EXPLAIN_URL = "http://127.0.0.1:8000/explain/"
FASTAPI_HEALTH_URL = "http://127.0.0.1:8000/"

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📉 Customer Churn Prediction Dashboard")

# Model info
try:
    info = requests.get(FASTAPI_HEALTH_URL).json()
    st.sidebar.success("✅ FastAPI Connected")
    st.sidebar.write(f"**Model Name:** {info['model_name']}")
    st.sidebar.write(f"**Alias:** {info['alias']}")
    st.sidebar.write(f"**Version:** v{info['version']}")
except:
    st.sidebar.error("❌ Could not connect to FastAPI")

# ----------------------------
# Input Fields
# ----------------------------
st.subheader("Enter Customer Info")
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔮 Predict Churn"):
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Prediction
    response = requests.post(FASTAPI_PREDICT_URL, json=customer_data)
    if response.status_code == 200:
        result = response.json()
        prediction = "❌ Customer Will Churn" if result["prediction"] == 1 else "✅ Customer Will Stay"
        st.success(f"Prediction: {prediction}")
        st.write(f"Probability of Churn: {result['probability']:.2f}")

        # Probability bar chart
        fig, ax = plt.subplots()
        ax.bar(["Stay", "Churn"], [1-result['probability'], result['probability']], color=["green", "red"])
        ax.set_ylabel("Probability")
        ax.set_title("Churn Probability Distribution")
        st.pyplot(fig)

        # ----------------------------
        # SHAP Explanation
        # ----------------------------
        explain_resp = requests.post(FASTAPI_EXPLAIN_URL, json=customer_data)
        if explain_resp.status_code == 200:
            shap_values = explain_resp.json()["explanation"]
            shap_df = pd.DataFrame(list(shap_values.items()), columns=["Feature", "Contribution"])
            shap_df = shap_df.sort_values("Contribution", key=abs, ascending=False).head(10)

            st.subheader("🔎 Top Feature Contributions (SHAP)")
            fig, ax = plt.subplots()
            shap_df.plot(kind="barh", x="Feature", y="Contribution", ax=ax, color="skyblue")
            ax.set_title("Top Features Driving Prediction")
            st.pyplot(fig)

    else:
        st.error("Prediction request failed. Check FastAPI server.")
