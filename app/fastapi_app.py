from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import json
import pickle
import shap
from mlflow.tracking import MlflowClient

app = FastAPI(title="Customer Churn API with Explainability")

MODEL_NAME = "ChurnModel"
ALIAS = "Production"

# -----------------------------
# Load model directly as sklearn
# -----------------------------
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

# Version info
client = MlflowClient()
versions = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
current_version = versions.version if versions else "Unknown"

# Preprocessing artifacts
scaler = pickle.load(open("models/scaler.pkl", "rb"))
columns = json.load(open("models/columns.json"))

@app.get("/")
def home():
    return {
        "model_name": MODEL_NAME,
        "alias": ALIAS,
        "version": current_version
    }

@app.post("/predict/")
def predict(customer: dict):
    df = pd.DataFrame([customer])
    df = pd.get_dummies(df, drop_first=True)

    # Ensure consistent features
    for col in columns:
        if col not in df:
            df[col] = 0
    df = df[columns]

    df_scaled = scaler.transform(df)

    # ✅ Now predict_proba works directly
    proba = model.predict_proba(df_scaled)[0][1]  # probability of churn (class=1)
    prediction = int(proba > 0.5)

    return {"prediction": prediction, "probability": float(proba)}

@app.post("/explain/")
def explain(customer: dict):
    df = pd.DataFrame([customer])
    df = pd.get_dummies(df, drop_first=True)

    # Ensure consistent features
    for col in columns:
        if col not in df:
            df[col] = 0
    df = df[columns]

    df_scaled = scaler.transform(df)

    # ✅ Use model directly with SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_scaled)

    explanation = dict(zip(columns, shap_values[0].tolist()))
    return {"explanation": explanation}
