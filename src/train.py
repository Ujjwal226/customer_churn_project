import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import pickle
import json
import os

from src.preprocess import preprocess_data  # your preprocessing script

# ----------------------------
# Load and preprocess data
# ----------------------------
df = pd.read_csv("data/dataset.csv")   # adjust path if needed
X_scaled, y, scaler, feature_names = preprocess_data(df)

# Train/Validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ----------------------------
# Optuna objective
# ----------------------------
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_int("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    return auc

# ----------------------------
# Run Optuna
# ----------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_trial.params
print("Best Params:", best_params)

# ----------------------------
# Train final model
# ----------------------------
best_model = XGBClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

preds = best_model.predict(X_valid)
probs = best_model.predict_proba(X_valid)[:, 1]

print(classification_report(y_valid, preds))
print("ROC-AUC:", roc_auc_score(y_valid, probs))

# ----------------------------
# Log to MLflow
# ----------------------------
mlflow.set_experiment("Customer Churn Prediction")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("roc_auc", roc_auc_score(y_valid, probs))

    # ✅ Log sklearn flavor (supports predict_proba)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="ChurnModel"
    )

    # Save preprocessing artifacts
    os.makedirs("models", exist_ok=True)
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    json.dump(list(feature_names), open("models/columns.json", "w"))

    # Register latest as Production
    client = MlflowClient()
    latest_version = client.get_latest_versions("ChurnModel")[0].version

    # ✅ Assign alias "Production" instead of stage
    client.set_registered_model_alias(
        name="ChurnModel",
        alias="Production",
        version=latest_version
    )
    print(f"Registered ChurnModel v{latest_version} with alias 'Production' ✅")
