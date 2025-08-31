# 📉 Customer Churn Prediction with MLOps  
**Tech Stack: Python, Scikit-learn, XGBoost, Optuna, MLflow, FastAPI, Streamlit, Docker, SHAP**

An end-to-end **Machine Learning + MLOps project** to predict customer churn (telecom use case).  
The system supports **training, experiment tracking, model registry, deployment via APIs, and explainable dashboards**.  

---

## 🚀 Features  

- **ML Model Development**  
  - Built **XGBoost classifier** for churn prediction.  
  - Preprocessing pipeline with scaling, encoding, missing value handling.  
  - Hyperparameter optimization using **Optuna**.  

- **MLOps with MLflow**  
  - Experiment tracking (params, metrics, ROC-AUC).  
  - Model Registry with versioning & aliasing (`Production`, `Staging`).  
  - Preprocessing artifacts (`scaler.pkl`, `columns.json`) logged alongside models.  

- **Deployment**  
  - **FastAPI**: REST API for model inference + SHAP explainability.  
  - **Streamlit**: Interactive UI for customer data entry, churn prediction, probability visualization, and top SHAP feature contributions.  
  - **Docker Compose**: Orchestrates FastAPI (port `8000`), Streamlit (port `8501`), and MLflow Tracking UI (port `5000`).  

---

## 📂 Project Structure  

customer_churn_project/
│
├── app/
│ ├── fastapi_app.py # FastAPI backend
│ ├── streamlit_app.py # Streamlit dashboard
│
├── src/
│ ├── train.py # Training + MLflow logging
│ ├── preprocess.py # Preprocessing pipeline
│
├── models/ # scaler.pkl, columns.json
├── data/ # dataset.csv
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml


---

## 🛠️ Setup & Run  

### 1️⃣ Clone Repo  
```bash
git clone https://github.com/Ujjwal226/customer-churn-mlops.git
cd customer-churn-mlops

Train model with :

 - python -m src.train

Run with Docker compose:

 - docker-compose build
 - docker-compose up

Access Services

 - FastAPI Docs → http://localhost:8000/docs

- Streamlit App → http://localhost:8501

  - MLflow UI → http://localhost:5000