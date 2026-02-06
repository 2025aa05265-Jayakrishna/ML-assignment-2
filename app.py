# ================================
# ML ASSIGNMENT – STREAMLIT APP
# Dataset: Breast Cancer Wisconsin
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    layout="centered"
)

st.title("Breast Cancer Prediction – ML Models")
st.write("Dataset: Breast Cancer Wisconsin (UCI / sklearn.datasets)")

# -------------------------------
# LOAD SCALER AND MODELS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

models = {
    "Logistic Regression": joblib.load(os.path.join(BASE_DIR, "models/logistic_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(BASE_DIR, "models/decision_tree.pkl")),
    "kNN": joblib.load(os.path.join(BASE_DIR, "models/knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(BASE_DIR, "models/naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "models/random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "models/xgboost.pkl")),
}

# -------------------------------
# MODEL SELECTION
# -------------------------------
st.subheader("Select ML Model")

model_name = st.selectbox(
    "Choose a trained model",
    list(models.keys())
)

selected_model = models[model_name]

# -------------------------------
# DATASET UPLOAD
# -------------------------------
st.subheader("Upload Test Dataset (CSV)")

st.info(
    "Upload ONLY test data.\n"
    "CSV must contain feature columns and a 'target' column."
)

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    test_df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded dataset:")
    st.dataframe(test_df.head())

    # -------------------------------
    # SPLIT FEATURES & TARGET
    # -------------------------------
    if "target" not in test_df.columns:
        st.error("CSV must contain a 'target' column.")
        st.stop()

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # -------------------------------
    # SCALE DATA
    # -------------------------------
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    y_pred = selected_model.predict(X_test_scaled)

    # -------------------------------
    # EVALUATION METRICS
    # -------------------------------
    st.subheader("Evaluation Metrics")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value": [acc, prec, rec, f1]
    })

    st.table(metrics_df)

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")

    st.pyplot(fig)

    # -------------------------------
    # FINAL MESSAGE
    # -------------------------------
    st.success("Prediction and evaluation completed successfully.")
