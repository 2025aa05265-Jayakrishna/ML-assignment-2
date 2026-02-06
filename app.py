import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.markdown("<h1 style='text-align:center;'>Breast Cancer Prediction – ML Models</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Dataset: Breast Cancer Wisconsin Dataset (UCI / sklearn.datasets)</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------------------
# LOAD MODELS AND SCALER
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

models = {
    "Logistic Regression": joblib.load(os.path.join(BASE_DIR, "models/logistic_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(BASE_DIR, "models/decision_tree.pkl")),
    "kNN": joblib.load(os.path.join(BASE_DIR, "models/knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(BASE_DIR, "models/naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "models/random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "models/xgboost.pkl"))
}

# --------------------------------------------------
# STEP 1: UPLOAD TEST DATASET
# --------------------------------------------------
st.subheader("Step 1: Upload Test Dataset (CSV)")
st.info("Upload ONLY test data. CSV must contain all feature columns and a `target` column.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Dataset")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
        st.stop()

    X_test = df.drop("target", axis=1)
    y_test = df["target"]

    # --------------------------------------------------
    # FEATURE VALIDATION
    # --------------------------------------------------
    expected_features = scaler.feature_names_in_
    missing_features = set(expected_features) - set(X_test.columns)

    if missing_features:
        st.error(f"Missing required columns: {missing_features}")
        st.stop()

    X_test = X_test[expected_features]
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # STEP 2: MODEL SELECTION
    # --------------------------------------------------
    st.subheader("Step 2: Select ML Model")

    selected_model_name = st.selectbox(
        "Choose trained model",
        list(models.keys())
    )

    model = models[selected_model_name]

    # --------------------------------------------------
    # STEP 3: EVALUATION
    # --------------------------------------------------
    st.subheader("Step 3: Model Evaluation Results")

    y_pred = model.predict(X_test_scaled)

    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = np.nan

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # --------------------------------------------------
    # METRICS DISPLAY
    # --------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{acc:.4f}")
    col1.metric("AUC", f"{auc:.4f}")

    col2.metric("Precision", f"{prec:.4f}")
    col2.metric("Recall", f"{rec:.4f}")

    col3.metric("F1 Score", f"{f1:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

else:
    st.warning("Please upload a test CSV file to continue.")
