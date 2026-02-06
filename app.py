import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction – ML",
    layout="wide"
)

st.title("Breast Cancer Prediction – ML Models")
st.write("Dataset: Breast Cancer Wisconsin Dataset (UCI / sklearn.datasets)")

# --------------------------------------------------
# LOAD SCALER AND MODELS
# --------------------------------------------------
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

# Feature names used during training
FEATURE_NAMES = scaler.feature_names_in_

# --------------------------------------------------
# STEP 1: DATASET UPLOAD (FIRST – AS PER ASSIGNMENT)
# --------------------------------------------------
st.subheader("Step 1: Upload Test Dataset (CSV)")
st.info("Upload ONLY test data. CSV must contain feature columns and a 'target' column.")

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Dataset")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
        st.stop()

    # Separate features and target
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    # --------------------------------------------------
    # FEATURE VALIDATION
    # --------------------------------------------------
    missing_cols = set(FEATURE_NAMES) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(FEATURE_NAMES)

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Drop extra columns and reorder correctly
    X_test = X_test[FEATURE_NAMES]

    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # STEP 2: MODEL SELECTION (AFTER DATASET)
    # --------------------------------------------------
    st.subheader("Step 2: Select ML Model")
    selected_model_name = st.selectbox("Choose trained model", list(models.keys()))
    model = models[selected_model_name]

    # --------------------------------------------------
    # STEP 3: PREDICTION & RESULTS
    # --------------------------------------------------
    st.subheader("Step 3: Model Evaluation Results")

    y_pred = model.predict(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1-score", f"{f1:.4f}")

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

    # --------------------------------------------------
    # PREDICTION TABLE
    # --------------------------------------------------
    st.subheader("Sample Predictions")
    result_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    st.dataframe(result_df.head())
