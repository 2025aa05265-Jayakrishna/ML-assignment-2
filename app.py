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
        st.error("CSV must contain a 'target' column.")
        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    expected_features = scaler.feature_names_in_

    missing = set(expected_features) - set(X.columns)
    extra = set(X.columns) - set(expected_features)

    if missing:
        st.error(f"Missing required columns: {list(missing)}")
        st.stop()

    if extra:
        st.warning(f"Extra columns ignored: {list(extra)}")
        X = X[expected_features]

    # --------------------------------------------------
    # STEP 2: MODEL SELECTION (AFTER DATASET)
    # --------------------------------------------------
    st.subheader("Step 2: Select ML Model")
    model_name = st.selectbox("Select ML model", list(models.keys()))
    model = models[selected_model_name]

    # --------------------------------------------------
    # STEP 3: PREDICTION & RESULTS
    # --------------------------------------------------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy, 4))
    col2.metric("Precision", round(precision, 4))
    col3.metric("Recall", round(recall, 4))
    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1, 4))
    col5.metric("MCC", round(mcc, 4))

    if auc is not None:
    	col6.metric("AUC", round(auc, 4))
    else:
    	col6.metric("AUC", "Not Supported")

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"])
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
