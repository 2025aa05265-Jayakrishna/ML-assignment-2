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
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.markdown("<h1 style='text-align:center;'>Breast Cancer Prediction – ML Models</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Dataset: Breast Cancer Wisconsin Dataset (UCI / sklearn.datasets)</p>",
    unsafe_allow_html=True
)
st.markdown("---")

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
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "models/xgboost.pkl"))
}

# --------------------------------------------------
# STEP 1: UPLOAD TEST DATASET
# --------------------------------------------------
st.subheader("Upload Test Dataset (CSV)")
st.info("Upload ONLY test data. CSV must contain feature columns and a 'target' column.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Preview of Uploaded Dataset")
st.dataframe(df.head())

if "target" not in df.columns:
    st.error("CSV must contain a 'target' column.")
    st.stop()

X_test = df.drop("target", axis=1)
y_test = df["target"]

expected_features = scaler.feature_names_in_
missing_cols = set(expected_features) - set(X_test.columns)

if missing_cols:
    st.error(f"Missing required feature columns: {missing_cols}")
    st.stop()

X_test = X_test[expected_features]
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# STEP 2: MODEL SELECTION 
# --------------------------------------------------
st.subheader("Select ML Model")

model_options = ["-- Select Model --"] + list(models.keys())
selected_model_name = st.selectbox("Choose trained model", model_options)

if selected_model_name == "-- Select Model --":
    st.warning("Please select a model to view results.")
    st.stop()

model = models[selected_model_name]

# --------------------------------------------------
# STEP 3: MODEL EVALUATION
# --------------------------------------------------
st.subheader(f"Evaluation Metrics – {selected_model_name}")

y_pred = model.predict(X_test_scaled)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
else:
    auc = np.nan

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ],
    "Value": [
        accuracy_score(y_test, y_pred),
        auc,
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ]
})

st.table(metrics_df)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader(f"Confusion Matrix – {selected_model_name}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(3.5, 3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)
