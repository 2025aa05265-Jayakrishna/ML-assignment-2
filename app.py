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
    confusion_matrix,
    classification_report
)

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="ML Classification App", layout="centered")

st.markdown(
    "<h1 style='text-align:center;'>Breast Cancer Classification</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;'>Upload Test Data → Select Model → View Results</h4>",
    unsafe_allow_html=True
)

# --------------------------------
# LOAD MODELS AND SCALER
# --------------------------------
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

# --------------------------------
# STEP 1: DATASET UPLOAD
# --------------------------------
st.subheader("Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV file (must include target column)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox(
        "Select target column",
        options=df.columns
    )

    X = df.drop(columns=[target_col])
    y_true = df[target_col]

    X_scaled = scaler.transform(X)

    # --------------------------------
    # STEP 2: MODEL SELECTION
    # --------------------------------
    st.subheader("Select Model")

    selected_model_name = st.selectbox(
        "Choose a model",
        options=["Select a model"] + list(models.keys())
    )

    if selected_model_name != "Select a model":

        model = models[selected_model_name]

        # --------------------------------
        # PREDICTIONS
        # --------------------------------
        y_pred = model.predict(X_scaled)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = np.nan

        # --------------------------------
        # METRICS
        # --------------------------------
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader(f"Evaluation Metrics – {selected_model_name}")

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1-score", "MCC"],
            "Value": [accuracy, auc, precision, recall, f1, mcc]
        })

        metrics_df["Value"] = metrics_df["Value"].round(4)

        st.dataframe(metrics_df, hide_index=True)

        # --------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------
        st.subheader("Classification Report")

        report_dict = classification_report(
            y_true,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.round(4)

        st.dataframe(report_df)

        # --------------------------------
        # CONFUSION MATRIX
        # --------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(selected_model_name)

        st.pyplot(fig)

else:
    st.info("Please upload a test CSV file to continue.")
