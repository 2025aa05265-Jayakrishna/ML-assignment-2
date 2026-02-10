# ML Classification Models and Deployment

## a. Problem Statement

This project implements and compares multiple machine learning classification models on the Breast Cancer Wisconsin Dataset.

The objective is to evaluate model performance using standard classification metrics and deploy the trained models using a Streamlit web application for interactive testing on unseen data.

The application covers:
- Data preprocessing and exploratory analysis
- Training and evaluation of multiple ML models
- Model persistence using joblib
- Deployment using Streamlit
- GitHub repository management

---

## b. Dataset Description

| Attribute | Details |
|----------|---------|
| Dataset Name | Breast Cancer Wisconsin Dataset |
| Source | UCI / sklearn.datasets |
| Total Samples | 569 |
| Total Features | 30 |
| Problem Type | Binary Classification |
| Primary Metric | F1-score |

### Target Classes

| Label | Class |
|------|-------|
| 0 | Malignant |
| 1 | Benign |

---

## Models Used

The following classification models were trained and evaluated:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- XGBoost

All models were trained on standardized features and saved for deployment.

---

## Evaluation Metrics

Each model is evaluated using:
- Accuracy
- ROC-AUC Score
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

---

## c. Comparison Table for Models

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9649 | 0.9960 | 0.9722 | 0.9722 | 0.9722 | 0.9246 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9123 | 0.9559 | 0.9429 | 0.9167 | 0.9296 | 0.8139 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost | 0.9561 | 0.9901 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

Confusion matrices and classification reports are generated for detailed performance analysis.

---

## Observations on Model Performance

| Model | Observation |
|------|-------------|
| Logistic Regression | Achieved the highest accuracy, F1-score, and MCC. Demonstrated excellent class separation and stable performance, making it a strong baseline. |
| Decision Tree | Showed lower accuracy due to overfitting. Performance dropped on test data, indicating sensitivity to depth and data variations. |
| KNN | Performed well with high recall and F1-score. Performance depends strongly on feature scaling and choice of k value. |
| Naive Bayes | Delivered balanced precision and recall. Feature independence assumption limits complex relationship modeling. |
| Random Forest | Improved performance over a single decision tree by reducing overfitting through ensemble learning. |
| XGBoost | Achieved excellent AUC and F1-score by learning complex feature interactions, with higher training cost. |

---

## Project Structure

```
project-folder/
│
├── ML_Assignment_2.ipynb
├── app.py
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── scaler.pkl
├── test_data.csv
├── requirements.txt
└── README.md
```

---

## Streamlit Application Features

- CSV dataset upload option for test data
- Model selection dropdown with no default selection
- Dataset information display
- Evaluation metrics in tabular format
- Confusion matrix visualization
- Classification report generation

### User Workflow

1. Upload test CSV file
2. Select a trained model
3. View metrics, confusion matrix, and classification report

---

## How to Run the Project Locally

### Step 1. Install Dependencies

```
pip install -r requirements.txt
```

### Step 2. Run the Streamlit App

```
streamlit run app.py
```

The application will open in your browser.
