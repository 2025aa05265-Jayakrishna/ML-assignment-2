ML Assignment 2 – Classification Models and Deployment



Project Overview



This project implements and compares multiple machine learning classification models on the Breast Cancer Wisconsin Dataset.

The objective is to evaluate model performance using standard classification metrics and deploy the trained models using a Streamlit web application for interactive testing on unseen data.



The assignment covers:

• Data preprocessing and exploratory analysis

• Training and evaluation of multiple ML models

• Model persistence using joblib

• Deployment using Streamlit

• GitHub repository management



Dataset Information



Dataset Name: Breast Cancer Wisconsin Dataset

Source: UCI Machine Learning Repository (via sklearn.datasets)



Dataset Characteristics:

• Total samples: 569

• Total features: 30 numeric features

• Target classes:



0: Malignant



1: Benign



The dataset satisfies the assignment requirement of more than 500 samples.



Machine Learning Models Implemented



The following classification models were trained and evaluated:



• Logistic Regression

• Decision Tree

• K-Nearest Neighbors (KNN)

• Naive Bayes

• Random Forest

• XGBoost



All models were trained on standardized features and saved for deployment.



Evaluation Metrics



Each model is evaluated using the following metrics:



• Accuracy

• AUC (ROC-AUC Score)

• Precision

• Recall

• F1-Score

• Matthews Correlation Coefficient (MCC)



Confusion matrices and classification reports are also generated for detailed performance analysis.



Project Structure



project-folder/

│

├── ML\_Assignment\_2.ipynb # Model training, evaluation, EDA

├── app.py # Streamlit application

├── models/ # Saved trained models

│ ├── logistic\_regression.pkl

│ ├── decision\_tree.pkl

│ ├── knn.pkl

│ ├── naive\_bayes.pkl

│ ├── random\_forest.pkl

│ └── xgboost.pkl

├── scaler.pkl # StandardScaler used during training

├── test\_data.csv # Sample test dataset for Streamlit upload

├── requirements.txt # Python dependencies

└── README.md # Project documentation



Streamlit Application Features



• CSV dataset upload option (test data only)

• Model selection dropdown (no default pre-selected model)

• Display of dataset information

• Display of evaluation metrics in tabular form

• Confusion matrix visualization

• Classification report generation



The user workflow is:



Upload test CSV file



Select a trained model



View metrics, confusion matrix, and classification report



How to Run the Project Locally



Step 1: Install dependencies

pip install -r requirements.txt



Step 2: Run the Streamlit app

streamlit run app.py



The app will open in your browser.





