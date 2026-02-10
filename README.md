ML Classification Models and Deployment



---------------------------------------

a.	Problem statement 

---------------------------------------



This project implements and compares multiple machine learning classification models on the Breast Cancer Wisconsin Dataset.



The objective is to evaluate model performance using standard classification metrics and deploy the trained models using a Streamlit web application for interactive testing on unseen data.



The application covers:



• Data preprocessing and exploratory analysis



• Training and evaluation of multiple ML models



• Model persistence using joblib



• Deployment using Streamlit



• GitHub repository management



------------------------------------------------

b.	Dataset description 

------------------------------------------------

Dataset name: Breast Cancer Wisconsin Dataset

Source: UCI / sklearn.datasets

Total samples: 569

Total features: 30

Problem type: Binary Classification

Primary metric: F1-score

Target classes:

0: Malignant

1: Benign

-------------------------------------------------



-------------------------------------------------

Models used: 

-------------------------------------------------

The following classification models were trained and evaluated:

• Logistic Regression

• Decision Tree

• K-Nearest Neighbors (KNN)

• Naive Bayes

• Random Forest

• XGBoost

-------------------------------------------------



All models were trained on standardized features and saved for deployment.

Evaluation Metrics

Each model is evaluated using the following metrics:

• Accuracy

• AUC (ROC-AUC Score)

• Precision

• Recall

• F1-Score

• Matthews Correlation Coefficient (MCC)

-------------------------------------------------





c.	Comparison Table for Models with the evaluation metrics 

------------------------------------------------------------------------------------------------------------------------

        Model			              Accuracy	          AUC		      Precision	      Recall	    	 F1	         	MCC

------------------------------------------------------------------------------------------------------------------------

0	  Logistic Regression	       0.964912	        0.996032	    0.972222	      0.972222	    0.972222	    0.924603

1	  Decision Tree		           0.912281	        0.915675	    0.955882	      0.902778	    0.928571	    0.817412

2	  KNN			                   0.912281	        0.955853	    0.942857	      0.916667	    0.929577	    0.813927

3	  Naive Bayes		             0.938596	        0.987765	    0.945205	      0.958333	    0.951724	    0.867553

4  	Random Forest		           0.956140	        0.993717	    0.958904	      0.972222	    0.965517	    0.905447

5	  XGBoost			               0.956140	        0.990079	    0.946667	      0.986111	    0.965986	    0.905824

-------------------------------------------------------------------------------------------------------------------------

&nbsp;



Confusion matrices and classification reports are also generated for detailed performance analysis.





-----------------------------------------------------------------------------------------------------------------------------

Observations on Model Performance: 

-----------------------------------------------------------------------------------------------------------------------------

index	ML Model Name		Observation about model performance

-----------------------------------------------------------------------------------------------------------------------------

0	Logistic Regression	Achieved the highest accuracy, F1-score, and MCC among all models. It showed excellent class 

&nbsp;				separation, stable performance, and fast convergence, making it a strong baseline for the dataset.

1	Decision Tree		Showed lower accuracy compared to other models due to overfitting on training data. Performance 

&nbsp;				dropped on test data, indicating sensitivity to data variations and depth settings.

2	kNN			Performed well with high recall and F1-score by capturing local data patterns. However, performance 

&nbsp;				depends strongly on feature scaling and choice of k value.

3	Naive Bayes		Delivered balanced performance with good recall and precision. Assumption of feature independence 

&nbsp;				limits its ability to model complex relationships, but it remains computationally efficient.

4	Random Forest (Ensemble)Improved performance over a single decision tree by reducing overfitting through ensemble learning. 

&nbsp;				Provided strong accuracy and robustness at moderate computational cost.

5	XGBoost (Ensemble)	Achieved excellent AUC and F1-score by effectively learning complex feature interactions. As a 

&nbsp;				boosting-based ensemble, it provided high predictive power but required more training time.



--------------------------------------------------------------------------------------------------------------------------



Project Structure



project-folder/



│



├── ML\\\_Assignment\\\_2.ipynb # Model training, evaluation, EDA



├── app.py # Streamlit application



├── models/ # Saved trained models



│ ├── logistic\\\_regression.pkl



│ ├── decision\\\_tree.pkl



│ ├── knn.pkl



│ ├── naive\\\_bayes.pkl



│ ├── random\\\_forest.pkl



│ └── xgboost.pkl



├── scaler.pkl # StandardScaler used during training



├── test\\\_data.csv # Sample test dataset for Streamlit upload



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
















