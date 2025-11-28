# AWS-Sagemaker-cardiovascular-disease-prediction 🫀
Cardiovascular disease prediction using sagemaker with user health data
Data Source:https://www.kaggle.com/sulianova/cardiovascular-disease-dataset

#  Cardiovascular Disease Prediction with PCA & XGBoost

[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-28A745?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![PCA](https://img.shields.io/badge/PCA-Dimensionality_Reduction-007ACC?style=for-the-badge)](https://scikit-learn.org/stable/modules/decomposition.html#pca)

This project leverages machine learning to classify cardiovascular disease presence based on objective, examination, and subjective features. It demonstrates an end-to-end workflow using **AWS SageMaker** for dimensionality reduction (PCA) and classification (XGBoost).

---

##  Project Overview

predicting the likelihood of CVD in patients.

We utilize a dataset containing patient information such as age, gender, blood pressure, cholesterol levels, and lifestyle habits. The core of this project involves:
1.  **Exploratory Data Analysis (EDA):** Understanding data distribution and correlations.
2.  **Dimensionality Reduction:** Applying **Principal Component Analysis (PCA)** to reduce feature space while retaining critical information.
3.  **Model Training:** Implementing the **XGBoost** algorithm, a powerful gradient boosting framework, specifically tuned for binary classification tasks.
4.  **Cloud Deployment:** Utilizing **Amazon SageMaker** for scalable model training and deployment.

##  Dataset Features

The dataset consists of 70,000 records of patients data, including:

| Feature Type | Features Included |
| :--- | :--- |
| **Objective** | Age, Height, Weight, Gender |
| **Examination** | Systolic Blood Pressure (`ap_hi`), Diastolic Blood Pressure (`ap_lo`), Cholesterol, Glucose |
| **Subjective** | Smoking, Alcohol Intake, Physical Activity |
| **Target** | Presence or Absence of Cardiovascular Disease (Binary: 0 or 1) |

*Data Source: [Kaggle - Cardiovascular Disease Dataset](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)*

## Tech Stack & Tools

* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Matplotlib, Scikit-Learn
* **Cloud Platform:** AWS SageMaker (Boto3, SageMaker SDK)
* **Algorithms:**
    * **PCA (Principal Component Analysis):** Used for unsupervised dimensionality reduction to distill features into principal components.
    * **XGBoost (eXtreme Gradient Boosting):** Used as the supervised learning classifier to predict disease presence.

## Steps in the Notebook

1.  **Data Cleaning & Preprocessing:** Handling duplicates, converting age from days to years, and splitting data.
2.  **EDA & Visualization:** Visualizing feature distributions (histograms) and relationships (correlation matrices/heatmaps) to gain insights.
3.  **Dimensionality Reduction with SageMaker PCA:**
    * Configuring a SageMaker PCA estimator.
    * Training the PCA model on the feature set.
    * Transforming the dataset into principal components.
4.  **Classification with XGBoost:**
    * Training an XGBoost classifier on the training set (local mode).
    * Training an XGBoost classifier using AWS SageMaker's built-in algorithm on the reduced features.
    * Hyperparameter tuning for optimal performance.
5.  **Model Evaluation:** Assessing model accuracy, precision, and recall on the testing set.

## 📊 Results & Insights

* **PCA Impact:** Successfully reduced the dimensionality of the dataset while capturing the most significant variance.
* **XGBoost Performance:** The classifier demonstrated robust performance in distinguishing between patients with and without cardiovascular disease.
* *(Add specific accuracy/metrics from your final run here, e.g., "Achieved an accuracy of ~73% on the test set.")*

*This project is for educational purposes and not medical, demonstrating ML workflows on AWS.*
