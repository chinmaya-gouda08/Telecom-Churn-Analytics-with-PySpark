# Telecom Churn Analytics with PySpark

## Project Overview

Customer churn is a major challenge in the telecommunications industry. This project focuses on predicting customer churn and identifying meaningful customer segments using **Apache PySpark** and Machine Learning.

The project uses the IBM Telco Customer Churn dataset, containing 7,043 customer records and 21 attributes. It combines supervised and unsupervised machine learning techniques to analyze customer behavior and support data-driven customer retention strategies.

## Objectives

- Predict customers who are likely to discontinue telecommunications services.
- Analyze customer characteristics and service usage patterns.
- Perform customer segmentation using clustering algorithms.
- Explore machine learning capabilities using Apache PySpark.
- Compare the performance of different classification and clustering algorithms.

## Technologies Used

- **Programming Language:** Python
- **Big Data Framework:** Apache Spark, PySpark
- **Machine Learning:** PySpark MLlib
- **Data Processing:** PySpark DataFrame API
- **Dataset:** IBM Telco Customer Churn

## Machine Learning Algorithms

### 1. Supervised Learning

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Customer churn prediction |
| Random Forest Classifier | Customer churn prediction |

### 2. Unsupervised Learning

| Algorithm | Purpose |
|---|---|
| K-Means Clustering | Customer segmentation |
| Bisecting K-Means | Hierarchical customer segmentation |

## Dataset

**IBM Telco Customer Churn Dataset**

- Total Records: 7,043
- Total Attributes: 21
- Target Variable: Churn

The dataset contains customer demographics, account information, subscribed services, billing details, and churn status.

Dataset Link: https://github.com/IBM/telco-customer-churn-on-icp4d

## Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Development
5. Model Training and Evaluation
6. Customer Segmentation
7. Performance Comparison
8. Result Analysis

## Model Performance

### Classification Results

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.74% | 79.85% |
| Precision | 79.82% | 78.76% |
| Recall | 80.74% | 79.85% |
| F1-Score | 79.87% | 78.72% |
| ROC-AUC | 0.8413 | 0.8395 |

### Clustering Results

| Evaluation Metric | K-Means | Bisecting K-Means |
|---|---:|---:|
| Number of Clusters | 2 | 2 |
| Silhouette Score | 0.4417 | 0.4417 |

## Key Findings

- Logistic Regression achieved an accuracy of 80.74%, slightly outperforming Random Forest.
- Both classification algorithms demonstrated their ability to identify customer churn patterns.
- K-Means and Bisecting K-Means generated comparable clustering results.
- Combining prediction and segmentation provides a broader understanding of customer behavior.

## Repository Structure

```text
Telecom-Churn-Analytics-with-PySpark/
│
├── README.md
├── Code/
│   ├── Logistic_Regression.py
│   ├── Random_Forest.py
│   ├── KMeans.py
│   └── Bisecting_KMeans.py
│
├── Report/
│   └── Customer_Churn_Project_Report.pdf
│
└── Results/
    └── Model_Performance.png
```

*Note: The structure above is a suggested organization. Update the filenames according to the files you actually upload.*

## Future Enhancements

- Hyperparameter tuning to improve model performance.
- Exploring additional machine learning algorithms.
- Improving customer segmentation through feature optimization.
- Developing an interactive dashboard for churn analysis.


---

⭐ If you find this project useful, consider giving this repository a star!
