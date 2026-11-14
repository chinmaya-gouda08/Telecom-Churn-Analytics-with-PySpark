
from pyspark.sql import SparkSession

# --------------------------------------------------
# 1. CREATE SPARK SESSION
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Final Model Comparison") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

try:

    # --------------------------------------------------
    # 2. CLASSIFICATION MODEL COMPARISON
    # --------------------------------------------------

    classification_data = [
        (
            "Logistic Regression",
            80.74, 69.50, 53.12, 60.22, 84.13
        ),
        (
            "Random Forest",
            79.85, 68.42, 49.32, 57.32, 83.95
        ),
        (
            "LR Threshold 0.4",
            79.03, 61.07, 65.04, 62.99, 84.13
        ),
        (
            "Class Weighted LR",
            74.28, 52.09, 77.51, 62.31, 84.20
        )
    ]

    classification_columns = [
        "Model",
        "Accuracy",
        "Churn_Precision",
        "Churn_Recall",
        "Churn_F1",
        "ROC_AUC"
    ]

    classification_df = spark.createDataFrame(
        classification_data,
        classification_columns
    )

    print("\n========================================")
    print(" FINAL CLASSIFICATION MODEL COMPARISON")
    print("========================================")

    classification_df.show(truncate=False)


    # --------------------------------------------------
    # 3. FIND BEST CLASSIFICATION MODELS
    # --------------------------------------------------

    print("\nBEST CLASSIFICATION RESULTS")

    best_accuracy = classification_df.orderBy(
        "Accuracy",
        ascending=False
    ).first()

    best_precision = classification_df.orderBy(
        "Churn_Precision",
        ascending=False
    ).first()

    best_recall = classification_df.orderBy(
        "Churn_Recall",
        ascending=False
    ).first()

    best_f1 = classification_df.orderBy(
        "Churn_F1",
        ascending=False
    ).first()

    best_auc = classification_df.orderBy(
        "ROC_AUC",
        ascending=False
    ).first()

    print(
        f"Best Accuracy: {best_accuracy['Model']} "
        f"({best_accuracy['Accuracy']}%)"
    )

    print(
        f"Best Churn Precision: {best_precision['Model']} "
        f"({best_precision['Churn_Precision']}%)"
    )

    print(
        f"Best Churn Recall: {best_recall['Model']} "
        f"({best_recall['Churn_Recall']}%)"
    )

    print(
        f"Best Churn F1: {best_f1['Model']} "
        f"({best_f1['Churn_F1']}%)"
    )

    print(
        f"Best ROC-AUC: {best_auc['Model']} "
        f"({best_auc['ROC_AUC']}%)"
    )


    # --------------------------------------------------
    # 4. CLUSTERING MODEL COMPARISON
    # --------------------------------------------------

    clustering_data = [
        ("K-Means", 2, 0.4417),
        ("Bisecting K-Means", 2, 0.4417)
    ]

    clustering_columns = [
        "Algorithm",
        "Best_K",
        "Silhouette_Score"
    ]

    clustering_df = spark.createDataFrame(
        clustering_data,
        clustering_columns
    )

    print("\n========================================")
    print(" FINAL CLUSTERING MODEL COMPARISON")
    print("========================================")

    clustering_df.show(truncate=False)


    # --------------------------------------------------
    # 5. FINAL MODEL SELECTION
    # --------------------------------------------------

    print("\n========================================")
    print(" FINAL MODEL SELECTION")
    print("========================================")

    print("\nClassification:")

    print(
        "Logistic Regression with threshold 0.4 "
        "has the highest F1-score among the tested "
        "classification experiments."
    )

    print(
        "Class-weighted Logistic Regression has "
        "the highest churn recall and ROC-AUC."
    )

    print("\nClustering:")

    print(
        "K-Means and Bisecting K-Means both achieved "
        "a silhouette score of 0.4417 at K=2."
    )

    print("\nModel comparison completed successfully!")

finally:

    # --------------------------------------------------
    # 6. STOP SPARK
    # --------------------------------------------------

    spark.stop()
