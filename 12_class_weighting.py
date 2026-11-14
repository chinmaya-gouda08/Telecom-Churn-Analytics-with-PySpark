
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim, lit

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)


# --------------------------------------------------
# 1. CREATE SPARK SESSION
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Customer Churn Class Weighting") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


try:

    # --------------------------------------------------
    # 2. LOAD DATASET
    # --------------------------------------------------

    df = spark.read.csv(
        "dataset/Telco-Customer-Churn.csv",
        header=True,
        inferSchema=True
    )

    print("\nDataset loaded successfully!")
    print("Total records:", df.count())


    # --------------------------------------------------
    # 3. CLEAN TOTAL CHARGES
    # --------------------------------------------------

    df = df.withColumn(
        "TotalCharges",
        when(
            col("TotalCharges").isNull() |
            (trim(col("TotalCharges").cast("string")) == ""),
            lit("0")
        ).otherwise(
            trim(col("TotalCharges").cast("string"))
        ).cast("double")
    )

    print("\nTotalCharges cleaned successfully!")


    # --------------------------------------------------
    # 4. SPLIT DATASET
    # --------------------------------------------------

    train_df, test_df = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_count = train_df.count()
    test_count = test_df.count()

    print("\nTraining records:", train_count)
    print("Testing records:", test_count)


    # --------------------------------------------------
    # 5. CALCULATE CLASS WEIGHTS
    # --------------------------------------------------

    class_counts = {
        row["Churn"]: row["count"]
        for row in train_df.groupBy("Churn").count().collect()
    }

    if "No" not in class_counts or "Yes" not in class_counts:
        raise ValueError(
            "Both No and Yes classes must be present in training data."
        )

    total_train = sum(class_counts.values())

    no_weight = total_train / (2.0 * class_counts["No"])
    yes_weight = total_train / (2.0 * class_counts["Yes"])

    print("\nClass Distribution:")
    print("No Churn:", class_counts["No"])
    print("Yes Churn:", class_counts["Yes"])

    print("\nCalculated Class Weights:")
    print(f"No: {no_weight:.4f}")
    print(f"Yes: {yes_weight:.4f}")


    # --------------------------------------------------
    # 6. ADD CLASS WEIGHT COLUMN
    # --------------------------------------------------

    train_df = train_df.withColumn(
        "class_weight",
        when(
            col("Churn") == "Yes",
            lit(yes_weight)
        ).otherwise(lit(no_weight))
    )

    test_df = test_df.withColumn(
        "class_weight",
        when(
            col("Churn") == "Yes",
            lit(yes_weight)
        ).otherwise(lit(no_weight))
    )


    # --------------------------------------------------
    # 7. DEFINE NUMERICAL AND CATEGORICAL FEATURES
    # --------------------------------------------------

    numeric_cols = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    categorical_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]


    # --------------------------------------------------
    # 8. STRING INDEXING
    # --------------------------------------------------

    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=c + "_index",
            handleInvalid="keep"
        )
        for c in categorical_cols
    ]


    # --------------------------------------------------
    # 9. ONE HOT ENCODING
    # --------------------------------------------------

    encoders = [
        OneHotEncoder(
            inputCol=c + "_index",
            outputCol=c + "_ohe",
            handleInvalid="keep"
        )
        for c in categorical_cols
    ]


    # --------------------------------------------------
    # 10. ASSEMBLE FEATURES
    # --------------------------------------------------

    assembler = VectorAssembler(
        inputCols=numeric_cols + [
            c + "_ohe" for c in categorical_cols
        ],
        outputCol="features",
        handleInvalid="error"
    )


    # --------------------------------------------------
    # 11. CONVERT TARGET INTO NUMERICAL LABELS
    # --------------------------------------------------

    label_indexer = StringIndexer(
        inputCol="Churn",
        outputCol="label",
        handleInvalid="error"
    )


    # --------------------------------------------------
    # 12. CLASS-WEIGHTED LOGISTIC REGRESSION
    # --------------------------------------------------

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0
    )


    # --------------------------------------------------
    # 13. BUILD PIPELINE
    # --------------------------------------------------

    pipeline = Pipeline(
        stages=
        indexers +
        encoders +
        [
            assembler,
            label_indexer,
            lr
        ]
    )


    # --------------------------------------------------
    # 14. TRAIN MODEL
    # --------------------------------------------------

    print("\nTraining class-weighted Logistic Regression...")

    model = pipeline.fit(train_df)

    print("Model trained successfully!")


    # --------------------------------------------------
    # 15. MAKE PREDICTIONS
    # --------------------------------------------------

    predictions = model.transform(test_df)

    print("\nPredictions generated successfully!")


    # --------------------------------------------------
    # 16. EVALUATE MODEL
    # --------------------------------------------------

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="precisionByLabel",
        metricLabel=1.0
    )

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="recallByLabel",
        metricLabel=1.0
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="fMeasureByLabel",
        metricLabel=1.0
    )

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )


    accuracy = accuracy_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)


    # --------------------------------------------------
    # 17. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n===================================")
    print(" CLASS-WEIGHTED MODEL PERFORMANCE")
    print("===================================")

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Churn Precision:{precision:.4f}")
    print(f"Churn Recall:   {recall:.4f}")
    print(f"Churn F1 Score: {f1:.4f}")
    print(f"ROC-AUC:        {auc:.4f}")


    # --------------------------------------------------
    # 18. CONFUSION MATRIX
    # --------------------------------------------------

    print("\nConfusion Matrix:")

    predictions.groupBy(
        "label",
        "prediction"
    ).count().orderBy(
        "label",
        "prediction"
    ).show()


    # --------------------------------------------------
    # 19. LABEL MAPPING
    # --------------------------------------------------

    label_model = model.stages[-2]

    print("\nLabel Mapping:")
    for index, label in enumerate(label_model.labels):
        print(f"{index} = {label}")


    print("\nClass Weighting Experiment Completed Successfully!")


finally:

    # --------------------------------------------------
    # 20. STOP SPARK
    # --------------------------------------------------

    spark.stop()
