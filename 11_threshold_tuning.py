
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, when, sum as spark_sum
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ==========================================
# 1. CREATE SPARK SESSION
# ==========================================

spark = SparkSession.builder \
    .appName("Customer Churn Threshold Tuning") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = spark.read.csv(
    "dataset/Telco-Customer-Churn.csv",
    header=True,
    inferSchema=True
)

print("\nDataset loaded successfully!")


# ==========================================
# 3. DATA CLEANING
# ==========================================

df = df.withColumn(
    "TotalCharges",
    when(
        trim(col("TotalCharges")) == "",
        "0"
    ).otherwise(trim(col("TotalCharges")))
)

df = df.withColumn(
    "TotalCharges",
    col("TotalCharges").cast("double")
)

df = df.fillna({"TotalCharges": 0.0})


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\n========== DATASET INFORMATION ==========")

print("Total records:", df.count())
print("Training records:", train_df.count())
print("Testing records:", test_df.count())


# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================

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

# String Indexing
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_Index",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

# One-Hot Encoding
encoder = OneHotEncoder(
    inputCols=[c + "_Index" for c in categorical_cols],
    outputCols=[c + "_OHE" for c in categorical_cols],
    handleInvalid="keep"
)

# Assemble Features
assembler = VectorAssembler(
    inputCols=numeric_cols + [
        c + "_OHE" for c in categorical_cols
    ],
    outputCol="features"
)

# Convert Churn into numeric labels
label_indexer = StringIndexer(
    inputCol="Churn",
    outputCol="label",
    handleInvalid="error"
)


# ==========================================
# 6. LOGISTIC REGRESSION
# ==========================================

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

pipeline = Pipeline(
    stages=indexers + [
        encoder,
        assembler,
        label_indexer,
        lr
    ]
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining Logistic Regression...")

model = pipeline.fit(train_df)

print("Model training completed!")

# Get the actual label index for Churn = Yes
label_model = model.stages[-2]
positive_index = label_model.labels.index("Yes")

print("\nLabel Mapping:", label_model.labels)
print("Positive class (Yes) index:", positive_index)


# ==========================================
# 8. GENERATE PREDICTIONS
# ==========================================

predictions = model.transform(test_df)

# Extract probability of Churn = Yes
predictions = predictions.withColumn(
    "churn_probability",
    vector_to_array(col("probability"))[positive_index]
)

print("\nSample Predictions:")

predictions.select(
    "Churn",
    "label",
    "churn_probability",
    "prediction"
).show(10, truncate=False)


# ==========================================
# 9. ROC-AUC EVALUATION
# ==========================================

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

roc_auc = auc_evaluator.evaluate(predictions)

print("\nROC-AUC Score:", round(roc_auc, 4))


# ==========================================
# 10. THRESHOLD TUNING
# ==========================================

thresholds = [0.3, 0.4, 0.5, 0.6]

print("\n========== THRESHOLD TUNING RESULTS ==========")

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1 Score':<12}"
)

print("-" * 60)

results = []

for threshold in thresholds:

    # Apply custom threshold
    evaluated = predictions.withColumn(
        "custom_prediction",
        when(
            col("churn_probability") >= threshold,
            float(positive_index)
        ).otherwise(
            float(1 - positive_index)
        )
    )

    # Calculate confusion matrix values
    metrics = evaluated.agg(
        spark_sum(
            when(
                (col("label") == positive_index) &
                (col("custom_prediction") == positive_index),
                1
            ).otherwise(0)
        ).alias("TP"),

        spark_sum(
            when(
                (col("label") != positive_index) &
                (col("custom_prediction") == positive_index),
                1
            ).otherwise(0)
        ).alias("FP"),

        spark_sum(
            when(
                (col("label") != positive_index) &
                (col("custom_prediction") != positive_index),
                1
            ).otherwise(0)
        ).alias("TN"),

        spark_sum(
            when(
                (col("label") == positive_index) &
                (col("custom_prediction") != positive_index),
                1
            ).otherwise(0)
        ).alias("FN")
    ).first()

    tp = metrics["TP"] or 0
    fp = metrics["FP"] or 0
    tn = metrics["TN"] or 0
    fn = metrics["FN"] or 0

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    })

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )


# ==========================================
# 11. BEST THRESHOLD
# ==========================================

best_result = max(
    results,
    key=lambda x: x["f1"]
)

print("\n========== BEST THRESHOLD ==========")

print("Best Threshold:", best_result["threshold"])
print("Accuracy:", round(best_result["accuracy"], 4))
print("Precision:", round(best_result["precision"], 4))
print("Churn Recall:", round(best_result["recall"], 4))
print("F1 Score:", round(best_result["f1"], 4))

print("\nConfusion Matrix:")

print("True Positive :", best_result["TP"])
print("False Positive:", best_result["FP"])
print("True Negative :", best_result["TN"])
print("False Negative:", best_result["FN"])


# ==========================================
# 12. STOP SPARK SESSION
# ==========================================

spark.stop()

print("\nThreshold tuning completed successfully!")
