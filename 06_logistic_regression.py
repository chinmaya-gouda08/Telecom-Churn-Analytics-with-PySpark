
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when

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
# 1. Create Spark Session
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Customer Churn - Logistic Regression") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n========== LOGISTIC REGRESSION ==========\n")


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

df = spark.read.csv(
    "dataset/Telco-Customer-Churn.csv",
    header=True,
    inferSchema=True
)

print("Total Records:", df.count())


# --------------------------------------------------
# 3. Clean TotalCharges
# --------------------------------------------------

df = df.withColumn(
    "TotalCharges",
    when(
        col("TotalCharges").isNull() |
        (trim(col("TotalCharges").cast("string")) == ""),
        0.0
    ).otherwise(
        trim(col("TotalCharges").cast("string")).cast("double")
    )
)


# --------------------------------------------------
# 4. Define Feature Columns
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
# 5. Split Dataset
# --------------------------------------------------

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\nTraining Records:", train_df.count())
print("Testing Records:", test_df.count())


# --------------------------------------------------
# 6. String Indexing
# --------------------------------------------------

indexers = []

indexed_cols = []

for column in categorical_cols:

    indexed_column = column + "_index"

    indexers.append(
        StringIndexer(
            inputCol=column,
            outputCol=indexed_column,
            handleInvalid="keep"
        )
    )

    indexed_cols.append(indexed_column)


# --------------------------------------------------
# 7. One-Hot Encoding
# --------------------------------------------------

encoded_cols = [
    column + "_encoded"
    for column in categorical_cols
]

encoder = OneHotEncoder(
    inputCols=indexed_cols,
    outputCols=encoded_cols,
    handleInvalid="keep",
    dropLast=True
)


# --------------------------------------------------
# 8. Feature Assembly
# --------------------------------------------------

assembler = VectorAssembler(
    inputCols=numeric_cols + encoded_cols,
    outputCol="features",
    handleInvalid="keep"
)


# --------------------------------------------------
# 9. Target Encoding
# --------------------------------------------------

label_indexer = StringIndexer(
    inputCol="Churn",
    outputCol="label",
    handleInvalid="error"
)


# --------------------------------------------------
# 10. Logistic Regression Model
# --------------------------------------------------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)


# --------------------------------------------------
# 11. Create Complete ML Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    stages=indexers + [
        encoder,
        assembler,
        label_indexer,
        lr
    ]
)


# --------------------------------------------------
# 12. Train Model
# --------------------------------------------------

print("\nTraining Logistic Regression Model...")

lr_model = pipeline.fit(train_df)

print("Model Training Completed!")


# --------------------------------------------------
# 13. Make Predictions
# --------------------------------------------------

predictions = lr_model.transform(test_df)

print("\n========== SAMPLE PREDICTIONS ==========")

predictions.select(
    "Churn",
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)


# --------------------------------------------------
# 14. Evaluate Model
# --------------------------------------------------

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
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
# 15. Display Evaluation Results
# --------------------------------------------------

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")


# --------------------------------------------------
# 16. Confusion Matrix
# --------------------------------------------------

print("\n========== CONFUSION MATRIX ==========")

confusion_matrix = predictions.groupBy(
    "label",
    "prediction"
).count().orderBy(
    "label",
    "prediction"
)

confusion_matrix.show()


# --------------------------------------------------
# 17. Stop Spark Session
# --------------------------------------------------

spark.stop()

print("\nLogistic Regression Execution Completed!")
