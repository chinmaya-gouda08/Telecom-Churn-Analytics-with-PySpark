
from pyspark.sql import SparkSession
from pyspark.sql.functions import trim, col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# 1. Create Spark Session
spark = SparkSession.builder \
    .appName("Customer_Churn_BisectingKMeans") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Load Dataset
df = spark.read.csv(
    "dataset/Telco-Customer-Churn.csv",
    header=True,
    inferSchema=True
)

print("\n========== BISECTING K-MEANS ==========")
print("Total Records:", df.count())

# 3. Clean TotalCharges
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

# 4. Select Features
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

# 5. Encode Categorical Features
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_Index",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

indexed_cols = [c + "_Index" for c in categorical_cols]

encoder = OneHotEncoder(
    inputCols=indexed_cols,
    outputCols=[c + "_Encoded" for c in categorical_cols],
    handleInvalid="keep",
    dropLast=True
)

encoded_cols = [c + "_Encoded" for c in categorical_cols]

# 6. Assemble Features
assembler = VectorAssembler(
    inputCols=numeric_cols + encoded_cols,
    outputCol="raw_features",
    handleInvalid="keep"
)

# 7. Scale Features
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=False,
    withStd=True
)

preprocessing_pipeline = Pipeline(
    stages=indexers + [encoder, assembler, scaler]
)

print("\nPreparing customer features...")

preprocessing_model = preprocessing_pipeline.fit(df)

prepared_df = preprocessing_model.transform(df)

final_df = prepared_df.select("features").cache()

print("Feature preparation completed!")
print("Total customers:", final_df.count())

# 8. Evaluate Different K Values
evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

k_values = [2, 3, 4, 5, 6]

results = []

print("\n========== BISECTING K-MEANS EVALUATION ==========")

for k in k_values:

    bisecting_kmeans = BisectingKMeans(
        featuresCol="features",
        predictionCol="prediction",
        k=k,
        maxIter=100,
        seed=42
    )

    model = bisecting_kmeans.fit(final_df)

    predictions = model.transform(final_df)

    silhouette = evaluator.evaluate(predictions)

    results.append((k, silhouette))

    print(
        f"K = {k} | Silhouette Score = {silhouette:.4f}"
    )

# 9. Identify Best K
best_k, best_score = max(
    results,
    key=lambda x: x[1]
)

print("\n========== BEST BISECTING K-MEANS MODEL ==========")
print("Best Number of Clusters:", best_k)
print("Best Silhouette Score:", round(best_score, 4))

# 10. Train Final Model
final_bisecting = BisectingKMeans(
    featuresCol="features",
    predictionCol="prediction",
    k=best_k,
    maxIter=100,
    seed=42
)

final_model = final_bisecting.fit(final_df)

clustered_df = final_model.transform(final_df)

print("\n========== CUSTOMER CLUSTER DISTRIBUTION ==========")

clustered_df.groupBy("prediction") \
    .count() \
    .orderBy("prediction") \
    .show()

print("\nBisecting K-Means Completed Successfully!")


from pyspark.sql.functions import avg, count, when

# Generate predictions with original customer information
profile_df = final_model.transform(prepared_df)

# Calculate cluster characteristics
cluster_profile = profile_df.groupBy("prediction").agg(
    count("*").alias("Total_Customers"),
    avg("tenure").alias("Average_Tenure"),
    avg("MonthlyCharges").alias("Average_MonthlyCharges"),
    avg("TotalCharges").alias("Average_TotalCharges"),
    avg("SeniorCitizen").alias("SeniorCitizen_Average"),
    avg(
        when(col("Churn") == "Yes", 1).otherwise(0)
    ).alias("Churn_Rate")
)

print("\n========== BISECTING K-MEANS CLUSTER PROFILING ==========")

cluster_profile.orderBy("prediction").show(
    truncate=False
)


final_df.unpersist()
spark.stop()
