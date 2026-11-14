
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

# --------------------------------------------------
# 1. Create Spark Session
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Customer Churn Feature Preparation") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n========== STEP 5: FEATURE PREPARATION ==========\n")


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

file_path = "dataset/Telco-Customer-Churn.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("Original Dataset Records:", df.count())


# --------------------------------------------------
# 3. Handle Blank Values in TotalCharges
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

print("TotalCharges data type:")
df.printSchema()


# --------------------------------------------------
# 4. Define Numerical and Categorical Columns
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

target_col = "Churn"


# --------------------------------------------------
# 5. Split Dataset into Training and Testing Data
# --------------------------------------------------

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\n========== DATASET SPLITTING ==========")

print("Training Records:", train_df.count())
print("Testing Records:", test_df.count())


# --------------------------------------------------
# 6. Create String Indexers for Categorical Columns
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
# 7. Apply One-Hot Encoding
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
# 8. Assemble All Features into One Vector
# --------------------------------------------------

assembler = VectorAssembler(
    inputCols=numeric_cols + encoded_cols,
    outputCol="features",
    handleInvalid="keep"
)


# --------------------------------------------------
# 9. Encode Target Column
# --------------------------------------------------

label_indexer = StringIndexer(
    inputCol=target_col,
    outputCol="label",
    handleInvalid="error"
)


# --------------------------------------------------
# 10. Create ML Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    stages=indexers + [
        encoder,
        assembler,
        label_indexer
    ]
)


# --------------------------------------------------
# 11. Fit Pipeline on Training Data Only
# --------------------------------------------------

pipeline_model = pipeline.fit(train_df)


# --------------------------------------------------
# 12. Transform Training and Testing Data
# --------------------------------------------------

train_data = pipeline_model.transform(train_df)

test_data = pipeline_model.transform(test_df)


# --------------------------------------------------
# 13. Display Prepared Training Data
# --------------------------------------------------

print("\n========== PREPARED TRAINING DATA ==========")

train_data.select(
    "features",
    "label"
).show(5, truncate=False)


# --------------------------------------------------
# 14. Display Prepared Testing Data
# --------------------------------------------------

print("\n========== PREPARED TESTING DATA ==========")

test_data.select(
    "features",
    "label"
).show(5, truncate=False)


# --------------------------------------------------
# 15. Display Feature Vector Information
# --------------------------------------------------

first_row = train_data.select("features").first()

print("\nTotal Features Generated:",
      first_row["features"].size)


# --------------------------------------------------
# 16. Display Target Label Mapping
# --------------------------------------------------

label_model = pipeline_model.stages[-1]

print("\n========== TARGET LABEL MAPPING ==========")

for index, label_name in enumerate(label_model.labels):
    print(label_name, "=", index)


# --------------------------------------------------
# 17. Cache Prepared Data in Memory
# --------------------------------------------------

train_data.cache()
test_data.cache()

print("\n========== FINAL DATASET SUMMARY ==========")

print("Total Records:", df.count())
print("Training Records:", train_data.count())
print("Testing Records:", test_data.count())

print("\nFeature Preparation Completed Successfully!")


# --------------------------------------------------
# 18. Stop Spark Session
# --------------------------------------------------

spark.stop()
