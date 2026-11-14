
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel

# --------------------------------------------------
# 1. CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("Customer_Churn_Final_Model")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nSpark session created successfully!")

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

dataset_path = os.path.join(
    "dataset",
    "Telco-Customer-Churn.csv"
)

df = spark.read.csv(
    dataset_path,
    header=True,
    inferSchema=True
)

print("\nDataset loaded successfully!")
print("Total records:", df.count())

# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

df = df.withColumn(
    "TotalCharges",
    when(
        trim(col("TotalCharges")) == "",
        0.0
    ).otherwise(col("TotalCharges").cast("double"))
)

df = df.withColumn(
    "TotalCharges",
    col("TotalCharges").cast("double")
)

print("\nDataset cleaned successfully!")

# --------------------------------------------------
# 4. DEFINE FEATURES
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
# 5. CREATE PREPROCESSING STAGES
# --------------------------------------------------

stages = []

indexed_cols = []

for column in categorical_cols:

    indexer = StringIndexer(
        inputCol=column,
        outputCol=column + "_indexed",
        handleInvalid="keep"
    )

    stages.append(indexer)
    indexed_cols.append(column + "_indexed")

encoder = OneHotEncoder(
    inputCols=indexed_cols,
    outputCols=[
        column + "_encoded"
        for column in categorical_cols
    ],
    handleInvalid="keep"
)

stages.append(encoder)

encoded_cols = [
    column + "_encoded"
    for column in categorical_cols
]

assembler = VectorAssembler(
    inputCols=numeric_cols + encoded_cols,
    outputCol="features",
    handleInvalid="keep"
)

stages.append(assembler)

# --------------------------------------------------
# 6. INDEX TARGET COLUMN
# --------------------------------------------------

label_indexer = StringIndexer(
    inputCol="Churn",
    outputCol="label",
    handleInvalid="error"
)

stages.append(label_indexer)

# --------------------------------------------------
# 7. LOGISTIC REGRESSION
# --------------------------------------------------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0,
    threshold=0.4
)

stages.append(lr)

# --------------------------------------------------
# 8. TRAIN FINAL MODEL
# --------------------------------------------------

pipeline = Pipeline(stages=stages)

print("\nTraining final model on complete dataset...")

final_model = pipeline.fit(df)

print("\nFinal model trained successfully!")

# --------------------------------------------------
# 9. EXTRACT FITTED MODEL INFORMATION
# --------------------------------------------------

lr_model = final_model.stages[-1]

encoder_model = None
indexer_models = {}

for stage in final_model.stages:

    if isinstance(stage, StringIndexerModel):

        indexer_models[stage.getInputCol()] = stage

    elif isinstance(stage, OneHotEncoderModel):

        encoder_model = stage

if encoder_model is None:
    raise RuntimeError("OneHotEncoderModel not found!")

# Get category information
category_sizes = list(encoder_model.categorySizes)

drop_last = encoder_model.getDropLast()

categorical_mappings = {}

for column in categorical_cols:

    indexer_model = indexer_models[column]

    categorical_mappings[column] = {
        "labels": list(indexer_model.labels),
        "indexed_column": column + "_indexed"
    }

# --------------------------------------------------
# 10. PREPARE MODEL ARTIFACT
# --------------------------------------------------

model_artifact = {
    "model_name": "Customer Churn Prediction",
    "algorithm": "Logistic Regression",
    "threshold": 0.4,

    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,

    "categorical_mappings": categorical_mappings,

    "category_sizes": category_sizes,
    "drop_last": drop_last,

    "coefficients": [
        float(value)
        for value in lr_model.coefficients
    ],

    "intercept": float(lr_model.intercept),

    "label_mapping": ["No", "Yes"],

    "training_records": df.count()
}

# --------------------------------------------------
# 11. SAVE JSON ARTIFACT
# --------------------------------------------------

save_directory = "saved_models"

os.makedirs(
    save_directory,
    exist_ok=True
)

model_path = os.path.join(
    save_directory,
    "final_churn_model.json"
)

with open(
    model_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        model_artifact,
        file,
        indent=4
    )

print("\n----------------------------------")
print("MODEL EXPORTED SUCCESSFULLY!")
print("----------------------------------")

print("Model location:", model_path)
print("Algorithm: Logistic Regression")
print("Threshold:", lr_model.getThreshold())
print("Feature count:", len(lr_model.coefficients))
print("Training records:", model_artifact["training_records"])

# --------------------------------------------------
# 12. VERIFY SAVED ARTIFACT
# --------------------------------------------------

with open(
    model_path,
    "r",
    encoding="utf-8"
) as file:

    loaded_artifact = json.load(file)

print("\nModel artifact loaded successfully!")

print(
    "Loaded coefficient count:",
    len(loaded_artifact["coefficients"])
)

print(
    "Loaded intercept:",
    loaded_artifact["intercept"]
)

print("\nStep 15 completed successfully!")

spark.stop()
