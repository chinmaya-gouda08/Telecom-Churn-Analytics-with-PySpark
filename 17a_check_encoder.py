
import json
import os

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexerModel, OneHotEncoder

# Load saved model
path = os.path.join(
    "saved_models",
    "final_churn_model.json"
)

with open(path, "r", encoding="utf-8") as f:
    model = json.load(f)

print("\n========== SAVED MODEL METADATA ==========")

print("Coefficient count:", len(model["coefficients"]))
print("Numeric features:", len(model["numeric_features"]))
print("Categorical features:", len(model["categorical_features"]))
print("Saved category sizes:", model["category_sizes"])
print("Saved dropLast:", model["drop_last"])

# Start Spark
spark = (
    SparkSession.builder
    .appName("Encoder_Diagnostic")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Create a small dataset with known categories
    rows = [{
        column: labels[0]
        for column, labels in (
            (c, model["categorical_mappings"][c]["labels"])
            for c in model["categorical_features"]
        )
    }]

    df = spark.createDataFrame(rows)

    indexed_columns = []
    encoded_columns = []

    for column in model["categorical_features"]:

        index_col = column + "_index"
        encoded_col = column + "_encoded"

        indexer = StringIndexerModel.from_labels(
            model["categorical_mappings"][column]["labels"],
            inputCol=column,
            outputCol=index_col,
            handleInvalid="keep"
        )

        df = indexer.transform(df)

        indexed_columns.append(index_col)
        encoded_columns.append(encoded_col)

    print("\n========== SPARK ENCODER TEST ==========")

    for drop_last in [True, False]:

        encoder = OneHotEncoder(
            inputCols=indexed_columns,
            outputCols=encoded_columns,
            handleInvalid="keep",
            dropLast=drop_last
        )

        encoder_model = encoder.fit(df)
        result = encoder_model.transform(df)

        vector_sizes = [
            len(result.select(c).first()[0].toArray())
            for c in encoded_columns
        ]

        print("\ndropLast:", drop_last)
        print("Encoder categorySizes:", encoder_model.categorySizes)
        print("Actual vector sizes:", vector_sizes)
        print(
            "Total feature count:",
            len(model["numeric_features"]) + sum(vector_sizes)
        )

finally:
    spark.stop()
