
import os
import json
import math

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when
from pyspark.ml.feature import (
    StringIndexerModel,
    OneHotEncoder,
    VectorAssembler
)

# ==================================================
# 1. LOAD SAVED MODEL
# ==================================================

MODEL_PATH = os.path.join(
    "saved_models",
    "final_churn_model.json"
)

DATASET_PATH = os.path.join(
    "dataset",
    "Telco-Customer-Churn.csv"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

with open(MODEL_PATH, "r", encoding="utf-8") as file:
    model = json.load(file)

numeric_features = model["numeric_features"]
categorical_features = model["categorical_features"]

categorical_mappings = model["categorical_mappings"]
category_sizes = model["category_sizes"]

coefficients = [
    float(value) for value in model["coefficients"]
]

intercept = float(model["intercept"])
threshold = float(model["threshold"])
drop_last = model["drop_last"]

print("\nSaved model loaded successfully!")

print("Coefficient count:", len(coefficients))
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ==================================================
# 2. SAMPLE CUSTOMERS
# ==================================================

customers = [

    {
        "customerID": "NEW001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 179.70
    },

    {
        "customerID": "NEW002",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.50,
        "TotalCharges": 3930.00
    },

    {
        "customerID": "NEW003",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.00,
        "TotalCharges": 1140.00
    }
]


# ==================================================
# 3. START SPARK SESSION
# ==================================================

spark = (
    SparkSession.builder
    .appName("Customer_Churn_Final_Validation")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:

    # ==================================================
    # 4. LOAD AND CLEAN DATASET
    # ==================================================

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(DATASET_PATH)
    )

    df = df.withColumn(
        "TotalCharges",
        when(
            trim(col("TotalCharges")) == "",
            0.0
        ).otherwise(
            col("TotalCharges").cast("double")
        )
    )

    print("\nDataset loaded:", df.count())


    # ==================================================
    # 5. RECONSTRUCT STRING INDEXERS
    # ==================================================

    indexed_columns = [
        column + "_index"
        for column in categorical_features
    ]

    encoded_columns = [
        column + "_encoded"
        for column in categorical_features
    ]

    indexed_df = df

    indexer_models = {}

    for column in categorical_features:

        labels = categorical_mappings[column]["labels"]

        indexer_model = StringIndexerModel.from_labels(
            labels,
            inputCol=column,
            outputCol=column + "_index",
            handleInvalid="keep"
        )

        indexed_df = indexer_model.transform(indexed_df)

        indexer_models[column] = indexer_model

    print("StringIndexer stages reconstructed!")


    # ==================================================
    # 6. RECONSTRUCT ONE-HOT ENCODER
    # ==================================================

    encoder = OneHotEncoder(
        inputCols=indexed_columns,
        outputCols=encoded_columns,
        handleInvalid="keep",
        dropLast=drop_last
    )

    encoder_model = encoder.fit(indexed_df)

    # Important:
    # With handleInvalid="keep" and dropLast=True,
    # actual output vector dimensions equal categorySizes.

    actual_sizes = list(encoder_model.categorySizes)

    print("\nSaved category sizes:", category_sizes)
    print("Spark category sizes:", actual_sizes)

    if list(category_sizes) != actual_sizes:
        raise ValueError(
            "Category sizes do not match between "
            "the saved model and Spark encoder."
        )

    encoded_df = encoder_model.transform(indexed_df)

    print("OneHotEncoder reconstructed!")


    # ==================================================
    # 7. RECONSTRUCT VECTOR ASSEMBLER
    # ==================================================

    assembler = VectorAssembler(
        inputCols=numeric_features + encoded_columns,
        outputCol="spark_features",
        handleInvalid="error"
    )


    # ==================================================
    # 8. PYTHON FEATURE ENCODING
    # ==================================================

    def custom_feature_vector(customer):

        features = []

        # Numeric features
        for column in numeric_features:

            value = float(customer[column])

            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid numeric value: {column}"
                )

            features.append(value)

        # Categorical features
        for i, column in enumerate(categorical_features):

            labels = categorical_mappings[column]["labels"]

            value = str(customer[column])

            # Reproduce StringIndexer
            if value in labels:
                category_index = labels.index(value)
            else:
                category_index = len(labels)

            # Use saved category size directly
            output_size = int(category_sizes[i])

            encoded = [0.0] * output_size

            # Reproduce OneHotEncoder
            if category_index < output_size:
                encoded[category_index] = 1.0

            features.extend(encoded)

        if len(features) != len(coefficients):
            raise ValueError(
                f"Feature mismatch: generated {len(features)}, "
                f"expected {len(coefficients)}"
            )

        return features


    # ==================================================
    # 9. PREPARE SAMPLE CUSTOMERS USING SPARK
    # ==================================================

    sample_df = spark.createDataFrame(customers)

    sample_indexed = sample_df

    for column in categorical_features:

        sample_indexed = indexer_models[column].transform(
            sample_indexed
        )

    sample_encoded = encoder_model.transform(sample_indexed)

    spark_predictions = assembler.transform(sample_encoded)


    # ==================================================
    # 10. COMPARE FEATURE VECTORS
    # ==================================================

    print("\n" + "=" * 60)
    print("             FEATURE VALIDATION")
    print("=" * 60)

    all_valid = True

    results = spark_predictions.select(
        "customerID",
        "spark_features"
    ).collect()

    for row in results:

        customer_id = row["customerID"]

        customer = next(
            c for c in customers
            if c["customerID"] == customer_id
        )

        # Spark feature vector
        spark_vector = (
            row["spark_features"]
            .toArray()
            .tolist()
        )

        # Python feature vector
        python_vector = custom_feature_vector(customer)

        if len(spark_vector) != len(python_vector):

            print(f"\nCustomer: {customer_id}")
            print("Spark dimensions:", len(spark_vector))
            print("Python dimensions:", len(python_vector))

            all_valid = False
            continue

        differences = [
            abs(a - b)
            for a, b in zip(spark_vector, python_vector)
        ]

        max_difference = max(differences, default=0.0)

        features_match = all(
            difference < 1e-9
            for difference in differences
        )

        print(f"\nCustomer: {customer_id}")
        print("Spark feature count:", len(spark_vector))
        print("Python feature count:", len(python_vector))
        print("Maximum difference:", max_difference)

        if features_match:
            print("Feature validation: PASSED")
        else:

            print("Feature validation: FAILED")

            mismatch_indices = [
                i for i, difference in enumerate(differences)
                if difference >= 1e-9
            ]

            print(
                "Mismatch indices:",
                mismatch_indices[:15]
            )

            all_valid = False


        # ==================================================
        # 11. COMPARE CHURN PROBABILITIES
        # ==================================================

        if features_match:

            # Spark feature vector probability
            z_spark = intercept + sum(
                a * b
                for a, b in zip(spark_vector, coefficients)
            )

            if z_spark >= 0:
                probability_spark = (
                    1.0 / (1.0 + math.exp(-z_spark))
                )
            else:
                exp_value = math.exp(z_spark)
                probability_spark = (
                    exp_value / (1.0 + exp_value)
                )

            # Python feature vector probability
            z_python = intercept + sum(
                a * b
                for a, b in zip(python_vector, coefficients)
            )

            if z_python >= 0:
                probability_python = (
                    1.0 / (1.0 + math.exp(-z_python))
                )
            else:
                exp_value = math.exp(z_python)
                probability_python = (
                    exp_value / (1.0 + exp_value)
                )

            print(
                "Spark churn probability:",
                f"{probability_spark * 100:.4f}%"
            )

            print(
                "Python churn probability:",
                f"{probability_python * 100:.4f}%"
            )

            probability_match = (
                abs(probability_spark - probability_python) < 1e-9
            )

            print(
                "Probability validation:",
                "PASSED" if probability_match else "FAILED"
            )

            if not probability_match:
                all_valid = False

            prediction = (
                "Yes"
                if probability_spark >= threshold
                else "No"
            )

            print("Predicted churn:", prediction)


    # ==================================================
    # 12. FINAL VALIDATION STATUS
    # ==================================================

    print("\n" + "=" * 60)
    print("             FINAL VALIDATION STATUS")
    print("=" * 60)

    if all_valid:
        print("SUCCESS: All feature vectors match.")
        print("SUCCESS: Custom preprocessing is validated.")
        print("SUCCESS: Prediction probabilities match.")
    else:
        print("VALIDATION FAILED.")
        print("Review the feature encoding differences.")

    print("=" * 60)


finally:
    spark.stop()
    print("\nSpark session stopped.")

