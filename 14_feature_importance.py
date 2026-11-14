
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim, lit

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import (
    RandomForestClassifier,
    LogisticRegression
)


# --------------------------------------------------
# 1. CREATE SPARK SESSION
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Customer Churn Feature Importance") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


try:

    # --------------------------------------------------
    # 2. LOAD AND CLEAN DATASET
    # --------------------------------------------------

    df = spark.read.csv(
        "dataset/Telco-Customer-Churn.csv",
        header=True,
        inferSchema=True
    )

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

    print("\nDataset loaded and cleaned successfully!")
    print("Total records:", df.count())


    # --------------------------------------------------
    # 3. SPLIT DATASET
    # --------------------------------------------------

    train_df, test_df = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print("Training records:", train_df.count())
    print("Testing records:", test_df.count())


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
    # 5. STRING INDEXING
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
    # 6. ONE HOT ENCODING
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
    # 7. FEATURE ASSEMBLY
    # --------------------------------------------------

    assembler = VectorAssembler(
        inputCols=numeric_cols + [
            c + "_ohe" for c in categorical_cols
        ],
        outputCol="features"
    )


    # --------------------------------------------------
    # 8. TARGET LABEL ENCODING
    # --------------------------------------------------

    label_indexer = StringIndexer(
        inputCol="Churn",
        outputCol="label",
        handleInvalid="error"
    )


    # --------------------------------------------------
    # 9. RANDOM FOREST MODEL
    # --------------------------------------------------

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=10,
        maxBins=32,
        seed=42
    )

    rf_pipeline = Pipeline(
        stages=indexers + encoders + [
            assembler,
            label_indexer,
            rf
        ]
    )

    print("\nTraining Random Forest...")

    rf_model = rf_pipeline.fit(train_df)

    print("Random Forest trained successfully!")


    # --------------------------------------------------
    # 10. EXTRACT FEATURE NAMES
    # --------------------------------------------------

    predictions = rf_model.transform(test_df)

    feature_metadata = predictions.schema[
        "features"
    ].metadata

    attributes = feature_metadata.get(
        "ml_attr", {}
    ).get("attrs", {})

    feature_names = {}

    for attribute_group in attributes.values():
        for attribute in attribute_group:
            feature_names[attribute["idx"]] = attribute["name"]

    rf_model_stage = rf_model.stages[-1]

    importances = rf_model_stage.featureImportances.toArray()

    print("\nTotal feature importance values:", len(importances))
    print("Total mapped feature names:", len(feature_names))


    # --------------------------------------------------
    # 11. DISPLAY RANDOM FOREST FEATURE IMPORTANCE
    # --------------------------------------------------

    importance_results = []

    for index, importance in enumerate(importances):

        feature_name = feature_names.get(
            index,
            f"Feature_{index}"
        )

        importance_results.append(
            (feature_name, float(importance))
        )

    importance_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n========================================")
    print(" TOP 15 RANDOM FOREST FEATURES")
    print("========================================")

    for rank, (name, score) in enumerate(
        importance_results[:15],
        start=1
    ):
        print(
            f"{rank:2d}. {name:<45} "
            f"Importance: {score:.6f}"
        )


    # --------------------------------------------------
    # 12. TRAIN LOGISTIC REGRESSION
    # --------------------------------------------------

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0
    )

    lr_pipeline = Pipeline(
        stages=indexers + encoders + [
            assembler,
            label_indexer,
            lr
        ]
    )

    print("\nTraining Logistic Regression...")

    lr_model = lr_pipeline.fit(train_df)

    print("Logistic Regression trained successfully!")


    # --------------------------------------------------
    # 13. EXTRACT LOGISTIC REGRESSION COEFFICIENTS
    # --------------------------------------------------

    lr_model_stage = lr_model.stages[-1]

    coefficients = lr_model_stage.coefficients.toArray()

    coefficient_results = []

    for index, coefficient in enumerate(coefficients):

        feature_name = feature_names.get(
            index,
            f"Feature_{index}"
        )

        coefficient_results.append(
            (feature_name, float(coefficient))
        )


    # --------------------------------------------------
    # 14. DISPLAY STRONGEST POSITIVE COEFFICIENTS
    # --------------------------------------------------

    coefficient_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n========================================")
    print(" TOP 10 POSITIVE LR COEFFICIENTS")
    print("========================================")

    for rank, (name, value) in enumerate(
        coefficient_results[:10],
        start=1
    ):
        print(
            f"{rank:2d}. {name:<45} "
            f"Coefficient: {value:.6f}"
        )


    # --------------------------------------------------
    # 15. DISPLAY STRONGEST NEGATIVE COEFFICIENTS
    # --------------------------------------------------

    coefficient_results.sort(
        key=lambda x: x[1]
    )

    print("\n========================================")
    print(" TOP 10 NEGATIVE LR COEFFICIENTS")
    print("========================================")

    for rank, (name, value) in enumerate(
        coefficient_results[:10],
        start=1
    ):
        print(
            f"{rank:2d}. {name:<45} "
            f"Coefficient: {value:.6f}"
        )


    print("\n========================================")
    print(" FEATURE INTERPRETATION COMPLETED")
    print("========================================")


finally:

    # --------------------------------------------------
    # 16. STOP SPARK
    # --------------------------------------------------

    spark.stop()
