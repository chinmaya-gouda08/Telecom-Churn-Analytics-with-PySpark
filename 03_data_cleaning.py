
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when

# 1. Create Spark Session
spark = SparkSession.builder \
    .appName("Customer Churn Data Cleaning") \
    .getOrCreate()

# 2. Load Dataset
file_path = "dataset/Telco-Customer-Churn.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

# 3. Display data type before cleaning
print("\n----- BEFORE CLEANING -----")
df.printSchema()

# 4. Identify blank TotalCharges records
print("\n----- BLANK TOTAL CHARGES RECORDS -----")

df.filter(
    trim(col("TotalCharges")) == ""
).select(
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
).show(20, truncate=False)

# 5. Handle blank values and convert to Double
df = df.withColumn(
    "TotalCharges",
    when(
        trim(col("TotalCharges")) == "",
        0.0
    ).otherwise(
        trim(col("TotalCharges")).cast("double")
    )
)

# 6. Display data type after cleaning
print("\n----- AFTER CLEANING -----")
df.printSchema()

# 7. Verify TotalCharges values
print("\n----- CLEANED TOTAL CHARGES -----")

df.select(
    "customerID",
    "tenure",
    "TotalCharges"
).show(10, truncate=False)

# 8. Check remaining NULL values
print("\n----- NULL CHECK AFTER CLEANING -----")

null_count = df.filter(
    col("TotalCharges").isNull()
).count()

print("NULL values in TotalCharges:", null_count)

# 9. Check total records
print("\nTotal Records After Cleaning:", df.count())

# 10. Stop Spark
spark.stop()
