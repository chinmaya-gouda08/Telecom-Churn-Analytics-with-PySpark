
# Import SparkSession
from pyspark.sql import SparkSession

# Create Spark Session
spark = SparkSession.builder \
    .appName("Customer Churn Prediction") \
    .getOrCreate()

# Load Dataset
file_path = "dataset/Telco-Customer-Churn.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

# Display first 5 records
print("\n----- FIRST 5 RECORDS -----")
df.show(5, truncate=False)

# Display Dataset Schema
print("\n----- DATASET SCHEMA -----")
df.printSchema()

# Display Dataset Dimensions
print("\n----- DATASET DIMENSIONS -----")
print("Total Rows:", df.count())
print("Total Columns:", len(df.columns))

# Display Column Names
print("\n----- COLUMN NAMES -----")
print(df.columns)

# Check Churn Distribution
print("\n----- CHURN DISTRIBUTION -----")
df.groupBy("Churn").count().show()

# Stop Spark Session
spark.stop()
