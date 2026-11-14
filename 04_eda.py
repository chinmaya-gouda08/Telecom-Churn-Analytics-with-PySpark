
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when, avg

# 1. Create Spark Session
spark = SparkSession.builder \
    .appName("Customer Churn EDA") \
    .getOrCreate()

# 2. Load Dataset
file_path = "dataset/Telco-Customer-Churn.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

# 3. Clean TotalCharges
df = df.withColumn(
    "TotalCharges",
    when(
        trim(col("TotalCharges")) == "",
        0.0
    ).otherwise(
        trim(col("TotalCharges")).cast("double")
    )
)

# 4. Customer Churn Distribution
print("\n----- CUSTOMER CHURN DISTRIBUTION -----")

df.groupBy("Churn").count().show()

# 5. Churn Based on Contract Type
print("\n----- CHURN BASED ON CONTRACT -----")

df.groupBy(
    "Contract",
    "Churn"
).count().orderBy(
    "Contract",
    "Churn"
).show()

# 6. Churn Based on Internet Service
print("\n----- CHURN BASED ON INTERNET SERVICE -----")

df.groupBy(
    "InternetService",
    "Churn"
).count().orderBy(
    "InternetService",
    "Churn"
).show()

# 7. Average Customer Charges
print("\n----- AVERAGE CUSTOMER CHARGES -----")

df.groupBy("Churn").agg(
    avg("MonthlyCharges").alias("Average Monthly Charges"),
    avg("TotalCharges").alias("Average Total Charges")
).show()

# 8. Average Tenure
print("\n----- AVERAGE TENURE BY CHURN -----")

df.groupBy("Churn").agg(
    avg("tenure").alias("Average Tenure")
).show()

# 9. Stop Spark
spark.stop()
