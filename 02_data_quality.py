
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when, sum as spark_sum

# Create Spark Session
spark = SparkSession.builder \
    .appName("Customer Churn Data Quality") \
    .getOrCreate()

# Load Dataset
file_path = "dataset/Telco-Customer-Churn.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

# 1. Check NULL values
print("\n----- NULL VALUE CHECK -----")

null_counts = df.select([
    spark_sum(
        when(col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
])

null_counts.show(truncate=False)


# 2. Check Blank Values
print("\n----- BLANK VALUE CHECK -----")

string_columns = [
    field.name
    for field in df.schema.fields
    if field.dataType.simpleString() == "string"
]

blank_counts = df.select([
    spark_sum(
        when(
            col(c).isNotNull() & (trim(col(c)) == ""),
            1
        ).otherwise(0)
    ).alias(c)
    for c in string_columns
])

blank_counts.show(truncate=False)


# 3. Check Duplicate Rows
print("\n----- DUPLICATE ROW CHECK -----")

total_rows = df.count()
unique_rows = df.dropDuplicates().count()

duplicate_rows = total_rows - unique_rows

print("Total Rows:", total_rows)
print("Unique Rows:", unique_rows)
print("Duplicate Rows:", duplicate_rows)


# 4. Check Duplicate Customer IDs
print("\n----- DUPLICATE CUSTOMER ID CHECK -----")

duplicate_ids = df.groupBy("customerID") \
    .count() \
    .filter(col("count") > 1)

print("Duplicate Customer IDs:", duplicate_ids.count())


# Stop Spark
spark.stop()
