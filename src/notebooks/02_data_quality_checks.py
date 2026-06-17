# Databricks notebook source
# MAGIC %md
# MAGIC # Data Quality Checks
# MAGIC 
# MAGIC **Purpose**: Validate the customer sales analytics table

# COMMAND ----------

from pyspark.sql import functions as F

# Get configuration
catalog = spark.conf.get("catalog", "dev_catalog")
schema = spark.conf.get("schema", "customer_analytics")
table_name = f"{catalog}.{schema}.customer_sales_analytics"

print(f"Checking table: {table_name}")

# COMMAND ----------

# Load the table
df = spark.table(table_name)

# COMMAND ----------

# Test 1: Row count
row_count = df.count()
assert row_count > 0, "Table is empty!"
print(f"✅ Test 1 Passed: Table has {row_count:,} rows")

# COMMAND ----------

# Test 2: No null customer IDs
null_customers = df.filter(F.col("customer_sk").isNull()).count()
assert null_customers == 0, f"Found {null_customers} null customer IDs"
print(f"✅ Test 2 Passed: No null customer IDs")

# COMMAND ----------

# Test 3: All sales are positive
negative_sales = df.filter(F.col("total_sales") < 0).count()
assert negative_sales == 0, f"Found {negative_sales} negative sales"
print(f"✅ Test 3 Passed: All sales are positive")

# COMMAND ----------

# Test 4: Valid customer segments
valid_segments = ["High Value", "Medium Value", "Low Value"]
invalid_segments = df.filter(~F.col("customer_segment").isin(valid_segments)).count()
assert invalid_segments == 0, f"Found {invalid_segments} invalid segments"
print(f"✅ Test 4 Passed: All customer segments are valid")

# COMMAND ----------

print("\n" + "="*60)
print("🎉 All Data Quality Checks Passed!")
print("="*60)