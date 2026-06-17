# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Sales Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Create widgets for configuration
dbutils.widgets.text("catalog", "dev_catalog", "Target Catalog")
dbutils.widgets.text("schema", "customer_analytics", "Target Schema")
dbutils.widgets.text("source_catalog", "samples", "Source Catalog")
dbutils.widgets.text("source_schema", "tpcds_sf1", "Source Schema")

# Get values from widgets
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
source_catalog = dbutils.widgets.get("source_catalog")
source_schema = dbutils.widgets.get("source_schema")

print("="*60)
print(f"Environment Configuration")
print("="*60)
print(f"Target Catalog: {catalog}")
print(f"Target Schema: {schema}")
print(f"Source Catalog: {source_catalog}")
print(f"Source Schema: {source_schema}")
print("="*60)

# COMMAND ----------

# MAGIC 
# MAGIC **Purpose**: Transform and enrich customer sales data from TPC-DS sample tables
# MAGIC 
# MAGIC **Source Tables**:
# MAGIC - `samples.tpcds_sf1.customer`
# MAGIC - `samples.tpcds_sf1.store_sales`
# MAGIC - `samples.tpcds_sf1.item`
# MAGIC 
# MAGIC **Output**: Customer sales analytics table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Import libraries
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

# Get environment variables from job configuration
catalog = spark.conf.get("catalog", "dev_catalog")
schema = spark.conf.get("schema", "customer_analytics")
source_catalog = spark.conf.get("source_catalog", "samples")
source_schema = spark.conf.get("source_schema", "tpcds_sf1")

print("="*60)
print(f"Environment Configuration")
print("="*60)
print(f"Target Catalog: {catalog}")
print(f"Target Schema: {schema}")
print(f"Source Catalog: {source_catalog}")
print(f"Source Schema: {source_schema}")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Customer Data

# COMMAND ----------

print("Loading customer data...")

# Load customer data
customers = spark.table(f"{source_catalog}.{source_schema}.customer")

# Select relevant columns and clean data
customers_clean = customers.select(
    F.col("c_customer_sk").alias("customer_sk"),
    F.col("c_customer_id").alias("customer_id"),
    F.concat(F.col("c_first_name"), F.lit(" "), F.col("c_last_name")).alias("customer_name"),
    F.col("c_email_address").alias("email"),
    F.col("c_birth_year").alias("birth_year"),
    F.col("c_birth_country").alias("country"),
    F.col("c_preferred_cust_flag").alias("preferred_customer")
).filter(F.col("c_customer_sk").isNotNull())

# Calculate customer age
current_year = datetime.now().year
customers_clean = customers_clean.withColumn(
    "age",
    F.lit(current_year) - F.col("birth_year")
)

customer_count = customers_clean.count()
print(f"✅ Customers loaded: {customer_count:,}")

# Show sample
display(customers_clean.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Sales Data

# COMMAND ----------

print("Loading sales data...")

# Load store sales
sales = spark.table(f"{source_catalog}.{source_schema}.store_sales")

# Clean and filter sales data
sales_clean = sales.select(
    F.col("ss_customer_sk").alias("customer_sk"),
    F.col("ss_item_sk").alias("item_sk"),
    F.col("ss_ticket_number").alias("ticket_number"),
    F.col("ss_quantity").alias("quantity"),
    F.col("ss_sales_price").alias("sales_price"),
    F.col("ss_net_paid").alias("net_paid"),
    F.col("ss_net_profit").alias("net_profit")
).filter(
    (F.col("ss_customer_sk").isNotNull()) &
    (F.col("ss_net_paid").isNotNull()) &
    (F.col("ss_net_paid") > 0)
)

sales_count = sales_clean.count()
print(f"✅ Sales records loaded: {sales_count:,}")

# Show sample
display(sales_clean.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Load Item Data

# COMMAND ----------

print("Loading item data...")

# Load item data
items = spark.table(f"{source_catalog}.{source_schema}.item")

# Select relevant item information
items_clean = items.select(
    F.col("i_item_sk").alias("item_sk"),
    F.col("i_item_id").alias("item_id"),
    F.col("i_product_name").alias("product_name"),
    F.col("i_category").alias("category"),
    F.col("i_class").alias("product_class"),
    F.col("i_brand").alias("brand"),
    F.col("i_current_price").alias("current_price")
).filter(F.col("i_item_sk").isNotNull())

items_count = items_clean.count()
print(f"✅ Items loaded: {items_count:,}")

# Show sample
display(items_clean.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Join Sales with Items

# COMMAND ----------

print("Joining sales with items...")

# Join sales with items
sales_with_items = sales_clean.join(
    items_clean,
    on="item_sk",
    how="inner"
)

sales_items_count = sales_with_items.count()
print(f"✅ Sales with items: {sales_items_count:,}")

# Show sample
display(sales_with_items.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Join with Customers

# COMMAND ----------

print("Joining with customers...")

# Join with customers
customer_sales = customers_clean.join(
    sales_with_items,
    on="customer_sk",
    how="inner"
)

customer_sales_count = customer_sales.count()
print(f"✅ Customer sales records: {customer_sales_count:,}")

# Show sample
display(customer_sales.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Calculate Customer Metrics

# COMMAND ----------

print("Calculating customer metrics...")

# Aggregate by customer
customer_metrics = customer_sales.groupBy(
    "customer_sk",
    "customer_id",
    "customer_name",
    "email",
    "age",
    "country",
    "preferred_customer"
).agg(
    F.sum("net_paid").alias("total_sales"),
    F.count("ticket_number").alias("purchase_count"),
    F.avg("net_paid").alias("avg_purchase_value"),
    F.sum("quantity").alias("total_items_purchased"),
    F.sum("net_profit").alias("total_profit"),
    F.collect_set("category").alias("categories_purchased"),
    F.max("category").alias("top_category")
)

# Calculate customer lifetime value (simple formula)
customer_metrics = customer_metrics.withColumn(
    "customer_lifetime_value",
    F.col("total_sales") + (F.col("avg_purchase_value") * 2)
)

# Add customer segment based on total sales
customer_metrics = customer_metrics.withColumn(
    "customer_segment",
    F.when(F.col("total_sales") > 10000, "High Value")
     .when(F.col("total_sales") > 5000, "Medium Value")
     .otherwise("Low Value")
)

# Add processing timestamp
customer_metrics = customer_metrics.withColumn(
    "processed_at",
    F.current_timestamp()
)

metrics_count = customer_metrics.count()
print(f"✅ Customer metrics calculated: {metrics_count:,}")

# Show top customers by sales
print("\nTop 10 Customers by Total Sales:")
display(customer_metrics.orderBy(F.desc("total_sales")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Save Results

# COMMAND ----------

print("Saving results...")

# Create catalog and schema if not exists
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# Define output table
output_table = f"{catalog}.{schema}.customer_sales_analytics"

# Write to target table
customer_metrics.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(output_table)

print(f"✅ Data successfully written to: {output_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Verify Results

# COMMAND ----------

print("Verifying results...")

# Read back the table
result = spark.table(output_table)

# Calculate summary statistics
total_customers = result.count()
total_sales = result.select(F.sum("total_sales")).collect()[0][0]
avg_customer_value = result.select(F.avg("total_sales")).collect()[0][0]
high_value_customers = result.filter("customer_segment = 'High Value'").count()

print("="*60)
print("Summary Statistics")
print("="*60)
print(f"Total Customers: {total_customers:,}")
print(f"Total Sales: ${total_sales:,.2f}")
print(f"Average Customer Value: ${avg_customer_value:,.2f}")
print(f"High Value Customers: {high_value_customers:,}")
print("="*60)

# Show sample of results
display(result.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Data Quality Checks

# COMMAND ----------

print("Running data quality checks...")

# Check for nulls in critical columns
null_checks = result.select([
    F.sum(F.when(F.col("customer_sk").isNull(), 1).otherwise(0)).alias("null_customer_sk"),
    F.sum(F.when(F.col("total_sales").isNull(), 1).otherwise(0)).alias("null_total_sales"),
    F.sum(F.when(F.col("purchase_count").isNull(), 1).otherwise(0)).alias("null_purchase_count")
])

print("\nNull Counts in Critical Columns:")
display(null_checks)

# Check for negative values
negative_checks = result.select([
    F.sum(F.when(F.col("total_sales") < 0, 1).otherwise(0)).alias("negative_sales"),
    F.sum(F.when(F.col("purchase_count") < 0, 1).otherwise(0)).alias("negative_purchases")
])

print("\nNegative Value Counts:")
display(negative_checks)

# Distribution by segment
segment_distribution = result.groupBy("customer_segment").count().orderBy("customer_segment")

print("\nCustomer Segment Distribution:")
display(segment_distribution)

print("\n✅ All data quality checks completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Complete! 🎉