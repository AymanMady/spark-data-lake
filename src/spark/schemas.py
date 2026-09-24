"""
Explicit data schemas (phases 3 and 5).

WHY NOT USE inferSchema=True?

1. COST: to deduce the types, Spark has to read the whole file ONE more time.
   On 10 GB of CSV, that is a full read wasted.
2. FRAGILITY: the inferred type depends on the data. If one batch has
   "quantity=3" everywhere, Spark infers int; if the next one has
   "quantity=3.0", it infers double. The pipeline breaks without warning.
3. SILENCE: an invalid value becomes NULL and nobody knows.

PROJECT RULE:
  Bronze = everything as StringType. Take the data as it comes, never risking
           losing any of it on read.
  Silver = explicit typing + validation. Whatever fails to convert is isolated
           and counted, not ignored.
"""

from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# BRONZE: everything as text, maximum fidelity to the source
# ---------------------------------------------------------------------------

RAW_CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id", StringType(), True),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("country", StringType(), True),
    StructField("created_at", StringType(), True),
])

RAW_PRODUCTS_SCHEMA = StructType([
    StructField("product_id", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("price", StringType(), True),
])

RAW_ORDERS_SCHEMA = StructType([
    StructField("order_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("quantity", StringType(), True),
    StructField("order_date", StringType(), True),
    StructField("status", StringType(), True),
])

RAW_SCHEMAS = {
    "customers": RAW_CUSTOMERS_SCHEMA,
    "products": RAW_PRODUCTS_SCHEMA,
    "orders": RAW_ORDERS_SCHEMA,
}

# ---------------------------------------------------------------------------
# SILVER: final types, the contract with the layers above
# ---------------------------------------------------------------------------

SILVER_CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id", LongType(), False),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("country", StringType(), True),
    StructField("created_at", DateType(), True),
])

SILVER_PRODUCTS_SCHEMA = StructType([
    StructField("product_id", LongType(), False),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("price", DoubleType(), False),
])

SILVER_ORDERS_SCHEMA = StructType([
    StructField("order_id", LongType(), False),
    StructField("customer_id", LongType(), False),
    StructField("product_id", LongType(), False),
    StructField("quantity", IntegerType(), False),
    StructField("order_ts", TimestampType(), True),
    StructField("order_date", DateType(), False),
    StructField("status", StringType(), False),
])

SILVER_SCHEMAS = {
    "customers": SILVER_CUSTOMERS_SCHEMA,
    "products": SILVER_PRODUCTS_SCHEMA,
    "orders": SILVER_ORDERS_SCHEMA,
}

# Allowed values for an order status (Silver validation).
VALID_STATUSES = ["delivered", "shipped", "pending", "cancelled", "returned"]

# Statuses that count as actual revenue (Gold aggregates).
REVENUE_STATUSES = ["delivered", "shipped"]
