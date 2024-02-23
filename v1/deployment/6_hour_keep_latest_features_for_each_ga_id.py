# Databricks notebook source
import re
import pandas as pd
import json
import datetime
import numpy as np
# from tqdm import tqdm
import requests

from pyspark.sql.functions import col, lit, udf, split, expr, concat, when, sum, mean, min, max, expr, to_timestamp, date_format, count, countDistinct
from pyspark.sql.types import MapType, StringType,StructType,StructField, FloatType
from functools import reduce
from pyspark.sql.window import Window

import pytz
from dateutil.relativedelta import relativedelta

days = lambda i: i * 86400

# COMMAND ----------

FEATURE_TABLE_NAME = 'data_science_metastore.nu_payout_production_tables.6_HOUR_FINAL_KAFKA_NU_DATA'

# COMMAND ----------

df = spark.sql(f"""\
SELECT * FROM {FEATURE_TABLE_NAME}
""")

# COMMAND ----------

# assuming your dataframe is called "df"
# group by ga_id and find the maximum timestamp for each group
max_timestamp_df = df.groupBy("ga_id").agg(max("timestamp").alias("max_timestamp"))

# join the original dataframe with the max_timestamp_df on ga_id and timestamp
df_with_max_timestamp = df.join(max_timestamp_df, ["ga_id"]).filter(df.timestamp == max_timestamp_df.max_timestamp)

# drop duplicates based on ga_id and max_timestamp
result_df = df_with_max_timestamp.dropDuplicates(["ga_id", "max_timestamp"]).drop("max_timestamp")

# COMMAND ----------

result_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.6_HOUR_LATEST_GA_ID_KAFKA_NU_DATA")

# COMMAND ----------


