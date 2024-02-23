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

MAIN_TABLE = 'data_science_metastore.nu_payout_production_tables.MAIN_PROD_LATEST_GA_ID_KAFKA_NU_DATA'
SIX_HOUR_TABLE_NAME = 'data_science_metastore.nu_payout_production_tables.6_HOUR_LATEST_GA_ID_KAFKA_NU_DATA'

# COMMAND ----------

df_main = spark.sql(f"""\
SELECT * FROM {MAIN_TABLE}
""")

df_six_hour = spark.sql(f"""\
SELECT * FROM {SIX_HOUR_TABLE_NAME}
""")

# df_main = df_main.alias('main')
# df_six_hour = df_six_hour.alias('six_hour')

columns = df_main.columns

main_columns = [f"main_{col}" for col in columns]
six_hour_columns = [f"six_hour_{col}" for col in columns]
                    
df_main = df_main.toDF(*main_columns)
df_six_hour = df_six_hour.toDF(*six_hour_columns)

result_df = df_main.join(df_six_hour, df_main.main_ga_id == df_six_hour.six_hour_ga_id, how='right')

# COMMAND ----------

if(df_six_hour.count() == 0):
    dbutils.notebook.exit("NUMBER OF ROWS IS 0")

# COMMAND ----------

df_main.count()

# COMMAND ----------

df_six_hour.count()

# COMMAND ----------

result_df = result_df.withColumn('number_of_sessions', col('six_hour_number_of_sessions') + col('main_number_of_sessions') + lit(1))\
        .withColumn('number_of_non_poc_actions', col('six_hour_number_of_non_poc_actions') + col('main_number_of_non_poc_actions') + lit(1))\
        .withColumn('number_of_non_poc_sessions', col('six_hour_number_of_non_poc_sessions') + col('main_number_of_non_poc_sessions') + lit(1))\
        .withColumn('seconds_since_first_session', col('six_hour_seconds_since_first_session') + col('main_seconds_since_first_session') + lit(1))

result_df = result_df.select(
    'six_hour_ga_id', 'six_hour_order_id', 'six_hour_category', 'six_hour_action', 'six_hour_timestamp', 'six_hour_date', 'six_hour_traffic_sourcemedium', 'six_hour_session_id', 'six_hour_session_start', 'six_hour_session_end', 'six_hour_session_time', 'six_hour_number_of_hits', 'six_hour_hit_number', 'six_hour_source', 'number_of_sessions', 'six_hour_mean_number_of_sessions', 'six_hour_median_number_of_sessions', 'six_hour_mean_session_time', 'six_hour_median_session_time', 'six_hour_first_transaction_time', 'six_hour_transaction_success_time', 'six_hour_first_session_start', 'number_of_non_poc_actions', 'number_of_non_poc_sessions', 'seconds_since_first_session', 'six_hour_seconds_after_transaction', 'six_hour_hits_after_transaction', 'six_hour_seconds_on_payment_gateway'
)

columns = result_df.columns

new_column_names = [col.replace("six_hour_", "") if col.startswith("six_hour_") else col for col in result_df.columns]

# Rename the columns in the DataFrame
result_df = result_df.toDF(*new_column_names)

# COMMAND ----------

columns = df_main.columns
new_column_names = [col.replace("main_", "") if col.startswith("main") else col for col in df_main.columns]

df_main = df_main.toDF(*new_column_names)

appended_df = df_main.union(result_df)

# COMMAND ----------

appended_df.count()

# COMMAND ----------

# assuming your dataframe is called "appended_df"
# group by ga_id and find the maximum timestamp for each group
max_timestamp_appended_df = appended_df.groupBy("ga_id").agg(max("timestamp").alias("max_timestamp"))

# join the original dataframe with the max_timestamp_appended_df on ga_id and timestamp
appended_df_with_max_timestamp = appended_df.join(max_timestamp_appended_df, ["ga_id"]).filter(appended_df.timestamp == max_timestamp_appended_df.max_timestamp)

# drop duplicates based on ga_id and max_timestamp
result_appended_df = appended_df_with_max_timestamp.dropDuplicates(["ga_id", "max_timestamp"]).drop("max_timestamp")

# COMMAND ----------

result_appended_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.MAIN_PROD_LATEST_GA_ID_KAFKA_NU_DATA")

# COMMAND ----------


