# Databricks notebook source
import re
import pandas as pd
import json
import datetime
import numpy as np
# from tqdm import tqdm
import requests

from pyspark.sql.functions import col, lit, udf, split, expr, concat, when, sum, mean, min, max, expr, to_timestamp, date_format, count, countDistinct, current_timestamp
from pyspark.sql.types import MapType, StringType,StructType,StructField, FloatType, TimestampType, DoubleType, LongType, IntegerType
from functools import reduce
from pyspark.sql.window import Window

import pytz
from dateutil.relativedelta import relativedelta

import mlflow

days = lambda i: i * 86400

# COMMAND ----------

MAIN_TABLE = 'data_science_metastore.nu_payout_production_tables.MAIN_PROD_LATEST_GA_ID_KAFKA_NU_DATA'
LIVE_FEATURES = 'data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA'

PAYOUT_THRESHOLD = 0.004

# COMMAND ----------

# MAGIC %md
# MAGIC ### UPDATING FEATURES FROM THE MAIN TABLE

# COMMAND ----------

df_main = spark.sql(f"""\
SELECT * FROM {MAIN_TABLE}
""")

try:
    df_rds = spark.sql(f"""\
    SELECT * FROM {LIVE_FEATURES}
    """)
except Exception as e:
    dbutils.notebook.exit("NO NU ORDERS OBTAINED FROM THIS JOB")

# df_main = df_main.alias('main')
# df_rds = df_rds.alias('six_hour')

columns = df_main.columns

main_columns = [f"main_{col}" for col in columns]
six_hour_columns = [f"rds_{col}" for col in columns]
                    
df_main = df_main.toDF(*main_columns)
df_rds = df_rds.toDF(*six_hour_columns)

result_df = df_main.join(df_rds, df_main.main_ga_id == df_rds.rds_ga_id, how='right')
result_df = result_df.na.fill(value=0, subset=['main_number_of_sessions', 'main_mean_number_of_sessions', 'main_mean_session_time', 'main_median_number_of_sessions', 'main_median_session_time', 'main_number_of_non_poc_actions', 'main_number_of_non_poc_sessions'])

# COMMAND ----------

df_main.count()

# COMMAND ----------

df_rds.count()

# COMMAND ----------

result_df = result_df.withColumn('number_of_sessions', col('rds_number_of_sessions') + col('main_number_of_sessions'))\
        .withColumn('number_of_non_poc_actions', col('rds_number_of_non_poc_actions') + col('main_number_of_non_poc_actions'))\
        .withColumn('number_of_non_poc_sessions', col('rds_number_of_non_poc_sessions') + col('main_number_of_non_poc_sessions'))\
        .withColumn('seconds_since_first_session', col('rds_seconds_since_first_session') + col('main_seconds_since_first_session'))

result_df = result_df.select(
    'rds_ga_id', 'rds_order_id', 'rds_category', 'rds_action', 'rds_timestamp', 'rds_date', 'rds_traffic_sourcemedium', 'rds_session_id', 'rds_session_start', 'rds_session_end', 'rds_session_time', 'rds_number_of_hits', 'rds_hit_number', 'rds_source', 'number_of_sessions', 'rds_mean_number_of_sessions', 'rds_median_number_of_sessions', 'rds_mean_session_time', 'rds_median_session_time', 'rds_first_transaction_time', 'rds_transaction_success_time', 'rds_first_session_start', 'number_of_non_poc_actions', 'number_of_non_poc_sessions', 'seconds_since_first_session', 'rds_seconds_after_transaction', 'rds_hits_after_transaction', 'rds_seconds_on_payment_gateway'
)

columns = result_df.columns

new_column_names = [col.replace("rds_", "") if col.startswith("rds_") else col for col in result_df.columns]

# Rename the columns in the DataFrame
result_df = result_df.toDF(*new_column_names)

# COMMAND ----------

display(result_df)

# COMMAND ----------

result_df = result_df.select(
    'ga_id', 'traffic_sourcemedium', 'session_time', 'number_of_hits', 'hit_number', 'number_of_sessions', 'number_of_non_poc_actions', 'number_of_non_poc_sessions', 'seconds_since_first_session', 'seconds_after_transaction', 'hits_after_transaction', 'seconds_on_payment_gateway'
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### ML INFERENCE

# COMMAND ----------

rds_df = result_df.toPandas()

rds_df = rds_df.rename(columns={
    'order_id' : 'rds_order_id',
    'source' : 'rds_source'
})

# COMMAND ----------

backend_df = pd.read_pickle('/dbfs/FileStore/harshul/nu_payout/api_response_dataframe.pkl')
backend_df = backend_df.drop(columns=['number_of_leads'])

# COMMAND ----------

profile_uuid_list = backend_df['profile_uuid'].to_list()
profile_uuids = ','.join([f"'{value}'" for value in profile_uuid_list])

print(profile_uuids)

# COMMAND ----------

query = f"""\
WITH base AS (
    select
        ld.profile_uuid, count(distinct lh.id) as number_of_leads
    from
        product_derived.leads_heavy lh
        inner join housing_leads_production.lead_details ld on lh.lead_details_id = ld.id
    where
        ld.profile_uuid IN ({profile_uuids})
    group by 1
)

SELECT base.profile_uuid, base.number_of_leads
FROM base
"""

num_leads_df = spark.sql(query).toPandas()

backend_df_with_leads = backend_df.merge(num_leads_df, on='profile_uuid', how='left')

# COMMAND ----------

num_leads_df.head(50)

# COMMAND ----------

df = backend_df_with_leads.merge(rds_df, on='ga_id', how='left')

df['city'] = df['city'].fillna('other')
df['referral_code'] = df['referral_code'].map({True: 'yes', False: 'no'}).astype('str')
df["profile_picture_url"] = df["profile_picture_url"].astype(int)

df['traffic_sourcemedium'] = df['traffic_sourcemedium'].replace('null', np.nan)
df['traffic_sourcemedium'] = df['traffic_sourcemedium'].astype('object')

# df = df.astype({
# 'traffic_sourcemedium' : str
# })

# COMMAND ----------

df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ### LOAD MODEL AND PREDICT

# COMMAND ----------

logged_model = 'runs:/46ff8b243aa945609f2504ceca1a8cba/rus_knn_1:4_PROD_v1'

# Load model as a PyFuncModel.
pipeline = mlflow.sklearn.load_model(logged_model)

# COMMAND ----------

def determine_payout(probability):
    if probability < PAYOUT_THRESHOLD:
        return 'GREEN'
    else:
        return 'RED'

# COMMAND ----------

# DBTITLE 1,DEBUGGING CODE IF YOU GET isnan ERROR
# df_copy = df.copy(deep=True)

# df_copy['traffic_sourcemedium'] = df_copy['traffic_sourcemedium'].astype('object')
# df_copy['traffic_sourcemedium'] = df_copy['traffic_sourcemedium'].replace('', np.nan)

# df_copy['ml_probability'] = pipeline.predict_proba(df_copy)[:,-1]
# df_copy['payout_decision'] = df_copy['ml_probability'].apply(determine_payout)

# COMMAND ----------

df['ml_probability'] = pipeline.predict_proba(df)[:,-1]
df['payout_decision'] = df['ml_probability'].apply(determine_payout)

# COMMAND ----------

df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ### STORE IN LOGS AND DELTE ALL INTERMEDIARY TABLES

# COMMAND ----------

spark_df = spark.createDataFrame(df)

# COMMAND ----------

spark_df = spark_df.withColumn('transaction_timestamp', (col('transactionTime') / 1000).cast(TimestampType()))
spark_df = spark_df.withColumn("log_timestamp", current_timestamp())

# COMMAND ----------

display(spark_df)

# COMMAND ----------

spark_df = spark_df.withColumn("number_of_leads", col("number_of_leads").cast(LongType()))
spark_df = spark_df.withColumn("session_time", col("session_time").cast(LongType()))
spark_df = spark_df.withColumn("number_of_hits", col("number_of_hits").cast(LongType()))
spark_df = spark_df.withColumn("hit_number", col("hit_number").cast(IntegerType()))
spark_df = spark_df.withColumn("number_of_sessions", col("number_of_sessions").cast(LongType()))
spark_df = spark_df.withColumn("number_of_non_poc_actions", col("number_of_non_poc_actions").cast(LongType()))
spark_df = spark_df.withColumn("number_of_non_poc_sessions", col("number_of_non_poc_sessions").cast(LongType()))
spark_df = spark_df.withColumn("seconds_since_first_session", col("seconds_since_first_session").cast(LongType()))
spark_df = spark_df.withColumn("seconds_after_transaction", col("seconds_after_transaction").cast(LongType()))
spark_df = spark_df.withColumn("hits_after_transaction", col("hits_after_transaction").cast(LongType()))
spark_df = spark_df.withColumn("seconds_on_payment_gateway", col("seconds_on_payment_gateway").cast(LongType()))

# COMMAND ----------

spark_df.write.format("delta").mode("append").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.nu_v1_logs")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DROP TABLE data_science_metastore.nu_payout_production_tables.RDS_P1_NU_DATA_WEB;
# MAGIC DROP TABLE data_science_metastore.nu_payout_production_tables.RDS_P1_NU_DATA_APP;
# MAGIC DROP TABLE data_science_metastore.nu_payout_production_tables.RDS_P2_NU_DATA_WEB;
# MAGIC DROP TABLE data_science_metastore.nu_payout_production_tables.RDS_P2_NU_DATA_APP;
# MAGIC DROP TABLE data_science_metastore.nu_payout_production_tables.RDS_P3_NU_DATA_WEB;
# MAGIC DROP TABLE data_science_metastore.nu_payout_production_tables.RDS_P3_NU_DATA_APP;
# MAGIC DROP TABLE data_science_metastore.nu_payout_production_tables.RDS_P4_NU_DATA_WEB;
# MAGIC DROP TABLE data_science_metastore.nu_payout_production_tables.RDS_P4_NU_DATA_APP;
# MAGIC DROP TABLE data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA;

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### SEND RESPONSE TO API 2

# COMMAND ----------

# curl --location --request POST 'https://beta7.housing.com/apollo/ml-param/v1/nu-payout-hold-webhook' \
# --header 'Content-Type: application/json' \
# --data-raw '{
#     "order_id" : "976512005571",
#     "user_type" : "GREEN",
#     "ml_probability" : "0.956"

# }'

# COMMAND ----------

selected_columns = df[['order_id', 'ml_probability', 'payout_decision']]
selected_columns = selected_columns.rename(columns={
    'payout_decision' : 'user_type'
})

ML_PREDICTIONS = selected_columns.to_dict(orient='records')

# COMMAND ----------

ML_PREDICTIONS

# COMMAND ----------

# HEADERS = {
#     'X-ML-KEY': '123456789abcde'
#     }
# URL = 'https://beta7.housing.com/apollo/ml-param/v1/nu-payout-hold-webhook'

# COMMAND ----------

HEADERS = {
    'X-ML-KEY': 'LKJSDOFIJERN234KN23LKJ2LK2'
    }
URL = 'http://apollo1.api.com/apollo/ml-param/v1/nu-payout-hold-webhook'

# COMMAND ----------

MAX_RETRY_LIMIT = 3

for prediction in ML_PREDICTIONS:
    retry_count = 0
    while retry_count < MAX_RETRY_LIMIT:
        response = requests.post(
            url= URL, headers = HEADERS, json= prediction)
        if response.status_code == 200:
            # Successful response, no need to retry
            prediction['retry_counter'] = retry_count
            break
        retry_count += 1
    else:
        # If the loop completes without a successful response, retry_counter is set to MAX_RETRY_LIMIT
        prediction['retry_counter'] = MAX_RETRY_LIMIT

# COMMAND ----------

ML_PREDICTIONS
