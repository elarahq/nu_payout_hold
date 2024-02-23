# Databricks notebook source
import requests
import pandas as pd
import json
import datetime
import numpy as np
import requests
import os
import time
# from tqdm import tqdm

from pyspark.sql.functions import col, lit, udf, split, expr, concat, when, sum, mean, min, max, expr, to_timestamp, date_format, lag, coalesce, lit, regexp_replace, from_json
from pyspark.sql.types import MapType, StringType,StructType,StructField, FloatType
from functools import reduce
from pyspark.sql.window import Window

import uuid
from datetime import timedelta, datetime
import pytz
from dateutil.relativedelta import relativedelta
import random
import psycopg2

days = lambda i: i * 86400

RDS_URL = 'jdbc:postgresql://ml-recommendation-db.cfgklq1b0ixg.ap-southeast-1.rds.amazonaws.com:5432/mlrecommendationdb'

# COMMAND ----------

CURRENT_DATE =(datetime.now(pytz.timezone('Asia/Kolkata'))+relativedelta(days=0)).strftime("%Y-%m-%d")
print(CURRENT_DATE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### GET DATA FROM BACKEND

# COMMAND ----------

# !curl --location --request GET 'https://beta7.housing.com/ping'

# !curl --location --request GET 'https://beta7.housing.com/apollo/ml-param/v1/nu-payout-hold' --header 'X-ML-KEY: 123456789abcde'

# !curl --location --request POST 'https://beta7.housing.com/apollo/ml-param/v1/nu-payout-hold-webhook' --header 'Content-Type: application/json' --header 'X-FORWARDED-FOR: 10.120.0.0/16' --data-raw '{"order_id" : "976512005571", "user_type" : "GREEN", "ml_probability" : "0.956"}'

# COMMAND ----------

# !curl --location --request GET 'http://apollo1.api.com/apollo/ping'
# !curl --location --request GET 'http://apollo1.api.com/apollo/ml-param/v1/nu-payout-hold' --header 'X-ML-KEY: LKJSDOFIJERN234KN23LKJ2LK2'

# COMMAND ----------

HEADERS = {
    'X-ML-KEY': 'LKJSDOFIJERN234KN23LKJ2LK2'
    }
URL = 'http://apollo1.api.com/apollo/ml-param/v1/nu-payout-hold'

# COMMAND ----------

# HEADERS = {
#     'X-ML-KEY': '123456789abcde'
#     }
# URL = 'https://beta7.housing.com/apollo/ml-param/v1/nu-payout-hold'

# COMMAND ----------

API_1_RESPONSE = []

try:
    while True:
        response = requests.request(method='GET', url= URL, headers= HEADERS).json()
        API_1_RESPONSE.extend(response['data']['mlPayoutHoldParams'])
        if response['data']['fetchNext'] is False:
            break
except Exception as e:
    dbutils.notebook.exit(f"API GAVE ERROR :: {e}")

# COMMAND ----------

if(len(API_1_RESPONSE) == 0):
    dbutils.notebook.exit("NO NU ORDERS OBTAINED FROM THIS JOB")

# COMMAND ----------

API_1_RESPONSE

# COMMAND ----------

# API_1_RESPONSE[0]['gaId'] = 'GA1.1.1017110592.1680589452'
# API_1_RESPONSE[2]['gaId'] = 'E97228EE-475A-46B2-A820-6E4ED4B59F32'

# API_1_RESPONSE[3]['gaId'] = 'GA1.2.2033514486.1673762900'
# API_1_RESPONSE[4]['gaId'] = 'b38922d4f244a7ef'

# API_1_RESPONSE[5]['gaId'] = 'GA1.1.54767700.1695015652'
# API_1_RESPONSE[6]['gaId'] = 'd7b9da0fb4e95d9f'

# COMMAND ----------

df = pd.DataFrame(API_1_RESPONSE)

# COMMAND ----------

df = df.rename(columns={
    'orderId' : 'order_id',
    'tenantId' : 'tenant_id',
    'profileUuid' : 'profile_uuid',
    'gaId' : 'ga_id',
    'recipientBank' : 'bank',
    'isOwner' : 'is_owner',
    'tenantName' : 'tenant_name',
    'landlordName' : 'landlord_name',
    'ageOnPlatformInMin' : 'tenant_age_in_seconds',
    'secondsSinceLastOrder' : 'seconds_since_last_transaction',
    'timeInConsecutiveTxn' : 'average_seconds_between_two_transactions',
    'avgTimeInConsecutiveTxn' : 'average_seconds_between_two_transactions',
    'pocCategory' : 'poc_category',
    'noOfFailedTxn' : 'number_of_failed_transactions',
    'averageFailedTxn' : 'average_number_of_transactions',
    'isReferral' : 'referral_code',
    'hourOfTheDay' : 'time_hour_ist',
    'noOfLeadsDropped' : 'number_of_leads',
    'isProfilePicture' : 'profile_picture_url'

})
df['transaction_timestamp'] = pd.to_datetime(df['transactionTime'], unit='ms')

df.to_pickle('/dbfs/FileStore/harshul/nu_payout/api_response_dataframe.pkl')

# COMMAND ----------

df.shape

# COMMAND ----------

df.head(100)

# COMMAND ----------

WEB_PLATFORMS = ['mWeb', 'dweb']

app_ga_id_list = df[~df['platform'].isin(WEB_PLATFORMS)]['ga_id'].to_list()
if(len(app_ga_id_list) == 0):
    app_ga_ids = "''"
else:
    app_ga_ids = ','.join([f"'{value}'" for value in app_ga_id_list])

web_ga_id_list = df[df['platform'].isin(WEB_PLATFORMS)]['ga_id'].to_list()
if(len(web_ga_id_list) == 0):
    web_ga_ids = "''"
else:
    web_ga_ids = ','.join([f"'{value}'" for value in web_ga_id_list])

# COMMAND ----------

# ## Harshul, Venky and Sadhana GA_IDs in order

# web_ga_ids = "'GA1.2.2046831729.1652445654','GA1.2.2033514486.1673762900', 'GA1.1.54767700.1695015652'"
# app_ga_ids = "'E97228EE-475A-46B2-A820-6E4ED4B59F32', 'b38922d4f244a7ef', 'cd5b190162e5bfce'"

# COMMAND ----------

# app_ga_id_list = ['2046831729', '2033514486']
# web_ga_id_list = ['b38922d4f244a7ef', 'E97228EE-475A-46B2-A820-6E4ED4B59F32']

# COMMAND ----------

web_ga_ids

# COMMAND ----------

app_ga_ids

# COMMAND ----------

# MAGIC %md
# MAGIC ### DUMP ALL (NO EVENTS FILTER)

# COMMAND ----------

# MAGIC %md
# MAGIC We will read ALL events from RDS web, RDS app, Housing Demands Web and Housing Demands App.\
# MAGIC We will append them all at once and write to a temporary table.\
# MAGIC \
# MAGIC \
# MAGIC NOTE :: We would be querying the data from Housing Demand Events for only the previous 2-3 months.

# COMMAND ----------

# DBTITLE 1,Query in RDS WEB
web_query_rds = f"""
SELECT
    dimension49 AS ga_id,
    category,
    action,
    timestamp,
    date,
    traffic_sourcemedium,
    event_label
FROM
    public.housing_demand_events_web
WHERE
    pk_id <= (
        select max(pk_id) from public.housing_demand_events_web
        )
    AND dimension49 IN ({web_ga_ids})
"""

rds_web_data = (spark.read.format("jdbc")
                   .option("driver", "org.postgresql.Driver")
                   .option("url", RDS_URL)
                   .option("query", web_query_rds)
                   .option("user","root")
                   .option("password", "ml#housing*")
                   .option("numPartitions",64).load()) #64 is optimal

# COMMAND ----------

rds_web_data = rds_web_data.withColumn("json_column", regexp_replace(rds_web_data["event_label"], "\\\\", ""))
json_schema = "struct<poc_price: int, orderId: string>"

rds_web_data = rds_web_data.withColumn("json_struct", from_json(col("json_column"), json_schema)).withColumn("orderId", col("json_struct.orderId")).drop('json_column', 'json_struct', 'event_label')

# COMMAND ----------

# DBTITLE 1,Query in RDS APP
app_query_rds = f"""
SELECT
    uid AS ga_id,
    category,
    action,
    timestamp,
    date,
    sourcemedium as traffic_sourcemedium,
    order_id
FROM
    public.housing_demand_events_app
WHERE
    pk_id <= (
        select max(pk_id) from public.housing_demand_events_web
        )
    AND uid IN ({app_ga_ids})
"""

rds_app_data = (spark.read.format("jdbc")
                   .option("driver", "org.postgresql.Driver")
                   .option("url", RDS_URL)
                   .option("query", app_query_rds)
                   .option("user","root")
                   .option("password", "ml#housing*")
                   .option("numPartitions",64).load()) #64 is optimal

# COMMAND ----------

rds_web_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_P1_NU_DATA_WEB")

# COMMAND ----------

rds_app_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_P1_NU_DATA_APP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CREATE SESSION ID, CALCULATE SESSION START, SESSION END, SESSION TIME, NUMBER OF HITS, HIT NUMBER AND DUMP AGAIN

# COMMAND ----------

query = """\
    SELECT * FROM data_science_metastore.nu_payout_production_tables.RDS_P1_NU_DATA_WEB
"""
df_web = spark.sql(query)

# COMMAND ----------

query = """\
    SELECT * FROM data_science_metastore.nu_payout_production_tables.RDS_P1_NU_DATA_APP
"""
df_app = spark.sql(query)

# COMMAND ----------

# Define window specification
window_spec = Window.partitionBy('ga_id').orderBy('timestamp')

# Calculate time differences and convert to seconds
df_web = df_web.withColumn('time_diff', (col('timestamp').cast('long') - lag('timestamp').over(window_spec).cast('long')))
df_web = df_web.withColumn('time_diff_seconds', coalesce(df_web['time_diff'], lit(0)))

# Identify session boundaries based on 30-minute threshold (1800 seconds)
df_web = df_web.withColumn('session_boundary', when(col('time_diff_seconds') > 1800, 1).otherwise(0))
df_web = df_web.withColumn('session_id', sum('session_boundary').over(window_spec))

# Create unique session IDs
df_web = df_web.withColumn('session_id', concat(col('ga_id'), lit('_'), col('session_id')))

# Drop intermediate columns
df_web = df_web.drop('time_diff', 'time_diff_seconds', 'session_boundary')

# COMMAND ----------

# Define window specification
window_spec = Window.partitionBy('ga_id').orderBy('timestamp')

# Calculate time differences and convert to seconds
df_app = df_app.withColumn('time_diff', (col('timestamp').cast('long') - lag('timestamp').over(window_spec).cast('long')))
df_app = df_app.withColumn('time_diff_seconds', coalesce(df_app['time_diff'], lit(0)))

# Identify session boundaries based on 30-minute threshold (1800 seconds)
df_app = df_app.withColumn('session_boundary', when(col('time_diff_seconds') > 1800, 1).otherwise(0))
df_app = df_app.withColumn('session_id', sum('session_boundary').over(window_spec))

# Create unique session IDs
df_app = df_app.withColumn('session_id', concat(col('ga_id'), lit('_'), col('session_id')))

# Drop intermediate columns
df_app = df_app.drop('time_diff', 'time_diff_seconds', 'session_boundary')

# COMMAND ----------

df_web.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_P2_NU_DATA_WEB")

# COMMAND ----------

df_app.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_P2_NU_DATA_APP")

# COMMAND ----------

p3_web_query = """\
    SELECT t.*, sub.session_start, sub.session_end,
    ROUND((unix_timestamp(sub.session_end)-unix_timestamp(sub.session_start))) as session_time,
    sub.number_of_hits,
    ROW_NUMBER() OVER (PARTITION BY t.session_id ORDER BY t.timestamp) AS hit_number
    FROM data_science_metastore.nu_payout_production_tables.RDS_P2_NU_DATA_WEB t
    JOIN (
        SELECT
            session_id,
            MIN(timestamp) AS session_start,
            MAX(timestamp) AS session_end,
            COUNT(*) AS number_of_hits
        FROM data_science_metastore.nu_payout_production_tables.RDS_P2_NU_DATA_WEB
        GROUP BY session_id
    ) sub ON t.session_id = sub.session_id;
"""

df_web = spark.sql(p3_web_query)

# COMMAND ----------

p3_app_query = """\
    SELECT t.*, sub.session_start, sub.session_end,
    ROUND((unix_timestamp(sub.session_end)-unix_timestamp(sub.session_start))) as session_time,
    sub.number_of_hits,
    ROW_NUMBER() OVER (PARTITION BY t.session_id ORDER BY t.timestamp) AS hit_number
    FROM data_science_metastore.nu_payout_production_tables.RDS_P2_NU_DATA_APP t
    JOIN (
        SELECT
            session_id,
            MIN(timestamp) AS session_start,
            MAX(timestamp) AS session_end,
            COUNT(*) AS number_of_hits
        FROM data_science_metastore.nu_payout_production_tables.RDS_P2_NU_DATA_APP
        GROUP BY session_id
    ) sub ON t.session_id = sub.session_id;
"""

df_app = spark.sql(p3_app_query)

# COMMAND ----------

df_web.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_P3_NU_DATA_WEB")

# COMMAND ----------

df_app.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_P3_NU_DATA_APP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### FILTER VIA EVENTS AND DUMP AGAIN

# COMMAND ----------

p4_web_query = f"""\
    SELECT
        *
    FROM
        data_science_metastore.nu_payout_production_tables.RDS_P3_NU_DATA_WEB
    WHERE
        session_id IN (
            SELECT
                session_id
            FROM
                data_science_metastore.nu_payout_production_tables.RDS_P3_NU_DATA_WEB
            WHERE
            (
                (category = 'poc_landing_page') 
                or (category = 'poc_home_page')
                or (category = 'poc_flow')
                or (category = 'transaction_success_page')
            )
        )
        AND
        (
            (category = 'poc_flow' and  action = 'pay_cta_clicked') 
            or  (category = 'poc_home_page' and  action = 'pay_again_clicked')
            or  (category = 'poc_home_page' and  action = 'total_pay_clicked')
            or (category = 'poc_flow' and  action = 'pay_cta_click')
            or (category = 'poc_landing_page' and  action = 'see_all_offers_clicked')
            or (category = 'poc_home_page' and  action = 'offer_clicked')
            or (category = 'poc_home_page' and  action = 'refer_now_clicked')
            or (category = 'poc_home_page' and  action = 'refer_and_earn_clicked')
            or (category = 'poc_landing_page' and  action = 'cta_clicked')
            or (category = 'poc_home_page' and  action = 'edit_clicked')
            or (category = 'transaction_success_page' and  action = 'payment_success')
            or (category = 'transaction_success_page' and  action = 'go_to_home_cta')
            or (category = 'transaction_success_page' and  action = 'go_to_transactions_cta')
            or (category = 'conversion' and  action = 'open_crf')
            or (category = 'conversion' and  action = 'filled_crf')
            or (category = 'conversion' and  action = 'submitted_crf')
            or (category = 'srp_card_clicked' and  action = 'details_page')
            or (category = 'auto_suggestion' and  action = 'select')
            or  (category = 'poc_edit_flow' and  action = 'total_pay_clicked')
            or  (category = 'final_screen' and  action = 'total_pay_clicked')
            or  (category = 'poc_home_page' and  action = 'total_pay_clicked')
            or  (category = 'pay_new_flow' and  action = 'total_pay_clicked')
        )
"""

p4_kafka_web_data = spark.sql(p4_web_query)
p4_kafka_web_data = p4_kafka_web_data.withColumn("source", lit("web"))

# COMMAND ----------

p4_app_query = f"""\
    SELECT
        *
    FROM
        data_science_metastore.nu_payout_production_tables.RDS_P3_NU_DATA_APP
    WHERE
        session_id IN (
            select
                session_id
            from
                data_science_metastore.nu_payout_production_tables.RDS_P3_NU_DATA_APP
            where 
                (
                    (category = 'poc_landing_page') 
                    or (category = 'poc_home_page')
                    or (category = 'poc_flow')
                    or (category = 'transaction_success_page')
                )
        )
        AND
        (
            (category = 'poc_flow' and  action = 'pay_cta_clicked') 
            or  (category = 'poc_home_page' and  action = 'pay_again_clicked' )
            or  (category = 'poc_home_page' and  action = 'total_pay_clicked')
            or (category = 'poc_flow' and  action = 'pay_cta_click')
            or (category = 'poc_landing_page' and  action = 'see_all_offers_clicked')
            or (category = 'poc_home_page' and  action = 'offer_clicked')
            or (category = 'poc_home_page' and  action = 'refer_now_clicked')
            or (category = 'poc_home_page' and  action = 'refer_and_earn_clicked')
            or (category = 'poc_landing_page' and  action = 'cta_clicked')
            or (category = 'poc_home_page' and  action = 'edit_clicked')
            or (category = 'transaction_success_page' and  action = 'payment_success')
            or (category = 'transaction_success_page' and  action = 'go_to_home_cta')
            or (category = 'transaction_success_page' and  action = 'refer_save_cta')
            or (category = 'transaction_success_page' and  action = 'gift_cta')
            or (category = 'transaction_success_page' and  action = 'coupon_copied')
            or (category = 'conversion' and  action = 'open_crf')
            or (category = 'conversion' and  action = 'filled_crf')
            or (category = 'conversion' and  action = 'submitted_crf')
            or (category = 'details_page' and  action = 'open')
            or (category = 'search' and  action = 'search')
            or  (category = 'poc_edit_flow' and  action = 'total_pay_clicked')
            or  (category = 'final_screen' and  action = 'total_pay_clicked')
            or  (category = 'poc_home_page' and  action = 'total_pay_clicked')
            or  (category = 'pay_new_flow' and  action = 'total_pay_clicked')
        )
"""

p4_kafka_app_data = spark.sql(p4_app_query)
p4_kafka_app_data = p4_kafka_app_data.withColumn("source", lit("app"))

# COMMAND ----------

p4_kafka_web_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_P4_NU_DATA_WEB")

# COMMAND ----------

p4_kafka_app_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_P4_NU_DATA_APP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CONCATENATE WEB AND APP

# COMMAND ----------

web = spark.sql("select * from data_science_metastore.nu_payout_production_tables.RDS_P4_NU_DATA_WEB")
app = spark.sql("select * from data_science_metastore.nu_payout_production_tables.RDS_P4_NU_DATA_APP")

combined = app.union(web)

# COMMAND ----------

combined.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ### COMPUTE NUMBER OF SESSIONS, MEAN NUMBER OF SESSIONS, MEAN SESSION TIME, ETC

# COMMAND ----------

query = """\
    WITH base AS (
        SELECT
        Z1.ga_id, Z1.session_id, Z1.timestamp, Z1.session_start, Z1.session_time, Z1.number_of_sessions,
        ROUND((sum(Z2.number_of_sessions) / count(Z2.number_of_sessions)), 2) as mean_number_of_sessions,
        ROUND((sum(Z2.session_time) / count(Z2.session_time)), 2) as mean_session_time,
        median(Z2.number_of_sessions) as median_number_of_sessions,
        median(Z2.session_time) as median_session_time
        FROM
        (
            SELECT T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time, COUNT(DISTINCT T2.session_id) AS number_of_sessions
            FROM (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA
            ) AS T1
            LEFT JOIN (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA
            ) AS T2 ON T2.ga_id = T1.ga_id AND T2.session_start < T1.session_start
            GROUP BY T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time
        ) AS Z1
        LEFT JOIN
        (
            SELECT T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time, COUNT(DISTINCT T2.session_id) AS number_of_sessions
            FROM (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA
            ) AS T1
            LEFT JOIN (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA
            ) AS T2 ON T2.ga_id = T1.ga_id AND T2.session_start < T1.session_start
            GROUP BY T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time
        ) AS Z2 ON Z2.ga_id = Z1.ga_id AND Z2.session_start < Z1.session_start
        GROUP BY Z1.ga_id, Z1.session_id, Z1.timestamp, Z1.session_start, Z1.session_time, Z1.number_of_sessions
    )
    
    SELECT NU.*, base.number_of_sessions, base.mean_number_of_sessions, base.median_number_of_sessions, base.mean_session_time, base.median_session_time
    FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA NU
    INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp AND base.session_id = NU.session_id
"""

nu_kafka = spark.sql(query)

# COMMAND ----------

nu_kafka = nu_kafka.na.fill(value=0, subset=['number_of_sessions', 'mean_number_of_sessions', 'mean_session_time', 'median_number_of_sessions', 'median_session_time'])

# COMMAND ----------

nu_kafka.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA")

# COMMAND ----------

# %sql
# SELECT * FROM feature_store.kafka_nu_data WHERE ga_id = '000239f747e82373'

# COMMAND ----------

# MAGIC %md
# MAGIC ### ADD FIRST TRANSACTION TIME + TRANSACTION SUCCESS TIME

# COMMAND ----------

query = """
WITH interim_table AS (
  SELECT 
    ga_id,
    order_id,
    category,
    action,
    timestamp,
    date,
    traffic_sourcemedium,
    session_id,
    session_start,
    session_end,
    session_time,
    number_of_hits,
    hit_number,
    source,
    number_of_sessions,
    mean_number_of_sessions,
    median_number_of_sessions,
    mean_session_time,
    median_session_time,
    MIN(CASE WHEN action IN ('pay_cta_clicked', 'pay_again_clicked', 'pay_cta_click', 'total_pay_clicked') THEN timestamp ELSE NULL END) OVER (PARTITION BY session_id) AS first_transaction_time,
    MIN(CASE WHEN action = 'payment_success' THEN timestamp ELSE NULL END) OVER (PARTITION BY session_id) AS transaction_success_time
  FROM 
    data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA
)
SELECT 
    ga_id,
    order_id,
    category,
    action,
    timestamp,
    date,
    traffic_sourcemedium,
    session_id,
    session_start,
    session_end,
    session_time,
    number_of_hits,
    hit_number,
    source,
    number_of_sessions,
    mean_number_of_sessions,
    median_number_of_sessions,
    mean_session_time,
    median_session_time,
  COALESCE(first_transaction_time, max(first_transaction_time) OVER (PARTITION BY ga_id, session_id)) AS first_transaction_time,
  COALESCE(transaction_success_time, max(transaction_success_time) OVER (PARTITION BY ga_id, session_id)) AS transaction_success_time
FROM 
  interim_table
"""

nu_kafka = spark.sql(query)

# COMMAND ----------

nu_kafka.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CALCULATE FIRST SESSION DATE AND DAYS SINCE FIRST SESSION 

# COMMAND ----------

query = """\
WITH base AS(
    SELECT ga_id, MIN(session_start) AS first_session_start
    FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA
    GROUP BY 1
)

SELECT NU.*, base.first_session_start
FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA AS NU
INNER JOIN base ON base.ga_id = NU.ga_id
"""

nu_kafka_first_session = spark.sql(query)

# COMMAND ----------

nu_kafka_first_session.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CALCULATE NUMBER OF NON-POC ACTIONS

# COMMAND ----------

query = """\
WITH base AS (
    SELECT  T1.ga_id, T1.session_id, T1.timestamp, T1.category, T1.action,
    COUNT(T2.category) AS number_of_non_poc_actions,
    COUNT(DISTINCT T2.session_id) AS number_of_non_poc_sessions
    FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA AS T1
    LEFT JOIN
    (
        SELECT ga_id, session_id, timestamp, category, action
        FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA
        WHERE category NOT IN ('poc_flow','poc_home_page','poc_landing_page','transaction_success_page')
    ) AS T2 ON T2.ga_id = T1.ga_id AND T2.timestamp < T1.timestamp
    GROUP BY 1,2,3,4,5
)

SELECT NU.*, base.number_of_non_poc_actions, base.number_of_non_poc_sessions
FROM
    data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA AS NU
    INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp AND base.category = NU.category AND base.action = NU.action
"""

nu_kafka_non_poc = spark.sql(query)

# COMMAND ----------

# display(nu_kafka_non_poc.filter(nu_kafka_non_poc['ga_id'] == '1fab5c3149e643cb').select('ga_id', 'session_id', 'timestamp', 'category', 'action', 'hit_number', 'number_of_non_poc_actions', 'number_of_non_poc_sessions'))

# COMMAND ----------

nu_kafka_non_poc.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CALCULATE TIME SINCE FIRST SESSSION, NUMBER OF HITS AFTER TRANSACTION, TIME SPENT AFTER TRANSACTION

# COMMAND ----------

query = """\
WITH base AS (
    SELECT
        ga_id, session_id, timestamp,
        ROUND((unix_timestamp(timestamp)-unix_timestamp(first_session_start)), 2) as seconds_since_first_session,
        ROUND((unix_timestamp(session_end)-unix_timestamp(first_transaction_time))) as seconds_after_transaction,
        (number_of_hits - hit_number) AS hits_after_transaction,
        ROUND((unix_timestamp(transaction_success_time)-unix_timestamp(first_transaction_time))) as seconds_on_payment_gateway
    FROM
        data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA
)

SELECT
    NU.*, base.seconds_since_first_session, base.seconds_after_transaction, base.hits_after_transaction, base.seconds_on_payment_gateway
FROM
    data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA NU
    INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp

"""

nu_kafka_last_features = spark.sql(query)

# COMMAND ----------

nu_kafka_last_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA")

# COMMAND ----------

# display(nu_kafka_last_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### FILTER GA_IDs (KEEP LATEST TIMESTAMP FOR EACH GA_ID)

# COMMAND ----------

df = spark.sql(f"""\
SELECT * FROM data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA
""")

# assuming your dataframe is called "df"
# group by ga_id and find the maximum timestamp for each group
max_timestamp_df = df.groupBy("ga_id").agg(max("timestamp").alias("max_timestamp"))

# join the original dataframe with the max_timestamp_df on ga_id and timestamp
df_with_max_timestamp = df.join(max_timestamp_df, ["ga_id"]).filter(df.timestamp == max_timestamp_df.max_timestamp)

# drop duplicates based on ga_id and max_timestamp
result_df = df_with_max_timestamp.dropDuplicates(["ga_id", "max_timestamp"]).drop("max_timestamp")

# COMMAND ----------

result_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.RDS_FINAL_NU_DATA")

# COMMAND ----------

display(result_df)

# COMMAND ----------


