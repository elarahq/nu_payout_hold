# Databricks notebook source
import re
import pandas as pd
import json
import datetime
import numpy as np
# from tqdm import tqdm

from pyspark.sql.functions import col, lit, udf, split, expr, concat, when, sum, mean, min, max, expr, to_timestamp, date_format, lag, coalesce, lit
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

date_1 = '2022-10-01'
date_2 = '2023-07-31'

# COMMAND ----------

# MAGIC %md
# MAGIC ### GET DATA FROM BACKEND

# COMMAND ----------

sample_data = {
    "statusCode": "2XX",
    "responseCode": "2XX",
    "version": "A",
    "data": {
        "mlPayoutHoldParams": [
            {
                "id": 0,
                "tenantId": 43434343,
                "profileUuid": "ewewewewe",
                "orderId": "3232323232",
                "gaId": "r3wdf432323",
                "source": "app",
                "platform": "erererer",
                "amount": 50000,
                "recipientBank": "ybl",
                "isOwner": False,
                "tenantName": "Sahistha Dagar",
                "landlordName": "Sahistha Dagar",
                "ageOnPlatformInMin": 30,
                "secondsSinceLastOrder": 343,
                "timeInConsecutiveTxn": 43,
                "city": "Hyderabad",
                "pocCategory": "House Rent",
                "noOfFailedTxn": 0,
                "averageFailedTxn": 10,
                "isReferral": True,
                "hourOfTheDay": 11,
                "noOfLeadsDropped": 50,
                "isProfilePicture": True
            },
            {
                "id": 1,
                "tenantId": 6987985,
                "profileUuid": "hjkgkj-hjkg-ghjf7",
                "orderId": "4457322",
                "gaId": "GA2.2046831729.35345304",
                "source": "web",
                "platform": "erererer",
                "amount": 10500,
                "recipientBank": "IDFC",
                "isOwner": True,
                "tenantName": "Yuvraj Singh ",
                "landlordName": "Rohit Sharma",
                "ageOnPlatformInMin": 30,
                "secondsSinceLastOrder": 343,
                "timeInConsecutiveTxn": 43,
                "city": "Faridabad",
                "pocCategory": "House Rent",
                "noOfFailedTxn": 10,
                "averageFailedTxn": 10,
                "isReferral": False,
                "hourOfTheDay": 11,
                "noOfLeadsDropped": 50,
                "isProfilePicture": False
            },
            {
                "id": 2,
                "tenantId": 9928852,
                "profileUuid": "hjkgkj-hjkg-ghjf7",
                "orderId": "97392243",
                "gaId": "GA2.8826134.222681",
                "source": "web",
                "platform": "erererer",
                "amount": 25000,
                "recipientBank": "paytm",
                "isOwner": True,
                "tenantName": "Megha Paul ",
                "landlordName": "Sonam Khampa",
                "ageOnPlatformInMin": 30.22,
                "secondsSinceLastOrder": 343,
                "timeInConsecutiveTxn": 43,
                "city": "Faridabad",
                "pocCategory": "House Rent",
                "noOfFailedTxn": 20,
                "averageFailedTxn": 10.23,
                "isReferral": True,
                "hourOfTheDay": 11,
                "noOfLeadsDropped": 50,
                "isProfilePicture": False
            }
        ]
    }
}

# COMMAND ----------

df = pd.DataFrame(sample_data['data']['mlPayoutHoldParams'])

# COMMAND ----------

app_ga_id_list = df[df['source'] == 'app']['gaId'].to_list()
app_ga_ids = ','.join([f"'{value}'" for value in app_ga_id_list])

web_ga_id_list = df[df['source'] == 'web']['gaId'].to_list()
web_ga_ids = ','.join([f"'{value}'" for value in web_ga_id_list])

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
    traffic_sourcemedium
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

# DBTITLE 1,Query in Housing Demand Events WEB
web_query = f"""\
    SELECT
        get_json_object(customDimensions, '$.dimension49') AS ga_id,
        category,
        action,
        timestamp,
        date,
        traffic_sourcemedium
    FROM
        kafka_stream_db.housing_demand_events_web
    WHERE
        date(date) between '{CURRENT_DATE}' - INTERVAL '90' DAY AND '{CURRENT_DATE}'
        AND get_json_object(customDimensions, '$.dimension49') IN ({web_ga_ids})
"""

kafka_web_data = spark.sql(web_query)

# COMMAND ----------

# DBTITLE 1,Query in RDS APP
app_query_rds = f"""
SELECT
    uid AS ga_id,
    category,
    action,
    timestamp,
    date,
    sourcemedium as traffic_sourcemedium
FROM
    public.housing_demand_events_app
WHERE
    pk_id <= (
        select max(pk_id) from public.housing_demand_events_web
        )
    AND uid IN ({app_ga_ids})
"""

rds_web_data = (spark.read.format("jdbc")
                   .option("driver", "org.postgresql.Driver")
                   .option("url", RDS_URL)
                   .option("query", app_query_rds)
                   .option("user","root")
                   .option("password", "ml#housing*")
                   .option("numPartitions",64).load()) #64 is optimal

# COMMAND ----------

# DBTITLE 1,Query in Housing Demand Events APP
app_query = f"""\
    SELECT
        uid AS ga_id,
        category,
        action,
        timestamp,
        date,
        sourcemedium as traffic_sourcemedium
    FROM
        kafka_stream_db.housing_demand_events_app
    WHERE
        date(date) between '{CURRENT_DATE}' - INTERVAL '90' DAY AND '{CURRENT_DATE}'
        AND get_json_object(customDimensions, '$.dimension49') IN ({app_ga_ids})
"""

kafka_app_data = spark.sql(app_query)

# COMMAND ----------

kafka_web_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "P1_KAFKA_NU_DATA_WEB")

# COMMAND ----------

kafka_app_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "P1_KAFKA_NU_DATA_APP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CREATE SESSION ID, CALCULATE SESSION START, SESSION END, SESSION TIME, NUMBER OF HITS, HIT NUMBER AND DUMP AGAIN

# COMMAND ----------

query = """\
    SELECT * FROM feature_store.P1_KAFKA_NU_DATA_WEB
"""
df_web = spark.sql(query)

# COMMAND ----------

query = """\
    SELECT * FROM feature_store.P1_KAFKA_NU_DATA_APP
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

df_web.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "P2_KAFKA_NU_DATA_WEB")

# COMMAND ----------

df_app.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "P2_KAFKA_NU_DATA_APP")

# COMMAND ----------

query = """\
    SELECT t.*, sub.session_start, sub.session_end,
    ROUND((unix_timestamp(sub.session_end)-unix_timestamp(sub.session_start))) as session_time,
    sub.number_of_hits,
    ROW_NUMBER() OVER (PARTITION BY t.session_id ORDER BY t.timestamp) AS hit_number
    FROM feature_store.P2_KAFKA_NU_DATA_WEB t
    JOIN (
        SELECT
            session_id,
            MIN(timestamp) AS session_start,
            MAX(timestamp) AS session_end,
            COUNT(*) AS number_of_hits
        FROM feature_store.P2_KAFKA_NU_DATA_WEB
        GROUP BY session_id
    ) sub ON t.session_id = sub.session_id;
"""

df_web = spark.sql(query)

# COMMAND ----------

query = """\
    SELECT t.*, sub.session_start, sub.session_end,
    ROUND((unix_timestamp(sub.session_end)-unix_timestamp(sub.session_start))) as session_time,
    sub.number_of_hits,
    ROW_NUMBER() OVER (PARTITION BY t.session_id ORDER BY t.timestamp) AS hit_number
    FROM feature_store.P2_KAFKA_NU_DATA_APP t
    JOIN (
        SELECT
            session_id,
            MIN(timestamp) AS session_start,
            MAX(timestamp) AS session_end,
            COUNT(*) AS number_of_hits
        FROM feature_store.P2_KAFKA_NU_DATA_APP
        GROUP BY session_id
    ) sub ON t.session_id = sub.session_id;
"""

df_app = spark.sql(query)

# COMMAND ----------

df_web.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "P2_KAFKA_NU_DATA_WEB")

# COMMAND ----------

df_app.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "P2_KAFKA_NU_DATA_APP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### FILTER VIA EVENTS AND DUMP AGAIN

# COMMAND ----------

p3_web_query = f"""\
    SELECT
        *
    FROM
        feature_store.P2_KAFKA_NU_DATA_WEB
    WHERE
        session_id IN (
            SELECT
                session_id
            FROM
                feature_store.P2_KAFKA_NU_DATA_WEB
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
        )
"""

p3_kafka_web_data = spark.sql(p3_web_query)
p3_kafka_web_data = p3_kafka_web_data.withColumn("source", lit("web"))

# COMMAND ----------

p3_app_query = f"""\
    SELECT
        *
    FROM
        feature_store.P2_KAFKA_NU_DATA_APP
    WHERE
        session_id IN (
            select
                session_id
            from
                feature_store.P2_KAFKA_NU_DATA_APP
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
            or (category = 'srp_card_clicked' and  action = 'details_page')
            or (category = 'auto_suggestion' and  action = 'select')
        )
"""

p3_kafka_app_data = spark.sql(p3_app_query)
p3_kafka_app_data = p3_kafka_app_data.withColumn("source", lit("app"))

# COMMAND ----------

p3_kafka_web_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("feature_store" + "." + "P3_KAFKA_NU_DATA_WEB")

# COMMAND ----------

p3_kafka_app_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("feature_store" + "." + "P3_KAFKA_NU_DATA_APP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CONCATENATE WEB AND APP

# COMMAND ----------

app = spark.sql("select * from feature_store.P3_KAFKA_NU_DATA_APP")
web = spark.sql("select * from feature_store.P3_KAFKA_NU_DATA_WEB")

combined = app.union(web)

# COMMAND ----------

combined.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "KAFKA_NU_DATA")

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
                FROM feature_store.KAFKA_NU_DATA
            ) AS T1
            LEFT JOIN (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM feature_store.KAFKA_NU_DATA
            ) AS T2 ON T2.ga_id = T1.ga_id AND T2.session_start < T1.session_start
            GROUP BY T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time
        ) AS Z1
        LEFT JOIN
        (
            SELECT T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time, COUNT(DISTINCT T2.session_id) AS number_of_sessions
            FROM (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM feature_store.KAFKA_NU_DATA
            ) AS T1
            LEFT JOIN (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM feature_store.KAFKA_NU_DATA
            ) AS T2 ON T2.ga_id = T1.ga_id AND T2.session_start < T1.session_start
            GROUP BY T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time
        ) AS Z2 ON Z2.ga_id = Z1.ga_id AND Z2.session_start < Z1.session_start
        GROUP BY Z1.ga_id, Z1.session_id, Z1.timestamp, Z1.session_start, Z1.session_time, Z1.number_of_sessions
    )
    
    SELECT NU.*, base.number_of_sessions, base.mean_number_of_sessions, base.median_number_of_sessions, base.mean_session_time, base.median_session_time
    FROM feature_store.KAFKA_NU_DATA NU
    INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp AND base.session_id = NU.session_id
"""

nu_kafka = spark.sql(query)

# COMMAND ----------

nu_kafka.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("feature_store" + "." + "KAFKA_NU_DATA")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM feature_store.kafka_nu_data WHERE ga_id = '000239f747e82373'

# COMMAND ----------

# MAGIC %md
# MAGIC ### ADD FIRST TRANSACTION TIME + TRANSACTION SUCCESS TIME

# COMMAND ----------

query = """
WITH interim_table AS (
  SELECT 
    ga_id,
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
    feature_store.kafka_nu_data
)
SELECT 
    ga_id,
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

nu_kafka.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "KAFKA_NU_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CALCULATE FIRST SESSION DATE AND DAYS SINCE FIRST SESSION 

# COMMAND ----------

query = """\
WITH base AS(
    SELECT ga_id, timestamp, category, action, MIN(session_start) AS first_session_start
    FROM feature_store.KAFKA_NU_DATA
    GROUP BY 1,2,3,4
)

SELECT NU.*, base.first_session_start
FROM feature_store.KAFKA_NU_DATA AS NU
INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp AND base.category = NU.category AND base.action = NU.action
"""

nu_kafka_first_session = spark.sql(query)

# COMMAND ----------

nu_kafka_first_session.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "KAFKA_NU_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CALCULATE NUMBER OF NON-POC ACTIONS

# COMMAND ----------

query = """\
WITH base AS (
    SELECT  T1.ga_id, T1.session_id, T1.timestamp, T1.category, T1.action,
    COUNT(T2.category) AS number_of_non_poc_actions,
    COUNT(DISTINCT T2.session_id) AS number_of_non_poc_sessions
    FROM feature_store.KAFKA_NU_DATA AS T1
    LEFT JOIN
    (
        SELECT ga_id, session_id, timestamp, category, action
        FROM feature_store.KAFKA_NU_DATA
        WHERE category NOT IN ('poc_flow','poc_home_page','poc_landing_page','transaction_success_page')
    ) AS T2 ON T2.ga_id = T1.ga_id AND T2.timestamp < T1.timestamp
    GROUP BY 1,2,3,4,5
)

SELECT NU.*, base.number_of_non_poc_actions, base.number_of_non_poc_sessions
FROM
    feature_store.KAFKA_NU_DATA AS NU
    INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp AND base.category = NU.category AND base.action = NU.action
"""

nu_kafka_non_poc = spark.sql(query)

# COMMAND ----------

display(nu_kafka_non_poc.filter(nu_kafka_non_poc['ga_id'] == '1fab5c3149e643cb').select('ga_id', 'session_id', 'timestamp', 'category', 'action', 'hit_number', 'number_of_non_poc_actions', 'number_of_non_poc_sessions'))

# COMMAND ----------

nu_kafka_non_poc.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "KAFKA_NU_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CALCULATE TIME SINCE FIRST SESSSION, NUMBER OF HITS AFTER TRANSACTION, TIME SPENT AFTER TRANSACTION

# COMMAND ----------

query = """\
WITH base AS (
    SELECT
        ga_id, session_id, timestamp,
        ROUND((unix_timestamp(timestamp)-unix_timestamp(first_session_start))/60*60, 2) as hours_since_first_session,
        ROUND((unix_timestamp(session_end)-unix_timestamp(first_transaction_time))) as seconds_after_transaction,
        (number_of_hits - hit_number) AS hits_after_transaction,
        ROUND((unix_timestamp(transaction_success_time)-unix_timestamp(first_transaction_time))) as seconds_on_payment_gateway
    FROM
        feature_store.KAFKA_NU_DATA
)

SELECT
    NU.*, base.hours_since_first_session, base.seconds_after_transaction, base.hits_after_transaction, base.seconds_on_payment_gateway
FROM
    feature_store.KAFKA_NU_DATA NU
    INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp

"""

nu_kafka_last_features = spark.sql(query)

# COMMAND ----------

nu_kafka_last_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("feature_store" + "." + "KAFKA_NU_DATA")

# COMMAND ----------


