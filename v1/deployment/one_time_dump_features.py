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
from datetime import timedelta
import pytz
from dateutil.relativedelta import relativedelta

days = lambda i: i * 86400

# COMMAND ----------

date_1 = '2022-10-01'
date_2 = '2023-11-07'

# COMMAND ----------

# MAGIC %md
# MAGIC ### DUMP ALL (NO EVENTS FILTER)

# COMMAND ----------

web_query = f"""\
    SELECT
        get_json_object(customDimensions, '$.dimension49') AS ga_id,
        category,
        action,
        timestamp,
        date,
        traffic_sourcemedium,
        get_json_object(event_label, '$.orderId') AS order_id
    FROM
        kafka_stream_db.housing_demand_events_web
    WHERE
        date(date) between '{date_1}' AND '{date_2}'
"""

kafka_web_data = spark.sql(web_query)

# COMMAND ----------

app_query = f"""\
    SELECT
        uid AS ga_id,
        category,
        action,
        timestamp,
        date,
        sourcemedium as traffic_sourcemedium,
        order_id
    FROM
        kafka_stream_db.housing_demand_events_app
    WHERE
        date(date) between '{date_1}' AND '{date_2}'
"""

kafka_app_data = spark.sql(app_query)

# COMMAND ----------

kafka_web_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_P1_KAFKA_NU_DATA_WEB")

# COMMAND ----------

kafka_app_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_P1_KAFKA_NU_DATA_APP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CREATE SESSION ID, CALCULATE SESSION START, SESSION END, SESSION TIME, NUMBER OF HITS, HIT NUMBER AND DUMP AGAIN

# COMMAND ----------

query = """\
    SELECT * FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_P1_KAFKA_NU_DATA_WEB
"""
df_web = spark.sql(query)

# COMMAND ----------

query = """\
    SELECT * FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_P1_KAFKA_NU_DATA_APP
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

df_web.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_P2_KAFKA_NU_DATA_WEB")

# COMMAND ----------

df_app.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_P2_KAFKA_NU_DATA_APP")

# COMMAND ----------

query = """\
    SELECT t.*, sub.session_start, sub.session_end,
    ROUND((unix_timestamp(sub.session_end)-unix_timestamp(sub.session_start))) as session_time,
    sub.number_of_hits,
    ROW_NUMBER() OVER (PARTITION BY t.session_id ORDER BY t.timestamp) AS hit_number
    FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_P2_KAFKA_NU_DATA_WEB t
    JOIN (
        SELECT
            session_id,
            MIN(timestamp) AS session_start,
            MAX(timestamp) AS session_end,
            COUNT(*) AS number_of_hits
        FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_P2_KAFKA_NU_DATA_WEB
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
    FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_P2_KAFKA_NU_DATA_APP t
    JOIN (
        SELECT
            session_id,
            MIN(timestamp) AS session_start,
            MAX(timestamp) AS session_end,
            COUNT(*) AS number_of_hits
        FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_P2_KAFKA_NU_DATA_APP
        GROUP BY session_id
    ) sub ON t.session_id = sub.session_id;
"""

df_app = spark.sql(query)

# COMMAND ----------

df_web.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_P3_KAFKA_NU_DATA_WEB")

# COMMAND ----------

df_app.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_P3_KAFKA_NU_DATA_APP")

# COMMAND ----------

# %sql
# SELECT * FROM feature_store.kafka_nu_data WHERE ga_id = '2033514486'

# COMMAND ----------

# MAGIC %md
# MAGIC ### FILTER VIA EVENTS AND DUMP AGAIN

# COMMAND ----------

p3_web_query = f"""\
    SELECT
        *
    FROM
        data_science_metastore.nu_payout_production_tables.ONE_TIME_P3_KAFKA_NU_DATA_WEB
    WHERE
        session_id IN (
            SELECT
                session_id
            FROM
                data_science_metastore.nu_payout_production_tables.ONE_TIME_P3_KAFKA_NU_DATA_WEB
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
               (category = 'page' and  action = 'landing')
            or (category = 'poc_home_page' and  action = 'open')
            or (category = 'poc_home_page' and  action = 'opened')
            or (category = 'poc_landing_page' and  action = 'opened')
            or (category = 'login_screen' and  action = 'login_screen')
            or (category = 'poc_flow' and  action = 'pay_cta_clicked') 
            or (category = 'poc_home_page' and  action = 'pay_again_clicked')
            or (category = 'poc_home_page' and  action = 'total_pay_clicked')
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
            or (category = 'poc_edit_flow' and  action = 'total_pay_clicked')
            or (category = 'final_screen' and  action = 'total_pay_clicked')
            or (category = 'poc_home_page' and  action = 'total_pay_clicked')
            or (category = 'pay_new_flow' and  action = 'total_pay_clicked')
        )
"""

p3_kafka_web_data = spark.sql(p3_web_query)
p3_kafka_web_data = p3_kafka_web_data.withColumn("source", lit("web"))

# COMMAND ----------

p3_app_query = f"""\
    SELECT
        *
    FROM
        data_science_metastore.nu_payout_production_tables.ONE_TIME_P3_KAFKA_NU_DATA_APP
    WHERE
        session_id IN (
            select
                session_id
            from
                data_science_metastore.nu_payout_production_tables.ONE_TIME_P3_KAFKA_NU_DATA_APP
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
               (category = 'app_open' and  action = 'app_opened')
            or (category = 'poc_landing_page' and  action = 'open')
            or (category = 'poc_home_page' and  action = 'open')
            or (category = 'poc_home_page' and  action = 'opened')
            or (category = 'login' and  action = 'login_completed')
            or (category = 'poc_flow' and  action = 'pay_cta_clicked') 
            or (category = 'poc_home_page' and  action = 'pay_again_clicked' )
            or (category = 'poc_home_page' and  action = 'total_pay_clicked')
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
            or (category = 'poc_edit_flow' and  action = 'total_pay_clicked')
            or (category = 'final_screen' and  action = 'total_pay_clicked')
            or (category = 'poc_home_page' and  action = 'total_pay_clicked')
            or (category = 'pay_new_flow' and  action = 'total_pay_clicked')
            or (category = 'log_out' and  action = 'clicked')
        )
"""

p3_kafka_app_data = spark.sql(p3_app_query)
p3_kafka_app_data = p3_kafka_app_data.withColumn("source", lit("app"))

# COMMAND ----------

p3_kafka_web_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_P4_KAFKA_NU_DATA_WEB")

# COMMAND ----------

p3_kafka_app_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_P4_KAFKA_NU_DATA_APP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CONCATENATE WEB AND APP

# COMMAND ----------

app = spark.sql("select * from data_science_metastore.nu_payout_production_tables.ONE_TIME_P4_KAFKA_NU_DATA_APP")
web = spark.sql("select * from data_science_metastore.nu_payout_production_tables.ONE_TIME_P4_KAFKA_NU_DATA_WEB")

combined = app.union(web)

# COMMAND ----------

combined.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA")

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
                FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
            ) AS T1
            LEFT JOIN (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
            ) AS T2 ON T2.ga_id = T1.ga_id AND T2.session_start < T1.session_start
            GROUP BY T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time
        ) AS Z1
        LEFT JOIN
        (
            SELECT T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time, COUNT(DISTINCT T2.session_id) AS number_of_sessions
            FROM (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
            ) AS T1
            LEFT JOIN (
                SELECT ga_id, session_id, timestamp, session_start, session_time
                FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
            ) AS T2 ON T2.ga_id = T1.ga_id AND T2.session_start < T1.session_start
            GROUP BY T1.ga_id, T1.session_id, T1.timestamp, T1.session_start, T1.session_time
        ) AS Z2 ON Z2.ga_id = Z1.ga_id AND Z2.session_start < Z1.session_start
        GROUP BY Z1.ga_id, Z1.session_id, Z1.timestamp, Z1.session_start, Z1.session_time, Z1.number_of_sessions
    )
    
    SELECT NU.*, base.number_of_sessions, base.mean_number_of_sessions, base.median_number_of_sessions, base.mean_session_time, base.median_session_time
    FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA NU
    INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp AND base.session_id = NU.session_id
"""

nu_kafka = spark.sql(query)

# COMMAND ----------

nu_kafka = nu_kafka.na.fill(value=0, subset=['number_of_sessions', 'mean_number_of_sessions', 'mean_session_time', 'median_number_of_sessions', 'median_session_time'])

# COMMAND ----------

nu_kafka.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA")

# COMMAND ----------

# %sql
# SELECT * FROM feature_store.kafka_nu_data WHERE ga_id = 'GA1.1.2046831729.1652445654'

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
    data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
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

nu_kafka.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CALCULATE FIRST SESSION DATE AND DAYS SINCE FIRST SESSION 

# COMMAND ----------

query = """\
WITH base AS(
    SELECT ga_id, MIN(session_start) AS first_session_start
    FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
    GROUP BY 1
)

SELECT NU.*, base.first_session_start
FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA AS NU
INNER JOIN base ON base.ga_id = NU.ga_id
"""

nu_kafka_first_session = spark.sql(query)

# COMMAND ----------

nu_kafka_first_session.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA")

# COMMAND ----------

# %sql
# select * from data_science_metastore.nu_payout_production_tables.one_time_final_kafka_nu_data where ga_id = 'E97228EE-475A-46B2-A820-6E4ED4B59F32';

# COMMAND ----------

# MAGIC %md
# MAGIC ### CALCULATE NUMBER OF NON-POC ACTIONS

# COMMAND ----------

query = """\
WITH base AS (
    SELECT  T1.ga_id, T1.session_id, T1.timestamp, T1.category, T1.action,
    COUNT(T2.category) AS number_of_non_poc_actions,
    COUNT(DISTINCT T2.session_id) AS number_of_non_poc_sessions
    FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA AS T1
    LEFT JOIN
    (
        SELECT ga_id, session_id, timestamp, category, action
        FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
        WHERE category NOT IN ('poc_flow','poc_home_page','poc_landing_page','transaction_success_page')
    ) AS T2 ON T2.ga_id = T1.ga_id AND T2.timestamp < T1.timestamp
    GROUP BY 1,2,3,4,5
)

SELECT NU.*, base.number_of_non_poc_actions, base.number_of_non_poc_sessions
FROM
    data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA AS NU
    INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp AND base.category = NU.category AND base.action = NU.action
"""

nu_kafka_non_poc = spark.sql(query)

# COMMAND ----------

# display(nu_kafka_non_poc.filter(nu_kafka_non_poc['ga_id'] == '1fab5c3149e643cb').select('ga_id', 'session_id', 'timestamp', 'category', 'action', 'hit_number', 'number_of_non_poc_actions', 'number_of_non_poc_sessions'))

# COMMAND ----------

nu_kafka_non_poc.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA")

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
        data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
)

SELECT
    NU.*, base.seconds_since_first_session, base.seconds_after_transaction, base.hits_after_transaction, base.seconds_on_payment_gateway
FROM
    data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA NU
    INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp
"""

nu_kafka_last_features = spark.sql(query)

# COMMAND ----------

nu_kafka_last_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CALCULATE WHETHER USER WAS LOGGED IN BEFORE LANDING ON POC

# COMMAND ----------

query = f"""\
WITH base AS (
    SELECT
        ga_id,
        timestamp,
        session_id,
        category,
        action,
        MAX(CASE WHEN category IN ('login_screen', 'login') AND action IN ('login_screen', 'login_completed') THEN 1 ELSE 0 END) OVER (PARTITION BY session_id) AS already_logged_in,
        MAX(CASE WHEN action IN ('refer_and_earn_clicked', 'refer_now_clicked', 'offer_clicked', 'see_all_offers_clicked') THEN 1 ELSE 0 END) OVER (PARTITION BY session_id) AS checked_offers,
        MAX(CASE WHEN category IN ('page', 'app_open') AND action IN ('landing', 'app_opened') THEN 1 ELSE 0 END) OVER (PARTITION BY session_id) AS reached_homepage_first
    FROM
        data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
    GROUP BY 1,2,3,4,5
)

SELECT NU.* except (NU.already_logged_in, NU.checked_offers, NU.reached_homepage_first),
base.already_logged_in, base.checked_offers, base.reached_homepage_first
FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA NU
INNER JOIN base ON base.ga_id = NU.ga_id AND base.timestamp = NU.timestamp 
"""

nu_kafka_v2_features = spark.sql(query)

# COMMAND ----------

nu_kafka_v2_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA")

# COMMAND ----------

# %sql
# select * from data_science_metastore.nu_payout_production_tables.one_time_final_kafka_nu_data where ga_id = 'E97228EE-475A-46B2-A820-6E4ED4B59F32';

# COMMAND ----------


