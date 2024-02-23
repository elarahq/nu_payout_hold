# Databricks notebook source
import re
import pandas as pd
import json
import datetime
import numpy as np
# from tqdm import tqdm

from pyspark.sql.functions import col, lit, udf, split, expr, concat, when, sum, mean, min, max, expr, to_timestamp, date_format
from pyspark.sql.types import MapType, StringType,StructType,StructField, FloatType
from functools import reduce
from pyspark.sql.window import Window

import pytz
from dateutil.relativedelta import relativedelta

days = lambda i: i * 86400

# COMMAND ----------

# MAGIC %md
# MAGIC # FUNCTIONS

# COMMAND ----------

def dump_session_ids(date_1, date_2):
    query = f"""\
        select distinct(concat_ws('-',fullVisitorId, visitId)) as session_id
        from housing_bq_data.ga_sessions_web
        where
        (
            (`hits-eventInfo`.eventCategory = 'poc_landing_page') 
            or (`hits-eventInfo`.eventCategory = 'poc_home_page')
            or (`hits-eventInfo`.eventCategory = 'poc_flow')
            or (`hits-eventInfo`.eventCategory = 'transaction_success_page')
        )
        and date(date) between '{date_1}' AND '{date_2}'
    """
    session_ids = spark.sql(query)
    return session_ids

# COMMAND ----------

def ga_web_data(date_1, date_2):

    web_query = f"""\
        select
            CASE WHEN array_contains(`hits-customDimensions`.index, 49) THEN 
            `hits-customDimensions`.value[array_position(`hits-customDimensions`.index, 49) - 1] END as ga_id,
            concat_ws('-',fullVisitorId, visitId) as session_id,
            date,
            from_unixtime((visitStartTime), 'yyyy-MM-dd HH:mm:ss.SSS') as time_stamp, -- NEED TO CONVERT IT INTO DATETIME
            from_unixtime((visitStartTime), 'yyyy-MM-dd HH:mm:ss.SSS') as session_start,
            `hits-eventInfo`.eventCategory as category,
            `hits-eventInfo`.eventAction as action,
            device.mobiledevicemarketingname as marketing_name,
            device.mobiledevicebranding as brand,
            geoNetwork.region as region,
            geoNetwork.city as city,
            geoNetwork.latitude as lat,
            geoNetwork.longitude as long,
            totals.hits as no_hits,
            totals.screenviews as screen_views,
            totals.uniquescreenviews as unique_screen_views,
            totals.pageviews as page_views,
            totals.bounces as bounces,
            totals.timeOnScreen as time_on_screen,
            totals.timeOnSite as time_on_site,
            `trafficSource`.source as traffic_source,
            `trafficSource`.campaign as traffic_campaign,
            concat_ws('-',`trafficSource`.source, `trafficSource`.medium) as traffic_source_medium,
            `hits-referer` as hits_referer,
            `hits-hitNumber` as hit_number
        from
            housing_bq_data.ga_sessions_web
        where
            concat_ws('-',fullVisitorId, visitId) IN (
                select session_id
                from feature_store.NU_GA_WEB_SESSION_IDS
            )
        and
        (
            (`hits-eventInfo`.eventCategory = 'poc_flow' and  `hits-eventInfo`.eventAction = 'pay_cta_clicked') 
            or  (`hits-eventInfo`.eventCategory = 'poc_home_page' and  `hits-eventInfo`.eventAction = 'pay_again_clicked' )
            or (`hits-eventInfo`.eventCategory = 'poc_flow' and  `hits-eventInfo`.eventAction = 'pay_cta_click')
            or (`hits-eventInfo`.eventCategory = 'poc_landing_page' and  `hits-eventInfo`.eventAction = 'see_all_offers_clicked')
            or (`hits-eventInfo`.eventCategory = 'poc_home_page' and  `hits-eventInfo`.eventAction = 'offer_clicked')
            or (`hits-eventInfo`.eventCategory = 'poc_home_page' and  `hits-eventInfo`.eventAction = 'refer_now_clicked')
            or (`hits-eventInfo`.eventCategory = 'poc_home_page' and  `hits-eventInfo`.eventAction = 'refer_and_earn_clicked')
            or (`hits-eventInfo`.eventCategory = 'poc_landing_page' and  `hits-eventInfo`.eventAction = 'cta_clicked')
            or (`hits-eventInfo`.eventCategory = 'poc_home_page' and  `hits-eventInfo`.eventAction = 'edit_clicked')
            or (`hits-eventInfo`.eventCategory = 'transaction_success_page' and  `hits-eventInfo`.eventAction = 'payment_success')
            or (`hits-eventInfo`.eventCategory = 'transaction_success_page' and  `hits-eventInfo`.eventAction = 'go_to_home_cta')
            or (`hits-eventInfo`.eventCategory = 'transaction_success_page' and  `hits-eventInfo`.eventAction = 'go_to_transactions_cta')
            or (`hits-eventInfo`.eventCategory = 'conversion' and  `hits-eventInfo`.eventAction = 'open_crf')
            or (`hits-eventInfo`.eventCategory = 'conversion' and  `hits-eventInfo`.eventAction = 'filled_crf')
            or (`hits-eventInfo`.eventCategory = 'conversion' and  `hits-eventInfo`.eventAction = 'submitted_crf')
            or (`hits-eventInfo`.eventCategory = 'srp_card_clicked' and  `hits-eventInfo`.eventAction = 'details_page')
            or (`hits-eventInfo`.eventCategory = 'auto_suggestion' and  `hits-eventInfo`.eventAction = 'select')
        )"""

    ga_web_data = spark.sql(web_query)
    ga_web_data = ga_web_data.withColumn("time_stamp", to_timestamp("time_stamp", "yyyy-MM-dd HH:mm:ss.SSS"))
    ga_web_data = ga_web_data.withColumn("session_start", to_timestamp("session_start", "yyyy-MM-dd HH:mm:ss.SSS"))
    ga_web_data = ga_web_data.withColumn("source", lit("web"))

    ga_web_data_v2 = ga_web_data.select('ga_id', 'session_id', 'date', 'category', 'action', 'time_stamp', 'session_start', 'marketing_name', 'brand', 'region', 'city', 'lat', 'long', 'no_hits', 'screen_views', 'unique_screen_views', 'page_views', 'bounces', 'time_on_screen', 'time_on_site', 'traffic_source', 'traffic_campaign', 'traffic_source_medium', 'hits_referer', 'hit_number', 'source')\
        .dropDuplicates(subset=['ga_id', 'date', 'category', 'action', 'time_stamp','marketing_name', 'brand', 'region', 'city', 'lat', 'long', 'no_hits', 'screen_views', 'unique_screen_views', 'page_views', 'bounces', 'time_on_screen', 'time_on_site', 'traffic_source', 'traffic_campaign', 'traffic_source_medium', 'hits_referer', 'source'])

# removed hit_number from drop duplicate condition as it was exploding the number of rows.
    return ga_web_data_v2

# COMMAND ----------

def ga_web_data_transformation(ga_web_data_v2):

    ga_web_data_v2.createOrReplaceTempView("ga_web_data_view")
    ga_web_session_features = spark.sql("""
    SELECT
    Z1.ga_id, Z1.session_id, Z1.date, Z1.category, Z1.action, Z1.time_stamp, Z1.session_start, Z1.marketing_name, Z1.brand, Z1.region, Z1.city, Z1.lat, Z1.long, Z1.no_hits, Z1.screen_views, Z1.unique_screen_views, Z1.page_views, Z1.bounces, Z1.time_on_screen, Z1.time_on_site, Z1.traffic_source, Z1.traffic_campaign, Z1.traffic_source_medium, Z1.hits_referer, Z1.hit_number, Z1.source,
    Z1.number_of_sessions,
    Z1.number_of_page_views,
    Z1.number_of_screen_views,
    Z1.number_of_devices_used,
    Z1.hit_occurence_wrt_total_hits,
    ROUND((sum(Z2.number_of_sessions) / count(Z2.number_of_sessions)), 2) as mean_number_of_sessions,
    ROUND((sum(Z2.time_on_site) / count(Z2.time_on_site)), 2) as mean_session_time,
    median(Z2.number_of_sessions) as median_number_of_sessions,
    median(Z2.time_on_site) as median_session_time
    FROM    
        (SELECT
            ord.ga_id, ord.session_id, ord.date, ord.category, ord.action, ord.time_stamp, ord.session_start, ord.marketing_name, ord.brand, ord.region, ord.city, ord.lat, ord.long, ord.no_hits, ord.screen_views, ord.unique_screen_views, ord.page_views, ord.bounces, ord.time_on_screen, ord.time_on_site, ord.traffic_source, ord.traffic_campaign, ord.traffic_source_medium, ord.hits_referer, ord.hit_number, ord.source,
            count(DISTINCT ord2.session_id) as number_of_sessions,
            sum(ord2.page_views) as number_of_page_views,
            sum(ord2.screen_views) as number_of_screen_views,
            count(DISTINCT ord2.brand) as number_of_devices_used,
            (ord.hit_number/ord.no_hits) as hit_occurence_wrt_total_hits
        FROM
          (
            SELECT
             T1.ga_id, T1.session_id, T1.date, T1.category, T1.action, T1.time_stamp, T1.session_start, T1.marketing_name, T1.brand, T1.region, T1.city, T1.lat, T1.long, T1.no_hits, T1.screen_views, T1.unique_screen_views, T1.page_views, T1.bounces, T1.time_on_screen, T1.time_on_site, T1.traffic_source, T1.traffic_campaign, T1.traffic_source_medium, T1.hits_referer, T1.hit_number, T1.source
            FROM
              ga_web_data_view T1) AS ord
            LEFT JOIN (
                        SELECT
                          T1.ga_id, T1.session_id, T1.date, T1.category, T1.action, T1.time_stamp, T1.session_start, T1.marketing_name, T1.brand, T1.region, T1.city, T1.lat, T1.long, T1.no_hits, T1.screen_views, T1.unique_screen_views, T1.page_views, T1.bounces, T1.time_on_screen, T1.time_on_site, T1.traffic_source, T1.traffic_campaign, T1.traffic_source_medium, T1.hits_referer, T1.hit_number, T1.source
                        FROM
                          ga_web_data_view T1) AS ord2
                                              on ord2.ga_id = ord.ga_id
                                              and ord2.time_stamp < ord.time_stamp
        GROUP BY ord.ga_id, ord.session_id, ord.date, ord.category, ord.action, ord.time_stamp, ord.session_start, ord.marketing_name, ord.brand, ord.region, ord.city, ord.lat, ord.long, ord.no_hits, ord.screen_views, ord.unique_screen_views, ord.page_views, ord.bounces, ord.time_on_screen, ord.time_on_site, ord.traffic_source, ord.traffic_campaign, ord.traffic_source_medium, ord.hits_referer, ord.hit_number, ord.source) Z1
    LEFT JOIN
        (SELECT
            ord.ga_id, ord.session_id, ord.date, ord.category, ord.action, ord.time_stamp, ord.session_start, ord.marketing_name, ord.brand, ord.region, ord.city, ord.lat, ord.long, ord.no_hits, ord.screen_views, ord.unique_screen_views, ord.page_views, ord.bounces, ord.time_on_screen, ord.time_on_site, ord.traffic_source, ord.traffic_campaign, ord.traffic_source_medium, ord.hits_referer, ord.hit_number, ord.source,
            count(DISTINCT ord2.session_id) as number_of_sessions,
            sum(ord2.page_views) as number_of_page_views,
            sum(ord2.screen_views) as number_of_screen_views,
            count(DISTINCT ord2.brand) as number_of_devices_used,
            (ord.hit_number/ord.no_hits) as hit_occurence_wrt_total_hits
         FROM
            (
                SELECT
                T1.ga_id, T1.session_id, T1.date, T1.category, T1.action, T1.time_stamp, T1.session_start, T1.marketing_name, T1.brand, T1.region, T1.city, T1.lat, T1.long, T1.no_hits, T1.screen_views, T1.unique_screen_views, T1.page_views, T1.bounces, T1.time_on_screen, T1.time_on_site, T1.traffic_source, T1.traffic_campaign, T1.traffic_source_medium, T1.hits_referer, T1.hit_number, T1.source
                FROM
                ga_web_data_view T1) AS ord
        LEFT JOIN (
            SELECT
            T1.ga_id, T1.session_id, T1.date, T1.category, T1.action, T1.time_stamp, T1.session_start, T1.marketing_name, T1.brand, T1.region, T1.city, T1.lat, T1.long, T1.no_hits, T1.screen_views, T1.unique_screen_views, T1.page_views, T1.bounces, T1.time_on_screen, T1.time_on_site, T1.traffic_source, T1.traffic_campaign, T1.traffic_source_medium, T1.hits_referer, T1.hit_number, T1.source
            FROM
            ga_web_data_view T1) AS ord2
            on ord2.ga_id = ord.ga_id
            and ord2.time_stamp < ord.time_stamp
        GROUP BY ord.ga_id, ord.session_id, ord.date, ord.category, ord.action, ord.time_stamp, ord.session_start, ord.marketing_name, ord.brand, ord.region, ord.city, ord.lat, ord.long, ord.no_hits, ord.screen_views, ord.unique_screen_views, ord.page_views, ord.bounces, ord.time_on_screen, ord.time_on_site, ord.traffic_source, ord.traffic_campaign, ord.traffic_source_medium, ord.hits_referer, ord.hit_number, ord.source) Z2
            on Z2.ga_id = Z1.ga_id
            and Z2.time_stamp < Z1.time_stamp

    GROUP BY
        Z1.ga_id, Z1.session_id, Z1.date, Z1.category, Z1.action, Z1.time_stamp, Z1.session_start, Z1.marketing_name, Z1.brand, Z1.region, Z1.city, Z1.lat, Z1.long, Z1.no_hits, Z1.screen_views, Z1.unique_screen_views, Z1.page_views, Z1.bounces, Z1.time_on_screen, Z1.time_on_site, Z1.traffic_source, Z1.traffic_campaign, Z1.traffic_source_medium, Z1.hits_referer, Z1.hit_number, Z1.source,
    Z1.number_of_sessions,
    Z1.number_of_page_views,
    Z1.number_of_screen_views,
    Z1.number_of_devices_used,
    Z1.hit_occurence_wrt_total_hits
    """)

    # ga_web_session_features = ga_web_session_features.withColumn('user_activity', lit(None).cast('integer')).withColumn("source", lit("web"))
    return ga_web_session_features

# COMMAND ----------

# MAGIC %md
# MAGIC # EXECUTION

# COMMAND ----------

spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

date_1 = '2023-01-01'
date_2 = '2023-07-31'

# COMMAND ----------

session_ids = dump_session_ids(date_1, date_2)
session_ids.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("feature_store" + "." + "NU_GA_WEB_SESSION_IDS")
del(session_ids)

# COMMAND ----------

web_data = ga_web_data(date_1, date_2)

# COMMAND ----------

web_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("feature_store" + "." + "NU_GA_WEB_INTERMEDIATE_TABLE")
del(web_data)

# COMMAND ----------

web_data = spark.sql("""SELECT * FROM feature_store.NU_GA_WEB_INTERMEDIATE_TABLE""")
web_session_features = ga_web_data_transformation(web_data)

# web_data = web_data.withColumn("source", lit("web")).select(
#                       col('date').alias('date'),
#                       col('ga_id').alias('ga_id'),
#                       col('time_stamp').alias('timestamp'),
#                       col('session_id').alias('session_id'),
#                       col('category').alias('category'),
#                       col('action').alias('action'),
#                       col('marketing_name').alias('mobileDeviceMarketingName'),
#                       col('brand').alias('mobileDeviceBranding'),
#                       col('region').alias('region'),
#                       col('city').alias('city'),
#                       col('lat').alias('latitude'),
#                       col('long').alias('longitude'),
#                       col('no_hits').alias('no_hits'),
#                       col('time_on_site').alias('time_on_site'),
#                       col('screen_views').alias('screen_views'),
#                       col('unique_screen_views').alias('unique_screen_views'),
#                       col('page_views').alias('page_views'),
#                       col('bounces').alias('bounces'),
#                       col('time_on_screen').alias('time_on_screen'),
#                       col('traffic_source').alias('traffic_source'),
#                       col('hits_referer').alias('hits_referer'),
#                       col('hit_number').alias('hit_number'),
#                       col('source').alias('source')
                    # )

web_session_features.repartition(10).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("feature_store" + "." + "NU_GA_FEATURE_STORE_more_features")
del(web_session_features)
# del(web_session_features)

# COMMAND ----------


