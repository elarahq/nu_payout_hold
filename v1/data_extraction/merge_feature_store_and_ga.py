# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, unix_timestamp, min, collect_set


START_DATE = '2023-01-01'
END_DATE = '2023-07-17'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# COMMAND ----------

new_users = spark.read.options(header='True').csv("dbfs:/FileStore/harshul/nu_payout/NU_feature_store_with_retroactive_marking.csv")

# COMMAND ----------

nu_ga_features_query = f"""\
SELECT *
FROM feature_store.NU_GA_FEATURE_STORE
"""
nu_ga_features = spark.sql(nu_ga_features_query)

# COMMAND ----------

nu_ga_features.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MERGE ON REAL-TIME (pyspark)

# COMMAND ----------

nu_ga_features = nu_ga_features.withColumnRenamed('city', 'city_ga')\
    .withColumnRenamed('ga_id', 'ga_id_ga')

# COMMAND ----------

df = new_users.join(nu_ga_features, new_users.ga_id == nu_ga_features.ga_id_ga, how='left')

# COMMAND ----------

df.repartition(10).write.format("delta").mode("overwrite").saveAsTable("feature_store" + "." + "NU_FULL_DATA")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## MERGE ON 2 DAYS PRIOR

# COMMAND ----------



# COMMAND ----------

merged = new_users.join(ga_features, on='ga_id', how='outer')

time_diff_sec = (unix_timestamp(col('created_at')) - unix_timestamp(col('timestamp')))
time_diff_hours = time_diff_sec / 3600

merged = merged.withColumn('duration_hours', time_diff_hours)
old_user = merged.where("duration_hours >= 48 and duration_hours <= 2160")

new_user_less_than_2day = merged.where("duration_hours < 48 and duration_hours >=0")
new_user_more_than_90day = merged.where("duration_hours >2160")

# COMMAND ----------

df_min_duration = old_user.groupby('order_id').agg(min(col('duration_hours')).alias('min_duration_hours'))
old_user = old_user.join(df_min_duration, on='order_id', how='inner')

df_result_2 = old_user.where("duration_hours == min_duration_hours")

# COMMAND ----------

def drop_duplicates_ignore_columns(df, ignore_columns):
    column_names = [col_name for col_name in df.columns if col_name not in ignore_columns]
    df_without_duplicates = df.dropDuplicates(subset=column_names)
    return df_without_duplicates

ignore_columns = ['date',
 'ga_id',
 'timestamp',
 'session_id',
 'mobileDeviceMarketingName',
 'mobileDeviceBranding',
 'region',
 'city',
 'latitude',
 'longitude',
 'no_hits',
 'time_on_site',
 'screen_views',
 'user_activity',
 'session_counts',
 'mean_session_counts',
 'median_session_counts',
 'mean_session_time',
 'median_session_time',
 'source',
 'duration_hours']
new_user_less_than_2day = drop_duplicates_ignore_columns(new_user_less_than_2day, ignore_columns)
new_user_more_than_90day = drop_duplicates_ignore_columns(new_user_more_than_90day, ignore_columns)

# COMMAND ----------

old_order_id = set(old_user.select(collect_set('order_id')).first()[0])

# COMMAND ----------

new_user_more_than_90day = new_user_more_than_90day.join(df_result_2, on=["order_id"], how="left_anti")
new_user_less_than_2day = new_user_less_than_2day.join(df_result_2, on=["order_id"], how="left_anti")

# COMMAND ----------

df_1= new_user_more_than_90day.toPandas()
df_1.to_csv("/dbfs/FileStore/harshul/nu_payout/new_user_more_than_90day_1.csv", index=False)

del df_1
df_2= new_user_less_than_2day.toPandas()
df_2.to_csv("/dbfs/FileStore/harshul/nu_payout/new_user_less_than_2day_1.csv", index=False)

del df_2
df_1= df_result_2.toPandas()
df_1.to_csv("/dbfs/FileStore/harshul/nu_payout/old_user_1.csv", index=False)


# COMMAND ----------

old_user= pd.read_csv("/dbfs/FileStore/harshul/nu_payout/old_user_1.csv")
new_user_less_than_2day =pd.read_csv("/dbfs/FileStore/harshul/nu_payout/new_user_less_than_2day_1.csv")
new_user_more_than_90day =pd.read_csv("/dbfs/FileStore/harshul/nu_payout/new_user_more_than_90day_1.csv")

# COMMAND ----------

new_user_combined = pd.concat([new_user_less_than_2day,new_user_more_than_90day])

ga_col=['date',
 'timestamp',
 'session_id',
 'mobileDeviceMarketingName',
 'mobileDeviceBranding',
 'region',
 'city',
 'latitude',
 'longitude',
 'no_hits',
 'time_on_site',
 'screen_views',
 'user_activity',
 'session_counts',
 'mean_session_counts',
 'median_session_counts',
 'mean_session_time',
 'median_session_time',
 'source']

for col in ga_col:
    new_user_combined[f"{col}"]=np.nan 

# COMMAND ----------

old_user.drop('min_duration_hours',axis=1,inplace=True)
df_final=pd.concat([new_user_combined,old_user])
df_final.to_csv("/dbfs/FileStore/harshul/nu_payout/NU_feature_store_merged_with_ga.csv", index=False)

# COMMAND ----------

df_final.shape

# COMMAND ----------

df_final.isna().sum()

# COMMAND ----------


