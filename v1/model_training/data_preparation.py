# Databricks notebook source
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

START_DATE_FULL = '2022-10-01'
END_DATE_FULL = '2023-10-15'

# COMMAND ----------

feature_store_query = f"""\
    select * from feature_store.nu_feature_store where date(created_at) between '{START_DATE_FULL}' AND '{END_DATE_FULL}'
"""
feature_store = spark.sql(feature_store_query).toPandas()

# COMMAND ----------

feature_store['fraud'].value_counts()

# COMMAND ----------

ga_features_query = f"""\
SELECT *
    FROM data_science_metastore.nu_payout_production_tables.ONE_TIME_FINAL_KAFKA_NU_DATA
    WHERE date(date) between '{START_DATE_FULL}' AND '{END_DATE_FULL}'
"""
ga_features = spark.sql(ga_features_query).toPandas()

LIST_OF_ACTIONS = [
    'pay_cta_clicked',
    'pay_again_clicked',
    'pay_cta_click',
    'total_pay_clicked'
]

ga_features = ga_features[ga_features['action'].isin(LIST_OF_ACTIONS)]
ga_features = ga_features.drop(columns=['order_id'])

# COMMAND ----------

feature_store = feature_store.sort_values(by='created_at', ascending=True)
ga_features = ga_features.sort_values(by='timestamp', ascending=True)

df = pd.merge_asof(feature_store, ga_features, by='ga_id',
                 left_on='created_at', right_on='timestamp',
                 tolerance=pd.Timedelta(minutes = 1))

# COMMAND ----------

feature_store.shape

# COMMAND ----------

df.shape

# COMMAND ----------

df[df['fraud'] == 1].isna().sum()

# COMMAND ----------

df['fraud'].value_counts()

# 2959 for Oct 2022 to Aug 31 2023
# 2982 for Oct 2022 to Sept 10 2023
# 3145 for Oct 22 to Oct 15 2023
# 3145 for Oct 22 to Oct 15 2023 (5 mins merge asof)


# COMMAND ----------

df.head()

# COMMAND ----------

fraud_tenants = df[df['fraud'] == 1]['tenant_id'].to_list()
fraud_phone_numbers = df[df['fraud'] == 1]['tenant_phone_number'].to_list()
fraud_ga_ids = df[df['fraud'] == 1]['ga_id'].to_list()

for index, row in df.iterrows():
    ga_id = str(row['ga_id'])
    if ga_id in fraud_ga_ids:
        ga_id_phone_number = str(row['tenant_phone_number'])
        fraud_phone_numbers.append(ga_id_phone_number)

df['fraud'] = df['tenant_id'].apply(lambda x: 1 if x in fraud_tenants else 0)
df['fraud'] = df['tenant_phone_number'].apply(lambda x: 1 if x in fraud_phone_numbers else 0)

# COMMAND ----------

df['fraud'].value_counts()

# 3546

# COMMAND ----------

df.head()

# COMMAND ----------

spark_df = spark.createDataFrame(df)

# COMMAND ----------

spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.nu_v1_training_data")

# COMMAND ----------


