# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, unix_timestamp, min, collect_set

START_DATE = '2023-01-01'
END_DATE = '2023-07-17'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# COMMAND ----------

# feature_store_query = f"""\
# SELECT
#     fs.*,
#     o.status as payment_status,
#     case when nt.tenant_id is null then 'NU' else 'RU' end as user_flag
# FROM
#     feature_store.rentpay_feature_store_v3 fs
#     INNER JOIN apollo.orders o ON fs.id = o.id
#     LEFT JOIN (
#         select tenant_id,
#         min(created_at) as first_transaction_date
#         from apollo.orders
#         where status in ('PAYOUT_SUCCESSFUL')group by 1
#     ) AS nt ON fs.tenant_id = nt.tenant_id and fs.created_at > nt.first_transaction_date
# WHERE date(fs.created_at) BETWEEN '{START_DATE}' AND '{END_DATE}'
# """
# feature_store = spark.sql(feature_store_query).toPandas()
# df = feature_store[feature_store['user_flag'] == "NU"]

# fraud_tenants = df[df['out'] == 1]['tenant_id'].to_list()
# fraud_phone_numbers = df[df['out'] == 1]['tenant_contact_number_encrypted'].to_list()
# fraud_ga_ids = df[df['out'] == 1]['ga_id'].to_list()

# for index, row in df.iterrows():
#     ga_id = str(row['ga_id'])
#     if ga_id in fraud_ga_ids:
#         ga_id_phone_number = str(row['tenant_contact_number_encrypted'])
#         fraud_phone_numbers.append(ga_id_phone_number)

# df['out'] = df['tenant_id'].apply(lambda x: 1 if x in fraud_tenants else 0)
# df['out'] = df['tenant_contact_number_encrypted'].apply(lambda x: 1 if x in fraud_phone_numbers else 0)

# COMMAND ----------

# df.to_csv("/dbfs/FileStore/harshul/nu_payout/NU_feature_store_with_retroactive_marking.csv", index=False)

# COMMAND ----------

df = spark.read.options(header='True').csv("dbfs:/FileStore/harshul/nu_payout/NU_feature_store_with_retroactive_marking.csv")

df.createOrReplaceTempView("NU")

# COMMAND ----------

profiles_query = f"""\
SELECT nu.order_id, nu.tenant_id, nu.profile_uuid, nu.out, p.is_owner, count(distinct uf.flat_id) as unique_listing_count
FROM NU AS nu
INNER JOIN housing_clients_production.profiles p ON p.profile_uuid = nu.profile_uuid
LEFT JOIN housing_production.user_flats uf ON uf.profile_uuid = p.profile_uuid
group by 1,2,3,4,5
"""

merged_with_profiles = spark.sql(profiles_query).toPandas()

# COMMAND ----------

merged_with_profiles['tenant_id'].nunique()

# COMMAND ----------

merged_with_profiles.shape

# COMMAND ----------

merged_with_profiles['out'].value_counts()

# COMMAND ----------

merged_with_profiles[merged_with_profiles['out'] == '1']['tenant_id'].nunique()

# COMMAND ----------

merged_with_profiles[merged_with_profiles['out'] == '0']['tenant_id'].nunique()

# COMMAND ----------

merged_with_profiles[merged_with_profiles['is_owner'] == 'true']['unique_listing_count'].describe()

# COMMAND ----------

merged_with_profiles[merged_with_profiles['is_owner'] == 'true']['unique_listing_count'].quantile(0.9)

# COMMAND ----------

merged_with_profiles[merged_with_profiles['out'] == '1']['tenant_id'].nunique()

# COMMAND ----------

# DBTITLE 1,Total NU
merged_with_profiles['tenant_id'].nunique()

# COMMAND ----------

# DBTITLE 1,NU that are owners
merged_with_profiles[merged_with_profiles['is_owner'] == "true"]['tenant_id'].nunique()

# COMMAND ----------

# DBTITLE 1,NU that are not owners
merged_with_profiles[merged_with_profiles['is_owner'] == "false"]['tenant_id'].nunique()

# COMMAND ----------

# DBTITLE 1,NU that are fraud owners
merged_with_profiles[(merged_with_profiles['is_owner'] == "true") & (merged_with_profiles['out'] == '1')]['tenant_id'].nunique()

# COMMAND ----------

merged_with_profiles[(merged_with_profiles['is_owner'] == "true") & (merged_with_profiles['out'] == '1')].shape

# COMMAND ----------

# DBTITLE 1,NU that are fraud non-owners
merged_with_profiles[(merged_with_profiles['is_owner'] == "false") & (merged_with_profiles['out'] == '1')]['tenant_id'].nunique()

# COMMAND ----------

# DBTITLE 1,NU that are non-fraud owners
merged_with_profiles[(merged_with_profiles['is_owner'] == "true") & (merged_with_profiles['out'] == '0')]['tenant_id'].nunique()

# COMMAND ----------

# DBTITLE 1,NU that are non-fraud non-owners
merged_with_profiles[(merged_with_profiles['is_owner'] == "false") & (merged_with_profiles['out'] == '0')]['tenant_id'].nunique()

# COMMAND ----------


