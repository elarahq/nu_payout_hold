# Databricks notebook source
# MAGIC %md
# MAGIC THIS FEATURE STORE CONTAINS FRAUD LABELS AND PHONE NUMBERS (WITHOUT ANY CARDS FEATURE)\
# MAGIC ALSO WILL CONTAIN TWO ADDITIONAL FEATURES : NO. OF RED TRANSACTIONS AND NO. OF YELLOW TRANSACTIONS IN LAST 30 DAYS

# COMMAND ----------

!pip install tqdm

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import col, max as max_, lit, when, split, isnan, count, hour, struct, sum, countDistinct
from pyspark.sql.types import StringType,BooleanType,DateType,IntegerType,TimestampType,FloatType
from pyspark.sql.functions import expr, create_map
from pyspark.sql import Window
from datetime import datetime
from itertools import chain
import pytz
from dateutil.relativedelta import relativedelta

import pickle
import requests
import json

# COMMAND ----------

current_date=(datetime.now(pytz.timezone('Asia/Kolkata'))+relativedelta(days=-1)).strftime("%Y-%m-%d")
print(current_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## FUNCTIONS FOR FEATURE STORE

# COMMAND ----------

def get_transactions_level_features():
  orders_query = '''
WITH base AS (
                select tenant_id,
                case when min_successful_date is null then max_payment_date else min_successful_date end as cutoff_date,
                first_transaction_date
                from (
                      select tenant_id,max(created_at) as max_payment_date, min(created_at) as first_transaction_date,
                      min(case when status = 'PAYOUT_SUCCESSFUL' then created_at end ) as min_successful_date
                      from apollo.orders
                      group by tenant_id
                    ) as x
                group by 1,2,3
              )
SELECT
    T1.id, T1.order_id, T1.created_at, T1.tenant_id, T2.profile_uuid, T1.ga_id, T1.tenant_name, T1.landlord_name, T1.tenant_contact_number_encrypted, T1.amount, T1.platform, T1.status, T1.landlord_bank_account_id, b.cutoff_date, b.first_transaction_date, T2.created_at as tenant_created_at,
    mrt.rent_type AS poc_category,
    CASE WHEN u.profile_picture_url IS NOT NULL THEN 1 ELSE 0 END AS profile_picture_url,
    CASE WHEN T1.referral_code IS NOT NULL THEN 'yes' ELSE 'no' END AS referral_code,
    case when NT.tenant_id is null then 'NU' else 'RU' end as user_flag,
    ROUND((unix_timestamp(T1.created_at)-unix_timestamp(T2.created_at))) as tenant_age_in_seconds,
    ROUND((unix_timestamp(T1.created_at)-unix_timestamp(ct.successful_at))) as seconds_on_payment_gateway_backend,
    CASE WHEN p.is_owner = 'true' THEN 1 ELSE 0 END AS is_owner,
    T3.fraud
  FROM
    apollo.orders AS T1
    LEFT JOIN (
          SELECT tenant_id, min(created_at) as first_transaction_date
          FROM apollo.orders
          WHERE status IN ('PAYOUT_SUCCESSFUL')
          GROUP BY 1
        ) AS NT ON T1.tenant_id = NT.tenant_id AND T1.created_at > NT.first_transaction_date
    INNER JOIN base b on T1.tenant_id = b.tenant_id and T1.created_at <= b.cutoff_date
    LEFT JOIN apollo.tenants AS T2 ON T2.id = T1.tenant_id
    INNER JOIN etl.pay_rent AS T3 ON T3.order_id = T1.order_id
    INNER JOIN apollo.master_rent_type mrt on mrt.id = T1.rent_type_id
    INNER JOIN housing_clients_production.profiles p ON p.profile_uuid = T2.profile_uuid
    INNER JOIN housing_clients_production.users u ON u.id = p.user_id
    LEFT JOIN fortuna.sub_credit_transactions sct ON T1.transaction_id = sct.sub_credit_txn_uuid
    LEFT JOIN fortuna.credit_transactions ct ON ct.id = sct.credit_transaction_id
  '''
  orders = spark.sql(orders_query)
  return orders

# COMMAND ----------



# COMMAND ----------

def get_sensible_kyc_logs_and_landlord_accounts():
  sensible_kyc_query = '''\
    SELECT * FROM apollo.sensible_kyc_logs
  '''
  
  landlord_bank_acc_query = '''\
    SELECT id AS landlord_bank_account_id, ifsc_code, upi_encrypted FROM apollo.landlord_bank_accounts
  '''
  
  sensible_logs = spark.sql(sensible_kyc_query)
  landlord_bank_accounts = spark.sql(landlord_bank_acc_query)
  
  return sensible_logs, landlord_bank_accounts

# COMMAND ----------

def get_number_of_different_locations():
  ip_query = '''\
    WITH base AS (
      SELECT
        ord1.order_id, ord1.tenant_id, ord1.created_at, ord1.city, count(DISTINCT ord2.city) as number_of_different_locations
      FROM
        (
          SELECT
            T1.order_id, T1.tenant_id, T1.created_at, T2.city as city
          FROM
            apollo.orders T1
            INNER JOIN apollo.user_ip_addresses T2 ON T2.order_id = T1.id
        ) AS ord1
          LEFT JOIN (
                      SELECT
                        T1.order_id, T1.tenant_id, T1.created_at, T2.city as city
                      FROM
                        apollo.orders T1
                        INNER JOIN apollo.user_ip_addresses T2 ON T2.order_id = T1.id
                    ) AS ord2 ON ord2.tenant_id = ord1.tenant_id and ord2.created_at < ord1.created_at
      GROUP BY
        ord1.order_id, ord1.tenant_id, ord1.created_at, ord1.city
      )

      SELECT base.order_id AS ip_details_order_id, base.city, base.number_of_different_locations
    FROM base
  '''
  ip_addr = spark.sql(ip_query)
  return ip_addr

# COMMAND ----------

def get_number_of_failed_transactions():
  failed_transactions_query = """\
    WITH base AS (
      SELECT order_id, tenant_id, created_at, amount, count(to_be_summed) as number_of_failed_transactions
      FROM
        (
          SELECT
            T1.order_id, T1.tenant_id, T1.created_at, T1.amount, T2.created_at AS to_be_summed
          FROM
            apollo.orders T1
            LEFT JOIN apollo.orders T2 ON T2.tenant_id = T1.tenant_id AND T2.created_at < T1.created_at AND T2.status = 'PAYMENT_FAILED'
        )
      GROUP BY
        order_id, tenant_id, created_at, amount
    )

    SELECT base.order_id AS failed_transactions_order_id, base.number_of_failed_transactions
    FROM base
  """
  failed_transactions_df = spark.sql(failed_transactions_query)
  return failed_transactions_df

# COMMAND ----------

def get_total_transactions_average_transactions_time_difference_average_time_difference():
  total_transactions_query = """\
    WITH base AS (
      SELECT
        BT1.order_id, BT1.total_number_of_transactions, BT1.average_number_of_transactions, BT1.seconds_since_last_transaction,
        ROUND((sum(BT2.seconds_since_last_transaction)/count(BT2.seconds_since_last_transaction)),2) as average_seconds_between_two_transactions
      FROM
        (
          SELECT
            T1.order_id AS order_id, T1.tenant_id as tenant_id, T1.created_at as created_at, T1.total_number_of_transactions,
            ROUND((sum(T2.total_number_of_transactions) / count(T2.total_number_of_transactions)), 2) as average_number_of_transactions,
            ROUND((unix_timestamp(T1.created_at) - unix_timestamp(LAG (T1.created_at, 1) OVER (PARTITION BY T1.tenant_id ORDER BY T1.created_at ASC))) ,2) as seconds_since_last_transaction
          FROM
            (
              SELECT order_id, tenant_id, created_at, amount, count(to_be_summed) as total_number_of_transactions
              FROM
                (
                  SELECT
                    T1.order_id, T1.tenant_id, T1.created_at, T1.amount, T2.created_at AS to_be_summed
                  FROM
                    apollo.orders T1
                    LEFT JOIN apollo.orders T2 ON T2.tenant_id = T1.tenant_id AND T2.created_at < T1.created_at
                )
              GROUP BY order_id, tenant_id, created_at, amount
            ) AS T1
            LEFT JOIN (
                SELECT order_id, tenant_id, created_at, amount, count(to_be_summed) as total_number_of_transactions
                FROM
                  (
                    SELECT
                      T1.order_id, T1.tenant_id, T1.created_at, T1.amount, T2.created_at AS to_be_summed
                    FROM
                      apollo.orders T1
                      LEFT JOIN apollo.orders T2 ON T2.tenant_id = T1.tenant_id AND T2.created_at < T1.created_at
                  )
                GROUP BY order_id, tenant_id, created_at, amount
            ) AS T2 ON T1.tenant_id = T2.tenant_id AND T2.created_at < T1.created_at
          GROUP BY T1.order_id, T1.tenant_id, T1.created_at, T1.total_number_of_transactions
        ) AS BT1
        LEFT JOIN (
              SELECT
                T1.order_id AS order_id, T1.tenant_id as tenant_id, T1.created_at as created_at, T1.total_number_of_transactions,
                ROUND((sum(T2.total_number_of_transactions) / count(T2.total_number_of_transactions)), 2) as average_number_of_transactions,
                ROUND((unix_timestamp(T1.created_at) - unix_timestamp(LAG (T1.created_at, 1) OVER (PARTITION BY T1.tenant_id ORDER BY T1.created_at ASC))) ,2) as seconds_since_last_transaction
              FROM
                (
                  SELECT order_id, tenant_id, created_at, amount, count(to_be_summed) as total_number_of_transactions
                  FROM
                    (
                      SELECT
                        T1.order_id, T1.tenant_id, T1.created_at, T1.amount, T2.created_at AS to_be_summed
                      FROM
                        apollo.orders T1
                        LEFT JOIN apollo.orders T2 ON T2.tenant_id = T1.tenant_id AND T2.created_at < T1.created_at
                    )
                  GROUP BY order_id, tenant_id, created_at, amount
                ) AS T1
                LEFT JOIN (
                    SELECT order_id, tenant_id, created_at, amount, count(to_be_summed) as total_number_of_transactions
                    FROM
                      (
                        SELECT
                          T1.order_id, T1.tenant_id, T1.created_at, T1.amount, T2.created_at AS to_be_summed
                        FROM
                          apollo.orders T1
                          LEFT JOIN apollo.orders T2 ON T2.tenant_id = T1.tenant_id AND T2.created_at < T1.created_at
                      )
                    GROUP BY order_id, tenant_id, created_at, amount
                ) AS T2 ON T1.tenant_id = T2.tenant_id AND T2.created_at < T1.created_at
              GROUP BY T1.order_id, T1.tenant_id, T1.created_at, T1.total_number_of_transactions
        ) AS BT2 ON BT2.tenant_id = BT1.tenant_id AND BT2.created_at < BT1.created_at
      GROUP BY BT1.order_id, BT1.total_number_of_transactions, BT1.average_number_of_transactions, BT1.seconds_since_last_transaction
    )

    SELECT
      base.order_id AS total_transactions_order_id, base.total_number_of_transactions,
      base.average_number_of_transactions, base.seconds_since_last_transaction,
      base.average_seconds_between_two_transactions
    FROM base
  """
  total_transactions_df = spark.sql(total_transactions_query)
  return total_transactions_df

# COMMAND ----------

def get_number_of_leads_dropped():
  query = """\
    WITH base AS (
        select
            ld.profile_uuid, count(distinct lh.id) as number_of_leads
        from
            product_derived.leads_heavy lh
            inner join housing_leads_production.lead_details ld on lh.lead_details_id = ld.id
        group by 1
    )

    SELECT base.profile_uuid AS num_leads_profile_uuid, base.number_of_leads
    FROM base
  """

  num_leads_df = spark.sql(query)
  return num_leads_df

# COMMAND ----------

# DBTITLE 1,Get number of bank accounts
def get_number_of_bank_accounts():
  query = f"""\
    WITH base AS (
        SELECT
                T1.order_id, T1.created_at, T1.tenant_id, T1.account_number_encrypted, COUNT(DISTINCT T2.account_number_encrypted) AS number_of_account_numbers,
                T1.upi_encrypted, COUNT(DISTINCT T2.upi_encrypted) AS number_of_upi_addresses 
        FROM
                (
                  SELECT
                    O.order_id, O.created_at, O.tenant_id, L.account_number_encrypted, L.upi_encrypted
                  FROM
                    apollo.orders O
                    INNER JOIN apollo.landlord_bank_accounts L ON L.id = O.landlord_bank_account_id
                  ) T1
              LEFT JOIN
                  (
                    SELECT
                      O.order_id, O.created_at, O.tenant_id, L.account_number_encrypted, L.upi_encrypted
                    FROM
                      apollo.orders O
                      INNER JOIN apollo.landlord_bank_accounts L ON L.id = O.landlord_bank_account_id
                  ) T2 ON T2.tenant_id = T1.tenant_id AND T2.created_at < T1.created_at
        GROUP BY 1,2,3,4,6
        )

    SELECT b.order_id AS num_banks_order_id, b.number_of_account_numbers, b.number_of_upi_addresses,
            (b.number_of_account_numbers + b.number_of_upi_addresses) AS number_of_banks
    FROM base as b;
    """

  num_banks_df = spark.sql(query)
  return num_banks_df

# COMMAND ----------

# DBTITLE 1,Get median transaction amount for all transactions / last x transactions
def get_mean_median_transaction_amount():
  query = f"""\
  WITH base AS (
    SELECT
          T1.order_id, T1.tenant_id, T1.created_at, T1.amount, 
          sum(T2.amount) AS total_transaction_amount,
          count(distinct T2.order_id) AS total_number_of_transactions,
          mean(T2.amount) AS mean_amount,
          median(T2.amount) AS median_amount
    FROM (
          SELECT order_id, tenant_id, created_at, amount
          FROM apollo.orders
        ) T1
        LEFT JOIN (
          SELECT order_id, tenant_id, created_at, amount
          FROM apollo.orders
        ) T2 ON T2.tenant_id = T1.tenant_id AND T2.created_at < T1.created_at
    GROUP BY 1,2,3,4
  )

  SELECT b.order_id AS mean_median_amount_order_id, b.total_transaction_amount, b.total_number_of_transactions AS total_transactions, b.mean_amount, b.median_amount
  FROM base as b;
  """

  mean_median_amount_df = spark.sql(query)
  return mean_median_amount_df

# COMMAND ----------

def get_mean_median_transaction_amount_of_last_3_transactions():
  query = f"""\
  WITH base AS (
    SELECT
      T1.order_id, T1.tenant_id, T1.created_at, T1.amount,
      sum(T2.amount) AS total_transaction_amount,
      count(distinct T2.order_id) AS total_number_of_transactions,
      mean(T2.amount) AS mean_amount,
      median(T2.amount) AS median_amount
    FROM (
          SELECT order_id, tenant_id, created_at, amount, ROW_NUMBER() OVER (PARTITION BY tenant_id ORDER BY created_at) AS row_num
          FROM apollo.orders
        ) T1
        LEFT JOIN (
          SELECT order_id, tenant_id, created_at, amount, ROW_NUMBER() OVER (PARTITION BY tenant_id ORDER BY created_at) AS row_num
          FROM apollo.orders
        ) T2 ON T2.tenant_id = T1.tenant_id AND T2.created_at < T1.created_at AND T2.row_num >= T1.row_num - 3
    GROUP BY 1,2,3,4
  )

  SELECT b.order_id AS mean_median_amount_last_3_order_id, b.total_transaction_amount AS transaction_amount_of_last_3_transactions, b.mean_amount AS mean_amount_of_last_3_transactions, b.median_amount AS median_amount_of_last_3_transactions
  FROM base as b;
  """
  mean_median_amount_last_3_df = spark.sql(query)
  return mean_median_amount_last_3_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## UPDATE PHONE NUMBERS IN JSON

# COMMAND ----------

orders = get_transactions_level_features()

sensible_logs, landlord_bank_accounts = get_sensible_kyc_logs_and_landlord_accounts()
different_locations_details = get_number_of_different_locations()

orders = orders.join(different_locations_details, orders.order_id == different_locations_details.ip_details_order_id, how='inner')
orders = orders.join(landlord_bank_accounts, orders.landlord_bank_account_id == landlord_bank_accounts.landlord_bank_account_id, how='inner')

# COMMAND ----------

orders = orders.na.fill(value='other', subset=['city'])

# COMMAND ----------

encrypted_upi = orders.select('upi_encrypted').rdd.flatMap(lambda x: x).collect()
encrypted_phone_numbers = orders.select('tenant_contact_number_encrypted').rdd.flatMap(lambda x: x).collect()

list_of_upi = list(set([x for x in encrypted_upi if x]))
list_of_phone_numbers = list(set([x for x in encrypted_phone_numbers if x]))

# COMMAND ----------

print(len(list_of_phone_numbers))
print(len(list_of_upi))

# 610433
# 295316

# COMMAND ----------

import requests
import json
from tqdm import tqdm

BIG_JSON_PHONE_NUMBERS = {}
chunks_of_50 = [list_of_phone_numbers[i:i + 50] for i in range(0, len(list_of_phone_numbers), 50)]

endpoint_url = 'https://rentpay.housing.com/apollo/encryption/decrypt/records'

for chunk in tqdm(chunks_of_50):
  body = {
    "records" : [x for x in chunk if '==' in x]
  }
  response = requests.post(endpoint_url, json = body, headers={'X-ML-KEY': 'LKJSDOFIJERN234KN23LKJ2LK2'})
#   print(json.loads(response.text)['data'])
  decrypted_chunk = json.loads(response.text)['data']
  for i,j in zip(chunk, decrypted_chunk):
    BIG_JSON_PHONE_NUMBERS[i] = j

# COMMAND ----------

import requests
import json
from tqdm import tqdm

BIG_JSON_UPI = {}
chunks_of_50 = [list_of_upi[i:i + 50] for i in range(0, len(list_of_upi), 50)]

endpoint_url = 'https://rentpay.housing.com/apollo/encryption/decrypt/records'

for chunk in tqdm(chunks_of_50):
  body = {
    "records" : [x for x in chunk if '==' in x]
  }
  response = requests.post(endpoint_url, json = body, headers={'X-ML-KEY': 'LKJSDOFIJERN234KN23LKJ2LK2'})
#   print(json.loads(response.text)['data'])
  decrypted_chunk = json.loads(response.text)['data']
  for i,j in zip(chunk, decrypted_chunk):
    BIG_JSON_UPI[i] = j

# COMMAND ----------

with open('/dbfs/FileStore/shared_uploads/harshul.kuhar@housing.com/rent_pay/ENCRYPTED_PHONE_NUMBER_DICT.b', 'rb') as read_file_phone_numbers:
    PHONE_NUMBER_LOOKUP = pickle.load(read_file_phone_numbers)

print(f"PREVIOUS PHONE NUMBERS IN JSON :: {len(PHONE_NUMBER_LOOKUP)}")
PHONE_NUMBER_LOOKUP.update(BIG_JSON_PHONE_NUMBERS)
print(f"LATEST PHONE NUMBERS IN JSON :: {len(PHONE_NUMBER_LOOKUP)}")

with open('/dbfs/FileStore/shared_uploads/harshul.kuhar@housing.com/rent_pay/ENCRYPTED_PHONE_NUMBER_DICT.b', 'wb') as write_file_phone_numbers:
    pickle.dump(PHONE_NUMBER_LOOKUP, write_file_phone_numbers)






with open('/dbfs/FileStore/shared_uploads/harshul.kuhar@housing.com/rent_pay/ENCRYPTED_UPI_DICT.b', 'rb') as read_file_upi:
    UPI_LOOKUP = pickle.load(read_file_upi)

print(f"PREVIOUS UPIs IN JSON :: {len(UPI_LOOKUP)}")
UPI_LOOKUP.update(BIG_JSON_UPI)
print(f"LATEST UPIs IN JSON :: {len(UPI_LOOKUP)}")

with open('/dbfs/FileStore/shared_uploads/harshul.kuhar@housing.com/rent_pay/ENCRYPTED_UPI_DICT.b', 'wb') as write_file_upi:
    pickle.dump(UPI_LOOKUP, write_file_upi)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXECUTION HERE

# COMMAND ----------

with open('/dbfs/FileStore/shared_uploads/harshul.kuhar@housing.com/rent_pay/ENCRYPTED_UPI_DICT.b', 'rb') as read_file:
    UPI_LOOKUP = pickle.load(read_file)

with open('/dbfs/FileStore/shared_uploads/harshul.kuhar@housing.com/rent_pay/ENCRYPTED_PHONE_NUMBER_DICT.b', 'rb') as read_file:
    PHONE_NUMBER_LOOKUP = pickle.load(read_file)

import pandas as pd

upi_df = pd.DataFrame(UPI_LOOKUP.items(), columns=['hash_upi','upi'])
upi_df_spark = spark.createDataFrame(upi_df)

phone_num_df = pd.DataFrame(PHONE_NUMBER_LOOKUP.items(), columns=['hash_num','tenant_phone_number'])
phone_num_df_spark = spark.createDataFrame(phone_num_df)

# COMMAND ----------

def build_feature_store():
  orders = get_transactions_level_features()

  ## JOIN WITH LANDLORD FOR UPI AND PHONE NUMBERS. ALSO DECRYPT UPI AND PHONE NUMBERS
  sensible_logs, landlord_bank_accounts = get_sensible_kyc_logs_and_landlord_accounts()
  orders = orders.join(landlord_bank_accounts, orders.landlord_bank_account_id == landlord_bank_accounts.landlord_bank_account_id, how='inner')
  orders = orders.join(upi_df_spark, orders.upi_encrypted == upi_df_spark.hash_upi, how='left')
  orders = orders.join(phone_num_df_spark, orders.tenant_contact_number_encrypted == phone_num_df_spark.hash_num, how='left')
  
  ## JOIN WITH DIFFERENT LOCATIONS. ALSO IMPUTE MISSING VALUES FOR CITY
  different_locations_details = get_number_of_different_locations()
  orders = orders.join(different_locations_details, orders.order_id == different_locations_details.ip_details_order_id, how='inner')
  orders = orders.na.fill(value='other', subset=['city'])


  ## INTERMEDIATE FILTERING OF ROWS
  orders = orders.select('order_id', 'id', 'ga_id', 'tenant_id', 'profile_uuid', 'tenant_name', 'landlord_name', 'is_owner', 'tenant_phone_number', 'tenant_age_in_seconds', 'poc_category', 'amount', 'created_at', 'status', 'cutoff_date', 'first_transaction_date', 'tenant_created_at', 'user_flag', 'platform', 'city', 'number_of_different_locations', 'ifsc_code', 'upi', 'profile_picture_url', 'referral_code', 'seconds_on_payment_gateway_backend', 'fraud')

  ## CONVERT TIMESTAMP TO IST
  orders = orders.withColumn("ist_time", col('created_at') + expr('INTERVAL 5 HOURS') + expr('INTERVAL 30 MINUTES'))

  ## JOIN WITH NUMBER OF FAILED TRANSACTIONS
  failed_transactions_df = get_number_of_failed_transactions()
  orders = orders.join(failed_transactions_df, orders.order_id == failed_transactions_df.failed_transactions_order_id, how='left')

  ## JOIN WITH TOTAL TRANSACTIONS, AVERAGE TRANSACTIONS, SECONDS SINCE LAST TRANSACTION AND AVERAGE TIME BETWEEN TWO CONSECUTIVE TRANSACTIONS
  total_transactions_df = get_total_transactions_average_transactions_time_difference_average_time_difference()
  orders = orders.join(total_transactions_df, orders.order_id == total_transactions_df.total_transactions_order_id, how='left')

  ## JOIN WITH NUMBER OF LEADS
  num_leads_df = get_number_of_leads_dropped()
  orders = orders.join(num_leads_df, orders.profile_uuid == num_leads_df.num_leads_profile_uuid, how='left')

  ## JOIN WITH NUMBER OF BANK ACCOUNTS
  num_bank_accounts_df = get_number_of_bank_accounts()
  orders = orders.join(num_bank_accounts_df, orders.order_id == num_bank_accounts_df.num_banks_order_id, how='left')

  ## JOIN WITH MEAN AND MEDIAN TRANSACTION AMOUNT (HISTORICAL AND LAST 3 TRANSACTIONS)
  mean_median_amount_df = get_mean_median_transaction_amount()
  orders = orders.join(mean_median_amount_df, orders.order_id == mean_median_amount_df.mean_median_amount_order_id, how='left')


  mean_median_amount_last_3_df = get_mean_median_transaction_amount_of_last_3_transactions()
  orders = orders.join(mean_median_amount_last_3_df, orders.order_id == mean_median_amount_last_3_df.mean_median_amount_last_3_order_id, how='left')

  ## EXTRACT BANK FROM IFSC AND UPI, ALSO EXTRACT TIME HOUR FROM TIMESTAMP
  orders = orders.withColumn('upi_bank', when(col("upi").contains("@"),split(orders['upi'], '@').getItem(1)).otherwise("other"))
  orders = orders.withColumn('account_bank',  when(col("ifsc_code").contains(""), orders['ifsc_code'].substr(1,4)).otherwise("other"))
  orders = orders.withColumn('time_hour_utc', hour(col("created_at")))
  orders = orders.withColumn('time_hour_ist', hour(col("ist_time")))

  ## COPY THE DATAFRAME ONTO FINAL_DF ####
  final_df = orders.alias('final_df') 
  
  final_df = final_df.select('order_id', 'id', 'ga_id', 'profile_uuid', 'tenant_name', 'landlord_name', 'is_owner', 'tenant_phone_number', 'tenant_id', 'created_at', 'ist_time', 'status', 'cutoff_date', 'first_transaction_date', 'tenant_created_at', 'user_flag', 'poc_category', 'amount', 'platform', 'upi', 'ifsc_code', 'tenant_age_in_seconds', 'city', 'number_of_different_locations', 'number_of_failed_transactions', 'total_number_of_transactions', 'average_number_of_transactions', 'seconds_since_last_transaction', 'average_seconds_between_two_transactions', 'time_hour_utc', 'time_hour_ist', 'upi_bank', 'account_bank', 'profile_picture_url', 'referral_code', 'number_of_leads', 'number_of_banks', 'total_transaction_amount', 'total_transactions', 'mean_amount', 'median_amount', 'transaction_amount_of_last_3_transactions', 'mean_amount_of_last_3_transactions', 'median_amount_of_last_3_transactions', 'seconds_on_payment_gateway_backend', 'fraud')
  
  return final_df

# COMMAND ----------

df = build_feature_store()
df = df.withColumn('fraud', df['fraud'].cast(IntegerType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## INITIALIZE FEATURE STORE AND POPULATE IT

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.nu_feature_store")

# COMMAND ----------



# COMMAND ----------

nu_features_query = f"""\
SELECT *
FROM data_science_metastore.nu_payout_production_tables.nu_feature_store
"""
nu_static_features = spark.sql(nu_features_query).toPandas()

print(nu_static_features['fraud'].value_counts())

# COMMAND ----------

# print(nu_static_features['is_owner'].value_counts())

# COMMAND ----------

fraud_tenants = nu_static_features[nu_static_features['fraud'] == 1]['tenant_id'].to_list()
nu_static_features['fraud'] = nu_static_features['tenant_id'].apply(lambda x: 1 if x in fraud_tenants else 0)

print(nu_static_features['fraud'].value_counts())

# COMMAND ----------

nu_static_features = spark.createDataFrame(nu_static_features)

nu_static_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("data_science_metastore.nu_payout_production_tables.nu_feature_store")

# COMMAND ----------



# COMMAND ----------


