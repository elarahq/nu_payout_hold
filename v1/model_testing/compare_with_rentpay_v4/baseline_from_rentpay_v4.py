# Databricks notebook source
import pandas as pd

START_DATE = '2023-07-03'
END_DATE = '2023-07-18'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM data_science_metastore.rentpay.rentpay_ml_logs WHERE ML_channel = 'RED' and date(date) BETWEEN '2023-07-01' AND '2023-07-25';

# COMMAND ----------

query = f"""\
SELECT * FROM data_science_metastore.rentpay.rentpay_ml_logs WHERE date(date) BETWEEN '{START_DATE}' AND '{END_DATE}'
"""

df = spark.sql(query).toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

y_pred_proba = df['ML_prob_fraud'].to_list()

# COMMAND ----------

count_75 = 0
count_80 = 0
count_82 = 0
count_83 = 0
count_85 = 0
count_90 = 0
count_95 = 0
count_99 = 0

for i in y_pred_proba:
    if i >= 0.75:
        count_75 += 1
        if i >= 0.80:
            count_80 += 1
            if i >=0.82:
                count_82 += 1
                if i >= 0.83:
                    count_83 += 1
            if i>=0.85:
                count_85 += 1
                if i>=0.90:
                    count_90 += 1
                    if i>=0.95:
                        count_95 += 1
                        if i>=0.99:
                            count_99 += 1

    

# COMMAND ----------

print(f" RED % above 75 :: {count_75 *100 / len(y_pred_proba)}")
print(f" RED % above 80 :: {count_80 *100 / len(y_pred_proba)}")
print(f" RED % above 82 :: {count_82 *100 / len(y_pred_proba)}")
print(f" RED % above 83 :: {count_83 *100 / len(y_pred_proba)}")
print(f" RED % above 85 :: {count_85 *100 / len(y_pred_proba)}")
print(f" RED % above 90 :: {count_90 *100 / len(y_pred_proba)}")
print(f" RED % above 95 :: {count_95 *100 / len(y_pred_proba)}")
print(f" RED % above 99 :: {count_99 *100 / len(y_pred_proba)}")


# COMMAND ----------

predictions = (y_pred_proba >= 0.75).astype(int)
df['fraud_probability'] = df['ML']

def map_prob_to_color(prob):
    if prob < 0.50:
        return 'GREEN'
    elif 0.50 <= prob < 0.75:
        return 'YELLOW'
    else:
        return 'RED'
df['ML_channel'] = data['fraud_probability'].apply(map_prob_to_color)

# COMMAND ----------

df_red = df[df['ML_channel'] == 'RED']
df_yellow = df[df['ML_channel'] == 'YELLOW']
df_green = df[df['ML_channel'] == 'GREEN']

# COMMAND ----------

print(f"Total number of transactions in this time period :: {len(df)}")
print()
print(f"Number of transactions marked RED :: {len(df_red)} ({len(df_red) * 100 / len(df)})")
print(f"Number of transactions marked YELLOW :: {len(df_yellow)} ({len(df_yellow) * 100 / len(df)})")
print(f"Number of transactions marked GREEN :: {len(df_green)} ({len(df_green) * 100 / len(df)})")

# COMMAND ----------

df_red['ML_prob_fraud'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Average RED prediction probability is 0.93

# COMMAND ----------

df_yellow['ML_prob_fraud'].describe()

# COMMAND ----------

df_green['ML_prob_fraud'].describe()

# COMMAND ----------

df_green['ML_prob_fraud'].quantile(0.90)

# COMMAND ----------

# MAGIC %md
# MAGIC 90% of GREEN prediction probability is below 0.11

# COMMAND ----------

# MAGIC %md
# MAGIC # REPORTED FRAUDS

# COMMAND ----------

# DBTITLE 1,Getting actuals from v4 and comparing them with the ML probability
v4_actuals = spark.sql("""
SELECT
    o.order_id AS order_id,
    o.tenant_id,
    o.ga_id,
    o.created_at AS order_created_at,
    krer.model_prob_value,
    pr.fraud
FROM
    apollo.orders o
    JOIN etl.pay_rent pr ON pr.case_id = o.id
    JOIN apollo.kyc_risk_engine_request krer ON o.kyc_request_id = krer.id
                                                AND krer.risk_engine = 'MLV4KycEngine2'
WHERE
    date(o.created_at) >= '2023-07-03'
    AND date(o.created_at) <= '2023-07-25'
    AND pr.fraud=1
""").toPandas()

# COMMAND ----------

v4_actuals.shape

# COMMAND ----------

v4_actuals['model_prob_value'].describe()

# COMMAND ----------

v4_actuals['model_prob_value'].quantile(0.99)

# COMMAND ----------

v4_actuals.head(20)

# COMMAND ----------

v4_actuals[v4_actuals['model_prob_value'] >= 0.83].shape

# COMMAND ----------

# MAGIC %md
# MAGIC 20 frauds reported, 13 of them are marked RED (probability above 0.75) \
# MAGIC Unresolved Frauds (8) :
# MAGIC 1. Probability : 0.503585
# MAGIC 2. Probability : 0.026612

# COMMAND ----------


SELECT
  o2.order_id, o2.tenant_id, o2.ga_id, T2.profile_uuid, o2.amount, o2.created_at, b.cutoff_date, b.first_payment_date, o2.payment_date, o2.status, pr.fraud
FROM
  apollo.orders o2
  INNER JOIN base b on o2.tenant_id = b.tenant_id and o2.created_at <= b.cutoff_date
  LEFT JOIN etl.pay_rent pr ON pr.case_id = o2.id
  LEFT JOIN apollo.tenants AS T2 ON T2.id = o2.tenant_id
WHERE o2.tenant_id = 550814
GROUP BY 1,2,3,4,5,6,7,8,9,10,11
ORDER BY o2.tenant_id,o2.created_at ASC

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH base AS (
# MAGIC                 select tenant_id,
# MAGIC                 case when min_successful_date is null then max_payment_date else min_successful_date end as cutoff_date,
# MAGIC                 first_payment_date
# MAGIC                 from (
# MAGIC                       select tenant_id,max(created_at) as max_payment_date, min(created_at) as first_payment_date,
# MAGIC                       min(case when status = 'PAYOUT_SUCCESSFUL' then created_at end ) as min_successful_date
# MAGIC                       from apollo.orders
# MAGIC                       group by tenant_id
# MAGIC                     ) as x
# MAGIC                 group by 1,2,3
# MAGIC               )
# MAGIC SELECT
# MAGIC     T1.id, T1.order_id, T1.created_at, T1.tenant_id, T2.profile_uuid, T1.tenant_contact_number_encrypted, T1.amount, T1.platform, T1.status, T1.city, T1.landlord_bank_account_id,
# MAGIC     ROUND((unix_timestamp(T1.created_at)-unix_timestamp(T2.created_at))/60,2) as tenant_age_in_minutes,
# MAGIC     T3.fraud
# MAGIC   FROM
# MAGIC     apollo.orders AS T1
# MAGIC     INNER JOIN base b on T1.tenant_id = b.tenant_id and T1.created_at <= b.cutoff_date
# MAGIC     LEFT JOIN apollo.tenants AS T2 ON T2.id = T1.tenant_id
# MAGIC     INNER JOIN etl.pay_rent AS T3 ON T3.order_id = T1.order_id
# MAGIC   WHERE
# MAGIC   T1.tenant_id = 550814

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select
# MAGIC o.order_id, o.tenant_id, o.status, o.created_at,
# MAGIC case when nt.tenant_id is null then 'NU' else 'RU' end as user_flag
# MAGIC  from apollo.orders o left join
# MAGIC (
# MAGIC   select tenant_id,
# MAGIC   min(created_at) as first_transaction
# MAGIC   from apollo.orders
# MAGIC   where status in ('PAYOUT_SUCCESSFUL')group by 1)  nt
# MAGIC ON o.tenant_id = nt.tenant_id and o.created_at > nt.first_transaction
# MAGIC where o.tenant_id = 550814

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     T1.id, T1.order_id, T1.created_at, T1.tenant_id, T2.profile_uuid,
# MAGIC     case when nt.tenant_id is null then 'NU' else 'RU' end as user_flag,
# MAGIC     T1.tenant_contact_number_encrypted, T1.amount, T1.platform, T1.status, T1.city, T1.landlord_bank_account_id,
# MAGIC     ROUND((unix_timestamp(T1.created_at)-unix_timestamp(T2.created_at))/60,2) as tenant_age_in_minutes,
# MAGIC     T3.fraud
# MAGIC   FROM
# MAGIC     apollo.orders AS T1
# MAGIC     LEFT JOIN (
# MAGIC                 select tenant_id,
# MAGIC                 min(created_at) as first_transaction
# MAGIC                 from apollo.orders
# MAGIC                 where status in ('PAYOUT_SUCCESSFUL')group by 1
# MAGIC               ) AS nt ON T1.tenant_id = nt.tenant_id and T1.created_at > nt.first_transaction
# MAGIC     LEFT JOIN apollo.tenants AS T2 ON T2.id = T1.tenant_id
# MAGIC     INNER JOIN etl.pay_rent AS T3 ON T3.order_id = T1.order_id
# MAGIC   WHERE
# MAGIC   T1.tenant_id = 550814

# COMMAND ----------


