# Databricks notebook source
import pandas as pd

START_DATE = '2023-07-03'
END_DATE = '2023-07-18'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# COMMAND ----------

v4_actuals = spark.sql("""
SELECT
    o.order_id AS order_id,
    o.tenant_id,
    o.created_at AS order_created_at,
    case when nt.tenant_id is null then 'NU' else 'RU' end as user_flag,
    krer.model_prob_value,
    pr.fraud
FROM
    apollo.orders o
    JOIN etl.pay_rent pr ON pr.case_id = o.id
    JOIN apollo.kyc_risk_engine_request krer ON o.kyc_request_id = krer.id
                                                AND krer.risk_engine = 'MLV4KycEngine2'
    LEFT JOIN (
            select tenant_id,
            min(created_at) as first_transaction_date
            from apollo.orders
            where status in ('PAYOUT_SUCCESSFUL')group by 1
        ) AS nt ON o.tenant_id = nt.tenant_id and o.created_at > nt.first_transaction_date
WHERE
    date(o.created_at) >= '2023-07-03'
    AND date(o.created_at) <= '2023-07-26'
""").toPandas()

# COMMAND ----------

v4_actuals = v4_actuals[v4_actuals['user_flag'] == 'NU']

# COMMAND ----------

v4_actuals['fraud'].value_counts()

# COMMAND ----------

v4_actuals[v4_actuals['fraud'] == 0]['model_prob_value'].describe()

# COMMAND ----------

v4_actuals[v4_actuals['fraud'] == 0]['model_prob_value'].plot(kind='box')

# COMMAND ----------

v4_actuals[v4_actuals['fraud'] == 1]['model_prob_value'].describe()

# COMMAND ----------

v4_actuals[v4_actuals['fraud'] == 1]['model_prob_value'].to_list()

# COMMAND ----------

v4_actuals[v4_actuals['fraud'] == 1].head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # CORRECT AND INCORRECT REDS ::

# COMMAND ----------

all_reds = v4_actuals[v4_actuals['model_prob_value'] >= 0.75]
correct_reds = all_reds[all_reds['fraud'] == 1]
incorrect_reds = all_reds[all_reds['fraud'] == 0]

# COMMAND ----------

correct_reds['model_prob_value'].to_list()

# COMMAND ----------

# DBTITLE 1,False Positives
incorrect_reds['model_prob_value'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # CORRECT AND INCORRECT NON-REDS ::

# COMMAND ----------

all_nonreds = v4_actuals[v4_actuals['model_prob_value'] < 0.75]
correct_nonreds = all_nonreds[all_nonreds['fraud'] == 0]
incorrect_nonreds = all_nonreds[all_nonreds['fraud'] == 1]

# COMMAND ----------

correct_nonreds['model_prob_value'].describe()

# COMMAND ----------

# DBTITLE 1,Bypassed Frauds
incorrect_nonreds['model_prob_value'].describe()

# COMMAND ----------


