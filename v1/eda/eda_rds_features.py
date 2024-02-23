# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, unix_timestamp, min, collect_set, lit

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

START_DATE = '2023-01-01'
END_DATE = '2023-07-17'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

%matplotlib inline
plt.rcParams["figure.figsize"] = (16,11)
plt.rcParams["font.size"] = '18'

# COMMAND ----------

nu_ga_features_query = f"""\
    SELECT
        nu_ga.*, nu_fs.order_id, nu_fs.tenant_id, nu_fs.created_at, nu_fs.fraud
    FROM 
        feature_store.KAFKA_NU_DATA AS nu_ga
        LEFT JOIN feature_store.nu_feature_store_increased_frauds AS nu_fs ON nu_fs.ga_id = nu_ga.ga_id
    WHERE
        date(date) between '2023-01-01' AND '2023-07-31'
        AND nu_ga.ga_id IN (
                    select ga_id
                    from feature_store.nu_feature_store_increased_frauds
                    where date(created_at) between '2023-01-01' AND '2023-07-31'
                    )
"""
nu_ga_features = spark.sql(nu_ga_features_query)

# nu_ga_features.createOrReplaceTempView("NU")

# df_main = nu_ga_features.toPandas()

# COMMAND ----------

# nu_ga_features.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("feature_store" + "." + "RDS_nu_ga_merged_with_frauds")

# COMMAND ----------

nu_ga_features_query = f"""\
    SELECT
        *
    FROM 
        feature_store.RDS_nu_ga_merged_with_frauds 
"""
nu_ga_features = spark.sql(nu_ga_features_query)
nu_ga_features.createOrReplaceTempView("NU")

df_main = nu_ga_features.toPandas()

df_main = df_main.drop_duplicates(subset=['ga_id', 'session_id', 'timestamp'])

# COMMAND ----------

df_main['fraud'].value_counts()

# COMMAND ----------

df_main['fraud'].value_counts(normalize=True) * 100

# COMMAND ----------

df_main['source'].value_counts(normalize=True)*100

# COMMAND ----------

# MAGIC %md
# MAGIC ### TIME BETWEEN SESSION START AND PAYMENT

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df['time_before_payment'] = (df['first_transaction_time'] - df['session_start']).dt.total_seconds() / 60

# COMMAND ----------

df[df['fraud'] == 0]['time_before_payment'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['time_before_payment'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['time_before_payment'].quantile(0.99)

# COMMAND ----------

df = df[df['time_before_payment'] <= 30]

# COMMAND ----------

sns.boxplot(x='fraud', y='time_before_payment', data=df)

# COMMAND ----------

df['log_time_before_payment'] = df['time_before_payment'].apply(lambda x: max(1, x))
df['log_time_before_payment'] = df['log_time_before_payment'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_time_before_payment', data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TIME BETWEEN PAYMENT AND SESSION END

# COMMAND ----------

df = df_main.copy(deep=True)

list_of_actions = [
    'pay_cta_clicked',
    'pay_again_clicked',
    'pay_cta_click'
]

df = df[df['action'].isin(list_of_actions)]

# COMMAND ----------

df['time_after_payment'] = (df['session_end'] - df['first_transaction_time']).dt.total_seconds() / 60

# COMMAND ----------

df[df['fraud'] == 0]['time_after_payment'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['time_after_payment'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='time_after_payment', data=df)

# COMMAND ----------

df['log_time_after_payment'] = df['time_after_payment'].apply(lambda x: max(1, x))
df['log_time_after_payment'] = df['log_time_after_payment'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_time_after_payment', data=df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### NUMBER OF HITS AFTER PAYMENT

# COMMAND ----------

df = df_main.copy(deep=True)

list_of_actions = [
    'pay_cta_clicked',
    'pay_again_clicked',
    'pay_cta_click'
]

df = df[df['action'].isin(list_of_actions)]

# COMMAND ----------

df['number_of_hits_after_payment'] = df['number_of_hits'] - df['hit_number']

# COMMAND ----------

df[df['fraud'] == 0]['number_of_hits_after_payment'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['number_of_hits_after_payment'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df['log_number_of_hits_after_payment'] = df['number_of_hits_after_payment'].apply(lambda x: max(1, x))
df['log_number_of_hits_after_payment'] = df['log_number_of_hits_after_payment'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_number_of_hits_after_payment', data=df)

# COMMAND ----------

df = df[df['number_of_hits_after_payment'] < 50]

# COMMAND ----------

sns.boxplot(x='fraud', y='number_of_hits_after_payment', data=df)

# COMMAND ----------

S_2 = df[df['number_of_hits_after_payment'] <= 100]
# nonfraud_sampled = df.sample(26884)
fig, ax = plt.subplots()
ax.hist(S_2[S_2["fraud"]==1]["number_of_hits_after_payment"], bins=10, alpha=0.5, color="red", label="Fraud", weights=np.ones(len(S_2[S_2["fraud"]==1])) / len(S_2))
ax.hist(S_2[S_2["fraud"]==0]["number_of_hits_after_payment"].sample(3935, random_state = 1), bins=10, alpha=0.5, color="green", label="Non-Fraud", weights=np.ones(len(S_2[S_2["fraud"]==0].sample(3935, random_state = 1))) / len(S_2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(0.01))
ax.set_xlabel("Number of Hits After Payment")
ax.set_ylabel("Percentage of Transaction")
fig.suptitle("Number of Hits After Payment")
ax.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### TIME SPENT ON PAYMENT GATEWAY

# COMMAND ----------

df = df_main.copy(deep=True)

list_of_actions = [
    'pay_cta_clicked',
    'pay_again_clicked',
    'pay_cta_click',
    'total_pay_clicked'
]

df = df[df['action'].isin(list_of_actions)]

# COMMAND ----------

df['time_on_gateway'] = (df['transaction_success_time'] - df['first_transaction_time']).dt.total_seconds() / 60
df = df[df['time_on_gateway'] > 0]

# COMMAND ----------

df[df['fraud'] == 0]['time_on_gateway'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['time_on_gateway'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='time_on_gateway', data=df)

# COMMAND ----------


