# Databricks notebook source
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# COMMAND ----------

# MAGIC %sql
# MAGIC select T1.tenant_id, case when T1.id = first_id then 'New User' else 'Repeat User' end as user_type from apollo.orders as T1
# MAGIC left join (select tenant_id, min(id) as first_id
# MAGIC from apollo.orders
# MAGIC where upper(status) in ('PAYOUT_SUCCESSFUL')
# MAGIC group by 1) first_payment_table
# MAGIC on first_payment_table.tenant_id = T1.tenant_id

# COMMAND ----------

query = f"""\
SELECT o1.id, o1.order_id, o1.tenant_id, o1.created_at, o1.amount, o1.status
FROM apollo.orders o1
WHERE o1.created_at <= (
    SELECT MIN(o2.created_at)
    FROM apollo.orders o2
    WHERE o2.tenant_id = o1.tenant_id
      AND o2.status = 'PAYOUT_SUCCESSFUL'
)
AND date(o1.created_at) >= '2023-01-01'
ORDER BY o1.tenant_id, o1.created_at;
"""

df = spark.sql(query).toPandas()

# COMMAND ----------

query_2 = f"""\
with base as (
select tenant_id,
case when min_successful_date is null then max_payment_date else min_successful_date end as cutoff_date,
first_payment_date
from
(select tenant_id,max(created_at) as max_payment_date, min(created_at) as first_payment_date,
min(case when status = 'PAYOUT_SUCCESSFUL' then created_at end ) as min_successful_date
from apollo.orders
group by tenant_id ) x
group by 1,2,3 )
select o2.order_id, o2.tenant_id, o2.ga_id, T2.profile_uuid, o2.amount, o2.created_at, b.cutoff_date, b.first_payment_date, o2.payment_date, o2.status, prr.failure_reason, pr.fraud
from apollo.orders o2
inner join base b on o2.tenant_id = b.tenant_id and o2.created_at <= b.cutoff_date
left join etl.pay_rent pr ON pr.case_id = o2.id
LEFT JOIN apollo.tenants AS T2 ON T2.id = o2.tenant_id
LEFT JOIN fortuna.payout_retry_records prr ON prr.transaction_id = o2.transaction_id 
group by 1,2,3,4,5,6,7,8,9,10,11,12
order by o2.tenant_id,o2.created_at asc
"""

df = spark.sql(query_2).toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

df.isna().sum()

# COMMAND ----------



# COMMAND ----------

df['failure_reason'].value_counts()

# COMMAND ----------

df=df.query("created_at >= '2023-01-01'")

# COMMAND ----------

df.shape

# COMMAND ----------

df['fraud'] = df['fraud'].fillna(0)

# COMMAND ----------

df['fraud'].value_counts(normalize=True) * 100

# COMMAND ----------

# DBTITLE 1,Retroactive fraud marking
fraud_tenants = df[df['fraud'] == 1]['tenant_id'].to_list()
df['fraud'] = df['tenant_id'].apply(lambda x: 1 if x in fraud_tenants else 0)

print(df['fraud'].value_counts(normalize=True) * 100)

# COMMAND ----------

# DBTITLE 1,Calculating time spent between first transaction and last transaction (successful or not)
df['first_payment_date'] = pd.to_datetime(df['first_payment_date'])
df['cutoff_date'] = pd.to_datetime(df['cutoff_date'])

df['days_elapsed'] = (df['cutoff_date'] - df['first_payment_date']).dt.total_seconds() / 3600

print(df['days_elapsed'].describe())

# COMMAND ----------

failed_df = df[df['status'] == 'PAYMENT_FAILED']
success_df = df[df['status'] == 'PAYOUT_SUCCESSFUL']

# COMMAND ----------

print(f"fraud_df length :: {len(fraud_df)}")
print(f"payment status counts :: {fraud_df['status'].value_counts()}")

# COMMAND ----------



# COMMAND ----------

all_tenants = df['tenant_id'].unique()
tenants_with_success = df[df['status'] == 'PAYOUT_SUCCESSFUL']['tenant_id'].unique()
tenants_with_refund = df[df['status'] == 'REFUND_SUCCESSFUL']['tenant_id'].unique()

tenants_with_no_success = [x for x in all_tenants if x not in tenants_with_success]

# COMMAND ----------

print(len(all_tenants))
print(len(tenants_with_no_success))
print(len(tenants_with_success))

# COMMAND ----------

tenants_with_refund

# COMMAND ----------

tenants_with_no_success

# COMMAND ----------

df[df['tenant_id'] == 1566374].head(200)

# COMMAND ----------

# MAGIC %md
# MAGIC ## TENANT DETAILS

# COMMAND ----------

df['tenant_id'].nunique()

# COMMAND ----------

df_fraud = df[df['fraud'] == 0]

# COMMAND ----------

tenant_activity_count = df_fraud['tenant_id'].value_counts().to_dict()
for key in tenant_activity_count.keys():
    tenant_activity_count[key] -= 1

# COMMAND ----------

tenant_activity_count

# COMMAND ----------

len(tenant_activity_count)

# COMMAND ----------

count = 0
tenants_with_history = []
for key in tenant_activity_count.keys():
    if tenant_activity_count[key] >= 1:
        tenants_with_history.append(key)
        count += 1

print(count)

# COMMAND ----------

df_atleast_1_transaction = df[df['tenant_id'].isin(tenants_with_history)]
df_atleast_1_transaction = df_atleast_1_transaction.drop_duplicates(subset=['tenant_id', 'days_elapsed'])
# df_atleast_1_transaction = df_atleast_1_transaction[df_atleast_1_transaction['status'] == 'PAYOUT_SUCCESSFUL']

# COMMAND ----------

df_atleast_1_transaction['fraud'].value_counts(normalize=True)* 100

# COMMAND ----------

df_atleast_1_transaction['days_elapsed'].describe()

# COMMAND ----------

num_transacs = []
for key in tenant_activity_count.keys():
    num_transacs.append(tenant_activity_count[key])

# COMMAND ----------

df[df['tenant_id'] == 2377213].head(1000)

# COMMAND ----------


