# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, unix_timestamp, min, collect_set
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

query = """\
    select * from feature_store.nu_feature_store_increased_frauds where date(created_at) between '2023-01-01' AND '2023-07-31'
"""
df_main = spark.sql(query)

df_main.createOrReplaceTempView("NU")

df_main = df_main.toPandas()

# COMMAND ----------

df_main.columns

# COMMAND ----------

print(df_main['fraud'].value_counts())

# COMMAND ----------

df_main.head()

# COMMAND ----------

df_main.shape

# COMMAND ----------

df_main[df_main['fraud'] == 1]['tenant_id'].nunique()

# COMMAND ----------

def paid_percentage_per_discrete_value(df, feature, class_num):
  json = {}
  val_list = sorted(df[feature].unique())
  for val in val_list:
    json[val] = (df[(df[feature] == val) & (df['out'] == class_num)].shape[0] / df[df[feature] == val].shape[0]) * 100
  
  return json

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from NU where number_of_failed_transactions >= 10 and fraud = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Platform

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

nonfraud_products = df[df['fraud'] == 0]['platform'].value_counts().to_dict()
fraud_products = df[df['fraud'] == 1]['platform'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_products.keys():
    if i in fraud_products.keys():
        percentage_of_frauds_dict[i] = (fraud_products[i] * 100) / nonfraud_products[i]

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(y=values, x=categories)
plt.xlabel('Platform')
plt.ylabel('Percentage of Fraud Transactions')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Amount

# COMMAND ----------

df = df_main.copy(deep=True)
# df[['amount']] = df[['amount']].apply(pd.to_numeric)

# COMMAND ----------

df[(df['out'] == '0')]['amount'].lt(100000).mean()

# COMMAND ----------

df[(df['out'] == '1')]['amount'].lt(100000).mean()

# COMMAND ----------

df = df.drop_duplicates(subset=['tenant_id'], keep='last')
df['median_amount'] = df.groupby('tenant_id')['amount'].transform('median')
df['average_amount'] = df.groupby('tenant_id')['amount'].transform('mean')

# COMMAND ----------

df[(df['out'] == '1')]['median_amount'].describe()

# COMMAND ----------

df[(df['out'] == '0')]['median_amount'].describe()

# COMMAND ----------

df = df[df['amount'] <= 100000]
plt.ticklabel_format(style='plain', axis='y')
sns.boxplot(x='out', y='amount', data=df)

# COMMAND ----------

sns.swarmplot(x='out', y='median_amount', data=df)

# COMMAND ----------

sns.violinplot(x='out', y='average_amount', data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Age on Housing Platform

# COMMAND ----------

df = df_main.copy(deep=True)
df[['tenant_age']] = df[['tenant_age']].apply(pd.to_numeric)
df['tenant_age_days'] = df['tenant_age'] / (60*24)
df['tenant_age_hours'] = df['tenant_age'] / (60)

# COMMAND ----------

fraud = df[df['out'] == '1']
nonfraud = df[df['out'] == '0']

# COMMAND ----------

fraud['tenant_age_days'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

nonfraud['tenant_age_days'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

nonfraud['tenant_age'].quantile(0.9)

# COMMAND ----------

df['log_tenant_age'] = df['tenant_age'].apply(lambda x: max(1, x))
df['log_tenant_age'] = df['log_tenant_age'].apply(lambda x: np.log(x))

# COMMAND ----------

sns.boxplot(x='out', y='log_tenant_age', data=df)

# COMMAND ----------

# sns.histplot(df[df['tenant_age_days'] < 60]['tenant_age_days'], discrete=True, stat='percent')
# plt.ylabel('Percentage')
# plt.xlabel('Tenant Age in Days')
# plt.show()


S_2 = df[(df['tenant_age_days'] <= 60)]
# nonfraud_sampled = df.sample(26884)
fig, ax = plt.subplots()
ax.hist(S_2[S_2["out"]=='1']["tenant_age_days"], bins=10, alpha=0.5, color="red", label="Fraud", weights=np.ones(len(S_2[S_2["out"]=='1'])) / len(S_2))
ax.hist(S_2[S_2["out"]=='0']["tenant_age_days"].sample(1023, random_state = 1), bins=10, alpha=0.5, color="green", label="Non-Fraud", weights=np.ones(len(S_2[S_2["out"]=='0'].sample(1023, random_state = 1))) / len(S_2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(0.01))
ax.set_xlabel("Tenant Age in Days")
ax.set_ylabel("Percentage of Tenants")
fig.suptitle("Tenant Age in Days")
ax.legend()
plt.show()


# json = paid_percentage_per_discrete_value(df[df['tenant_age_days'] <50], 'tenant_age', '1')
# plt.scatter(*zip(*json.items()))
# plt.xlabel('Tenant Age in Days')
# plt.ylabel('Percentage of tenants that are fraud')
# plt.show()

# json = paid_percentage_per_discrete_value(df[df['tenant_age_days'] <50], 'tenant_age', '0')
# plt.scatter(*zip(*json.items()))
# plt.xlabel('Tenant Age in Days')
# plt.ylabel('Percentage of tenants that are NOT fraud')
# plt.show()

# COMMAND ----------

df[['out']] = df[['out']].apply(pd.to_numeric)

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]

labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']

# Bin the 'tenant_age' column
df['tenant_age_days_bins'] = pd.cut(df['tenant_age_days'], bins=bins, labels=labels, right=False)

# Calculate the percentage of 'out' == 1 for each bin
percentage_out_1 = df.groupby('tenant_age_days_bins')['out'].mean() * 100

# Plot the percentages in a bar plot
plt.bar(percentage_out_1.index, percentage_out_1.values)
plt.xlabel('Tenant Age (Days)')
plt.ylabel('% of Fraud Transactions')
plt.show()

# COMMAND ----------

df[['out']] = df[['out']].apply(pd.to_numeric)


bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]

labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']

# Bin the 'tenant_age' column
df['tenant_age_hours_bins'] = pd.cut(df['tenant_age_hours'], bins=bins, labels=labels, right=False)

# Calculate the percentage of 'out' == 1 for each bin
percentage_out_1 = df.groupby('tenant_age_hours_bins')['out'].mean() * 100

# Plot the percentages in a bar plot
plt.bar(percentage_out_1.index, percentage_out_1.values)
plt.xlabel('Tenant Age (Hours)')
plt.ylabel('% of Fraud Transactions')
plt.show()

# COMMAND ----------

df[['out']] = df[['out']].apply(pd.to_numeric)


bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, float('inf')]

labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300+']

# Bin the 'tenant_age' column
df['tenant_age_bins'] = pd.cut(df['tenant_age'], bins=bins, labels=labels, right=False)

# Calculate the percentage of 'out' == 1 for each bin
percentage_out_1 = df.groupby('tenant_age_bins')['out'].mean() * 100

# Plot the percentages in a bar plot
plt.bar(percentage_out_1.index, percentage_out_1.values)
plt.xlabel('Tenant Age (Minutes)')
plt.ylabel('% of Fraud Transactions')
plt.show()

# COMMAND ----------

df[(df['out'] == 1) & (df['payment_status'] == 'PAYOUT_SUCCESSFUL')]['tenant_age'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[(df['out'] == 0) & (df['payment_status'] == 'PAYOUT_SUCCESSFUL')]['tenant_age'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df['out'].value_counts(normalize=True) * 100

# COMMAND ----------

df[df['tenant_age'] <= 1]['out'].value_counts(normalize=True) * 100

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time spent on Housing before first transaction

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df['age_till_first_transaction'] = (df['first_transaction_date'] - df['tenant_created_at']).dt.total_seconds()

# COMMAND ----------

df = df[df['age_till_first_transaction'] >= 0]

df['age_till_first_transaction'] = df['age_till_first_transaction'] / 60

# COMMAND ----------

df[(df['fraud'] == 0)]['age_till_first_transaction'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[(df['fraud'] == 1)]['age_till_first_transaction'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]

labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']

# Bin the 'tenant_age' column
df['age_first_bins'] = pd.cut(df['age_till_first_transaction'], bins=bins, labels=labels, right=False)

# Calculate the percentage of 'out' == 1 for each bin
percentage_out_1 = df.groupby('age_first_bins')['fraud'].mean() * 100

# Plot the percentages in a bar plot
plt.bar(percentage_out_1.index, percentage_out_1.values)
plt.xlabel('Tenant Age Before their first Transaction')
plt.ylabel('% of Fraud Transactions')
plt.show()

# COMMAND ----------

sns.histplot(df[df['age_till_first_transaction'] < 1000]['age_till_first_transaction'], discrete=True, stat='probability')
plt.ylabel('Percentage')
plt.xlabel('Age till first transaction')
plt.show()


S_2 = df[df['age_till_first_transaction'] <= 1000]
# nonfraud_sampled = df.sample(26884)
fig, ax = plt.subplots()
ax.hist(S_2[S_2["fraud"]==1]["age_till_first_transaction"], bins=10, alpha=0.5, color="red", label="Fraud", weights=np.ones(len(S_2[S_2["fraud"]==1])) / len(S_2))
ax.hist(S_2[S_2["fraud"]==0]["age_till_first_transaction"].sample(1023, random_state = 1), bins=10, alpha=0.5, color="green", label="Non-Fraud", weights=np.ones(len(S_2[S_2["fraud"]==0].sample(1023, random_state = 1))) / len(S_2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(0.01))
ax.set_xlabel("Age till first transaction (in seconds)")
ax.set_ylabel("Percentage of Tenants")
fig.suptitle("Age till first transaction")
ax.legend()
plt.show()


# json = paid_percentage_per_discrete_value(df[df['age_till_first_transaction'] <1000], 'age_till_first_transaction', 1)
# plt.scatter(*zip(*json.items()))
# plt.xlabel('Age till first transaction')
# plt.ylabel('Percentage of tenants that are fraud')
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Days since first transaction

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df['earliest_created_at'] = df.groupby('tenant_id')['created_at'].transform('min')

# COMMAND ----------

df.head()

# COMMAND ----------

df['days_elapsed'] = (df['created_at'] - df['earliest_created_at']).dt.days

# COMMAND ----------

df[df['fraud'] == 1]['days_elapsed'].describe()

# COMMAND ----------

df[df['fraud'] == 0]['days_elapsed'].describe()

# COMMAND ----------

# sns.histplot(df[df['days_elapsed'] < 100]['days_elapsed'], discrete=True, stat='probability')
# plt.ylabel('Percentage')
# plt.xlabel('Days since first transaction')
# plt.show()


S_2 = df[df['days_elapsed'] <= 100]
# nonfraud_sampled = df.sample(26884)
fig, ax = plt.subplots()
ax.hist(S_2[S_2["fraud"]==1]["days_elapsed"], bins=10, alpha=0.5, color="red", label="Fraud", weights=np.ones(len(S_2[S_2["fraud"]==1])) / len(S_2))
ax.hist(S_2[S_2["fraud"]==0]["days_elapsed"].sample(1023, random_state = 1), bins=10, alpha=0.5, color="green", label="Non-Fraud", weights=np.ones(len(S_2[S_2["fraud"]==0].sample(1023, random_state = 1))) / len(S_2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(0.01))
ax.set_xlabel("Days since first transaction")
ax.set_ylabel("Percentage of Tenants")
fig.suptitle("Days since first transaction")
ax.legend()
plt.show()


# json = paid_percentage_per_discrete_value(df[df['days_elapsed'] <100], 'days_elapsed', 1)
# plt.scatter(*zip(*json.items()))
# plt.xlabel('Days since first transaction')
# plt.ylabel('Percentage of tenants that are fraud')
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time since last transaction

# COMMAND ----------

df = df.sort_values(['tenant_id', 'created_at'])
df['hours_since_last_transaction'] = df.groupby('tenant_id')['created_at'].diff()

# Step 3: Convert the time difference to days
df['hours_since_last_transaction'] = df['hours_since_last_transaction'].dt.total_seconds() / 3600

# COMMAND ----------

df[df['out'] == 1]['hours_since_last_transaction'].describe()

# COMMAND ----------

df[df['out'] == 0]['hours_since_last_transaction'].describe()

# COMMAND ----------

df[df['tenant_id'] == '1010540'].head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of GA IDs, phone numbers of a tenant_id and ga_id

# COMMAND ----------

df['tenant_id'].nunique()

# COMMAND ----------

df['ga_id'].nunique()

# COMMAND ----------

df['tenant_phone_number'].nunique()

# COMMAND ----------

df['number_of_ga_ids'] = df.groupby('tenant_id')['ga_id'].transform('nunique')

# COMMAND ----------

df[df['fraud'] == 0]['number_of_ga_ids'].describe()

# COMMAND ----------

df[df['fraud'] == 1]['number_of_ga_ids'].describe()

# COMMAND ----------

df['number_of_phone_numbers_for_this_ga_id'] = df.groupby('ga_id')['tenant_phone_number'].transform('nunique')

# COMMAND ----------

df[df['fraud'] == 0]['number_of_phone_numbers_for_this_ga_id'].describe()

# COMMAND ----------

df[df['fraud'] == 0]['number_of_phone_numbers_for_this_ga_id'].quantile(0.95)

# COMMAND ----------

df[df['fraud'] == 1]['number_of_phone_numbers_for_this_ga_id'].describe()

# COMMAND ----------

df[df['fraud'] == 1]['number_of_phone_numbers_for_this_ga_id'].quantile(0.95)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Is owner

# COMMAND ----------

profiles_query = f"""\
    with base as (
        SELECT nu.order_id, p.is_owner, count(distinct uf.flat_id) as unique_listing_count
        FROM NU AS nu
        INNER JOIN housing_clients_production.profiles p ON p.profile_uuid = nu.profile_uuid
        LEFT JOIN housing_production.user_flats uf ON uf.profile_uuid = p.profile_uuid
        group by 1,2
    )

    SELECT NU.*, base.is_owner, base.unique_listing_count
    FROM NU
    INNER JOIN base ON base.order_id = NU.order_id
"""

df = spark.sql(profiles_query).toPandas()

# COMMAND ----------



# COMMAND ----------

df[(df['out'] == 1) & (df['is_owner'] == 'true')]['unique_listing_count'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of leads dropped

# COMMAND ----------

leads_query = f"""\
    WITH base AS (
        select p.profile_uuid, count(distinct lm.id) as no_of_leads_dropped
        from housing_leads_production.lead_masters lm
        inner join housing_leads_production.lead_demand_users ldu on ldu.id = lm.lead_demand_user_id
        left join housing_clients_production.profiles_phone_numbers ppn on ppn.number = ldu.phone
        left join housing_clients_production.profiles p on p.id = ppn.profile_id
        group by p.profile_uuid    
    )

    SELECT NU.*, base.no_of_leads_dropped
    FROM NU
    LEFT JOIN base ON base.profile_uuid = NU.profile_uuid
"""

df = spark.sql(leads_query).toPandas()

# COMMAND ----------

df.shape

# COMMAND ----------

df['no_of_leads_dropped'] = df['no_of_leads_dropped'].fillna(0)
df = df[df['no_of_leads_dropped'] <= 100]

# COMMAND ----------

df[(df['fraud'] == 0)]['no_of_leads_dropped'].describe()

# COMMAND ----------

df[(df['fraud'] == 1)]['no_of_leads_dropped'].describe()

# COMMAND ----------

sns.boxplot(data= df, y='no_of_leads_dropped', x='fraud')

# COMMAND ----------

df['log_no_of_leads_dropped'] = df['no_of_leads_dropped'].apply(lambda x: max(1, x))
df['log_no_of_leads_dropped'] = df['log_no_of_leads_dropped'].apply(lambda x: np.log(x))

# COMMAND ----------

df[(df['fraud'] == 1) & (df['tenant_age_in_seconds'] <=3)]['no_of_leads_dropped'].quantile(0.99)

# COMMAND ----------

df[df['no_of_leads_dropped'] == 1753].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### is profile marked fraud in demand fraud

# COMMAND ----------

demand_fraud_query = """
SELECT NU.*, sf.flag as marked_demand_fraud
FROM NU
LEFT JOIN demand_fraud_db_v2.demand_users_4w_fraud_predictions_profileuuid_phone_complied AS sf ON sf.phone = NU.tenant_phone_number
"""

df = spark.sql(demand_fraud_query).toPandas()

# COMMAND ----------



# COMMAND ----------

df[df['fraud'] == 0]['marked_demand_fraud'].value_counts()

# COMMAND ----------

df[df['fraud'] == 1]['marked_demand_fraud'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### social media presence

# COMMAND ----------

social_media_query = f"""\
    SELECT
        NU.*,
        CASE WHEN u.profile_picture_url IS NOT NULL THEN 1 ELSE 0 END AS profile_picture_url,
        CASE WHEN u.fb_id IS NOT NULL THEN 1 ELSE 0 END AS fb_id,
        CASE WHEN u.gplus_id IS NOT NULL THEN 1 ELSE 0 END AS gplus_id,
        u.seen_properties
    FROM
        NU
        JOIN housing_clients_production.profiles p ON p.profile_uuid = NU.profile_uuid
        JOIN housing_clients_production.users u ON u.id = p.user_id
"""

df = spark.sql(social_media_query).toPandas()

# COMMAND ----------

df[df['fraud'] == 0]['profile_picture_url'].value_counts(normalize=True) * 100

# COMMAND ----------

df[df['fraud'] == 1]['profile_picture_url'].value_counts(normalize=True) * 100

# COMMAND ----------

sns.countplot(data=df, x='profile_picture_url', hue='fraud')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hour of the day

# COMMAND ----------

df[['time_hour']] = df[['time_hour']].apply(pd.to_numeric)

# COMMAND ----------

nonfraud = df[df['fraud'] == 0]
fraud = df[df['fraud'] == 1]

nonfraud = nonfraud.sample(1023 * 5)

df_resampled = pd.concat([nonfraud, fraud])

# COMMAND ----------

sns.countplot(data=df_resampled, x="time_hour_ist", hue="fraud", order= sorted(df_resampled['time_hour_ist'].unique()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Payment Location

# COMMAND ----------

df = df_main.copy(deep=True)
# df = df[df['city'] != 'Secaucus']

# COMMAND ----------

nonfraud_cities = df[df['fraud'] == 0]['city'].value_counts().to_dict()
fraud_cities = df[df['fraud'] == 1]['city'].value_counts().to_dict()

# COMMAND ----------

percentage_of_frauds_dict = {}

for i in nonfraud_cities.keys():
    if i in fraud_cities.keys():
        percentage_of_frauds_dict[i] = (fraud_cities[i] * 100) / (fraud_cities[i] + nonfraud_cities[i])

# COMMAND ----------

dict(sorted(percentage_of_frauds_dict.items(), key=lambda item: item[1], reverse=True))

# COMMAND ----------

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(x=values, y=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('City')
plt.show()

# COMMAND ----------

df[df['out'] == 1]['city'].value_counts(normalize = True).iloc[:11] * 100

# COMMAND ----------

df[df['out'] == 0]['city'].value_counts(normalize = True).iloc[:11] * 100

# COMMAND ----------

sns.countplot(data = df_resampled, x = 'city', hue='out', order=df_resampled['city'].value_counts().iloc[:10].index)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Referral Code

# COMMAND ----------

ref_query = f"""\
SELECT NU.*, CASE WHEN o.referral_code IS NOT NULL THEN 'yes' ELSE 'no' END AS referral_code
FROM NU
INNER JOIN apollo.orders o ON o.order_id = NU.order_id    
"""

df = spark.sql(ref_query).toPandas()

# COMMAND ----------

df[df['fraud'] == 0]['referral_code'].value_counts(normalize = True) * 100

# COMMAND ----------

df[df['fraud'] == 1]['referral_code'].value_counts(normalize = True) * 100

# COMMAND ----------

sns.countplot(data=df, x= 'referral_code', hue='fraud')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Number of properties seen

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC    WITH base AS (
# MAGIC         select
# MAGIC         id,
# MAGIC         uuid,
# MAGIC         seen_properties,
# MAGIC         (coalesce(seen_buy ,0) + coalesce(seen_rent ,0) + coalesce(seen_new_projects ,0) + coalesce(seen_pg ,0)) as num_properties_seen
# MAGIC         from (
# MAGIC         select id, uuid, seen_properties,
# MAGIC         json_array_length(get_json_object(seen_properties, '$.buy')) as seen_buy,
# MAGIC         json_array_length(get_json_object(seen_properties, '$.rent')) as seen_rent,
# MAGIC         json_array_length(get_json_object(seen_properties, '$.new-projects')) as seen_new_projects,
# MAGIC         json_array_length(get_json_object(seen_properties, '$.pg')) as seen_pg
# MAGIC         from housing_clients_production.users
# MAGIC         )
# MAGIC     )
# MAGIC
# MAGIC     SELECT NU.*, base.seen_properties, base.num_properties_seen
# MAGIC     FROM NU
# MAGIC     INNER JOIN housing_clients_production.profiles p ON p.profile_uuid = NU.profile_uuid
# MAGIC     INNER JOIN base ON base.id = p.user_id
# MAGIC     WHERE base.num_properties_seen >= 1

# COMMAND ----------

seen_properties_query = f"""\
    WITH base AS (
        select
        id,
        uuid,
        seen_properties,
        (coalesce(seen_buy ,0) + coalesce(seen_rent ,0) + coalesce(seen_new_projects ,0) + coalesce(seen_pg ,0)) as num_properties_seen
        from (
        select id, uuid, seen_properties,
        json_array_length(get_json_object(seen_properties, '$.buy')) as seen_buy,
        json_array_length(get_json_object(seen_properties, '$.rent')) as seen_rent,
        json_array_length(get_json_object(seen_properties, '$.new-projects')) as seen_new_projects,
        json_array_length(get_json_object(seen_properties, '$.pg')) as seen_pg
        from housing_clients_production.users
        )
    )

    SELECT NU.*, base.seen_properties, base.num_properties_seen
    FROM NU
    INNER JOIN housing_clients_production.profiles p ON p.profile_uuid = NU.profile_uuid
    INNER JOIN base ON base.id = p.user_id
"""

df = spark.sql(seen_properties_query).toPandas()

# COMMAND ----------

df['num_properties_seen'].describe()

# COMMAND ----------

df[df['num_properties_seen'] >= 1]['out'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### PayOnCredit Product Used

# COMMAND ----------

profiles_query = f"""\
select NU.*, mrt.rent_type AS poc_product
from NU
    inner join apollo.orders o ON o.id = NU.id
    inner join apollo.master_rent_type mrt on mrt.id = o.rent_type_id
"""

df = spark.sql(profiles_query).toPandas()

# COMMAND ----------

nonfraud_products = df[df['fraud'] == 0]['poc_product'].value_counts().to_dict()
fraud_products = df[df['fraud'] == 1]['poc_product'].value_counts().to_dict()

# COMMAND ----------

percentage_of_frauds_dict = {}

for i in nonfraud_products.keys():
    if i in fraud_products.keys():
        percentage_of_frauds_dict[i] = (fraud_products[i] * 100) / nonfraud_products[i]

# COMMAND ----------

dict(sorted(percentage_of_frauds_dict.items(), key=lambda item: item[1], reverse=True))

# COMMAND ----------

sns.countplot(data=df, y='poc_product', hue='fraud')

# COMMAND ----------

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(x=values, y=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('PayOnCredit Product Used')
plt.show()

# COMMAND ----------

df['time_month'] = df['created_at'].dt.month

# COMMAND ----------

df[df['poc_product'] == 'EDUCATION_FEES']['time_month'].value_counts(normalize=True)

# COMMAND ----------

df[(df['fraud'] == 1) & (df['poc_product'] == 'EDUCATION_FEES')]['time_month'].value_counts(normalize=True)

# COMMAND ----------

df[(df['fraud'] == 0) & (df['poc_product'] == 'EDUCATION_FEES')]['time_month'].value_counts(normalize=True)

# COMMAND ----------

df_edu = df[df['poc_product'] == 'EDUCATION_FEES']

nonfraud_products = df_edu[df_edu['fraud'] == 0]['time_month'].value_counts().to_dict()
fraud_products = df_edu[df_edu['fraud'] == 1]['time_month'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_products.keys():
    if i in fraud_products.keys():
        percentage_of_frauds_dict[i] = (fraud_products[i] * 100) / nonfraud_products[i]

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(y=values, x=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Month of Education Fees Transactions')
plt.show()

# COMMAND ----------

df_edu = df[df['poc_product'] == 'HOUSE_RENT']

nonfraud_products = df_edu[df_edu['fraud'] == 0]['time_month'].value_counts().to_dict()
fraud_products = df_edu[df_edu['fraud'] == 1]['time_month'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_products.keys():
    if i in fraud_products.keys():
        percentage_of_frauds_dict[i] = (fraud_products[i] * 100) / nonfraud_products[i]

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(y=values, x=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Month of House Rent Transactions')
plt.show()

# COMMAND ----------

df_edu = df[df['poc_product'] == 'OFFICE_RENT']

nonfraud_products = df_edu[df_edu['fraud'] == 0]['time_month'].value_counts().to_dict()
fraud_products = df_edu[df_edu['fraud'] == 1]['time_month'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_products.keys():
    if i in fraud_products.keys():
        percentage_of_frauds_dict[i] = (fraud_products[i] * 100) / nonfraud_products[i]

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(y=values, x=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Month of OfficeRent Transactions')
plt.show()

# COMMAND ----------

df_edu = df[df['poc_product'] == 'SECURITY_DEPOSIT']

nonfraud_products = df_edu[df_edu['fraud'] == 0]['time_month'].value_counts().to_dict()
fraud_products = df_edu[df_edu['fraud'] == 1]['time_month'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_products.keys():
    if i in fraud_products.keys():
        percentage_of_frauds_dict[i] = (fraud_products[i] * 100) / nonfraud_products[i]

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(y=values, x=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Month of Security Deposit Transactions')
plt.show()

# COMMAND ----------

df_edu = df[df['poc_product'] == 'MAINTENANCE']

nonfraud_products = df_edu[df_edu['fraud'] == 0]['time_month'].value_counts().to_dict()
fraud_products = df_edu[df_edu['fraud'] == 1]['time_month'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_products.keys():
    if i in fraud_products.keys():
        percentage_of_frauds_dict[i] = (fraud_products[i] * 100) / nonfraud_products[i]

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(y=values, x=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Month of Maintenace Transactions')
plt.show()

# COMMAND ----------

df_edu = df[df['poc_product'] == 'BROKERAGE']

nonfraud_products = df_edu[df_edu['fraud'] == 0]['time_month'].value_counts().to_dict()
fraud_products = df_edu[df_edu['fraud'] == 1]['time_month'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_products.keys():
    if i in fraud_products.keys():
        percentage_of_frauds_dict[i] = (fraud_products[i] * 100) / nonfraud_products[i]

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(y=values, x=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Month of Brokerage Transactions')
plt.show()

# COMMAND ----------

df_edu = df[df['poc_product'] == 'BOOKING_TOKEN']

nonfraud_products = df_edu[df_edu['fraud'] == 0]['time_month'].value_counts().to_dict()
fraud_products = df_edu[df_edu['fraud'] == 1]['time_month'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_products.keys():
    if i in fraud_products.keys():
        percentage_of_frauds_dict[i] = (fraud_products[i] * 100) / nonfraud_products[i]

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(y=values, x=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Month of Booking Token Transactions')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of Listings uploaded

# COMMAND ----------

profiles_query = f"""\
    with base as (
        SELECT nu.order_id, p.is_owner, count(distinct uf.flat_id) as unique_listing_count
        FROM NU AS nu
        INNER JOIN housing_clients_production.profiles p ON p.profile_uuid = nu.profile_uuid
        LEFT JOIN housing_production.user_flats uf ON uf.profile_uuid = p.profile_uuid
        group by 1,2
    )

    SELECT NU.*, base.is_owner, base.unique_listing_count
    FROM NU
    INNER JOIN base ON base.order_id = NU.order_id
"""

df = spark.sql(profiles_query).toPandas()

# COMMAND ----------

df[(df['fraud'] == 1) & (df['is_owner'] == 'true')]['unique_listing_count'].value_counts()

# COMMAND ----------

df[(df['fraud'] == 0) & (df['is_owner'] == 'true')]['unique_listing_count'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of Failed Transactions

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df[df['fraud'] == 1]['number_of_failed_transactions'].describe()

# COMMAND ----------

df[df['fraud'] == 0]['number_of_failed_transactions'].describe()

# COMMAND ----------

S_2 = df[(df['number_of_failed_transactions'] <= 200)]
# nonfraud_sampled = df.sample(26884)
fig, ax = plt.subplots()
ax.hist(S_2[S_2["fraud"]==1]["number_of_failed_transactions"], bins=10, alpha=0.5, color="red", label="Fraud", weights=np.ones(len(S_2[S_2["fraud"]==1])) / len(S_2))
ax.hist(S_2[S_2["fraud"]==0]["number_of_failed_transactions"].sample(1023, random_state = 1), bins=10, alpha=0.5, color="green", label="Non-Fraud", weights=np.ones(len(S_2[S_2["fraud"]==0].sample(1023, random_state = 1))) / len(S_2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(0.01))
ax.set_xlabel("Number of Failed Transactions")
ax.set_ylabel("Percentage of Users")
fig.suptitle("Number of Failed Transactions")
ax.legend()
plt.show()

# COMMAND ----------

sns.boxplot(x='fraud', y='number_of_failed_transactions', data=df)

# COMMAND ----------

df['log_number_of_failed_transactions'] = df['number_of_failed_transactions'].apply(lambda x: max(1, x))
df['log_number_of_failed_transactions'] = df['log_number_of_failed_transactions'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_number_of_failed_transactions', data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MEDIAN NUMBER OF FAILED TRANSACTIONS

# COMMAND ----------

query = """\
    WITH base AS(
        SELECT
            T1.order_id, T1.tenant_id, T1.created_at, T1.number_of_failed_transactions,
            median(T2.number_of_failed_transactions) as median_number_of_failed_transactions
        FROM (
            SELECT
                order_id, tenant_id, created_at, number_of_failed_transactions
            FROM
                NU
            ) T1
            LEFT JOIN (
                SELECT
                    order_id, tenant_id, created_at, number_of_failed_transactions
                FROM
                    NU
            ) T2 ON T2.tenant_id = T1.tenant_id AND T2.created_at < T1.created_at
        GROUP BY T1.order_id, T1.tenant_id, T1.created_at, T1.number_of_failed_transactions
    )
    
    SELECT NU.*, base.median_number_of_failed_transactions
    FROM NU
    INNER JOIN base ON base.order_id = NU.order_id
"""

df = spark.sql(query).toPandas()

# COMMAND ----------

df[df['fraud'] == 1]['median_number_of_failed_transactions'].describe()

# COMMAND ----------

df[df['fraud'] == 0]['median_number_of_failed_transactions'].describe()

# COMMAND ----------

sns.boxplot(x='fraud', y='median_number_of_failed_transactions', data=df)

# COMMAND ----------

df['log_median_number_of_failed_transactions'] = df['median_number_of_failed_transactions'].apply(lambda x: max(1, x))
df['log_median_number_of_failed_transactions'] = df['log_median_number_of_failed_transactions'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_median_number_of_failed_transactions', data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TIME DIFFERENCE BETWEEN TWO CONSECUTIVE TRANSACTIONS

# COMMAND ----------

df = df_main.copy(deep=True)

df['minutes_since_last_transaction'] = df['seconds_since_last_transaction']/60
df['hours_since_last_transaction'] = df['seconds_since_last_transaction']/(60*60)
df['days_since_last_transaction'] = df['seconds_since_last_transaction']/(60*60*24)

df['log_seconds_since_last_transaction'] = df['seconds_since_last_transaction'].apply(lambda x: max(1, x))
df['log_seconds_since_last_transaction'] = df['log_seconds_since_last_transaction'].apply(lambda x: np.log(x))


df['log_minutes_since_last_transaction'] = df['minutes_since_last_transaction'].apply(lambda x: max(1, x))
df['log_minutes_since_last_transaction'] = df['log_minutes_since_last_transaction'].apply(lambda x: np.log(x))

df['log_hours_since_last_transaction'] = df['hours_since_last_transaction'].apply(lambda x: max(1, x))
df['log_hours_since_last_transaction'] = df['log_hours_since_last_transaction'].apply(lambda x: np.log(x))

df['log_days_since_last_transaction'] = df['days_since_last_transaction'].apply(lambda x: max(1, x))
df['log_days_since_last_transaction'] = df['log_days_since_last_transaction'].apply(lambda x: np.log(x))

# COMMAND ----------

df[df['fraud'] == 0]['hours_since_last_transaction'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['hours_since_last_transaction'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='log_hours_since_last_transaction', data=df)

# COMMAND ----------

sns.boxplot(x='fraud', y='log_minutes_since_last_transaction', data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### AVERAGE TIME DIFFERENCE BETWEEN TWO TRANSACTIONS

# COMMAND ----------

df = df_main.copy(deep=True)

df['average_minutes_between_two_transactions'] = df['average_seconds_between_two_transactions']/60
df['average_hours_between_two_transactions'] = df['average_seconds_between_two_transactions']/(60*60)
df['average_days_between_two_transactions'] = df['average_seconds_between_two_transactions']/(60*60*24)

df['log_average_seconds_between_two_transactions'] = df['average_seconds_between_two_transactions'].apply(lambda x: max(1, x))
df['log_average_seconds_between_two_transactions'] = df['log_average_seconds_between_two_transactions'].apply(lambda x: np.log(x))


df['log_average_minutes_between_two_transactions'] = df['average_minutes_between_two_transactions'].apply(lambda x: max(1, x))
df['log_average_minutes_between_two_transactions'] = df['log_average_minutes_between_two_transactions'].apply(lambda x: np.log(x))

df['log_average_hours_between_two_transactions'] = df['average_hours_between_two_transactions'].apply(lambda x: max(1, x))
df['log_average_hours_between_two_transactions'] = df['log_average_hours_between_two_transactions'].apply(lambda x: np.log(x))

df['log_average_days_between_two_transactions'] = df['average_days_between_two_transactions'].apply(lambda x: max(1, x))
df['log_average_days_between_two_transactions'] = df['log_average_days_between_two_transactions'].apply(lambda x: np.log(x))

# COMMAND ----------

df[df['fraud'] == 0]['average_hours_between_two_transactions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['average_hours_between_two_transactions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='log_average_minutes_between_two_transactions', data=df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of different locations

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df[df['fraud'] ==0]['number_of_different_locations'].describe()

# COMMAND ----------

df[df['fraud'] ==1]['number_of_different_locations'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### UPI Bank

# COMMAND ----------

df = df_main.copy(deep=True)
df['upi_bank'] = df['upi_bank'].str.lower()

# COMMAND ----------

nonfraud_cities = df[df['fraud'] == 0]['upi_bank'].value_counts().to_dict()
fraud_cities = df[df['fraud'] == 1]['upi_bank'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_cities.keys():
    if i in fraud_cities.keys():
        percentage_of_frauds_dict[i] = (fraud_cities[i] * 100) / (fraud_cities[i] + nonfraud_cities[i])

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(x=values, y=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('UPI BANK')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### IFSC BANK

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

nonfraud_cities = df[df['fraud'] == 0]['account_bank'].value_counts().to_dict()
fraud_cities = df[df['fraud'] == 1]['account_bank'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_cities.keys():
    if i in fraud_cities.keys():
        percentage_of_frauds_dict[i] = (fraud_cities[i] * 100) / (fraud_cities[i] + nonfraud_cities[i])

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(x=values, y=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('IFSC BANK')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### IS SELF-PAYOUT

# COMMAND ----------

query = """\
    select NU.*, o.tenant_name, o.landlord_name
    FROM NU
    INNER JOIN apollo.orders o ON NU.order_id = o.order_id
"""

df = spark.sql(query).toPandas()

# COMMAND ----------

from difflib import SequenceMatcher

def similar(x):
    a, b = x
    thresh = 0.7
    lower = lambda k: k.lower()
    a_words = map(lower, a.split())
    b_words = map(lower, b.split())
    for w1 in a_words:
        for w2 in b_words:
            if SequenceMatcher(None, w1, w2).ratio() >= thresh:
                return True
            if w1 in w2 or w2 in w1:
                return True
            
    return False

# COMMAND ----------

df['self_payout'] = df[['tenant_name', 'landlord_name']].apply(similar, axis=1)

# COMMAND ----------

df[df['fraud'] == 0]['self_payout'].value_counts(normalize=True)*100

# COMMAND ----------

df[df['fraud'] == 1]['self_payout'].value_counts(normalize=True)*100

# COMMAND ----------

value_counts = df[df['fraud'] == 0]['self_payout'].value_counts(normalize=True) * 100

# COMMAND ----------

plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of self-payout for non-Fraud Transactions')
plt.axis('equal')

# COMMAND ----------

# MAGIC %md
# MAGIC ### NUMBER OF LEADS DROPPED (IMPROVED)

# COMMAND ----------

query = """\
    WITH base AS (
        select
            ld.profile_uuid, count(distinct lh.id) as number_of_leads
        from
            product_derived.leads_heavy lh
            inner join housing_leads_production.lead_details ld on lh.lead_details_id = ld.id
        group by 1
    )

    SELECT NU.*, base.number_of_leads
    FROM NU INNER JOIN base ON NU.profile_uuid = base.profile_uuid
"""

df = spark.sql(query).toPandas()

# COMMAND ----------

query = """\
    WITH base AS (
        select
            NU.profile_uuid, count(distinct lh.id) as number_of_leads
        from
            NU
            LEFT JOIN housing_leads_production.lead_details ld on ld.profile_uuid = NU.profile_uuid and ld.created_at < NU.created_at
            INNER JOIN product_derived.leads_heavy lh on lh.lead_details_id = ld.id
        group by 1
    )

    SELECT NU.*, base.number_of_leads
    FROM NU INNER JOIN base ON NU.profile_uuid = base.profile_uuid
"""

df = spark.sql(query).toPandas()

# COMMAND ----------

df[df['fraud'] ==0]['number_of_leads'].describe()

# COMMAND ----------

df[df['fraud'] ==1]['number_of_leads'].describe()

# COMMAND ----------

sns.boxplot(x='fraud', y='number_of_leads', data=df[df['number_of_leads'] <= 50])

# COMMAND ----------

# MAGIC %md
# MAGIC ### TIME SPENT ON PAYMENT GATEWAY

# COMMAND ----------

query = """
SELECT
    NU.order_id, NU.tenant_id, NU.created_at, o.status, NU.fraud, ct.successful_at, timestampdiff(MINUTE, o.created_at, ct.successful_at) as difference
FROM
    NU
    INNER JOIN apollo.orders o ON o.order_id = NU.order_id
    LEFT JOIN fortuna.sub_credit_transactions sct ON o.transaction_id = sct.sub_credit_txn_uuid
    LEFT JOIN fortuna.credit_transactions ct ON ct.id = sct.credit_transaction_id
GROUP BY 1,2,3,4,5,6,7
"""

df = spark.sql(query).toPandas()

# COMMAND ----------

df[df['fraud'] == 0]['difference'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['difference'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df = df[df['difference'] < 60]

# COMMAND ----------

sns.boxplot(x='fraud', y='difference', data=df)

# COMMAND ----------


