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

nu_features_query = f"""\
SELECT *
FROM feature_store.nu_feature_store_increased_frauds
WHERE date(created_at) between '2023-01-01' AND '2023-07-31'
"""
nu_static_features = spark.sql(nu_features_query).toPandas()

# COMMAND ----------

nu_static_features['fraud'].value_counts()

# COMMAND ----------

# spark.sql("""\
#     SELECT
#         nu_ga.*, nu_fs.fraud
#     FROM 
#         feature_store.nu_ga_feature_store AS nu_ga
#         LEFT JOIN feature_store.nu_feature_store_increased_frauds AS nu_fs ON nu_fs.ga_id = nu_ga.ga_id
#     WHERE
#         date(date) between '2023-01-01' AND '2023-07-31'
#         AND nu_ga.ga_id IN (
#                     select ga_id
#                     from feature_store.nu_feature_store_increased_frauds
#                     where date(created_at) between '2023-01-01' AND '2023-07-31'
#                     )
# """).toPandas().shape


# # WITHOUT GA_ID FILTER :: 1,07,16,624
# # WITH GA_ID FILTER :: 16,69,064
# # AFTER ADDING FRAUD LABELS :: 54,67,460

# COMMAND ----------

nu_ga_features_query = f"""\
    SELECT
        nu_ga.*, nu_fs.order_id, nu_fs.tenant_id, nu_fs.fraud
    FROM 
        feature_store.nu_ga_feature_store_more_features AS nu_ga
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

# nu_ga_features.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("feature_store" + "." + "nu_ga_merged_with_frauds")

# COMMAND ----------

nu_ga_features_query = f"""\
    SELECT
        *
    FROM 
        feature_store.nu_ga_merged_with_frauds 
"""
nu_ga_features = spark.sql(nu_ga_features_query)
nu_ga_features.createOrReplaceTempView("NU")

df_main = nu_ga_features.toPandas()

# COMMAND ----------

df_main['fraud'].value_counts()

# COMMAND ----------

df_main['fraud'].value_counts(normalize=True) * 100

# COMMAND ----------

def paid_percentage_per_discrete_value(df, feature, class_num):
  json = {}
  val_list = sorted(df[feature].unique())
  for val in val_list:
    json[val] = (df[(df[feature] == val) & (df['fraud'] == class_num)].shape[0] / df[df[feature] == val].shape[0]) * 100
  
  return json

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of sessions

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

query = """\
    with base as (
        SELECT
            NU1.ga_id, NU1.tenant_id, NU1.session_id, NU1.time_stamp, NU1.category, NU1.action,
            count(DISTINCT NU2.session_id) as total_number_of_sessions,
            COUNT(DISTINCT CASE WHEN NU2.source = 'app' THEN NU2.session_id END) AS number_of_app_sessions
        FROM
            (
            SELECT
                ga_id, tenant_id, session_id, time_stamp, category, action, source
            FROM
                feature_store.nu_ga_merged_with_frauds) AS NU1
            LEFT JOIN (
                        SELECT
                            ga_id, tenant_id, session_id, time_stamp, category, action, source
                        FROM
                            feature_store.nu_ga_merged_with_frauds) AS NU2 on NU2.tenant_id = NU1.tenant_id and NU2.time_stamp < NU1.time_stamp
        GROUP BY NU1.ga_id, NU1.tenant_id, NU1.session_id, NU1.time_stamp, NU1.category, NU1.action
    )

    SELECT
        NU.order_id, NU.ga_id, NU.tenant_id, NU.time_stamp, NU.session_id, NU.category, NU.action, base.total_number_of_sessions, base.number_of_app_sessions, NU.fraud
    FROM
        feature_store.nu_ga_merged_with_frauds AS NU
        INNER JOIN base ON base.ga_id = NU.ga_id AND base.time_stamp = NU.time_stamp AND base.category = NU.category AND base.action = NU.action
"""
df = spark.sql(query)

# COMMAND ----------

# df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("data_science_metastore.nu_payout_eda_features.number_of_sessions")

# COMMAND ----------

df = spark.sql("select * from data_science_metastore.nu_payout_eda_features.number_of_sessions").toPandas()

# COMMAND ----------

df[df['fraud'] == 0]['total_number_of_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['total_number_of_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='total_number_of_sessions', data=df)

# COMMAND ----------

df['log_total_number_of_sessions'] = df['total_number_of_sessions'].apply(lambda x: max(1, x))
df['log_total_number_of_sessions'] = df['log_total_number_of_sessions'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_total_number_of_sessions', data=df)

# COMMAND ----------

S_2 = df[(df['total_number_of_sessions'] <= 200)]
# nonfraud_sampled = df.sample(26884)
fig, ax = plt.subplots()
ax.hist(S_2[S_2["fraud"]==1]["total_number_of_sessions"], bins=10, alpha=0.5, color="red", label="Fraud", weights=np.ones(len(S_2[S_2["fraud"]==1])) / len(S_2))
ax.hist(S_2[S_2["fraud"]==0]["total_number_of_sessions"].sample(17584*2, random_state = 1), bins=10, alpha=0.5, color="green", label="Non-Fraud", weights=np.ones(len(S_2[S_2["fraud"]==0].sample(17584*2, random_state = 1))) / len(S_2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(0.01))
ax.set_xlabel("Number of Sessions")
ax.set_ylabel("Percentage of Tenants")
fig.suptitle("Number of Sessions")
ax.legend()
plt.show()

# COMMAND ----------

json = paid_percentage_per_discrete_value(df[df['total_number_of_sessions'] <200], 'total_number_of_sessions', 1)
plt.scatter(*zip(*json.items()))
plt.xlabel('Number of Sessions')
plt.ylabel('Percentage of tenants that are fraud')
plt.show()

# COMMAND ----------

df_one_session_atleast = df[df['total_number_of_sessions'] > 0]

df_one_session_atleast['percentage_of_app_sessions'] = df_one_session_atleast['number_of_app_sessions'] * 100 / df_one_session_atleast['total_number_of_sessions']
df_one_session_atleast['percentage_of_web_sessions'] = 100 - df_one_session_atleast['percentage_of_app_sessions']

# COMMAND ----------

df_one_session_atleast[df_one_session_atleast['fraud'] == 0]['percentage_of_app_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df_one_session_atleast[df_one_session_atleast['fraud'] == 1]['percentage_of_app_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df_one_session_atleast[df_one_session_atleast['fraud'] == 0]['percentage_of_web_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df_one_session_atleast[df_one_session_atleast['fraud'] == 0]['percentage_of_web_sessions'].quantile(0.99)

# COMMAND ----------

df_one_session_atleast[df_one_session_atleast['fraud'] == 1]['percentage_of_web_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df_one_session_atleast[df_one_session_atleast['fraud'] == 1]['percentage_of_web_sessions'].quantile(0.94)

# COMMAND ----------

df_one_session_atleast['log_percentage_of_web_sessions'] = df_one_session_atleast['percentage_of_web_sessions'].apply(lambda x: max(1, x))
df_one_session_atleast['log_percentage_of_web_sessions'] = df_one_session_atleast['log_percentage_of_web_sessions'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='percentage_of_web_sessions', data=df_one_session_atleast)

# COMMAND ----------

df_one_session_atleast['log_percentage_of_app_sessions'] = df_one_session_atleast['percentage_of_app_sessions'].apply(lambda x: max(1, x))
df_one_session_atleast['log_percentage_of_app_sessions'] = df_one_session_atleast['log_percentage_of_app_sessions'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='percentage_of_app_sessions', data=df_one_session_atleast)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df[df['fraud'] == 0]['number_of_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 0]['number_of_sessions'].quantile(0.95)

# COMMAND ----------

df[df['fraud'] == 1]['number_of_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='number_of_sessions', data=df)

# COMMAND ----------

df['log_number_of_sessions'] = df['number_of_sessions'].apply(lambda x: max(1, x))
df['log_number_of_sessions'] = df['log_number_of_sessions'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_number_of_sessions', data=df)

# COMMAND ----------

S_2 = df[(df['number_of_sessions'] <= 100)]
# nonfraud_sampled = df.sample(26884)
fig, ax = plt.subplots()
ax.hist(S_2[S_2["fraud"]==1]["number_of_sessions"], bins=10, alpha=0.5, color="red", label="Fraud", weights=np.ones(len(S_2[S_2["fraud"]==1])) / len(S_2))
ax.hist(S_2[S_2["fraud"]==0]["number_of_sessions"].sample(17584, random_state = 1), bins=10, alpha=0.5, color="green", label="Non-Fraud", weights=np.ones(len(S_2[S_2["fraud"]==0].sample(17584, random_state = 1))) / len(S_2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(0.01))
ax.set_xlabel("Number of Sessions")
ax.set_ylabel("Percentage of Tenants")
fig.suptitle("Number of Sessions")
ax.legend()
plt.show()

# COMMAND ----------

json = paid_percentage_per_discrete_value(df[df['number_of_sessions'] <60], 'number_of_sessions', 1)
plt.scatter(*zip(*json.items()))
plt.xlabel('Number of Sessions')
plt.ylabel('Percentage of tenants that are fraud')
plt.show()

# COMMAND ----------

df[(df['fraud'] == 1) & (df['number_of_sessions'] == 1)].shape

# COMMAND ----------

df[(df['fraud'] == 0) & (df['number_of_sessions'] == 1)].shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mean Number of Sessions

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df[df['fraud'] == 0]['mean_number_of_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['mean_number_of_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='mean_number_of_sessions', data=df)

# COMMAND ----------

df['log_mean_number_of_sessions'] = df['mean_number_of_sessions'].apply(lambda x: max(1, x))
df['log_mean_number_of_sessions'] = df['log_mean_number_of_sessions'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_mean_number_of_sessions', data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Session Time

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df[df['fraud'] == 0]['time_on_site'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['time_on_site'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='time_on_site', data=df)

# COMMAND ----------

S_2 = df[(df['time_on_site'] <= 10000)]
# nonfraud_sampled = df.sample(26884)
fig, ax = plt.subplots()
ax.hist(S_2[S_2["fraud"]==1]["time_on_site"], bins=10, alpha=0.5, color="red", label="Fraud", weights=np.ones(len(S_2[S_2["fraud"]==1])) / len(S_2))
ax.hist(S_2[S_2["fraud"]==0]["time_on_site"].sample(17584, random_state = 1), bins=10, alpha=0.5, color="green", label="Non-Fraud", weights=np.ones(len(S_2[S_2["fraud"]==0].sample(17584, random_state = 1))) / len(S_2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(0.01))
ax.set_xlabel("Session Time (in seconds)")
ax.set_ylabel("Percentage of Tenants")
fig.suptitle("Session Time")
ax.legend()
plt.show()

# COMMAND ----------

json = paid_percentage_per_discrete_value(df[df['time_on_site'] <10000], 'time_on_site', 1)
plt.scatter(*zip(*json.items()))
plt.xlabel('Time On Site (in seconds)')
plt.ylabel('Percentage of tenants that are fraud')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### % of app sessions

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Time since first session

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df['first_session_date'] = df.groupby('ga_id')['session_start'].transform('min')

# COMMAND ----------

df['days_elapsed'] = (df['session_start'] - df['first_session_date']).dt.days

# COMMAND ----------

sns.boxplot(data=df, x='fraud', y='days_elapsed')

# COMMAND ----------

df[df['fraud'] == 0]['days_elapsed'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['days_elapsed'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

S_2 = df[df['days_elapsed'] <= 100]
# nonfraud_sampled = df.sample(26884)
fig, ax = plt.subplots()
ax.hist(S_2[S_2["fraud"]==1]["days_elapsed"], bins=10, alpha=0.5, color="red", label="Fraud", weights=np.ones(len(S_2[S_2["fraud"]==1])) / len(S_2))
ax.hist(S_2[S_2["fraud"]==0]["days_elapsed"].sample(17584, random_state = 1), bins=10, alpha=0.5, color="green", label="Non-Fraud", weights=np.ones(len(S_2[S_2["fraud"]==0].sample(17584, random_state = 1))) / len(S_2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(0.01))
ax.set_xlabel("Days since first session")
ax.set_ylabel("Percentage of Tenants")
fig.suptitle("Days since first session")
ax.legend()
plt.show()

# COMMAND ----------

df[df['days_elapsed'] == 0]['fraud'].value_counts(normalize=True)*100

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of devices used

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df[df['fraud'] == 0]['number_of_devices_used'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['number_of_devices_used'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Traffic Source

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

nonfraud_cities = df[df['fraud'] == 0]['traffic_source'].value_counts().to_dict()
fraud_cities = df[df['fraud'] == 1]['traffic_source'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_cities.keys():
    if i in fraud_cities.keys():
        percentage_of_frauds_dict[i] = (fraud_cities[i] * 100) / nonfraud_cities[i]

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(x=values, y=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Traffic Source')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Traffic Campaign

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

nonfraud_cities = df[df['fraud'] == 0]['traffic_campaign'].value_counts().to_dict()
fraud_cities = df[df['fraud'] == 1]['traffic_campaign'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_cities.keys():
    if i in fraud_cities.keys():
        percentage_of_frauds_dict[i] = (fraud_cities[i] * 100) / (nonfraud_cities[i] + fraud_cities[i])

percentage_of_frauds_dict = dict(sorted(percentage_of_frauds_dict.items(), key=lambda item: item[1], reverse=True)[:30])
categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(x=values, y=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Traffic Campaign')
plt.show()

# COMMAND ----------

dict(sorted(percentage_of_frauds_dict.items(), key=lambda item: item[1], reverse=True)[:20])

# COMMAND ----------

df[df['traffic_campaign'] == 'DSA_Locality_deficit_Faridabad_Exper']['time_stamp'].describe()

# COMMAND ----------

df[df['traffic_campaign'] == 'DSA_Locality_deficit_Faridabad_Exper']['fraud'].value_counts(normalize=True) * 100

# COMMAND ----------

# MAGIC %md
# MAGIC ### Traffic source medium

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

nonfraud_cities = df[df['fraud'] == 0]['traffic_source_medium'].value_counts().to_dict()
fraud_cities = df[df['fraud'] == 1]['traffic_source_medium'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_cities.keys():
    if i in fraud_cities.keys():
        percentage_of_frauds_dict[i] = (fraud_cities[i] * 100) / (nonfraud_cities[i] + fraud_cities[i])

percentage_of_frauds_dict = dict(sorted(percentage_of_frauds_dict.items(), key=lambda item: item[1], reverse=True)[:30])
categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(x=values, y=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Traffic Source-Medium')
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of Screen Views / Page Views

# COMMAND ----------

df = df_main.copy(deep=True)

df['screen_or_page_views'] = np.where(df['number_of_page_views'].notnull(), df['number_of_page_views'], df['number_of_screen_views'])

# COMMAND ----------

df[df['fraud'] == 0]['screen_or_page_views'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['screen_or_page_views'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(data=df, x='fraud', y='screen_or_page_views')

# COMMAND ----------

df['log_screen_or_page_views'] = df['screen_or_page_views'].apply(lambda x: max(1, x))
df['log_screen_or_page_views'] = df['log_screen_or_page_views'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_screen_or_page_views', data=df)

# COMMAND ----------

df['average_screen_time'] = df['time_on_screen'] / df['screen_or_page_views']

# COMMAND ----------

df[df['fraud'] == 0]['average_screen_time'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['average_screen_time'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df['log_average_screen_time'] = df['average_screen_time'].apply(lambda x: max(1, x))
df['log_average_screen_time'] = df['log_average_screen_time'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_average_screen_time', data=df)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of app/web sessions

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

df_main['source'].value_counts(normalize=True) * 100

# COMMAND ----------

df_main[df_main['fraud'] == 0]['source'].value_counts(normalize=True) * 100

# COMMAND ----------

df_main[df_main['fraud'] == 1]['source'].value_counts(normalize=True) * 100

# COMMAND ----------

value_counts = df_main[df_main['fraud'] == 0]['source'].value_counts(normalize=True) * 100

# COMMAND ----------

plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of App/Web Sessions for Non-Fraud Transactions')
plt.axis('equal')

# COMMAND ----------

sns.countplot(data=df_main, x='source', hue='fraud')

# COMMAND ----------

df_main[df_main['source'] == 'app']['fraud'].value_counts(normalize=True) * 100

# COMMAND ----------

df_main[df_main['source'] == 'web']['fraud'].value_counts(normalize=True) * 100

# COMMAND ----------

query = """\
    SELECT ord1.order_id, ord1.tenant_id, ord1.ga_id, ord1.session_id, ord1.time_stamp, ord1.fraud, ord1.number_of_sessions,
            COUNT(DISTINCT CASE WHEN ord2.source = 'app' THEN ord2.session_id END) AS number_of_app_sessions
    FROM (
        SELECT order_id, tenant_id, ga_id, session_id, time_stamp, number_of_sessions, source, fraud
        From feature_store.nu_ga_merged_with_frauds
    ) ord1
    LEFT JOIN
    (
        SELECT order_id, tenant_id, ga_id, session_id, time_stamp, number_of_sessions, source, fraud
        From feature_store.nu_ga_merged_with_frauds
    ) ord2 ON ord2.tenant_id = ord1.tenant_id AND ord2.time_stamp < ord1.time_stamp
    GROUP BY ord1.order_id, ord1.tenant_id, ord1.ga_id, ord1.session_id, ord1.time_stamp, ord1.fraud, ord1.number_of_sessions
"""

df = spark.sql(query)

# COMMAND ----------

# df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("data_science_metastore.nu_payout_eda_features.number_of_app_sessions")

# COMMAND ----------

df_app_counts = spark.sql("select * from data_science_metastore.nu_payout_eda_features.number_of_app_sessions").toPandas()

# COMMAND ----------

df_second = df.merge(df_app_counts, on='ga_id')

# COMMAND ----------



# COMMAND ----------

df = df_main.copy(deep=True)

app_source_counts = df[df['source'] == 'app'].groupby('tenant_id')['source'].count().reset_index()
app_source_counts.rename(columns={'source': 'number_of_app_sessions'}, inplace=True)

# COMMAND ----------

df = pd.merge(df, app_source_counts, on='tenant_id', how='left')

# COMMAND ----------

df[df['number_of_app_sessions'] > df['number_of_sessions']].head()

# COMMAND ----------

df['fraud'].value_counts()

# COMMAND ----------

df['number_of_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df_one_session_atleast = df[df['number_of_sessions'] > 0]

# COMMAND ----------

df_one_session_atleast['percentage_of_app_sessions'] = df_one_session_atleast['number_of_app_sessions'] * 100 / df_one_session_atleast['number_of_sessions']
df_one_session_atleast['percentage_of_web_sessions'] = 100 - df_one_session_atleast['percentage_of_app_sessions']

# COMMAND ----------

df_one_session_atleast['percentage_of_app_sessions'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df_main.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Region

# COMMAND ----------

df = df_main.copy(deep=True)

# COMMAND ----------

nonfraud_cities = df[df['fraud'] == 0]['region'].value_counts().to_dict()
fraud_cities = df[df['fraud'] == 1]['region'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_cities.keys():
    if i in fraud_cities.keys():
        percentage_of_frauds_dict[i] = (fraud_cities[i] * 100) / (fraud_cities[i] + nonfraud_cities[i])

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(x=values, y=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('Region')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CITY

# COMMAND ----------

nonfraud_cities = df[df['fraud'] == 0]['city'].value_counts().to_dict()
fraud_cities = df[df['fraud'] == 1]['city'].value_counts().to_dict()

percentage_of_frauds_dict = {}

for i in nonfraud_cities.keys():
    if i in fraud_cities.keys():
        percentage_of_frauds_dict[i] = (fraud_cities[i] * 100) / (fraud_cities[i] + nonfraud_cities[i])

categories = list(percentage_of_frauds_dict.keys())
values = list(percentage_of_frauds_dict.values())

# Plotting using a bar chart with Seaborn
sns.barplot(x=values, y=categories)
plt.xlabel('Percentage of Fraud Transactions')
plt.ylabel('City')
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of hits outside PayOnCredit

# COMMAND ----------

query = """\
    with base as (
        SELECT
            NU1.ga_id, NU1.tenant_id, NU1.session_id, NU1.time_stamp, NU1.category, NU1.action,
            count(DISTINCT NU2.session_id) as total_number_of_sessions,
            COUNT(DISTINCT CASE WHEN NU2.category NOT IN ('poc_flow','poc_home_page','poc_landing_page','transaction_success_page') THEN NU2.session_id END) AS number_of_non_poc_sessions
        FROM
            (
            SELECT
                ga_id, tenant_id, session_id, time_stamp, category, action, source
            FROM
                feature_store.nu_ga_merged_with_frauds) AS NU1
            LEFT JOIN (
                        SELECT
                            ga_id, tenant_id, session_id, time_stamp, category, action, source
                        FROM
                            feature_store.nu_ga_merged_with_frauds) AS NU2 on NU2.tenant_id = NU1.tenant_id and NU2.time_stamp < NU1.time_stamp
        GROUP BY NU1.ga_id, NU1.tenant_id, NU1.session_id, NU1.time_stamp, NU1.category, NU1.action
    )

    SELECT
        NU.order_id, NU.ga_id, NU.tenant_id, NU.time_stamp, NU.session_id, NU.category, NU.action, base.total_number_of_sessions, base.number_of_non_poc_sessions, NU.fraud
    FROM
        feature_store.nu_ga_merged_with_frauds AS NU
        INNER JOIN base ON base.ga_id = NU.ga_id AND base.time_stamp = NU.time_stamp AND base.category = NU.category AND base.action = NU.action
"""
df = spark.sql(query)

# COMMAND ----------

# df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("data_science_metastore.nu_payout_eda_features.number_of_sessions_with_non_poc_activity")

# COMMAND ----------

df = spark.sql("select * from data_science_metastore.nu_payout_eda_features.number_of_sessions_with_non_poc_activity").toPandas()

# COMMAND ----------

df_one_session_atleast = df[df['total_number_of_sessions'] > 0]

df_one_session_atleast['percentage_of_non_poc_sessions'] = df_one_session_atleast['number_of_non_poc_sessions'] * 100 / df_one_session_atleast['total_number_of_sessions']
# df_one_session_atleast['percentage_of_web_sessions'] = 100 - df_one_session_atleast['percentage_of_app_sessions']

# COMMAND ----------

df_one_session_atleast['log_percentage_of_non_poc_sessions'] = df_one_session_atleast['percentage_of_non_poc_sessions'].apply(lambda x: max(1, x))
df_one_session_atleast['log_percentage_of_non_poc_sessions'] = df_one_session_atleast['log_percentage_of_non_poc_sessions'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_percentage_of_non_poc_sessions', data=df_one_session_atleast)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hit Number of Transaction

# COMMAND ----------

df = df_main.copy(deep=True)

list_of_actions = [
    'pay_cta_clicked',
    'pay_again_clicked',
    'pay_cta_click'
]

df = df[df['action'].isin(list_of_actions)]

# COMMAND ----------

df[df['fraud'] ==0]['hit_number'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] ==1]['hit_number'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='hit_number', data=df)

# COMMAND ----------

df['log_hit_number'] = df['hit_number'].apply(lambda x: max(1, x))
df['log_hit_number'] = df['log_hit_number'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_hit_number', data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of hits per session

# COMMAND ----------

df = df_main.copy(deep=True)

list_of_actions = [
    'pay_cta_clicked',
    'pay_again_clicked',
    'pay_cta_click'
]

df = df[df['action'].isin(list_of_actions)]

# COMMAND ----------

df[df['fraud'] ==0]['no_hits'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] ==1]['no_hits'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df['log_no_hits'] = df['no_hits'].apply(lambda x: max(1, x))
df['log_no_hits'] = df['log_no_hits'].apply(lambda x: np.log(x))

sns.boxplot(x='fraud', y='log_no_hits', data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### % of journey when transaction attempted

# COMMAND ----------

df = df_main.copy(deep=True)

list_of_actions = [
    'pay_cta_clicked',
    'pay_again_clicked',
    'pay_cta_click'
]

df = df[df['action'].isin(list_of_actions)]

# COMMAND ----------

df = df[df['hit_number'] <= df['no_hits']]

df['journey_percentage'] = df['hit_number'] * 100 / df['no_hits']
# df_one_session_atleast['percentage_of_web_sessions'] = 100 - df_one_session_atleast['percentage_of_app_sessions']

# COMMAND ----------

df[df['fraud'] ==0]['journey_percentage'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

df[df['fraud'] == 1]['journey_percentage'].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

sns.boxplot(x='fraud', y='journey_percentage', data=df)

# COMMAND ----------


