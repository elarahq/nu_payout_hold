# Databricks notebook source
import pandas as pd
import numpy as np
import pickle
from pyspark.sql.functions import col, unix_timestamp, min, collect_set

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

START_DATE_FULL = '2023-06-16'
END_DATE_FULL = '2023-07-31'

# COMMAND ----------

with open("/dbfs/FileStore/harshul/nu_payout/june_16_to_july_31_order_ids.pkl", 'rb') as file:
    order_ids = pickle.load(file)

# COMMAND ----------

len(order_ids)

# COMMAND ----------

feature_store_query = f"""\
SELECT
    fs.*,
    o.status as payment_status,
    case when nt.tenant_id is null then 'NU' else 'RU' end as user_flag
FROM
    feature_store.rentpay_feature_store_v3 fs
    INNER JOIN apollo.orders o ON fs.id = o.id
    LEFT JOIN (
        select tenant_id,
        min(created_at) as first_transaction_date
        from apollo.orders
        where status in ('PAYOUT_SUCCESSFUL')group by 1
    ) AS nt ON fs.tenant_id = nt.tenant_id and fs.created_at > nt.first_transaction_date
"""
feature_store = spark.sql(feature_store_query).toPandas()

# COMMAND ----------

feature_store = feature_store[feature_store['order_id'].isin(order_ids)]
ga_ids = feature_store['ga_id'].to_list()

# COMMAND ----------

feature_store.shape

# COMMAND ----------

ga_features_query = f"""\
SELECT *
    FROM feature_store.rentpay_ga_features
"""
ga_features_pd = spark.sql(ga_features_query).toPandas()

ga_features_pd = ga_features_pd[ga_features_pd['ga_id'].isin(ga_ids)]

# COMMAND ----------

ga_features_pd.shape

# COMMAND ----------

# feature_store = feature_store.sort_values(by='created_at', ascending=True)
# ga_features_pd = ga_features_pd.sort_values(by='timestamp', ascending=True)

# df = pd.merge_asof(feature_store, ga_features_pd, by='ga_id',
#                  left_on='created_at', right_on='timestamp',
#                  tolerance=pd.Timedelta(minutes = 60))

# COMMAND ----------

feature_store['out'].value_counts()

# COMMAND ----------

new_users = spark.createDataFrame(feature_store)
ga_features = spark.createDataFrame(ga_features_pd)

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

new_user_more_than_90day = new_user_more_than_90day.join(df_result_2, on=["order_id"], how="left_anti")
new_user_less_than_2day = new_user_less_than_2day.join(df_result_2, on=["order_id"], how="left_anti")

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
df_final.to_csv("/dbfs/FileStore/harshul/nu_payout//v4_new_data_1.csv", index=False)

# COMMAND ----------

df_final['out'].value_counts()

# COMMAND ----------

fraud_tenants = df_final[df_final['out'] == 1]['tenant_id'].to_list()
fraud_phone_numbers = df_final[df_final['out'] == 1]['tenant_contact_number_encrypted'].to_list()
fraud_ga_ids = df_final[df_final['out'] == 1]['ga_id'].to_list()

for index, row in df_final.iterrows():
    ga_id = str(row['ga_id'])
    if ga_id in fraud_ga_ids:
        ga_id_phone_number = str(row['tenant_contact_number_encrypted'])
        fraud_phone_numbers.append(ga_id_phone_number)

df_final['out'] = df_final['tenant_id'].apply(lambda x: 1 if x in fraud_tenants else 0)
df_final['out'] = df_final['tenant_contact_number_encrypted'].apply(lambda x: 1 if x in fraud_phone_numbers else 0)

# COMMAND ----------

df_final['out'].value_counts()

# COMMAND ----------

data=df_final[['amount','platform','tenant_age','gtv_15_day','gtv_30_day','city','different_location_30','diff','number_failed_30days','number_failed_15days','number_sucessfull_30days','num_attempts_30_days','avg_number_attemps_past30day','time_hour','upi_bank','account_bank','mobileDeviceBranding','latitude','longitude','session_counts','mean_session_counts','mean_session_time','median_session_counts','median_session_time','num_red_predictions','num_yellow_predictions','user_activity', 'out']]

# COMMAND ----------

### FILLNA GA features numerical with  0  and string feature with "other"
# 
data['mobileDeviceBranding']=data['mobileDeviceBranding'].fillna("other")
data['latitude']=data['latitude'].fillna(0)
data['longitude']=data['longitude'].fillna(0)
data['session_counts']=data['session_counts'].fillna(0)
data['mean_session_counts']=data['mean_session_counts'].fillna(0)
data['mean_session_time']=data['mean_session_time'].fillna(0)
data['median_session_counts']=data['median_session_counts'].fillna(0)
data['median_session_time']=data['median_session_time'].fillna(0)
data['num_red_predictions']=data['num_red_predictions'].fillna(0)
data['num_yellow_predictions']=data['num_yellow_predictions'].fillna(0)

## feature_store Features
data['gtv_15_day']=data['gtv_15_day'].fillna("other")
data['gtv_30_day']=data['gtv_30_day'].fillna(0)
data['number_failed_30days']=data['number_failed_30days'].fillna(0)
data['number_failed_15days']=data['number_failed_15days'].fillna(0)

data['city']=data['city'].fillna("other")
data['user_activity']=data['user_activity'].fillna(0)

data=data[data['upi_bank'].notna()]

# COMMAND ----------

data.rename(columns = {
    'city':'city_name',
    'gtv_15_day':'gtv_of_successful_transaction_in_last_fifteen_days',
    'gtv_30_day':'gtv_of_successful_transaction_in_last_thirty_days', 
    'different_location_30':'different_location_in_past_thirty_days',
    'number_failed_30days':'failed_attempt_in_last_thirty_days',
    'number_failed_15days':'failed_attempt_in_last_fifteen_days',
    'number_sucessfull_30days':'number_of_success_attemp_30days',
    'num_attempts_30_days':'number_attempt_30days',
    'avg_number_attemps_past30day':'average_attempts_in_past_thirty_days',
    'time_hour':'hour_of_the_day',
    'num_red_predictions':'red_count_in_last_thirty_days',
    'num_yellow_predictions':'yellow_count_in_last_thirty_days'
}, inplace = True)

actuals = data[['out']]
data.drop(['out'],axis=1,inplace=True)


# COMMAND ----------

import mlflow
logged_model = 'runs:/3d0941e92ba643a0a0248db84dd17ab4/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
y_pred_proba = loaded_model.predict_proba(pd.DataFrame(data))

# COMMAND ----------

predictions = (y_pred_proba[:,-1] >= 0.02).astype(int)
# data['fraud_probability'] = loaded_model.predict_proba(pd.DataFrame(data))[:,-1]

# def map_prob_to_color(prob):
#     if prob < 0.50:
#         return 'GREEN'
#     elif 0.50 <= prob < 0.75:
#         return 'YELLOW'
#     else:
#         return 'RED'
# data['ML_channel'] = data['fraud_probability'].apply(map_prob_to_color)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true = actuals, y_pred = predictions))

cm = confusion_matrix(y_true = actuals, y_pred = predictions)
print(cm)

from sklearn.metrics import precision_score
print(f"{precision_score(actuals, predictions, pos_label=0, average='binary') * 100}%")

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
disp.plot(cmap=plt.cm.Blues, values_format='d')
