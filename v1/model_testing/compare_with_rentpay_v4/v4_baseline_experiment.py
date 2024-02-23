# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, unix_timestamp, min, collect_set

START_DATE = '2023-06-16'
END_DATE = '2023-07-31'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

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
WHERE date(fs.created_at) BETWEEN '{START_DATE}' AND '{END_DATE}'
"""
feature_store = spark.sql(feature_store_query)

# COMMAND ----------

# new_users = feature_store[feature_store['user_flag'] == 'NU']

new_users = feature_store.filter(feature_store['user_flag'] == "NU")

# COMMAND ----------

ga_features_query = f"""\
SELECT *
    FROM feature_store.rentpay_ga_features
    WHERE date(timestamp) BETWEEN '{START_DATE}' AND '{END_DATE}'
"""
ga_features = spark.sql(ga_features_query)

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

old_order_id = set(old_user.select(collect_set('order_id')).first()[0])

# COMMAND ----------

new_user_more_than_90day = new_user_more_than_90day.join(df_result_2, on=["order_id"], how="left_anti")
new_user_less_than_2day = new_user_less_than_2day.join(df_result_2, on=["order_id"], how="left_anti")

# COMMAND ----------

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

df_final.columns

# COMMAND ----------

df_final['out'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## INCREASING FRAUD COUNTS

# COMMAND ----------

df_final =pd.read_csv("/dbfs/FileStore/harshul/nu_payout/v4_new_data_1.csv", low_memory=False)
df_final = df_final[((df_final['out'] == 0) & (df_final['payment_status'] == 'PAYOUT_SUCCESSFUL') | (df_final['out'] == 1))]

# COMMAND ----------

df_final['created_at'] = pd.to_datetime(df_final['created_at'])

# COMMAND ----------

df_final['out'].value_counts()

# COMMAND ----------

fraud_tenant_no = set(df_final[df_final['out']==1]['tenant_contact_number_encrypted'])
fraud_ga_id = set(df_final[df_final['out']==1]['ga_id'])
print("before",len(fraud_tenant_no))
for index, row in df_final.iterrows():
    ga_id = str(row['ga_id'])
    if ga_id in fraud_ga_id:
        fraud_tenant_no.add(row['tenant_contact_number_encrypted'])
print("after",len(fraud_tenant_no))

# COMMAND ----------

def mark_fraud(t_no):
    if t_no in fraud_tenant_no:
        return 1
    else:
        return 0
df_final["f_1"]= df_final.tenant_contact_number_encrypted.apply(mark_fraud)
df_final["f_1"].value_counts()

# COMMAND ----------

fraud_tenant_no = set(df_final[df_final['f_1']==1]['tenant_contact_number_encrypted'])
fraud_ga_id = set(df_final[df_final['f_1']==1]['ga_id'])
print("before",len(fraud_tenant_no))
for index, row in df_final.iterrows():
    ga_id = str(row['ga_id'])
    if ga_id in fraud_ga_id:
        fraud_tenant_no.add(row['tenant_contact_number_encrypted'])
print("after",len(fraud_tenant_no))

def mark_fraud(t_no):
    if t_no in fraud_tenant_no:
        return 1
    else:
        return 0
df_final["f_1"]= df_final.tenant_contact_number_encrypted.apply(mark_fraud)
df_final["f_1"].value_counts()

# COMMAND ----------

def mark_new_old(duration):
    if duration>=48 and duration<=2160:
        return "old"
    else:
        return "new"
df_final['user_type']=df_final['duration_hours'].apply(mark_new_old)
df_final['user_type'].value_counts()

# COMMAND ----------

data=df_final[['amount','platform','tenant_age','gtv_15_day','gtv_30_day','city','different_location_30','diff','number_failed_30days','number_failed_15days','number_sucessfull_30days','num_attempts_30_days','avg_number_attemps_past30day','time_hour','upi_bank','account_bank','mobileDeviceBranding','latitude','longitude','f_1','session_counts','mean_session_counts','mean_session_time','median_session_counts','median_session_time','num_red_predictions','num_yellow_predictions','ist_time','user_activity','user_type']]
new_user= data[data['user_type']=="new"]
data.drop(['user_type'],axis=1,inplace=True)

f_df=data[data['f_1']==1]
nf_df=data[data['f_1']==0]

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

data.columns

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
    'num_yellow_predictions':'yellow_count_in_last_thirty_days',
    'f_2':'all_f2',
}, inplace = True)

actuals = data[['f_1']]
data.drop(['ist_time', 'f_1'],axis=1,inplace=True)


# COMMAND ----------

import mlflow
logged_model = 'runs:/3d0941e92ba643a0a0248db84dd17ab4/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
y_pred_proba = loaded_model.predict_proba(pd.DataFrame(data))

# COMMAND ----------

predictions = (y_pred_proba[:,-1] >= 0.20).astype(int)
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

# COMMAND ----------

from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
def plot_PR_curve(y_test, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_test.astype('int'), y_pred_proba[:, 1])
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()

def find_threshold_for_max_recall(recall_threshold, y_test, y_pred_proba):
        """
        This function makes the precision at particular recall value by plotting PR curve.
        """
        recall_threshold = recall_threshold
        precision, recall, thresholds = precision_recall_curve(y_test.astype('int'), y_pred_proba[:, 1])
        prediction_threshold = thresholds[np.where(precision == precision[recall >= recall_threshold][-1])[0][0]]
        return prediction_threshold

def find_threshold_for_max_precision(precision_threshold, y_test, y_pred_proba):
        """
        This function makes the precision at particular recall value by plotting PR curve.
        """
        precision_threshold = precision_threshold
        precision, recall, thresholds = precision_recall_curve(y_test.astype('int'), y_pred_proba[:, 1])
        prediction_threshold = thresholds[np.where(recall == recall[precision >= precision_threshold][0])[0][0]]
        return prediction_threshold

# COMMAND ----------

find_threshold_for_max_precision(1, actuals, y_pred_proba)
# find_threshold_for_max_recall(0.65, y_test, y_pred_proba)

# COMMAND ----------

y_pred_x = (y_pred_proba[:,-1] >= 0.9992138).astype(int)
print(classification_report(y_true = actuals, y_pred = y_pred_x))

# COMMAND ----------

cm = confusion_matrix(y_true = actuals, y_pred = predictions)

class_1_index = 1  # Assuming class 1 is the positive class (adjust the index if needed)
TP_class_1 = cm[class_1_index, class_1_index]
FP_class_1 = cm[0, class_1_index]
FN_class_1 = cm[class_1_index, 0]
TN_class_1 = np.sum(cm) - (TP_class_1 + FP_class_1 + FN_class_1)

print("True Positives (Class 1):", TP_class_1)
print("False Positives (Class 1):", FP_class_1)
print("False Negatives (Class 1):", FN_class_1)
print("True Negatives (Class 1):", TN_class_1)


# COMMAND ----------

# data['fraud_probability'] = y_pred_proba
data['actual'] = actuals
# data['prediction'] = predictions

# COMMAND ----------

all_red_preds = data[data['ML_channel'] == 'RED']
incorrect_red_preds = all_red_preds[all_red_preds['actual'] != 1]
correct_red_preds = all_red_preds[all_red_preds['actual'] == 1]

# COMMAND ----------

all_red_preds['fraud_probability'].describe()

# COMMAND ----------

incorrect_red_preds['fraud_probability'].describe()

# COMMAND ----------

correct_red_preds['fraud_probability'].describe()

# COMMAND ----------

all_non_red_preds = data[(data['ML_channel'] == 'GREEN') | ((data['ML_channel'] == 'YELLOW'))]
incorrect_non_red_preds = all_non_red_preds[all_non_red_preds['actual'] != 0]
correct_non_red_preds = all_non_red_preds[all_non_red_preds['actual'] == 0]

# COMMAND ----------

incorrect_non_red_preds['fraud_probability'].describe()

# COMMAND ----------

correct_non_red_preds['fraud_probability'].describe()

# COMMAND ----------


