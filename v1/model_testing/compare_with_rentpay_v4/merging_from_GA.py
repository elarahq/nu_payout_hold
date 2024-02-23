# Databricks notebook source
import pandas as pd
import numpy as np
min_date = '2022-08-01'

# COMMAND ----------

feature_store_query = f"""\
SELECT *
FROM feature_store.rentpay_feature_store_v3
WHERE date(created_at) > '{min_date}'
"""
feature_store = spark.sql(feature_store_query)

# COMMAND ----------

ga_features_query = f"""\
SELECT *
    FROM feature_store.90_days_RENTPAY_GA_FEATURES
    WHERE timestamp >= '{min_date}'
"""
ga_features = spark.sql(ga_features_query)

# COMMAND ----------

from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Perform full outer join
merged = feature_store.join(ga_features, on='ga_id', how='outer')

# Display the merged DataFrame
# merged.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Assuming you have a DataFrame called 'df' with columns 'created_at' and 'timestamp'

# Calculate the time difference in seconds
time_diff_sec = (unix_timestamp(col('created_at')) - unix_timestamp(col('timestamp')))

# Calculate the time difference in hours
time_diff_hours = time_diff_sec / 3600

# Add a new column 'Duration_hours' to the DataFrame with the calculated time difference
merged = merged.withColumn('Duration_hours', time_diff_hours)

# Display the updated DataFrame
# merged.show()

# COMMAND ----------

display(merged)

# COMMAND ----------

old_user = merged.where("Duration_hours >= 48 and Duration_hours <= 2160")

new_user_less_than_2day = merged.where("Duration_hours < 48 and Duration_hours >=0")
new_user_more_than_90day = merged.where("Duration_hours >2160")


# COMMAND ----------

# from pyspark.sql.functions import collect_set

# column_name = "ga_id"
# ga_id_old = set(old_user.select(collect_set(column_name)).first()[0])
# ga_id_new = set(new_user_less_than_2day.select(collect_set(column_name)).first()[0])

# COMMAND ----------

###Code to calculate Min duration gap between feature store and ga store for old user
from pyspark.sql.functions import col, min

df_min_duration = old_user.groupby('order_id').agg(min(col('Duration_hours')).alias('min_duration_hours'))
old_user = old_user.join(df_min_duration, on='order_id', how='inner')
# df_result_2 = old_user.where("Duration_hours == min_duration_hours")



# COMMAND ----------

df_result_2 = old_user.where("Duration_hours == min_duration_hours")


# COMMAND ----------

# print(new_user_less_than_2day.count())
# print(new_user_more_than_90day.count())
# # print(old_user.count())

# COMMAND ----------

def drop_duplicates_ignore_columns(df, ignore_columns):
    # Get the list of column names excluding the ignore columns
    column_names = [col_name for col_name in df.columns if col_name not in ignore_columns]

    # Drop duplicate rows while ignoring the specified columns
    df_without_duplicates = df.dropDuplicates(subset=column_names)

    return df_without_duplicates

# Call the function to drop duplicates while ignoring 'col3' and 'col4'
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
 'Duration_hours']
new_user_less_than_2day = drop_duplicates_ignore_columns(new_user_less_than_2day, ignore_columns)
new_user_more_than_90day = drop_duplicates_ignore_columns(new_user_more_than_90day, ignore_columns)


# COMMAND ----------

##Remove old_user order_id from new_user_less_than_2day and new_user_more_than_90day
from pyspark.sql.functions import collect_set

old_order_id = set(old_user.select(collect_set('order_id')).first()[0])


# COMMAND ----------

old_user.count()

# COMMAND ----------

new_user_less_than_2day.count()

# COMMAND ----------

new_user_less_than_2day = new_user_less_than_2day.join(df_result_2, on=["order_id"], how="left_anti")


# COMMAND ----------

display(new_user_less_than_2day)

# COMMAND ----------

new_user_less_than_2day.count()

# COMMAND ----------

new_user_more_than_90day = new_user_more_than_90day.join(df_result_2, on=["order_id"], how="left_anti")


# COMMAND ----------

df_1= new_user_more_than_90day.toPandas()

# COMMAND ----------

df_final["f_1"].value_counts()

# COMMAND ----------

old_user.head()

# COMMAND ----------

# Now we will merge new_user_1 , new_user_2 and old_user into 1 data frame and increse fraud marked

# COMMAND ----------

# ga_col.remove('ga_id')

# COMMAND ----------

len(new_user_combined)

# COMMAND ----------

df_final.to_csv("/dbfs/FileStore/Badal/rent_pay/v4_new_data_1.csv")

# COMMAND ----------

new_user_combined.columns

# COMMAND ----------

import numpy as np
for col in ga_col:
    new_user_combined[f"{col}"]=np.nan

# COMMAND ----------

del df_2

# COMMAND ----------

new_user_combined.head()

# COMMAND ----------

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

# COMMAND ----------

##Meak extra fraud
df_final.out.value_counts()

# COMMAND ----------

new_user_combined.columns

# COMMAND ----------

fraud_tenant_no = set(df_final[df_final['f_1']==1]['tenant_contact_number_encrypted'])
fraud_ga_id = set(df_final[df_final['f_1']==1]['ga_id'])
print("before",len(fraud_tenant_no))
for index, row in df_final.iterrows():
    ga_id = str(row['ga_id'])
    if ga_id in fraud_ga_id:
        fraud_tenant_no.add(row['tenant_contact_number_encrypted'])
print("after",len(fraud_tenant_no))

# COMMAND ----------

import pandas as pd
df_final=pd.read_csv("/dbfs/FileStore/Badal/rent_pay/v4_new_data_1.csv")

# COMMAND ----------

df_1.to_csv("/dbfs/FileStore/Badal/rent_pay/old_user_1.csv")

# COMMAND ----------

# # for col in ga_features.columns:
# #     new_user_combined[f"{col}"]=np.nan
# ga_col=ga_features.columns
# # 

# COMMAND ----------

df_final=pd.concat([new_user_combined,old_user])

# COMMAND ----------

del df_1
df_2= new_user_less_than_2day.toPandas()

# COMMAND ----------

df_2.to_csv("/dbfs/FileStore/Badal/rent_pay/new_user_less_than_2day_1.csv")

# COMMAND ----------

new_user_more_than_90day.count()

# COMMAND ----------

old_user.columns

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

import pandas as pd
old_user= pd.read_csv("/dbfs/FileStore/Badal/rent_pay/old_user_1.csv")
new_user_less_than_2day =pd.read_csv("/dbfs/FileStore/Badal/rent_pay/new_user_less_than_2day_1.csv")
new_user_more_than_90day =pd.read_csv("/dbfs/FileStore/Badal/rent_pay/new_user_more_than_90day_1.csv")

# COMMAND ----------

df_1= df_result_2.toPandas()

# COMMAND ----------

old_user.drop('min_duration_hours',axis=1,inplace=True)

# COMMAND ----------

df_1.to_csv("/dbfs/FileStore/Badal/rent_pay/new_user_more_than_90day_1.csv")

# COMMAND ----------

def mark_fraud(t_no):
    if t_no in fraud_tenant_no:
        return 1
    else:
        return 0
df_final["f_1"]= df_final.tenant_contact_number_encrypted.apply(mark_fraud)

# COMMAND ----------

new_user_combined = pd.concat([new_user_less_than_2day,new_user_more_than_90day])

# COMMAND ----------

def mark_fraud(t_no):
    if t_no in fraud_tenant_no:
        return 1
    else:
        return 0
df_final["f_1"]= df_final.tenant_contact_number_encrypted.apply(mark_fraud)

# COMMAND ----------

df_final.head()

# COMMAND ----------

df_final["f_1"].value_counts()

# COMMAND ----------

def mark_new_old(duration):
    if duration>=48 and duration<=2160:
        return "old"
    else:
        return "new"
df_final['user_type']=df_final.Duration_hours.apply(mark_new_old)

# COMMAND ----------

new_user= data[data['user_type']=="new"]

# COMMAND ----------



# COMMAND ----------

f_df=data[data['f_1']==1]
nf_df=data[data['f_1']==0]


# COMMAND ----------

data=df_final[['amount','platform','tenant_age','gtv_15_day','gtv_30_day','city','different_location_30','diff','number_failed_30days','number_failed_15days','number_sucessfull_30days','num_attempts_30_days','avg_number_attemps_past30day','time_hour','upi_bank','account_bank','mobileDeviceBranding','latitude','longitude','f_1','session_counts','mean_session_counts','mean_session_time','median_session_counts','median_session_time','num_red_predictions','num_yellow_predictions','ist_time','user_activity','user_type']]


# COMMAND ----------

new_user

# COMMAND ----------

df_final.to_csv("/dbfs/FileStore/Badal/rent_pay/v4_new_data_1.csv")

# COMMAND ----------

df_final.user_type.value_counts()

# COMMAND ----------

data.drop(['user_type'],axis=1,inplace=True)

# COMMAND ----------

import pandas as pd
# df_final=pd.read_csv("/dbfs/FileStore/Badal/rent_pay/v4_new_data.csv")
# df_final.head()
df_final =pd.read_csv("/dbfs/FileStore/Badal/rent_pay/v4_new_data_1.csv")

# COMMAND ----------

f_df.isnull().sum()

# COMMAND ----------

f_df.head()

# COMMAND ----------

# data=df_final[['amount','platform','tenant_age','gtv_15_day','gtv_30_day','city','different_location_30','diff','number_failed_30days','number_failed_30days','number_sucessfull_30days','num_attempts_30_days','avg_number_attemps_past30day','time_hour','upi_bank','account_bank','mobileDeviceBranding','latitude','longitude','f_1','session_counts','mean_session_counts','mean_session_time','median_session_counts','median_session_time','num_red_predictions','num_yellow_predictions','ist_time','user_activity','user_type']]


# COMMAND ----------

### FILLNA GA features numerical with  0  and string feature with "other"
# 
f_df['mobileDeviceBranding']=f_df['mobileDeviceBranding'].fillna("other")
f_df['latitude']=f_df['latitude'].fillna(0)
f_df['longitude']=f_df['longitude'].fillna(0)
f_df['session_counts']=f_df['session_counts'].fillna(0)
f_df['mean_session_counts']=f_df['mean_session_counts'].fillna(0)
f_df['mean_session_time']=f_df['mean_session_time'].fillna(0)
f_df['median_session_counts']=f_df['median_session_counts'].fillna(0)
f_df['median_session_time']=f_df['median_session_time'].fillna(0)
f_df['num_red_predictions']=f_df['num_red_predictions'].fillna(0)
f_df['num_yellow_predictions']=f_df['num_yellow_predictions'].fillna(0)

## feature_store Features
f_df['gtv_15_day']=f_df['gtv_15_day'].fillna("other")
f_df['gtv_30_day']=f_df['gtv_30_day'].fillna(0)
f_df['number_failed_30days']=f_df['number_failed_30days'].fillna(0)
f_df['number_failed_15days']=f_df['number_failed_15days'].fillna(0)

f_df['city']=f_df['city'].fillna("other")
f_df['user_activity']=f_df['user_activity'].fillna(0)

f_df=f_df[f_df['upi_bank'].notna()]









# COMMAND ----------

f_df.isnull().sum()

# COMMAND ----------

### FILLNA GA features numerical with  0  and string feature with "other"
# 
nf_df['mobileDeviceBranding']=nf_df['mobileDeviceBranding'].fillna("other")
nf_df['latitude']=nf_df['latitude'].fillna(0)
nf_df['longitude']=nf_df['longitude'].fillna(0)
nf_df['session_counts']=nf_df['session_counts'].fillna(0)
nf_df['mean_session_counts']=nf_df['mean_session_counts'].fillna(0)
nf_df['mean_session_time']=nf_df['mean_session_time'].fillna(0)
nf_df['median_session_counts']=nf_df['median_session_counts'].fillna(0)
nf_df['median_session_time']=nf_df['median_session_time'].fillna(0)
nf_df['num_red_predictions']=nf_df['num_red_predictions'].fillna(0)
nf_df['num_yellow_predictions']=nf_df['num_yellow_predictions'].fillna(0)

## feature_store Features
nf_df['gtv_15_day']=nf_df['gtv_15_day'].fillna("other")
nf_df['gtv_30_day']=nf_df['gtv_30_day'].fillna(0)
nf_df['number_failed_30days']=nf_df['number_failed_30days'].fillna(0)
nf_df['number_failed_15days']=nf_df['number_failed_15days'].fillna(0)
nf_df['city']=nf_df['city'].fillna("other")
nf_df['user_activity']=nf_df['user_activity'].fillna(0)

nf_df=nf_df[nf_df['upi_bank'].notna()]


# COMMAND ----------

f_df.columns

# COMMAND ----------

# data = data.rename(columns={'tenant_age': 'tenant_age', 'gtv_15_days': 'gtv_of_successful_transaction_in_last_fifteen_days',
# 'gtv_30_days':'gtv_of_successful_transaction_in_last_thirty_days','city_y':'city_name','diff_location_30_days':'different_location_in_past_thirty_days','num_failed_transactions_30_days':'failed_attempt_in_last_thirty_days',
# 'num_failed_transactions_15_days':'failed_attempt_in_last_fifteen_days',
# 'num_successful_transactions_30_days':'number_of_success_attemp_30days',
# 'num_attempts_30_days':'number_attempt_30days',
# 'avg_num_attempts_30_days':'average_attempts_in_past_thirty_days',
# 'time_hour_ist':'hour_of_the_day',
# 'num_yellow_predictions':'yellow_count_in_last_thirty_days',
# 'num_red_predictions':'red_count_in_last_thirty_days'})


# COMMAND ----------

categorical_features=['platform','city','upi_bank','account_bank','mobileDeviceBranding']
ignore= ['different_location_30','time_hour','number_failed_30days','number_failed_15days','number_sucessfull_30days','num_attempts_30_days','avg_number_attemps_past30day','num_red_predictions','num_yellow_predictions']
numerical_features=['amount','tenant_age','gtv_15_day','gtv_30_day','diff','latitude','longitude','session_counts','mean_session_counts','mean_session_time','median_session_counts','median_session_time']

# COMMAND ----------

# categorical_features=['platform','city','upi_bank','account_bank','mobileDeviceBranding']
# ignore= ['different_location_30','time_hour','number_failed_30days','failed_attempt_in_last_fifteen_days','number_of_success_attemp_30days','number_attempt_30days','average_attempts_in_past_thirty_days','red_count_in_last_thirty_days','red_count_in_last_thirty_days']
# numerical_features=['amount','tenant_age','gtv_of_successful_transaction_in_last_fifteen_days','gtv_of_successful_transaction_in_last_fifteen_days','diff','latitude','longitude','session_counts','mean_session_counts','mean_session_time','median_session_counts','median_session_time']

# COMMAND ----------

nf_df.isnull().sum()

# COMMAND ----------

len(nf_df),len(f_df)

# COMMAND ----------



# COMMAND ----------

f_df.head()

# COMMAND ----------

f_df['ist_time']=pd.to_datetime(f_df['ist_time'])
nf_df['ist_time']=pd.to_datetime(nf_df['ist_time'])
f_df=f_df.query('ist_time < "2023-06-15"')
nf_df=nf_df.query('ist_time < "2023-06-15"')
f_testing = f_df.query('ist_time > "2023-04-15" and ist_time <="2023-06-15"')
nf_testing = nf_df.query('ist_time > "2023-04-15" and ist_time <="2023-06-15"')


# COMMAND ----------

nf_df_sample = nf_df.sample(n=len(f_df)*3, random_state=108)

# COMMAND ----------

# f_df=pd.concat([f_df,f_testing[:200]])

# COMMAND ----------

len(f_df)

# COMMAND ----------


f_df=f_df.drop(f_testing.index)
nf_df=nf_df.drop(nf_testing.index)

# COMMAND ----------

nf_df_sample.drop(['f_1','ist_time'],axis=1,inplace=True)
f_df.drop(['f_1','ist_time'],axis=1,inplace=True)

# COMMAND ----------

nf_df_sample['fraud']=[0]*len(nf_df_sample)
f_df['fraud']=[1]*len(f_df)
# cc['fraud']=[1]*len(cc)

# COMMAND ----------

train_data

# COMMAND ----------

train_data.head()

# COMMAND ----------

len(train_data.query('mobileDeviceBranding =="other" and latitude==0 and longitude==0 and session_counts==0 '))


# COMMAND ----------

train_data=pd.concat([nf_df_sample,f_df])

# COMMAND ----------

len(train_data)

# COMMAND ----------

len(train_data)-14294

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["amount", "avg_number_attemps_past30day", "diff", "different_location_30", "gtv_15_day", "gtv_30_day", "latitude", "longitude", "mean_session_counts", "mean_session_time", "num_attempts_30_days", "number_failed_15days", "number_sucessfull_30days", "tenant_age", "time_hour"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["mean_session_counts", "longitude", "diff", "number_sucessfull_30days", "avg_number_attemps_past30day", "mean_session_time", "time_hour", "amount", "tenant_age", "num_attempts_30_days", "gtv_30_day", "latitude", "gtv_15_day", "different_location_30", "number_failed_30days","number_failed_15days"])]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers + categorical_one_hot_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=1)

# COMMAND ----------

from sklearn.model_selection import train_test_split

target_col='fraud'
X = train_data.drop([target_col], axis=1)
Y = train_data[target_col]


# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["account_bank", "city", "mobileDeviceBranding", "platform", "upi_bank","user_activity"])]

# COMMAND ----------

len(X),len(Y)

# COMMAND ----------

#old paramters

# Cuurent Best

from xgboost import XGBClassifier

xgbc_classifier = XGBClassifier(
  colsample_bytree=0.2571106642575727,
  learning_rate=0.07118674126515383,
  max_depth=14,
  min_child_weight=8,
  n_estimators=451,
  n_jobs=100,
  subsample=0.6602075725462673,
  verbosity=0,
  random_state=627083932,
  reg_lambda=0.3,
  reg_alpha =0.3,
  gamma=0.3,
  
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgbc_classifier),
])


xgbc_classifier_1 = XGBClassifier(
 colsample_bytree=0.2571106642575727,
  learning_rate=0.07118674126515383,
  max_depth=10,
  min_child_weight=8,
  n_estimators=451,
  n_jobs=100,
  subsample=0.6602075725462673,
  verbosity=0,
  random_state=627083932,
  

)

model2=Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgbc_classifier_1),
])

# COMMAND ----------

X

# COMMAND ----------

X.drop(['f_1','ist_time'],axis=1, inplace=True)

# COMMAND ----------

len(X),len(Y)

# COMMAND ----------

mdl=model.fit(X,Y)
mdl1=model2.fit(X,Y)

# COMMAND ----------

testing_data = pd.concat([f_testing,nf_testing])

# COMMAND ----------

testing_data

# COMMAND ----------

y_true = testing_data['f_1']
testing_data.drop(['f_1','ist_time'],axis=1, inplace=True)

# COMMAND ----------

# from xgboost import XGBClassifier
# for 
# for n_estimators in [100,200,300,400,451,500]:
#     for  reg_lambda  in  [0.1,0.30]:
#         for reg_alpha in [0.1,0.2,0.3]:
#             for gamma in [0,0.1,0.2,0.3]:
#                 print("reg_lambda,reg_alpha" ,reg_lambda,reg_alpha,gamma)
#                 xgbc_classifier = XGBClassifier(
#                 colsample_bytree=0.2571106642575727,
#                 learning_rate=0.07118674126515383,
#                 max_depth=14,
#                 min_child_weight=8,
#                 n_estimators=451,
#                 n_jobs=100,
#                 subsample=0.6602075725462673,
#                 verbosity=0,
#                 reg_lambda=reg_lambda,
#                 reg_alpha=reg_alpha,
#                 gamma =gamma,
#                 random_state=627083932,

#                 )

#                 model = Pipeline([
#                     ("preprocessor", preprocessor),
#                     ("classifier", xgbc_classifier),
#                 ])
#                 mdl=model.fit(X,Y)
#                 y_pred =mdl.predict(testing_data)
#                 print(classification_report(y_true, y_pred))

# COMMAND ----------

testing_data

# COMMAND ----------

from sklearn.metrics import classification_report
y_pred =mdl.predict(testing_data)
print(classification_report(y_true, y_pred))

# COMMAND ----------

y_pred =mdl1.predict(testing_data)
print(classification_report(y_true, y_pred))

# COMMAND ----------

y_pred_proba =mdl.predict_proba(testing_data)


# COMMAND ----------

y =[]
for p in y_pred_proba:
    # print(p)
    a,b=p[0],p[1]
    if b>=0.75:
        y.append(1)
    else:
        y.append(0)

print(classification_report(y_true, y))


# COMMAND ----------

from sklearn.metrics import precision_recall_curve
import numpy as np
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

y_pred_proba=mdl.predict_proba(testing_data)
print(find_threshold_for_max_recall(0.65, y_true, y_pred_proba))

# COMMAND ----------

# y_pred_proba
y_true_new_threshold =[]
for p in y_pred_proba:
    # print(p)
    a,b=p[0],p[1]
    if b>=0.603439:
        y_true_new_threshold.append(1)
    else:
        y_true_new_threshold.append(0)


    # print(b)

# COMMAND ----------



# COMMAND ----------

# y_pred_threshold = (np.max(y_pred_proba, axis=1) >= 0.43).astype(int)  # Predicted classes with threshold

print(classification_report(y_true, y_true_new_threshold))
# print(report)

# COMMAND ----------

testing_data.head()

# COMMAND ----------

len(testing_data)

# COMMAND ----------

testing_data = pd.concat([f_testing,nf_testing])

# COMMAND ----------

new_user = testing_data.query('mobileDeviceBranding =="other" and latitude==0 and longitude==0 and session_counts==0 ')


# COMMAND ----------

new_user.head()

# COMMAND ----------

old_user =testing_data.drop(new_user.index)

# COMMAND ----------

old_user.head()

# COMMAND ----------

old_user_true= old_user['f_1']

# COMMAND ----------

old_user.drop(['f_1','ist_time'],axis=1,inplace=True)

# COMMAND ----------

old_predict_proba=mdl.predict_proba(old_user)
old_predict = mdl.predict(old_user)


# COMMAND ----------

# print(find_threshold_for_max_recall(0.65, old_user_true, old_predict_proba))

# COMMAND ----------

y =[]
for p in old_predict_proba:
    # print(p)
    a,b=p[0],p[1]
    if b>=0.75:
        y.append(1)
    else:
        y.append(0)

print(classification_report(old_user_true, y))



# COMMAND ----------

new_user.head()

# COMMAND ----------

new_user_true=new_user['f_1']
new_user.drop(['f_1','ist_time'],axis=1, inplace=True)

# COMMAND ----------

new_user.head()

# COMMAND ----------

new_user_proba=mdl.predict_proba(new_user)
# old_predict = mdl.predict(old_user)
# print(find_threshold_for_max_recall(0.65, new_user_true, new_user_proba))

# COMMAND ----------

y =[]
for p in new_user_proba:
    # print(p)
    a,b=p[0],p[1]
    if b>=0.75:
        y.append(1)
    else:
        y.append(0)

print(classification_report(new_user_true, y))


# COMMAND ----------

### FILLNA GA features numerical with  0  and string feature with "other"
# 
new_user['mobileDeviceBranding']=new_user['mobileDeviceBranding'].fillna("other")
new_user['latitude']=new_user['latitude'].fillna(0)
new_user['longitude']=new_user['longitude'].fillna(0)
new_user['session_counts']=new_user['session_counts'].fillna(0)
new_user['mean_session_counts']=new_user['mean_session_counts'].fillna(0)
new_user['mean_session_time']=new_user['mean_session_time'].fillna(0)
new_user['median_session_counts']=new_user['median_session_counts'].fillna(0)
new_user['median_session_time']=new_user['median_session_time'].fillna(0)
new_user['num_red_predictions']=new_user['num_red_predictions'].fillna(0)
new_user['num_yellow_predictions']=new_user['num_yellow_predictions'].fillna(0)

## feature_store Features
new_user['gtv_15_day']=new_user['gtv_15_day'].fillna("other")
new_user['gtv_30_day']=new_user['gtv_30_day'].fillna(0)
new_user['number_failed_30days']=new_user['number_failed_30days'].fillna(0)
new_user['number_failed_15days']=new_user['number_failed_15days'].fillna(0)
new_user['city']=new_user['city'].fillna("other")
new_user['user_activity']=new_user['user_activity'].fillna(0)

new_user=new_user[new_user['upi_bank'].notna()]


# COMMAND ----------

old_user.head()

# COMMAND ----------

new_user_true = new_user['f_1']


# COMMAND ----------

new_user.drop(['f_1','ist_time'],axis=1, inplace=True)

# COMMAND ----------

y_pred_prob=mdl.predict_proba(new_user)

y =[]
for p in y_pred_prob:
    # print(p)
    a,b=p[0],p[1]
    if b>0.70190954:
        y.append(1)
    else:
        y.append(0)

# COMMAND ----------

from sklearn.metrics import classification_report
print(classification_report(new_user_true, y))

# COMMAND ----------

# y_prob=mdl.predict_proba(old_user)

y_true_oldnew_threshold =[]
for p in y_prob:
    # print(p)
    a,b=p[0],p[1]
    if b>=0.65:
        y_true_oldnew_threshold.append(1)
    else:
        y_true_oldnew_threshold.append(0)


# COMMAND ----------

from sklearn.metrics import classification_report
print(classification_report(old_user_true, y_true_oldnew_threshold))

# COMMAND ----------

y_prob_new =  mdl.predict_proba(new_user)

# COMMAND ----------

print(find_threshold_for_max_recall(0.65, new_user_true,y_prob_new))

# COMMAND ----------

####View data
old_data =pd.read_csv("/dbfs/FileStore/Badal/rent_pay/v4_new_data.csv")
new_data = pd.read_csv("/dbfs/FileStore/Badal/rent_pay/v4_new_data_1.csv")


# COMMAND ----------

old_data.f_1.value_counts()

# COMMAND ----------

new_data.f_1.value_counts()

# COMMAND ----------


