# Databricks notebook source
import numpy as np
import pandas as pd
# from sklearn.cluster import DBSCAN
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
from imblearn.under_sampling import RandomUnderSampler

import category_encoders as ce

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, precision_recall_curve, PrecisionRecallDisplay

import mlflow
from mlflow.utils.environment import _mlflow_conda_env

# import shap

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# COMMAND ----------

# data_query = f"""\
# SELECT * FROM feature_store.nu_v1_model_data
# """
# df = spark.sql(data_query).toPandas()

# # df = pd.read_csv('/dbfs/FileStore/Kajal/rent_pay/v4_data_increased.csv',low_memory=False)

# COMMAND ----------

def fetch_data():
    nonfraud_train_query = f"""\
    SELECT * FROM feature_store.nu_v1_model_data
    WHERE fraud = 0
    AND status = 'PAYOUT_SUCCESSFUL'
    AND date(created_at) BETWEEN '2023-01-01' AND '2023-04-30'
    """

    nonfraud_test_query = f"""\
    SELECT * FROM feature_store.nu_v1_model_data
    WHERE fraud = 0
    AND status = 'PAYOUT_SUCCESSFUL'
    AND date(created_at) BETWEEN '2023-05-01' AND '2023-07-31'
    """

    fraud_train_query = f"""\
    SELECT * FROM feature_store.nu_v1_model_data
    WHERE fraud = 1
    AND date(created_at) BETWEEN '2022-10-01' AND '2023-04-30'
    """

    fraud_test_query = f"""\
    SELECT * FROM feature_store.nu_v1_model_data
    WHERE fraud = 1
    AND date(created_at) BETWEEN '2023-05-01' AND '2023-07-31'
    """

    nonfraud_train = spark.sql(nonfraud_train_query).toPandas()
    nonfraud_test = spark.sql(nonfraud_test_query).toPandas()

    fraud_train = spark.sql(fraud_train_query).toPandas()
    fraud_test = spark.sql(fraud_test_query).toPandas()

    X_train = pd.concat([nonfraud_train, fraud_train])
    X_test = pd.concat([nonfraud_test, fraud_test])

    X_train = X_train.dropna(subset=['ga_id', 'session_id'])
    X_test = X_test.dropna(subset=['ga_id', 'session_id'])
    
    X_train['bank'] = X_train['upi_bank'].fillna(X_train['account_bank'])
    X_test['bank'] = X_test['upi_bank'].fillna(X_test['account_bank'])
    
    return X_train, X_test

# COMMAND ----------

X_train, X_test = fetch_data()

# COMMAND ----------

print(f"TRAINING DATA :: {X_train.shape[0]}")
print(f"TESTING DATA :: {X_test.shape[0]}")

# COMMAND ----------

X_train['fraud'].value_counts()

# COMMAND ----------

X_test['fraud'].value_counts()

# COMMAND ----------

SELECTED_FEATURES = [
    'amount',
    'platform',
    'poc_category',
    'tenant_age_in_seconds',
    'city',
    'number_of_failed_transactions',
    'seconds_since_last_transaction',
    'average_seconds_between_two_transactions',
    'average_number_of_transactions',
    'time_hour_ist',
    'bank',
    'profile_picture_url',
    'referral_code',
    'number_of_leads',
    'number_of_sessions',
    'session_time',
    'number_of_hits',
    'hit_number',
    'number_of_non_poc_actions',
    'number_of_non_poc_sessions',
    'traffic_sourcemedium',
    'hours_since_first_session',
    'seconds_after_transaction',
    'hits_after_transaction',
    'seconds_on_payment_gateway'
]
y_train = X_train[['fraud']]
y_test = X_test[['fraud']]

X_train = X_train[SELECTED_FEATURES]
X_test = X_test[SELECTED_FEATURES]

# COMMAND ----------

numerical_columns = [
    'amount',
    'tenant_age_in_seconds',
    'number_of_failed_transactions',
    'seconds_since_last_transaction',
    'average_seconds_between_two_transactions',
    'average_number_of_transactions',
    'number_of_leads',
    'number_of_sessions',
    'session_time',
    'number_of_hits',
    'hit_number',
    'number_of_non_poc_actions',
    'number_of_non_poc_sessions',
    'hours_since_first_session',
    'seconds_after_transaction',
    'hits_after_transaction',
    'seconds_on_payment_gateway'
]

categorical_columns = [
    'platform',
    'poc_category',
    'city',
    'bank',
    'profile_picture_url',
    'referral_code',
    'time_hour_ist',
    'traffic_sourcemedium'
]

# COMMAND ----------

X_train['seconds_since_last_transaction'] = X_train['seconds_since_last_transaction'].fillna(0)
X_train['average_seconds_between_two_transactions'] = X_train['average_seconds_between_two_transactions'].fillna(0)
X_train['average_number_of_transactions'] = X_train['average_number_of_transactions'].fillna(0)
X_train['number_of_leads'] = X_train['number_of_leads'].fillna(0)
X_train['traffic_sourcemedium'] = X_train['traffic_sourcemedium'].fillna("other")
X_train['seconds_on_payment_gateway'] = X_train['seconds_on_payment_gateway'].fillna(0)

# COMMAND ----------

X_test['seconds_since_last_transaction'] = X_test['seconds_since_last_transaction'].fillna(0)
X_test['average_seconds_between_two_transactions'] = X_test['average_seconds_between_two_transactions'].fillna(0)
X_test['average_number_of_transactions'] = X_test['average_number_of_transactions'].fillna(0)
X_test['number_of_leads'] = X_test['number_of_leads'].fillna(0)
X_test['traffic_sourcemedium'] = X_test['traffic_sourcemedium'].fillna("other")
X_test['seconds_on_payment_gateway'] = X_test['seconds_on_payment_gateway'].fillna(0)

# COMMAND ----------

def scale_yeo_johnson(df, to_scale):
  yeo = PowerTransformer(method = 'yeo-johnson')
  _ = yeo.fit(df[to_scale])

  return yeo, pd.DataFrame(yeo.transform(df[to_scale]), columns=to_scale)

def encode_one_hot(df, columns):
  ohe = ce.OneHotEncoder(cols = columns, return_df = True)
  return ohe, ohe.fit_transform(df)

def get_feature_transformation_test(X_test, yeo, to_yeo, ohe, to_one_hot):
  X_test[to_yeo] = pd.DataFrame(yeo.transform(X_test[to_yeo]), columns=to_yeo)
  X_test = ohe.transform(X_test)
  
  return X_test


yeo, X_train[numerical_columns] = scale_yeo_johnson(X_train, numerical_columns)
ohe, X_train = encode_one_hot(X_train, categorical_columns)

X_test = get_feature_transformation_test(X_test, yeo, numerical_columns, ohe, categorical_columns)

# COMMAND ----------



# COMMAND ----------

fraud_count = len(y_train[y_train['fraud'] == 1])
print(f"FRAUD ORDERS :: {fraud_count}")
sampling_ratio = {0: 4 * fraud_count}

# COMMAND ----------

rus = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=42)
# rus = RandomUnderSampler(sampling_strategy="majority", random_state=42)

X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# COMMAND ----------

X_train_rus.shape

# COMMAND ----------

y_train_rus['fraud'].value_counts()

# COMMAND ----------



# COMMAND ----------

model = XGBClassifier(
    random_state = 42
    )
model.fit(X_train_rus, y_train_rus.values.ravel())

# COMMAND ----------

y_pred_proba = model.predict_proba(X_test)

# COMMAND ----------

# actuals = y_train
actuals = y_test

predictions = (y_pred_proba[:,-1] >= 0.015).astype(int)

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


