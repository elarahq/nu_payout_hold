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

def fetch_data():
    nonfraud_train_query = f"""\
    SELECT * FROM feature_store.nu_v1_model_data
    WHERE fraud = 0
    AND status = 'PAYOUT_SUCCESSFUL'
    AND date(created_at) BETWEEN '2023-01-01' AND '2023-06-15'
    """

    nonfraud_test_query = f"""\
    SELECT * FROM feature_store.nu_v1_model_data
    WHERE fraud = 0
    AND status = 'PAYOUT_SUCCESSFUL'
    AND date(created_at) BETWEEN '2023-06-16' AND '2023-07-31'
    """

    fraud_train_query = f"""\
    SELECT * FROM feature_store.nu_v1_model_data
    WHERE fraud = 1
    AND date(created_at) BETWEEN '2022-10-01' AND '2023-06-15'
    """

    fraud_test_query = f"""\
    SELECT * FROM feature_store.nu_v1_model_data
    WHERE fraud = 1
    AND date(created_at) BETWEEN '2023-06-16' AND '2023-07-31'
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

logged_model = 'runs:/365b9609a2314ec899b76f3a13b4c5ec/rus_knn_1:4_JUNE_16_v4_COMPARISON'
pipeline = mlflow.sklearn.load_model(logged_model)

y_pred_proba = pipeline.predict_proba(X_test)

# COMMAND ----------

predictions = (y_pred_proba[:,-1] >= 0.003).astype(int)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true = y_test, y_pred = predictions))

cm = confusion_matrix(y_true = y_test, y_pred = predictions)
print(cm)

from sklearn.metrics import precision_score
print(f"{precision_score(y_test, predictions, pos_label=0, average='binary') * 100}%")

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
disp.plot(cmap=plt.cm.Blues, values_format='d')

# COMMAND ----------


