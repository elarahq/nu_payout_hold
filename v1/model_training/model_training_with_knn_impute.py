# Databricks notebook source
import numpy as np
import pandas as pd
# from sklearn.cluster import DBSCAN
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
from imblearn.under_sampling import RandomUnderSampler
from difflib import SequenceMatcher

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

def is_self_payout(x):
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

df = spark.sql(
    """
    SELECT * FROM data_science_metastore.nu_payout_production_tables.nu_v1_training_data
    """
)
df.createOrReplaceTempView("NU")

# COMMAND ----------

def fetch_data_NU():
    nonfraud_train_query = f"""\
    SELECT * FROM NU
    WHERE fraud = 0
    AND status = 'PAYOUT_SUCCESSFUL'
    AND date(created_at) BETWEEN '2023-01-01' AND '2023-09-30'
    """

    nonfraud_test_query = f"""\
    SELECT * FROM NU
    WHERE fraud = 0
    AND status = 'PAYOUT_SUCCESSFUL'
    AND date(created_at) BETWEEN '2023-10-01' AND '2023-10-15'
    """

    fraud_train_query = f"""\
    SELECT * FROM NU
    WHERE fraud = 1
    AND date(created_at) BETWEEN '2022-10-01' AND '2023-09-30'
    """

    fraud_test_query = f"""\
    SELECT * FROM NU
    WHERE fraud = 1
    AND date(created_at) BETWEEN '2023-10-01' AND '2023-10-15'
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

    X_train['is_self_payout'] = X_train[['tenant_name', 'landlord_name']].apply(is_self_payout, axis=1)
    X_test['is_self_payout'] = X_test[['tenant_name', 'landlord_name']].apply(is_self_payout, axis=1)
    
    return X_train, X_test

# COMMAND ----------

X_train, X_test = fetch_data_NU()

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
    # 'is_owner',
    # 'is_self_payout',
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
    'seconds_since_first_session',
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
    'seconds_since_first_session',
    'seconds_after_transaction',
    'hits_after_transaction',
    'seconds_on_payment_gateway'
]

categorical_columns = [
    'platform',
    'poc_category',
    'city',
    # 'is_owner',
    # 'is_self_payout',
    'bank',
    'profile_picture_url',
    'referral_code',
    'time_hour_ist',
    'traffic_sourcemedium'
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### RUS + SKLEARN PIPELINE

# COMMAND ----------

fraud_count = len(y_train[y_train['fraud'] == 1])
print(f"FRAUD ORDERS :: {fraud_count}")
sampling_ratio = {0: 4 * fraud_count}

# COMMAND ----------

rus = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=42)

X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# COMMAND ----------

X_train_rus.shape

# COMMAND ----------

y_train_rus['fraud'].value_counts()

# COMMAND ----------

preprocessor = ColumnTransformer(transformers = [
  ('log_transformer', PowerTransformer(method = 'yeo-johnson'), numerical_columns),
  ('encoder', OneHotEncoder(handle_unknown="ignore"), categorical_columns)
  ], remainder="passthrough", sparse_threshold=0)

pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('imputer', KNNImputer(n_neighbors=2)),
      ('ml_model', XGBClassifier(random_state = 1))
    ])

# COMMAND ----------

_ = pipeline.fit(X_train_rus, y_train_rus)

# COMMAND ----------

# import mlflow
# # logged_model = 'runs:/29bae76f322b451ea54680af6a3c6791/rus_knn_1:4' ## FIRST MODEL THAT WAS TRAINED
# # logged_model = 'runs:/46ff8b243aa945609f2504ceca1a8cba/rus_knn_1:4_PROD_v1' ## CONTENDER TO GO LIVE
# logged_model = 'runs:/46ff8b243aa945609f2504ceca1a8cba/rus_knn_1:4_PROD_v1'

# # Load model as a PyFuncModel.
# pipeline = mlflow.sklearn.load_model(logged_model)

# COMMAND ----------

# X_test = X_test.rename(columns={
#     'seconds_since_first_session' : 'hours_since_first_session'
# })

X_test = X_test.rename(columns={
    'hours_since_first_session' : 'seconds_since_first_session'
})

# COMMAND ----------

y_pred_proba = pipeline.predict_proba(X_test)

# COMMAND ----------

predictions = (y_pred_proba[:,-1] >= 0.004).astype(int)

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

mlflow.sklearn.log_model(pipeline, "rus_knn_1:4_PROD_v1")

# COMMAND ----------


