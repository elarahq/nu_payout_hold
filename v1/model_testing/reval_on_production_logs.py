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

PAYOUT_THRESHOLD = 0.004

# COMMAND ----------

# query = "SELECT * FROM data_science_metastore.nu_payout_production_tables.nu_v1_logs"

# df = spark.sql(query).drop('source')

# COMMAND ----------

query = """select NU.*, F.fraud
from data_science_metastore.nu_payout_production_tables.nu_v1_logs NU
INNER JOIN etl.pay_rent AS F ON F.order_id = NU.order_id
WHERE date(log_timestamp) <= '2023-10-15'
"""

df = spark.sql(query).drop('source')

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

df.shape

# COMMAND ----------

def determine_payout(probability):
    if probability < PAYOUT_THRESHOLD:
        return 'GREEN'
    else:
        return 'RED'

# COMMAND ----------

# logged_model = 'runs:/46ff8b243aa945609f2504ceca1a8cba/rus_knn_1:4_PROD_v1'
# logged_model = 'runs:/9972efa5222c49d597a4cdc10480e5a7/rus_knn_1:4_PROD_v1.0.1'
# logged_model = 'runs:/a45c866936c847f68e3504bb5f0c2b48/model' # NEW MODEL - TRAINED TILL OCT
# logged_model = 'runs:/d9c5b16306c64a3dabf9df0152b6b6ac/model'
# logged_model = 'runs:/8d5b92cba50f45149f7b98ea2815a06d/model' # FRAUD NOV 2022 to OCT 2023, NONFRAUD FEB 2023 to OCT 2023
# logged_model = 'runs:/2fe3c04d9c0849f8bf72e7b6729095ee/model' # NONFRAUD FEB 2023 to OCT 2023
# logged_model = 'runs:/106efbbc74444335b2ac9227dbbd619f/model' # SAME AS V1
logged_model = 'runs:/bee5271af8ef464597648130b3e7633a/model' # SAMPLED 1x from Sept2023, 3x from Jan-Aug2023
pipeline = mlflow.sklearn.load_model(logged_model)

df['ml_probability'] = pipeline.predict_proba(df)[:,-1]
df['payout_decision'] = df['ml_probability'].apply(determine_payout)

# COMMAND ----------

df['payout_decision'].value_counts(normalize=True) * 100

# COMMAND ----------

y_pred_proba = df[['ml_probability']]
y_test = df[['fraud']]

# COMMAND ----------

predictions = (y_pred_proba >= 0.004).astype(int)

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


