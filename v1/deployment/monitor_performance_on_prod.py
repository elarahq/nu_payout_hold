# Databricks notebook source
import numpy as np
import pandas as pd


from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, precision_recall_curve, PrecisionRecallDisplay

import mlflow

# import shap

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# COMMAND ----------

query = """select NU.* except(source), F.fraud
from data_science_metastore.nu_payout_production_tables.nu_v1_logs NU
INNER JOIN etl.pay_rent AS F ON F.order_id = NU.order_id
WHERE date(NU.log_timestamp) <= '2023-10-30'
"""

df = spark.sql(query)

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

len(df)

# COMMAND ----------

y_pred_proba = df[['ml_probability']]
y_test = df[['fraud']]

# COMMAND ----------

predictions = (y_pred_proba >= 0.01).astype(int)

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


