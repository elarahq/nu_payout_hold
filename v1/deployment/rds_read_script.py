# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import MapType, StringType,StructType,StructField,IntegerType
from functools import reduce
from pyspark.sql.window import *
import datetime
import pyspark.sql.functions as F

import random
import psycopg2

RDS_URL = 'jdbc:postgresql://ml-recommendation-db.cfgklq1b0ixg.ap-southeast-1.rds.amazonaws.com:5432/mlrecommendationdb'

# COMMAND ----------

# MAGIC %md
# MAGIC ### WEB

# COMMAND ----------

# RDS pull time from previous_online_run(success) till current_run_time 
realtime_df_web = (spark.read.format("jdbc")
                   .option("driver", "org.postgresql.Driver")
                   .option("url", RDS_URL)
                   .option("query", 
                           """
                           select
                           *                                    
                           from public.housing_demand_events_web
                           where pk_id <= (
                                select max(pk_id) from public.housing_demand_events_web
                                )
                            and dimension49 = 'GA1.1.2046831729.1652445654'
                           """)
                   .option("user","root")
                   .option("password", "ml#housing*")
                   .option("numPartitions",64).load()) #64 is optimal

# realtime_df_web.count()

# COMMAND ----------

display(realtime_df_web)

# COMMAND ----------

result = realtime_df_web.filter(col('uid')==2033514486)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### APP

# COMMAND ----------

# (Replace(event_label, '\', '')::json->>'orderId')::bigint AS order_id

# RDS pull time from previous_online_run(success) till current_run_time 
realtime_df_app = (spark.read.format("jdbc")
                   .option("driver", "org.postgresql.Driver")
                   .option("url", RDS_URL)
                   .option("query", 
                           """
                           select
                           *                                        
                           from public.housing_demand_events_app
                           where pk_id <= (
                                select max(pk_id) from public.housing_demand_events_app
                                )
                            and uid = 'd7b9da0fb4e95d9f'
                            order by timestamp asc
                           """)
                   .option("user","root")
                   .option("password", "ml#housing*")
                   .option("numPartitions",64).load()) #64 is optimal

# realtime_df_web.count()

# COMMAND ----------

display(realtime_df_app)

# COMMAND ----------


