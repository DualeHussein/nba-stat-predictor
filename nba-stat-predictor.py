# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

df = spark.table("default.nba_player_stats_and_salaries_2000_2025")

df = df.filter(
    (F.col("G") > 0) &
    (F.col("MP") > 0) &
    (F.col("PTS") > 0)
)

df.count()

# COMMAND ----------

df = df.withColumn("PPG", F.col("PTS") / F.col("G"))

df.select("PPG").show(5)

# COMMAND ----------

df = df.withColumn("FG_pg", F.col("FG") / F.col("G"))
df = df.withColumn("FGA_pg", F.col("FGA") / F.col("G"))
df = df.withColumn("3P_pg", F.col("3P") / F.col("G"))
df = df.withColumn("3PA_pg", F.col("3PA") / F.col("G"))
df = df.withColumn("FT_pg", F.col("FT") / F.col("G"))
df = df.withColumn("FTA_pg", F.col("FTA") / F.col("G"))
df = df.withColumn("TRB_pg", F.col("TRB") / F.col("G"))
df = df.withColumn("AST_pg", F.col("AST") / F.col("G"))
df = df.withColumn("STL_pg", F.col("STL") / F.col("G"))
df = df.withColumn("BLK_pg", F.col("BLK") / F.col("G"))
df = df.withColumn("TOV_pg", F.col("TOV") / F.col("G"))

# COMMAND ----------

feature_cols = [
    "Age",
    "MP",
    "FGA_pg",
    "3PA_pg",
    "FTA_pg",
    "TRB_pg",
    "AST_pg",
    "STL_pg",
    "BLK_pg",
    "TOV_pg"
]

# COMMAND ----------

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

lr = LinearRegression(
    featuresCol="features",
    labelCol="PPG"
)

pipeline = Pipeline(stages=[assembler, lr])

# COMMAND ----------

train, test = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train)

predictions = model.transform(test)

# COMMAND ----------

evaluator = RegressionEvaluator(
    labelCol="PPG",
    predictionCol="prediction",
    metricName="r2"
)

r2 = evaluator.evaluate(predictions)

print("R2:", r2)

# COMMAND ----------

metrics = ["r2", "rmse", "mae"]

for metric in metrics:
    evaluator = RegressionEvaluator(
        labelCol="PPG",
        predictionCol="prediction",
        metricName=metric
    )
    print(metric, evaluator.evaluate(predictions))

# COMMAND ----------

lr_model = model.stages[-1]

for feature, coef in zip(feature_cols, lr_model.coefficients):
    print(feature, coef)

# COMMAND ----------

display(
    predictions.select("PPG", "prediction")
)

# COMMAND ----------

predictions = predictions.withColumn(
    "residual",
    F.col("PPG") - F.col("prediction")
)

display(
    predictions.select("prediction", "residual")
)

# COMMAND ----------

import pandas as pd

coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)
display(coef_df)

# COMMAND ----------

