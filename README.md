# nba-stat-predictor
Performance forecasting model using Databricks and Spark MLlib

Overview
This project builds a forecasting model using 25 years of NBA player data (2000–2025) to predict scoring output (PPG) using opportunity and role-based performance metrics.

Objective
To identify key performance drivers influencing player scoring and evaluate predictive accuracy using Spark MLlib.

Technologies
- Databricks
- PySpark
- Spark MLlib
- SQL

Methodology
- Data ingestion into Delta tables
- Feature engineering (per-game normalization)
- Removal of data leakage features
- Linear regression modeling
- Model evaluation using R², RMSE, MAE
- Coefficient interpretation

Results
- R² ≈ 0.87
- Shot volume (FGA_pg) was the strongest scoring predictor
- Role indicators (blocks, turnovers) also influenced output

Key Insight
Scoring output is strongly driven by usage-based metrics, while positional indicators influence scoring patterns across player types.
