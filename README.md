# Iowa Retail Liquor Sales Forecasting

A time series forecasting capstone using BigQuery for large-scale SQL aggregation (34M+ raw
transactions) and Python for modeling, evaluation, and diagnosis.

## The problem

Using 9+ years of Iowa liquor sales data, this project forecasts statewide weekly sales and
top-category sales trends, aimed at the kind of demand-planning problem central to retail
operations (e.g., Amazon inventory planning, Apple retail, Netflix content demand).

## What's in this repo

| File | What it covers |
|---|---|
| `docs/decisions_log.md` | Every judgment call made in this project, with reasoning |
| `sql/weekly_aggregation.sql` | The BigQuery SQL that aggregates 34M raw transactions into the weekly modeling dataset |
| `notebooks/01_data_exploration.ipynb` | Data validation, trend/seasonality analysis, anomaly investigation |
| `notebooks/02_baseline_models.ipynb` | Time-based train/test split; Naive, Seasonal Naive, and Drift baseline forecasts with MAPE/RMSE/MAE evaluation |

*(More notebooks to come: SARIMA, Prophet, and ML-based forecasting models, evaluated against the same holdout, followed by a final diagnosis/write-up)*

## Data

This project uses Google BigQuery's public `bigquery-public-data.iowa_liquor_sales.sales`
dataset (34M+ transactions, 2012-2026). The aggregation query in `sql/weekly_aggregation.sql`
can be run directly in BigQuery (free tier) to reproduce `data/raw/weekly_sales_by_category.csv`.

## Key findings so far

- Strong growth 2017-2024 (~44% increase), but 2025 saw the first year-over-year decline
- Moderate seasonality: ~39% swing between the strongest (December) and weakest (January) months
- The second week of October is reliably one of the strongest weeks every year -- confirmed as
  a real recurring pattern, not a data anomaly, after direct investigation
- Baseline forecasts (52-week holdout): Seasonal Naive is the benchmark to beat at 8.29% MAPE
  (vs. 8.44% for Naive, 9.73% for Drift). Drift performing worst is a signal in itself -- it
  assumes growth continues, which contradicts the 2025 downturn above

## Setup

```bash
conda create -n retailforecast python=3.9 -y
conda activate retailforecast
pip install -r requirements.txt
python -m notebook
```
