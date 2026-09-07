# Iowa Retail Liquor Sales Forecasting

A time series forecasting capstone using BigQuery for large-scale SQL aggregation (34M+ raw
transactions) and Python for modeling, evaluation, and diagnosis.

## The problem

Using 9+ years of Iowa liquor sales data, this project forecasts statewide weekly sales and
top-category sales trends, aimed at the kind of demand-planning problem central to retail
operations (e.g., Amazon inventory planning, Apple retail, Netflix content demand).

## Results

Six forecasting approaches were evaluated on an identical 52-week holdout (the last full year
of data, held out from every model in every stage of tuning):

| Model | Test MAPE | Notes |
|---|---|---|
| **SARIMA(1,1,2)(1,1,1,52)** | **6.61%** | Winner. Order selected via AIC on training data only |
| Prophet | 6.72% | `changepoint_prior_scale` tuned via a validation split, not the test set |
| Seasonal Naive | 8.29% | The benchmark to beat -- confirms exploitable yearly seasonality |
| Naive / ML (gradient boosting) | 8.44% | ML model tied Naive statewide, but is the only model with category-level forecasts |
| Drift | 9.73% | Worst performer -- extrapolates growth that reversed in 2025 |

**Recommendation:** SARIMA for statewide forecasting; the ML model for category-level forecasts,
where SARIMA/Prophet were never extended. Full reasoning and the two known failure modes
(a recurring Memorial Day-July 4th swing SARIMA can't phase-align to, and category difficulty
that only tracks intrinsic volatility at the extremes) are written up in
`docs/business_memo.md`.

## What's in this repo

| File | What it covers |
|---|---|
| `docs/business_memo.md` | Stakeholder-facing summary: recommendation, business impact, known limitations |
| `docs/decisions_log.md` | Every judgment call made in this project, with reasoning (16 entries) |
| `sql/weekly_aggregation.sql` | The BigQuery SQL that aggregates 34M raw transactions into the weekly modeling dataset |
| `notebooks/01_data_exploration.ipynb` | Data validation, trend/seasonality analysis, anomaly investigation |
| `notebooks/02_baseline_models.ipynb` | Time-based train/test split; Naive, Seasonal Naive, and Drift baselines |
| `notebooks/03_sarima.ipynb` | SARIMA order selection via AIC, final model, evaluation |
| `notebooks/04_prophet.ipynb` | Prophet with validation-split changepoint tuning (avoids test-set leakage) |
| `notebooks/05_ml_model.ipynb` | Global gradient boosting model across statewide + 5 top categories, with genuine recursive multi-step forecasting |
| `notebooks/06_diagnosis.ipynb` | Where SARIMA (the winning model) actually struggles, and why |
| `build_dashboard.py` | Builds `dashboard.html`, a self-contained interactive summary of all of the above |

## Data

This project uses Google BigQuery's public `bigquery-public-data.iowa_liquor_sales.sales`
dataset (34M+ transactions, 2012-2026). The aggregation query in `sql/weekly_aggregation.sql`
can be run directly in BigQuery (free tier) to reproduce `data/raw/weekly_sales_by_category.csv`
-- 487 weeks (2017-2026) x statewide + top-5 categories by revenue (American Vodkas, Canadian
Whiskies, Straight Bourbon Whiskies, Whiskey Liqueur, 100% Agave Tequila), everything else
grouped as `ALL_OTHER`.

## Key findings

- Strong growth 2017-2024, but 2025 saw the first year-over-year decline in the series ($424M
  vs. $451M in 2024) -- the reversal every model had to represent, not just extrapolate
- The second week of October is reliably one of the strongest weeks every year -- confirmed as
  a real recurring pattern, not a data anomaly, after direct investigation
- SARIMA and Prophet both clear the Seasonal Naive benchmark by a real margin; the ML model,
  despite validation-based tuning, lands just short of it -- explicit statistical structure
  outperformed a feature-based approach on this single, moderately-sized weekly series
- SARIMA's forecast error shows no correlation with how far out the forecast reaches
  (r=-0.028), even though its confidence interval mechanically widens with horizon (r=0.992) --
  the CI band is not a reliable guide to where the model actually struggles
- A violent week-over-week sales swing recurs every year in the Memorial Day-July 4th window.
  SARIMA's fixed 52-week seasonal lag can't phase-align to it, since Memorial Day falls on a
  different calendar week each year -- the single largest identified weakness in the winning model
- Category-level forecast difficulty tracks intrinsic volatility only at the extremes (lowest-CV
  Whiskey Liqueur is easiest to forecast at 8.94% MAPE, highest-CV Tequila is hardest at 16.96%),
  but the three middle categories don't rank consistently by volatility alone

## Setup

```bash
conda create -n retailforecast python=3.9 -y
conda activate retailforecast
pip install -r requirements.txt
python -m notebook
```

To rebuild the dashboard after running the notebooks:

```bash
python build_dashboard.py
open dashboard.html
```
