# Decisions Log: Iowa Retail Liquor Sales Forecasting

| # | Decision | Rationale |
|---|---|---|
| 1 | Restrict analysis to 2017-01-01 onward | Category taxonomy changed around 2016-2017 (71-96 distinct categories pre-2017 vs. a consistent 44-48 post-2017) -- using pre-2017 data would mix incompatible category definitions |
| 2 | Exclude rows with sale_dollars < 0 (returns/corrections) | Only 0.032% of rows (10,734 of 34M) -- negligible impact, keeps target variable interpretable as non-negative sales volume |
| 3 | Exclude rows with category_name IS NULL | Small number of unlabeled transactions in the raw data; can't be assigned to a category |
| 4 | Forecasting granularity: statewide weekly totals + top-5 categories by revenue (American Vodkas, Canadian Whiskies, Straight Bourbon Whiskies, Whiskey Liqueur, 100% Agave Tequila), with everything else grouped as ALL_OTHER | Weekly balances noise reduction with enough data points (487 weeks) for seasonality modeling; per-store/per-county forecasting was ruled out as unnecessary scope creep for this project's goals |
| 5 | Investigated Oct 2022 spike ($11.67M, single highest week in the series) before treating as an anomaly | Confirmed the second week of October is elevated every year 2017-2025, broad-based across categories -- a real recurring signal, not a data error. Kept in the modeling data rather than removed. |
| 6 | Noted 2025 sales declined year-over-year for the first time in the series ($424M vs $451M in 2024) | Real finding that should inform model design -- growth should not be assumed to continue indefinitely; relevant for the final business write-up |
