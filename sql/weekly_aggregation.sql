-- Aggregates raw Iowa liquor transaction data (34M+ rows) into weekly totals,
-- split by top-5 category by revenue, with everything else grouped as ALL_OTHER.
-- Run directly in BigQuery against bigquery-public-data.iowa_liquor_sales.sales

SELECT
  DATE_TRUNC(date, WEEK(MONDAY)) AS week_start,
  CASE
    WHEN category_name IN ('AMERICAN VODKAS', 'CANADIAN WHISKIES', 'STRAIGHT BOURBON WHISKIES', 'WHISKEY LIQUEUR', '100% AGAVE TEQUILA')
    THEN category_name
    ELSE 'ALL_OTHER'
  END AS category_group,
  ROUND(SUM(sale_dollars), 2) AS total_sales_dollars,
  SUM(bottles_sold) AS total_bottles_sold
FROM `bigquery-public-data.iowa_liquor_sales.sales`
WHERE date >= '2017-01-01'
  AND sale_dollars >= 0
  AND category_name IS NOT NULL
GROUP BY week_start, category_group
ORDER BY week_start, category_group;
