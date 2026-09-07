# Memo: Statewide & Category-Level Liquor Sales Forecasting

**To:** VP, Merchandising & Demand Planning
**From:** Data Science
**Re:** Recommended forecasting approach for weekly sales planning

## Bottom line

We tested six forecasting approaches against a full year of sales we held back from every
model during training. A SARIMA model came out ahead, missing actual weekly sales by 6.61%
on average -- meaningfully better than the simplest reasonable benchmark (8.29%) and worth
deploying for statewide weekly planning. For category-level forecasts, where SARIMA wasn't
extended, a separate machine-learning model provides working (if less accurate) estimates.

## Why this matters in dollar terms

Statewide sales ran about $424M in 2025, or roughly $8.15M/week on average. A 6.61% MAPE
means a typical week's forecast is off by somewhere in the neighborhood of $540K -- a useful
number for sizing inventory buffers and flagging when actual sales are running unusually far
from plan, though it should be read as a rough planning figure rather than a hard guarantee,
since some weeks miss by much more than others (see Known Limitations).

## What we compared

Every model was scored the same way: trained on ~8.5 years of weekly data (2017 through early
2025), then asked to forecast the next full year with zero access to the real outcomes --
the same constraint a production system would face. Comparing five forecasting methods against
that identical, untouched holdout is what makes the numbers below comparable at all.

| Approach | Weekly forecast error (MAPE) | Read |
|---|---|---|
| SARIMA | 6.61% | **Recommended.** Classical time-series model, tuned on training data alone |
| Prophet | 6.72% | Close second; more configurable if we later add holiday effects |
| Seasonal Naive ("same week as last year") | 8.29% | The bar any real model needs to clear |
| Naive ("same as last week") | 8.44% | |
| Machine learning (gradient boosting) | 8.44% | Didn't beat the simple seasonal benchmark statewide, but is the only model that also forecasts each of our top 5 categories individually |
| Drift ("keep extrapolating the trend") | 9.73% | Worst performer -- and revealing: it assumes growth continues, which broke in 2025 |

## Known limitations -- read before relying on this for a specific week

**A recurring blind spot around Memorial Day through July 4th.** Every year in our data, sales
swing sharply during this stretch -- and 2025 saw the sharpest swing yet. SARIMA forecasts a
fixed pattern by calendar week number, but Memorial Day itself lands on a different week number
every year, so the model can't consistently anticipate exactly when the swing will hit. This is
a real, identified gap, not a hypothetical one -- treat SARIMA's forecasts for that six-week
window each year with extra caution, and lean on Prophet (which supports explicit holiday
inputs, not yet built out here) if that window matters for a specific planning decision.

**Forecast error doesn't grow the further out you look -- but the model's own stated
uncertainty does.** SARIMA reports a wider confidence interval the further into the future you
ask it to forecast, which looks intuitive but is a mechanical side effect of how the model is
built, not a signal we validated against real outcomes. In our test year, forecast accuracy at
week 52 was statistically no worse than at week 8. Don't treat a "tighter" near-term confidence
band as proof the near-term number is more trustworthy.

**Category-level forecasts are uneven, and not for a fully understood reason.** Our best
category forecast (Whiskey Liqueur, 8.94% error) and our worst (100% Agave Tequila, 16.96%
error) line up with which category is intrinsically more volatile -- but the three categories
in between don't rank the way their volatility alone would predict. We can't yet say with
confidence which categories will forecast well beyond "the most stable and least stable ones
are roughly as expected." Category forecasts should be used as a working estimate, not a
precise commitment, especially for the three middle categories (American Vodkas, Canadian
Whiskies, Straight Bourbon Whiskies).

## Recommended next steps

1. **Deploy SARIMA for statewide weekly planning**; use the ML model's category-level output
   as directional guidance rather than a precise number, particularly for the three
   inconsistently-ranked middle categories.
2. **Build out Prophet's holiday/event regressor support** specifically for the Memorial
   Day-July 4th window before relying on any model's forecast for that period.
3. **Revisit category-level modeling** with features aimed at separating seasonal, learnable
   variance from idiosyncratic noise -- the open question this project surfaced but didn't
   resolve.

Full methodology, every modeling decision and its reasoning, and the diagnostic work behind
each finding above are in `docs/decisions_log.md` and the numbered notebooks in this repo.
