"""
Iowa Retail Liquor Sales Forecasting -- Dashboard
Builds a self-contained dashboard.html summarizing the forecasting project.
Run from the project root: python build_dashboard.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------------------------------------------
# Palette -- validated set, from the dataviz skill's reference palette
# (references/palette.md). Swap these hex values only if re-branding;
# nothing else in the file needs to change.
# -----------------------------------------------------------------
SURFACE       = "#fcfcfb"
PAGE          = "#f9f9f7"
INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRIDLINE      = "#e1e0d9"
BASELINE      = "#c3c2b7"

BLUE   = "#2a78d6"   # categorical slot 1 -- emphasis / primary series
ORANGE = "#eb6834"   # categorical slot 2 -- secondary metric (CV)
RED    = "#e34948"   # categorical slot 8 -- accent (2025 highlight)
GRAY   = "#c3c2b7"   # de-emphasis gray for "rest" in emphasis charts

# 8-step sequential blue ramp, ordinal (2017-2024). Lightest step is >=250
# so it still clears 2:1 contrast on the light surface (palette.md rule).
YEAR_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95"]

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Chart titles are set as an HTML heading above each plot (see fig_card()
# below), not as Plotly's own `title`, so a chart's legend never has to
# compete with in-plot title text for the same sliver of vertical space.
CHART_LAYOUT = dict(
    plot_bgcolor=SURFACE,
    paper_bgcolor=SURFACE,
    font=dict(family=FONT, color=INK_SECONDARY, size=13),
    margin=dict(l=60, r=20, t=20, b=50),
    hoverlabel=dict(bgcolor=SURFACE, font=dict(family=FONT, color=INK_PRIMARY)),
)


def layout_with(**overrides):
    """CHART_LAYOUT merged with per-chart overrides (e.g. legend, margin),
    without the duplicate-keyword clash of passing both **CHART_LAYOUT and
    an explicit kwarg to the same update_layout() call."""
    layout = dict(CHART_LAYOUT)
    layout.update(overrides)
    return layout


def style_line_axes(fig, y_title=None, y_prefix=None):
    """Vertical/time-series charts: hairline solid gridlines on y, clean x."""
    fig.update_xaxes(showgrid=False, showline=True, linecolor=BASELINE,
                      linewidth=1, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, gridwidth=1,
                      griddash="solid", zeroline=False,
                      tickfont=dict(color=INK_MUTED), title=y_title,
                      tickprefix=y_prefix or "")
    return fig


def style_hbar_axes(fig, x_title=None, max_val=None):
    """Horizontal bar charts: hairline solid gridlines on x, clean y.
    `max_val` (the largest bar value) pads the x-axis range so an
    outside-positioned data label on the longest bar has room to render
    instead of getting clipped at the plot edge (anti-pattern: a label
    clipped by a too-small axis range)."""
    fig.update_xaxes(showgrid=True, gridcolor=GRIDLINE, gridwidth=1,
                      griddash="solid", zeroline=False,
                      tickfont=dict(color=INK_MUTED), title=x_title,
                      range=[0, max_val * 1.18] if max_val else None)
    fig.update_yaxes(showgrid=False, showline=False,
                      tickfont=dict(color=INK_PRIMARY))
    return fig


# -----------------------------------------------------------------
# Data
# -----------------------------------------------------------------
df = pd.read_csv("data/raw/weekly_sales_by_category.csv", parse_dates=["week_start"])

statewide = df.groupby("week_start")["total_sales_dollars"].sum().reset_index()
statewide.columns = ["week_start", "total_sales"]
statewide = statewide.sort_values("week_start").reset_index(drop=True)
history_start_year = statewide["week_start"].min().year
history_end_year = statewide["week_start"].max().year
history_years = round((statewide["week_start"].max() - statewide["week_start"].min()).days / 365.25, 1)

TEST_WEEKS = 52
train = statewide.iloc[:-TEST_WEEKS].copy()
test = statewide.iloc[-TEST_WEEKS:].copy()

sarima_model = SARIMAX(train["total_sales"].values, order=(1, 1, 2),
                        seasonal_order=(1, 1, 1, 52),
                        enforce_stationarity=True, enforce_invertibility=True)
sarima_fit = sarima_model.fit(disp=False)
sarima_result = sarima_fit.get_forecast(steps=TEST_WEEKS)
sarima_forecast = sarima_result.predicted_mean
ci = sarima_result.conf_int(alpha=0.05)

# -----------------------------------------------------------------
# KPI row -- the handful of headline numbers this dashboard leads with
# -----------------------------------------------------------------
kpis = [
    ("Best model MAPE", "6.61%", "SARIMA, final test year"),
    ("Series forecast", "6", "statewide + 5 top categories"),
    ("Test horizon", "52 wks", "full year, no peeking"),
    ("History used", f"{history_years:g} yrs", f"{history_start_year}-{history_end_year} weekly"),
]

# -----------------------------------------------------------------
# Fig 1 -- statewide trend (single series: no legend needed)
# -----------------------------------------------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=statewide["week_start"], y=statewide["total_sales"],
    mode="lines", line=dict(color=BLUE, width=2), showlegend=False,
    hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>",
))
fig1.update_layout(**CHART_LAYOUT)
style_line_axes(fig1, y_title="Weekly sales ($)", y_prefix="$")

# -----------------------------------------------------------------
# Fig 2 -- model comparison. Emphasis form: one accent + gray the rest.
# Color follows the entity (SARIMA), not the sort rank.
# -----------------------------------------------------------------
models = ["Drift", "Naive", "Seasonal Naive", "ML (gradient boosting)", "Prophet", "SARIMA"]
mape = [9.73, 8.44, 8.29, 8.44, 6.72, 6.61]
colors = [GRAY, GRAY, GRAY, GRAY, GRAY, BLUE]

order = np.argsort(mape)[::-1]  # largest MAPE at top, best at bottom
models_sorted = [models[i] for i in order]
mape_sorted = [mape[i] for i in order]
colors_sorted = [colors[i] for i in order]

fig2 = go.Figure(go.Bar(
    x=mape_sorted, y=models_sorted, orientation="h",
    marker=dict(color=colors_sorted, cornerradius=4),
    text=[f"{v:.2f}%" for v in mape_sorted], textposition="outside",
    textfont=dict(color=INK_SECONDARY),
    hovertemplate="%{y}: %{x:.2f}% MAPE<extra></extra>",
    showlegend=False,
))
fig2.update_layout(**CHART_LAYOUT)
style_hbar_axes(fig2, x_title="MAPE (%)", max_val=max(mape_sorted))

# -----------------------------------------------------------------
# Fig 3 -- SARIMA actual vs forecast. CI band is a blue wash matching
# the forecast line's own color (the series' own uncertainty).
# -----------------------------------------------------------------
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=test["week_start"], y=ci[:, 1], mode="lines",
                           line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig3.add_trace(go.Scatter(x=test["week_start"], y=ci[:, 0], mode="lines",
                           line=dict(width=0), fill="tonexty",
                           fillcolor="rgba(42,120,214,0.10)",
                           name="95% CI", hoverinfo="skip"))
fig3.add_trace(go.Scatter(x=test["week_start"], y=test["total_sales"], mode="lines",
                           line=dict(color=INK_PRIMARY, width=2), name="Actual",
                           hovertemplate="%{x|%b %d, %Y}<br>Actual: $%{y:,.0f}<extra></extra>"))
fig3.add_trace(go.Scatter(x=test["week_start"], y=sarima_forecast, mode="lines",
                           line=dict(color=BLUE, width=2), name="SARIMA forecast",
                           hovertemplate="%{x|%b %d, %Y}<br>Forecast: $%{y:,.0f}<extra></extra>"))
fig3.update_layout(**layout_with(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                font=dict(color=INK_SECONDARY)),
    margin=dict(l=60, r=20, t=40, b=50),
))
style_line_axes(fig3, y_title="Weekly sales ($)", y_prefix="$")

# -----------------------------------------------------------------
# Fig 4a / 4b -- category MAPE and category volatility (CV).
# Was one dual-axis chart; split into two single-axis charts (anti-pattern
# fix -- a shared y-scale between MAPE% and CV% is an arbitrary alignment).
# Same category order (sorted by MAPE) in both, so they stay comparable
# side by side without faking a shared axis.
# -----------------------------------------------------------------
category_mape = {
    "WHISKEY LIQUEUR": 8.94,
    "STRAIGHT BOURBON WHISKIES": 10.82,
    "AMERICAN VODKAS": 13.57,
    "CANADIAN WHISKIES": 13.69,
    "100% AGAVE TEQUILA": 16.96,
}
cv_table = (df.groupby("category_group")["total_sales_dollars"]
              .agg(lambda s: s.std() / s.mean() * 100))
category_cv = {cat: cv_table.get(cat, np.nan) for cat in category_mape}

cats = list(category_mape.keys())
mape_vals = [category_mape[c] for c in cats]
cv_vals = [category_cv[c] for c in cats]

order = np.argsort(mape_vals)
cats_sorted = [cats[i] for i in order]
mape_sorted2 = [mape_vals[i] for i in order]
cv_sorted = [cv_vals[i] for i in order]

fig4a = go.Figure(go.Bar(
    x=mape_sorted2, y=cats_sorted, orientation="h",
    marker=dict(color=BLUE, cornerradius=4),
    text=[f"{v:.2f}%" for v in mape_sorted2], textposition="outside",
    textfont=dict(color=INK_SECONDARY),
    hovertemplate="%{y}: %{x:.2f}% MAPE<extra></extra>",
))
fig4a.update_layout(**CHART_LAYOUT)
style_hbar_axes(fig4a, x_title="MAPE (%)", max_val=max(mape_sorted2))

fig4b = go.Figure(go.Bar(
    x=cv_sorted, y=cats_sorted, orientation="h",
    marker=dict(color=ORANGE, cornerradius=4),
    text=[f"{v:.1f}%" for v in cv_sorted], textposition="outside",
    textfont=dict(color=INK_SECONDARY),
    hovertemplate="%{y}: %{x:.1f}% CV<extra></extra>",
))
fig4b.update_layout(**CHART_LAYOUT)
style_hbar_axes(fig4b, x_title="CV (%)", max_val=max(cv_sorted))

# -----------------------------------------------------------------
# Fig 5 -- Memorial Day - July 4 window, year over year. Was 9 overlaid
# lines using a generated 9-color palette (exceeds the 8-hue categorical
# cap). Redesigned as an 8-step sequential blue ordinal ramp for the
# historical years 2017-2024, plus a red accent for 2025 -- the one
# series that's the point, the rest are context.
# -----------------------------------------------------------------
fig5 = go.Figure()
years = list(range(2017, 2026))
for i, year in enumerate(years):
    yearly = statewide[statewide["week_start"].dt.year == year].copy()
    mask = (
        ((yearly["week_start"].dt.month == 5) & (yearly["week_start"].dt.day >= 10))
        | (yearly["week_start"].dt.month == 6)
        | ((yearly["week_start"].dt.month == 7) & (yearly["week_start"].dt.day <= 20))
    )
    yearly = yearly[mask].sort_values("week_start").reset_index(drop=True)
    if yearly.empty:
        continue
    yearly["pct_change"] = yearly["total_sales"].pct_change() * 100

    is_current = (year == 2025)
    color = RED if is_current else YEAR_RAMP[i]
    fig5.add_trace(go.Scatter(
        x=list(range(len(yearly))), y=yearly["pct_change"],
        mode="lines", name=str(year),
        line=dict(color=color, width=3 if is_current else 2),
        hovertemplate=f"{year}, week %{{x}}<br>%{{y:.1f}}% w/w<extra></extra>",
    ))
fig5.update_layout(**layout_with(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                font=dict(color=INK_SECONDARY)),
    margin=dict(l=60, r=20, t=40, b=50),
))
style_line_axes(fig5, y_title="Week-over-week change (%)")
fig5.update_xaxes(title="Weeks into window")

# -----------------------------------------------------------------
# Assemble the page
# -----------------------------------------------------------------
def fig_card(fig, title, caption=None, include_js=False, height=380):
    # Embed plotly.js inline (once, on the first chart) rather than loading it
    # from a CDN, so the dashboard is a single offline-viewable file with no
    # network dependency. The chart title lives in this HTML heading, not in
    # Plotly's own `title` layout property, so it never has to compete with a
    # chart's legend for the same sliver of vertical space.
    #
    # Plotly's exported div is `responsive: true` with style="height:100%;
    # width:100%;" -- it needs an ancestor with a resolved pixel height, or it
    # collapses toward 0 once it isn't the sole child of `.chart` (a sibling
    # <h3> makes `.chart`'s own height ambiguous inside the flex/grid parent).
    # Giving the wrapper a fixed height sidesteps that entirely.
    body = fig.to_html(full_html=False, include_plotlyjs=(True if include_js else False))
    caption_html = f'<p class="chart-caption">{caption}</p>' if caption else ""
    return (f'<div class="chart"><h3 class="chart-title">{title}</h3>'
            f'<div class="plot-wrap" style="height:{height}px;">{body}</div>'
            f'{caption_html}</div>')


def row_caption(text):
    """A caption spanning a full row of two charts -- used where the note
    covers both charts together (the split MAPE/CV pair), rather than
    duplicating it inside each card."""
    return f'<p class="row-caption">{text}</p>'

kpi_html = "".join(
    f'''
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>'''
    for label, value, sub in kpis
)

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Iowa Retail Liquor Sales Forecasting</title>
<style>
  :root {{
    --surface: {SURFACE};
    --page: {PAGE};
    --ink-primary: {INK_PRIMARY};
    --ink-secondary: {INK_SECONDARY};
    --ink-muted: {INK_MUTED};
    --gridline: {GRIDLINE};
    --border: rgba(11,11,11,0.10);
    --accent: {BLUE};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--page);
    font-family: {FONT};
    color: var(--ink-secondary);
  }}
  header {{
    padding: 32px 40px 20px;
  }}
  header h1 {{
    margin: 0 0 6px;
    font-size: 22px;
    font-weight: 600;
    color: var(--ink-primary);
  }}
  header p {{
    margin: 0;
    color: var(--ink-muted);
    font-size: 14px;
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 2px;
    background: var(--gridline);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin: 0 40px 24px;
  }}
  .kpi {{
    background: var(--surface);
    padding: 18px 20px;
  }}
  .kpi-label {{
    font-size: 12px;
    color: var(--ink-muted);
    margin-bottom: 6px;
  }}
  .kpi-value {{
    font-size: 28px;
    font-weight: 600;
    color: var(--ink-primary);
    font-variant-numeric: proportional-nums;
  }}
  .kpi-sub {{
    font-size: 12px;
    color: var(--ink-muted);
    margin-top: 4px;
  }}
  main {{
    padding: 0 40px 48px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }}
  .row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  .chart {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 16px 16px;
  }}
  .chart-title {{
    margin: 0 0 4px 4px;
    font-size: 15px;
    font-weight: 600;
    color: var(--ink-primary);
  }}
  .plot-wrap {{
    width: 100%;
  }}
  .chart-caption {{
    margin: 10px 4px 0;
    font-size: 13px;
    line-height: 1.5;
    color: var(--ink-secondary);
  }}
  .row-caption {{
    margin: -4px 4px 0;
    font-size: 13px;
    line-height: 1.5;
    color: var(--ink-secondary);
  }}
  @media (max-width: 900px) {{
    .row, .kpi-row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Iowa Retail Liquor Sales Forecasting</h1>
  <p>SARIMA(1,1,2)(1,1,1,52): 6.61% MAPE on a 52-week statewide holdout, ahead of Prophet (6.72%), the ML model (8.44%), and the Seasonal Naive benchmark (8.29%).</p>
</header>

<div class="kpi-row">{kpi_html}
</div>

<main>
  {fig_card(
      fig1, f"Statewide weekly liquor sales, {history_start_year}-{history_end_year}",
      caption="Strong growth 2017-2024, then the first year-over-year decline in 2025 -- "
              "the reversal every subsequent model had to represent.",
      include_js=True, height=420,
  )}
  <div class="row">
    {fig_card(
        fig2, "Model comparison -- test-set MAPE (lower is better)",
        caption="SARIMA and Prophet both clear the Seasonal Naive benchmark by a real "
                "margin; the ML model, despite validation-based tuning, lands just short of it.",
    )}
    {fig_card(
        fig3, "SARIMA: actual vs. forecast (final test year, 95% CI)",
        caption="The winning model's actual forecast against the holdout it was never "
                "shown during training.",
    )}
  </div>
  <div class="row">
    {fig_card(fig4a, "ML model MAPE by category")}
    {fig_card(fig4b, "Series volatility by category (coefficient of variation)")}
  </div>
  {row_caption(
      "Forecast difficulty tracks intrinsic volatility only at the extremes "
      "(Tequila worst/highest-CV, Whiskey Liqueur best/lowest-CV) -- the middle "
      "three don't rank consistently."
  )}
  {fig_card(
      fig5, "Week-over-week % change, Memorial Day-July 4 window, by year",
      caption="A violent swing recurs every year in this window; 2025 (highlighted in red) "
              "is the most extreme instance, and SARIMA's fixed 52-week seasonal lag can't "
              "phase-align to a driver whose calendar position shifts yearly.",
      height=420,
  )}
</main>
</body>
</html>
"""

with open("dashboard.html", "w") as f:
    f.write(page)

print("Dashboard written to dashboard.html -- open it in a browser.")
