"""Generate notebooks/smog_forecast_analysis.ipynb; interpretation comes from scripts/findings.md."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def findings(section: str) -> str:
    path = ROOT / "scripts" / "findings.md"
    text = path.read_text() if path.exists() else ""
    marker = f"## {section}\n"
    return text.split(marker, 1)[1].split("\n## ", 1)[0].strip() if marker in text else "_Interpretation pending._"


cells = [
    md("""# Pakistan Smog Forecast: Analysis and Backtest

**Author:** Anas Khan · Data Science student

Every winter, smog makes the air in Lahore and other Pakistani cities among the most polluted in the world. This notebook:

1. describes the seasonal pattern of PM2.5 in **Lahore, Peshawar, Islamabad and Karachi** (August 2022 – September 2026),
2. examines the **weather conditions** that trap pollution,
3. builds **1-, 2- and 3-day forecasts** and backtests them against simple baselines on a period the model never saw,
4. explains what the model relies on.

**Data caveat:** PM2.5 comes from the Copernicus (CAMS) global atmospheric model via Open-Meteo, not ground monitors. It tracks seasonal and day-to-day changes but underestimates the worst ground-level peaks, so the absolute numbers here are conservative."""),
    code("""import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from smog import model as M

ROOT = Path("..").resolve()
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
CITY_COLORS = {"Lahore": "#E4572E", "Peshawar": "#4C7AF2", "Islamabad": "#29A36A", "Karachi": "#8E5BD9"}
def save(fig, name): fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")

daily = pd.read_csv(ROOT / "data" / "processed" / "daily.csv", parse_dates=["date"])
print(f"{len(daily):,} city-days · {daily['date'].min().date()} to {daily['date'].max().date()} · "
      f"{daily['pm25'].notna().mean():.1%} of days have PM2.5")"""),
    md("## 1. The smog season"),
    code("""monthly = daily.assign(month=daily["date"].dt.month).groupby(["city", "month"])["pm25"].mean().unstack()
monthly = monthly.loc[["Lahore", "Peshawar", "Islamabad", "Karachi"]]
fig, ax = plt.subplots(figsize=(12, 3.8))
sns.heatmap(monthly, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "µg/m³"}, ax=ax,
            xticklabels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set(title="Average daily PM2.5 by month (Aug 2022 – Sep 2026)", xlabel="", ylabel="")
save(fig, "01_monthly_heatmap")
plt.show()"""),
    code("""cats = daily.dropna(subset=["pm25"]).assign(category=lambda d: M.aqi_category(d["pm25"].to_numpy()).to_numpy())
share = pd.crosstab(cats["city"], cats["category"], normalize="index").reindex(columns=M.AQI_LABELS, fill_value=0) * 100
share.loc[["Lahore", "Peshawar", "Islamabad", "Karachi"]].round(1)"""),
    code("""fig, ax = plt.subplots(figsize=(13, 5))
for city, g in daily.groupby("city"):
    ax.plot(g["date"], g["pm25"].rolling(7, min_periods=4).mean(), color=CITY_COLORS[city], lw=1.6, label=city)
ax.axhline(15, color="grey", ls="--", lw=1); ax.text(daily["date"].min(), 17, "WHO 24-h guideline (15 µg/m³)", color="grey", fontsize=9)
ax.set(title="PM2.5, 7-day rolling average", ylabel="µg/m³", xlabel="")
ax.legend(frameon=False, ncol=4)
save(fig, "02_timeseries")
plt.show()"""),
    md(findings("season")),
    md("## 2. What traps the smog? Weather drivers"),
    code("""lahore = daily[daily["city"] == "Lahore"].dropna(subset=["pm25", "blh_night"])
winter = lahore["date"].dt.month.isin([11, 12, 1])
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(lahore.loc[~winter, "blh_night"], lahore.loc[~winter, "pm25"], s=10, alpha=.35, color="#9CA3AF", label="Other months")
ax.scatter(lahore.loc[winter, "blh_night"], lahore.loc[winter, "pm25"], s=12, alpha=.6, color="#E4572E", label="Nov–Jan")
ax.set(xscale="log", title="Lahore: low night-time mixing height goes with high PM2.5",
       xlabel="Night-time boundary layer height (m, log scale)", ylabel="Daily PM2.5 (µg/m³)")
ax.legend(frameon=False)
save(fig, "03_blh_vs_pm25")
plt.show()

weather = ["blh_night", "blh_mean", "wind_mean", "temp_mean", "rh_mean", "temp_range", "precip_sum"]
corr = pd.DataFrame({city: g[weather + ["pm25"]].corr(method="spearman")["pm25"].drop("pm25")
                     for city, g in daily.groupby("city")})
corr[["Lahore", "Peshawar", "Islamabad", "Karachi"]].round(2)"""),
    md(findings("drivers")),
    md("""## 3. Forecasting: backtest

**Setup.** For each issue day *t*, predict the daily mean PM2.5 on day *t+1*, *t+2* and *t+3* from recent PM2.5 (today, 1, 2 and 6 days ago, and the 7-day mean), today's weather, the **target day's weather**, the season, and the city. One gradient-boosting model per horizon, trained on log(PM2.5).

**Honest split.** Train on everything whose *target* falls before **1 August 2025**; test on every forecast issued from 1 August 2025 to September 2026. That covers the full 2025–26 smog winter and is strictly later than the training data.

**Baselines.** *Persistence* (the future equals today, which is hard to beat at short range) and *seasonal average* (the training-period mean for that city and time of year)."""),
    code("""scores = pd.read_csv(ROOT / "data" / "processed" / "backtest_scores.csv")
pred = pd.read_csv(ROOT / "data" / "processed" / "backtest_predictions.csv", parse_dates=["date", "target_date"])
table = scores.pivot_table(index="method", columns="horizon", values="MAE").loc[
    ["Persistence", "Climatology", "Ridge regression", "Gradient boosting"]]
table.index = ["Persistence", "Seasonal average", "Ridge regression", "Gradient boosting"]
print("Mean absolute error (µg/m³), test period Aug 2025 – Sep 2026")
table.round(2)"""),
    code("""rows = []
for h, g in pred.groupby("horizon"):
    for label, mask in {"All months": slice(None), "Smog season (Nov–Jan)": g["target_date"].dt.month.isin([11, 12, 1])}.items():
        s = g.loc[mask]
        for method in ["Persistence", "Gradient boosting"]:
            rows.append({"horizon": h, "period": label, "method": method, "MAE": (s[method] - s["target"]).abs().mean()})
mae = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (period, g) in zip(axes, mae.groupby("period")):
    sns.barplot(data=g, x="horizon", y="MAE", hue="method", palette=["#9CA3AF", "#E4572E"], ax=ax)
    ax.set(title=period, xlabel="Days ahead", ylabel="MAE (µg/m³)"); ax.legend(frameon=False)
save(fig, "04_mae_by_horizon")
plt.show()
mae.pivot_table(index=["period", "horizon"], columns="method", values="MAE").assign(
    improvement=lambda t: (1 - t["Gradient boosting"] / t["Persistence"]) * 100).round(1)"""),
    code("""season = pred[(pred["city"] == "Lahore") & pred["target_date"].between("2025-10-15", "2026-02-15")]
fig, ax = plt.subplots(figsize=(13, 5))
one = season[season["horizon"] == 1].sort_values("target_date")
three = season[season["horizon"] == 3].sort_values("target_date")
ax.plot(one["target_date"], one["target"], color="black", lw=2, label="Actual (CAMS)")
ax.plot(one["target_date"], one["Gradient boosting"], color="#E4572E", lw=1.6, label="Forecast, 1 day ahead")
ax.plot(three["target_date"], three["Gradient boosting"], color="#F3A712", lw=1.4, ls="--", label="Forecast, 3 days ahead")
ax.set(title="Lahore, 2025–26 smog season: forecasts vs. actual (never seen in training)", ylabel="µg/m³", xlabel="")
ax.legend(frameon=False)
save(fig, "05_lahore_winter")
plt.show()"""),
    md(findings("backtest")),
    md("## 4. What does the model rely on?"),
    code("""split = pd.Timestamp("2025-08-01")
frame = M.make_features(daily, 1).dropna(subset=["target", "pm25_lag0"])
cols = M.feature_columns(frame)
train = frame[frame["target_date"] < split].dropna(subset=cols)
test = frame[frame["date"] >= split].dropna(subset=cols)
gb = M.LogTarget(M.models()["Gradient boosting"]).fit(train[cols], train["target"])

imp = permutation_importance(gb, test[cols], test["target"], scoring="neg_mean_absolute_error",
                             n_repeats=10, random_state=42, n_jobs=-1)
importance = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False).head(12)
fig, ax = plt.subplots(figsize=(9, 5))
importance[::-1].plot.barh(ax=ax, color="#E4572E")
ax.set(title="Permutation importance, 1-day model (increase in MAE when shuffled, µg/m³)", xlabel="")
save(fig, "06_importance")
plt.show()
importance.round(2)"""),
    md(findings("importance")),
    md("## 5. Conclusions and limitations\n\n" + findings("conclusions")),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "smog_forecast_analysis.ipynb"
nbf.write(nb, out)
print("wrote", out)
