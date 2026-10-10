"""Render the public forecast page (docs/index.html)."""

from __future__ import annotations

import html
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

AQI_COLORS = {
    "Good": "#00E400", "Moderate": "#FFFF00", "Unhealthy for sensitive groups": "#FF7E00",
    "Unhealthy": "#FF0000", "Very unhealthy": "#8F3F97", "Hazardous": "#7E0023",
}
ADVICE = {
    "Good": "Air quality is satisfactory.",
    "Moderate": "Unusually sensitive people should consider reducing long or heavy outdoor exertion.",
    "Unhealthy for sensitive groups": "Children, older adults and people with heart or lung disease should reduce long or heavy outdoor exertion.",
    "Unhealthy": "Everyone should reduce long or heavy outdoor exertion; sensitive groups should avoid it.",
    "Very unhealthy": "Everyone should avoid long or heavy outdoor exertion. Consider a well-fitting N95 mask outdoors.",
    "Hazardous": "Everyone should avoid all outdoor physical activity and keep windows closed.",
}
CITY_ORDER = ["Lahore", "Peshawar", "Islamabad", "Karachi"]


def _text_color(category: str) -> str:
    return "#111" if category in ("Good", "Moderate", "Unhealthy for sensitive groups") else "#fff"


def _day_label(day: str, issue_day: str) -> str:
    offset = (date.fromisoformat(day) - date.fromisoformat(issue_day)).days
    names = {1: "Today", 2: "Tomorrow", 3: date.fromisoformat(day).strftime("%A")}
    return f"{names.get(offset, day)} · {date.fromisoformat(day):%d %b}"


def city_cards(payload: dict) -> str:
    cards = []
    for city in CITY_ORDER:
        days = sorted((f for f in payload["forecasts"] if f["city"] == city), key=lambda f: f["horizon"])
        cells = []
        for f in days:
            if f["pm25"] is None:
                cells.append(f'<div class="day"><div class="label">{_day_label(f["date"], payload["issue_day"])}</div>'
                             '<div class="value na">n/a</div></div>')
                continue
            bg, fg = AQI_COLORS[f["category"]], _text_color(f["category"])
            cells.append(
                f'<div class="day" style="background:{bg};color:{fg}" title="{html.escape(ADVICE[f["category"]])}">'
                f'<div class="label">{_day_label(f["date"], payload["issue_day"])}</div>'
                f'<div class="value">{f["pm25"]:.0f}<span> µg/m³</span></div>'
                f'<div class="cat">{html.escape(f["category"])}</div></div>')
        first = next((f for f in days if f["category"]), None)
        advice = f'<p class="advice">{html.escape(ADVICE[first["category"]])}</p>' if first else ""
        cards.append(f'<section class="city"><h2>{city}</h2><div class="days">{"".join(cells)}</div>{advice}</section>')
    return "\n".join(cards)


def trend_chart(payload: dict) -> str:
    observed = pd.DataFrame(payload["observed"])
    forecasts = pd.DataFrame(payload["forecasts"]).dropna(subset=["pm25"])
    fig = go.Figure()
    palette = {"Lahore": "#E4572E", "Peshawar": "#4C7AF2", "Islamabad": "#29A36A", "Karachi": "#8E5BD9"}
    for city in CITY_ORDER:
        o = observed[observed["city"] == city].sort_values("date")
        f = forecasts[forecasts["city"] == city].sort_values("date")
        fig.add_trace(go.Scatter(x=o["date"], y=o["pm25"], name=city, mode="lines+markers",
                                 line=dict(color=palette[city], width=2.5), legendgroup=city))
        if not o.empty and not f.empty:
            bridge = pd.concat([o.tail(1)[["date", "pm25"]], f[["date", "pm25"]]])
            fig.add_trace(go.Scatter(x=bridge["date"], y=bridge["pm25"], name=f"{city} forecast", mode="lines+markers",
                                     line=dict(color=palette[city], width=2.5, dash="dot"), legendgroup=city,
                                     showlegend=False))
    for low, high, cat in [(0, 9, "Good"), (9, 35.4, "Moderate"), (35.4, 55.4, "Unhealthy for sensitive groups"),
                           (55.4, 125.4, "Unhealthy"), (125.4, 225.4, "Very unhealthy")]:
        fig.add_hrect(y0=low, y1=high, fillcolor=AQI_COLORS[cat], opacity=0.10, line_width=0)
    fig.add_vline(x=payload["issue_day"], line_dash="dash", line_color="grey")
    fig.update_layout(template="plotly_white", height=460, margin=dict(l=40, r=20, t=50, b=40),
                      title="Daily mean PM2.5: last two weeks (solid) and 3-day forecast (dotted)",
                      yaxis_title="PM2.5 (µg/m³)", legend=dict(orientation="h", y=-0.15))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def skill_table(metrics: pd.DataFrame | None) -> str:
    if metrics is None:
        return ""
    rows = []
    for h in (1, 2, 3):
        m = metrics[metrics["horizon"] == h].set_index("method")
        model_mae, persist = m.loc["Gradient boosting", "MAE"], m.loc["Persistence", "MAE"]
        rows.append(f"<tr><td>{h} day{'s' if h > 1 else ''}</td><td>{model_mae:.1f}</td><td>{persist:.1f}</td>"
                    f"<td>{m.loc['Climatology', 'MAE']:.1f}</td><td>{(1 - model_mae / persist) * 100:.0f}%</td>"
                    f"<td>{m.loc['Gradient boosting', 'AQI category accuracy'] * 100:.0f}%</td></tr>")
    return ("<table><thead><tr><th>Horizon</th><th>Model MAE</th><th>Persistence MAE</th><th>Seasonal-average MAE</th>"
            "<th>Improvement vs persistence</th><th>Correct AQI category</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def write(payload: dict, metrics: pd.DataFrame | None, path: Path) -> Path:
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pakistan Smog Forecast</title>
<meta name="description" content="Daily 3-day PM2.5 air pollution forecast for Lahore, Peshawar, Islamabad and Karachi.">
<style>
  :root {{ --bg:#F4F6FB; --card:#fff; --ink:#1C2333; --muted:#5A6478; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--ink); }}
  header {{ background: linear-gradient(120deg,#1F2937,#7C2D12); color:#fff; padding:32px 20px; }}
  header h1 {{ margin:0 0 6px; font-size:28px; }}
  header p {{ margin:4px 0; opacity:.9; max-width:900px; line-height:1.5; }}
  main {{ max-width:1160px; margin:0 auto; padding:22px 16px 48px; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:16px; }}
  .city, .panel {{ background:var(--card); border-radius:14px; padding:16px; box-shadow:0 2px 10px rgba(15,23,42,.08); }}
  .city h2 {{ margin:0 0 12px; font-size:20px; }}
  .days {{ display:grid; grid-template-columns: repeat(3,1fr); gap:8px; }}
  .day {{ border-radius:10px; padding:10px 8px; text-align:center; background:#E5E7EB; }}
  .label {{ font-size:12px; font-weight:600; }}
  .value {{ font-size:26px; font-weight:800; margin:4px 0; }}
  .value span {{ font-size:11px; font-weight:600; }}
  .value.na {{ font-size:18px; color:var(--muted); }}
  .cat {{ font-size:11px; font-weight:600; line-height:1.2; }}
  .advice {{ font-size:13px; color:var(--muted); margin:12px 0 0; line-height:1.45; }}
  .panel {{ margin-top:18px; overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid #E5E7EB; text-align:left; }}
  .note {{ font-size:13.5px; color:var(--muted); line-height:1.55; }}
  footer {{ text-align:center; font-size:13px; color:var(--muted); padding:0 16px 32px; }}
  a {{ color:#1D4ED8; }}
</style></head>
<body>
<header>
  <h1>🌫️ Pakistan Smog Forecast</h1>
  <p>3-day forecast of daily average PM2.5 (fine particulate pollution) for four Pakistani cities, updated automatically every morning.</p>
  <p style="font-size:13px">Issued {payload['generated_at']} · last observed day {payload['issue_day']} · colours follow the US EPA air-quality categories.</p>
</header>
<main>
  <div class="grid">{city_cards(payload)}</div>
  <div class="panel">{trend_chart(payload)}</div>
  <div class="panel">
    <h3 style="margin-top:0">How accurate is it?</h3>
    <p class="note">Backtest on August 2025 – September 2026 (including the full 2025–26 smog winter), a period the model never saw in training. MAE is the average error in µg/m³; "persistence" assumes tomorrow equals today.</p>
    {skill_table(metrics)}
  </div>
  <div class="panel note">
    <h3 style="margin-top:0;color:var(--ink)">About the data, and its limits</h3>
    <p>PM2.5 values are estimates from the Copernicus Atmosphere Monitoring Service (CAMS) global model, served by <a href="https://open-meteo.com/">Open-Meteo</a>, not ground sensors. CAMS captures day-to-day swings and the winter smog season, but it is known to <b>underestimate the most extreme readings</b> that ground monitors record in Lahore and Peshawar, so real conditions can be worse than shown.</p>
    <p>The model is a gradient-boosting regressor using recent pollution levels and the weather forecast for each day, especially <b>night-time mixing height</b> (low mixing traps smoke near the ground), wind, humidity and temperature. This is a student data-science project, not an official advisory: follow guidance from the Punjab and Khyber Pakhtunkhwa Environmental Protection Agencies.</p>
    <p>Code, analysis and backtest: <a href="https://github.com/anascs12/pakistan-smog-forecast">github.com/anascs12/pakistan-smog-forecast</a></p>
  </div>
</main>
<footer>Weather and air-quality data: Open-Meteo (CC BY 4.0), Copernicus Atmosphere Monitoring Service, ERA5. Model trained through {payload['trained_through']}. Built by Anas Khan.</footer>
</body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path
