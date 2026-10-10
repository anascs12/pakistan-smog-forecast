"""Download hourly PM2.5 and weather from Open-Meteo and aggregate to daily values.

PM2.5 comes from the Open-Meteo Air Quality API, which serves the Copernicus
Atmosphere Monitoring Service (CAMS) global model. These are model estimates,
not ground measurements: they capture the seasonal pattern and day-to-day swings
well, but are known to underestimate the most extreme ground-level peaks in
South Asian cities. Weather comes from Open-Meteo's historical archive (ERA5)
and, for live forecasts, its forecast API.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CITIES = {
    "Lahore": (31.5497, 74.3436),
    "Peshawar": (34.0151, 71.5249),
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
}
TIMEZONE = "Asia/Karachi"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "boundary_layer_height", "precipitation"]
MIN_HOURS = 18  # a day needs at least 18 hourly PM2.5 values to count
# CAMS occasionally serves near-zero values for every city at once (e.g. 14-15 Dec 2022,
# peak smog season), a feed artifact rather than clean air; daily means below this are dropped.
MIN_PLAUSIBLE_PM25 = 2.0


def _get(url: str, params: dict, cache: Path | None = None) -> dict:
    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(6):
        response = requests.get(url, params=params, timeout=120)
        if response.status_code == 200 and response.text.lstrip().startswith("{"):
            payload = response.json()
            break
        if response.status_code not in (200, 429, 500, 502, 503, 504):
            response.raise_for_status()
        time.sleep(5 * 2 ** attempt)  # free API: back off on rate limits and transient errors
    else:
        raise RuntimeError(f"Open-Meteo kept failing for {url} ({response.status_code})")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
    return payload


def _hourly_frame(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.set_index("time")


def hourly_history(city: str, start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    lat, lon = CITIES[city]
    base = {"latitude": lat, "longitude": lon, "timezone": TIMEZONE,
            "start_date": start.isoformat(), "end_date": end.isoformat()}
    tag = f"{city}_{start:%Y%m%d}_{end:%Y%m%d}"
    aq = _get(AQ_URL, {**base, "hourly": "pm2_5"}, cache_dir / f"aq_{tag}.json")
    wx = _get(ARCHIVE_URL, {**base, "hourly": ",".join(WEATHER_VARS)}, cache_dir / f"wx_{tag}.json")
    return _hourly_frame(aq).join(_hourly_frame(wx), how="outer")


def hourly_recent(city: str, past_days: int = 14, forecast_days: int = 4) -> pd.DataFrame:
    """Recent PM2.5 plus weather for the past days and the coming forecast days (not cached)."""
    lat, lon = CITIES[city]
    base = {"latitude": lat, "longitude": lon, "timezone": TIMEZONE,
            "past_days": past_days, "forecast_days": forecast_days}
    aq = _get(AQ_URL, {**base, "hourly": "pm2_5"})
    wx = _get(FORECAST_URL, {**base, "hourly": ",".join(WEATHER_VARS)})
    return _hourly_frame(aq).join(_hourly_frame(wx), how="outer")


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily features. Night-time mixing height (00-06 h) captures winter inversions."""
    day = hourly.index.normalize()
    grouped = hourly.groupby(day)
    night = hourly[hourly.index.hour < 6].groupby(hourly.index[hourly.index.hour < 6].normalize())
    daily = pd.DataFrame({
        "pm25": grouped["pm2_5"].mean(),
        "pm25_hours": grouped["pm2_5"].count(),
        "temp_mean": grouped["temperature_2m"].mean(),
        "temp_range": grouped["temperature_2m"].max() - grouped["temperature_2m"].min(),
        "rh_mean": grouped["relative_humidity_2m"].mean(),
        "wind_mean": grouped["wind_speed_10m"].mean(),
        "blh_mean": grouped["boundary_layer_height"].mean(),
        "blh_night": night["boundary_layer_height"].mean(),
        "precip_sum": grouped["precipitation"].sum(min_count=1),
    })
    daily.loc[daily["pm25_hours"] < MIN_HOURS, "pm25"] = np.nan
    daily.loc[daily["pm25"] < MIN_PLAUSIBLE_PM25, "pm25"] = np.nan
    daily.index.name = "date"
    return daily.drop(columns="pm25_hours")


def build_history(start: date, end: date, root: Path = Path(".")) -> pd.DataFrame:
    frames = []
    for city in CITIES:
        daily = to_daily(hourly_history(city, start, end, root / "data" / "raw"))
        frames.append(daily.assign(city=city).reset_index())
        print(f"  {city:10} {daily['pm25'].notna().sum():5} days with PM2.5")
    history = pd.concat(frames, ignore_index=True)
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    history.to_csv(out / "daily.csv", index=False)
    return history


if __name__ == "__main__":
    # The ERA5 archive lags real time by about a week, so stop a little before today.
    history = build_history(date(2022, 8, 5), date(2026, 9, 30))
    print(f"Saved {len(history):,} city-days to data/processed/daily.csv")
