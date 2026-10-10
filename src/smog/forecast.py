"""Train the final models and publish a live 3-day forecast page to docs/.

    python -m smog.forecast --train   # refit on all history, save models/forecaster.joblib
    python -m smog.forecast           # fetch latest data, forecast, write docs/index.html

The daily GitHub Action runs the second command and commits the updated page.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from . import data, model, page

ROOT = Path(".")
MODEL_PATH = ROOT / "models" / "forecaster.joblib"
TZ = ZoneInfo(data.TIMEZONE)


def train() -> None:
    history = pd.read_csv(ROOT / "data" / "processed" / "daily.csv", parse_dates=["date"])
    fitted = model.fit_final(history)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"models": fitted, "trained_through": str(history["date"].max().date())}, MODEL_PATH)
    print(f"saved {MODEL_PATH} (trained through {history['date'].max().date()})")


def recent_daily(today: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for city in data.CITIES:
        daily = data.to_daily(data.hourly_recent(city))
        # Only complete past days count as observed; the API also returns CAMS's own
        # forecast hours for today onward, which must not be used as inputs.
        daily.loc[daily.index >= today, "pm25"] = np.nan
        frames.append(daily.assign(city=city).reset_index())
    return pd.concat(frames, ignore_index=True)


def run() -> None:
    bundle = joblib.load(MODEL_PATH)
    now = datetime.now(TZ)
    today = pd.Timestamp(now.date())
    issue_day = today - timedelta(days=1)  # last complete observed day
    daily = recent_daily(today)

    forecasts = []
    for h, entry in bundle["models"].items():
        frame = model.make_features(daily, h)
        rows = frame[frame["date"] == issue_day]
        for _, row in rows.iterrows():
            X = row[entry["columns"]].to_frame().T.astype(float)
            missing = [c for c in entry["columns"] if pd.isna(X[c].iloc[0])]
            value = None if missing else float(entry["model"].predict(X)[0])
            forecasts.append({
                "city": row["city"], "horizon": int(h), "date": str(row["target_date"].date()),
                "pm25": None if value is None else round(value, 1),
                "category": None if value is None else str(model.aqi_category([value]).iloc[0]),
            })

    observed = daily[(daily["date"] < today) & daily["pm25"].notna()][["city", "date", "pm25"]]
    observed = observed.assign(date=observed["date"].dt.date.astype(str), pm25=observed["pm25"].round(1))
    payload = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M %Z"),
        "issue_day": str(issue_day.date()),
        "trained_through": bundle["trained_through"],
        "forecasts": forecasts,
        "observed": observed.to_dict(orient="records"),
    }
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "forecast.json").write_text(json.dumps(payload, indent=1))
    metrics_path = ROOT / "data" / "processed" / "backtest_scores.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    page.write(payload, metrics, docs / "index.html")
    ok = sum(f["pm25"] is not None for f in forecasts)
    print(f"{ok}/{len(forecasts)} forecasts issued from {issue_day.date()} -> docs/index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="refit and save the models")
    args = parser.parse_args()
    train() if args.train else run()
