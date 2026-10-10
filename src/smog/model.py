"""Features, models, baselines and evaluation for 1-3 day PM2.5 forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WEATHER = ["temp_mean", "temp_range", "rh_mean", "wind_mean", "blh_mean", "blh_night", "precip_sum"]
HORIZONS = (1, 2, 3)

# US EPA PM2.5 24-hour AQI categories (breakpoints revised in 2024), in µg/m³.
AQI_BREAKS = [9.0, 35.4, 55.4, 125.4, 225.4]
AQI_LABELS = ["Good", "Moderate", "Unhealthy for sensitive groups", "Unhealthy", "Very unhealthy", "Hazardous"]


def aqi_category(pm25) -> pd.Series:
    values = pd.Series(pm25, dtype=float)
    codes = np.digitize(values.to_numpy(), AQI_BREAKS, right=True)
    return pd.Series(pd.Categorical.from_codes(codes, categories=AQI_LABELS, ordered=True), index=values.index)


def make_features(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per (city, issue date t): what is known at the end of day t, plus the
    weather forecast for the target day t + horizon.

    In training, target-day weather comes from the ERA5 archive (observed), so the
    backtest assumes a perfect weather forecast. Live forecasts use real weather
    forecasts, which adds error, especially at 3 days.
    """
    rows = []
    for city, g in daily.sort_values("date").groupby("city"):
        g = g.set_index("date").asfreq("D")
        f = pd.DataFrame(index=g.index)
        f["city"] = city
        for lag in (0, 1, 2, 6):
            f[f"pm25_lag{lag}"] = g["pm25"].shift(lag)
        f["pm25_mean7"] = g["pm25"].rolling(7, min_periods=5).mean()
        f["pm25_change1"] = g["pm25"] - g["pm25"].shift(1)
        for col in WEATHER:
            f[f"{col}_today"] = g[col]
            f[f"{col}_target"] = g[col].shift(-horizon)
        target_day = g.index + pd.Timedelta(days=horizon)
        f["doy_sin"] = np.sin(2 * np.pi * target_day.dayofyear / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * target_day.dayofyear / 365.25)
        f["target_date"] = target_day
        f["target"] = g["pm25"].shift(-horizon)
        rows.append(f.reset_index().rename(columns={"index": "date"}))
    frame = pd.concat(rows, ignore_index=True)
    for city in sorted(daily["city"].unique()):
        frame[f"is_{city.lower()}"] = (frame["city"] == city).astype(int)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    skip = {"date", "city", "target", "target_date"}
    return [c for c in frame.columns if c not in skip]


def models() -> dict:
    return {
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=31,
                                                           min_samples_leaf=20, l2_regularization=1.0, random_state=42),
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


def LogTarget(model) -> TransformedTargetRegressor:
    """Fit on log1p(PM2.5): pollution is right-skewed and errors scale with its level."""
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1, check_inverse=False)


def climatology(train: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Mean PM2.5 per city and day of year (±window days), from the training period only."""
    t = train.dropna(subset=["pm25"]).assign(doy=lambda d: d["date"].dt.dayofyear)
    rows = []
    for city, g in t.groupby("city"):
        for doy in range(1, 367):
            dist = np.abs(((g["doy"] - doy + 183) % 366) - 183)
            rows.append({"city": city, "doy": doy, "clim": g.loc[dist <= window, "pm25"].mean()})
    return pd.DataFrame(rows)


def evaluate(pred: pd.Series, actual: pd.Series) -> dict:
    ok = actual.notna() & pred.notna()
    p, a = pred[ok], actual[ok]
    return {
        "MAE": float(np.mean(np.abs(p - a))),
        "RMSE": float(np.sqrt(np.mean((p - a) ** 2))),
        "AQI category accuracy": float((aqi_category(p.to_numpy()).to_numpy() == aqi_category(a.to_numpy()).to_numpy()).mean()),
        "n": int(ok.sum()),
    }


def backtest(daily: pd.DataFrame, split: str = "2025-08-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on everything before `split`, test on everything after (time-ordered, no shuffling)."""
    split_ts = pd.Timestamp(split)
    clim = climatology(daily[daily["date"] < split_ts])
    scores, predictions = [], []
    for h in HORIZONS:
        frame = make_features(daily, h).dropna(subset=["target", "pm25_lag0"])
        cols = feature_columns(frame)
        # The target must also fall before the split, so no test-period day leaks into training.
        train = frame[frame["target_date"] < split_ts].dropna(subset=cols)
        test = frame[frame["date"] >= split_ts].dropna(subset=cols)
        out = test[["city", "date", "target_date", "target"]].copy()
        out["Persistence"] = test["pm25_lag0"]
        out["Climatology"] = (test.assign(doy=test["target_date"].dt.dayofyear)
                              .merge(clim, on=["city", "doy"], how="left")["clim"].to_numpy())
        for name, model in models().items():
            out[name] = LogTarget(model).fit(train[cols], train["target"]).predict(test[cols])
        out["horizon"] = h
        predictions.append(out)
        for method in ["Persistence", "Climatology", *models()]:
            scores.append({"horizon": h, "method": method, **evaluate(out[method], out["target"])})
    return pd.DataFrame(scores), pd.concat(predictions, ignore_index=True)


def fit_final(daily: pd.DataFrame) -> dict:
    """Fit the gradient-boosting model on all data, one per horizon, for live forecasts."""
    fitted = {}
    for h in HORIZONS:
        frame = make_features(daily, h).dropna(subset=["target", "pm25_lag0"])
        cols = feature_columns(frame)
        frame = frame.dropna(subset=cols)
        fitted[h] = {"model": LogTarget(models()["Gradient boosting"]).fit(frame[cols], frame["target"]), "columns": cols}
    return fitted
