import numpy as np
import pandas as pd
import pytest

from smog.data import to_daily
from smog.model import aqi_category, climatology, evaluate, make_features


def synthetic_daily(days: int = 40, city: str = "Lahore") -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "date": dates, "city": city, "pm25": np.arange(days, dtype=float) + 10,
        "temp_mean": 15.0, "temp_range": 8.0, "rh_mean": 60.0, "wind_mean": 5.0,
        "blh_mean": 400.0, "blh_night": np.arange(days, dtype=float) * 10, "precip_sum": 0.0,
    })


@pytest.mark.parametrize("value, expected", [
    (9.0, "Good"), (9.1, "Moderate"), (35.4, "Moderate"), (35.5, "Unhealthy for sensitive groups"),
    (55.5, "Unhealthy"), (125.5, "Very unhealthy"), (300.0, "Hazardous"),
])
def test_aqi_categories_follow_epa_breakpoints(value, expected):
    assert aqi_category([value]).iloc[0] == expected


def test_features_use_only_past_pollution_and_target_day_weather():
    frame = make_features(synthetic_daily(), horizon=2).set_index("date")
    row = frame.loc["2025-01-10"]
    assert row["pm25_lag0"] == 19 and row["pm25_lag1"] == 18 and row["pm25_lag6"] == 13
    assert row["target"] == 21  # value two days later
    assert row["blh_night_today"] == 90 and row["blh_night_target"] == 110
    assert row["target_date"] == pd.Timestamp("2025-01-12")


def test_no_target_beyond_the_data():
    frame = make_features(synthetic_daily(days=10), horizon=3)
    assert frame.sort_values("date")["target"].tail(3).isna().all()


def test_to_daily_masks_days_with_too_few_hours():
    idx = pd.date_range("2025-01-01", periods=48, freq="h")
    hourly = pd.DataFrame({"pm2_5": 50.0, "temperature_2m": 10.0, "relative_humidity_2m": 70.0,
                           "wind_speed_10m": 3.0, "boundary_layer_height": 300.0, "precipitation": 0.0}, index=idx)
    hourly.loc["2025-01-02 00:00":"2025-01-02 09:00", "pm2_5"] = np.nan  # only 14 valid hours left
    daily = to_daily(hourly)
    assert daily.loc["2025-01-01", "pm25"] == 50.0
    assert np.isnan(daily.loc["2025-01-02", "pm25"])


def test_to_daily_drops_implausible_near_zero_days():
    idx = pd.date_range("2025-01-01", periods=24, freq="h")
    hourly = pd.DataFrame({"pm2_5": 0.7, "temperature_2m": 10.0, "relative_humidity_2m": 70.0,
                           "wind_speed_10m": 3.0, "boundary_layer_height": 300.0, "precipitation": 0.0}, index=idx)
    assert np.isnan(to_daily(hourly).loc["2025-01-01", "pm25"])


def test_climatology_wraps_around_the_new_year():
    train = pd.DataFrame({"city": "Lahore", "date": pd.to_datetime(["2024-12-30", "2025-01-02"]), "pm25": [100.0, 200.0]})
    clim = climatology(train, window=3).set_index(["city", "doy"])
    assert clim.loc[("Lahore", 1), "clim"] == pytest.approx(150.0)


def test_evaluate_ignores_missing_values():
    scores = evaluate(pd.Series([10.0, np.nan, 40.0]), pd.Series([12.0, 5.0, np.nan]))
    assert scores["n"] == 1 and scores["MAE"] == pytest.approx(2.0)
