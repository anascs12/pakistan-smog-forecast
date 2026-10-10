## season
- **Lahore stands apart.** Its average PM2.5 is **140 µg/m³ in January**, 117 in December and 103 in November, against about 36 in April, and it never drops to "Good". On **52% of all days** Lahore's air is *Unhealthy* or worse by US EPA categories, and in November–January **41% of days reach "Very unhealthy" or "Hazardous"** levels (above 125.4 µg/m³).
- Peshawar and Islamabad follow a milder version of the same winter pattern (about 55–58 µg/m³ in January), while **Karachi is cleanest** (three-quarters of days "Moderate"), helped by sea breezes.
- Every winter repeats the same build-up. Lahore's worst 7-day average reached 211 µg/m³ in winter 2022–23 and **219 µg/m³ in winter 2025–26**.
- Even Karachi's cleanest months stay above the WHO 24-hour guideline of 15 µg/m³ on most days.

These are CAMS model estimates, which are known to underestimate ground-level extremes in South Asia, so the real picture, especially in Peshawar, is likely worse.

## drivers
- In **Lahore**, PM2.5 rises sharply when the **night-time mixing height** is low (Spearman ρ = −0.54) and when **wind** is weak (ρ = −0.49). On calm winter nights the boundary layer can collapse to just tens of metres, trapping smoke from vehicles, industry, brick kilns and crop burning near the ground: a temperature inversion.
- Cold days are dirtier (ρ = −0.47 with temperature), mostly because cold days are calm, stable winter days rather than because of temperature itself.
- **Karachi** behaves differently: there, *humid* days are cleaner (ρ = −0.43) because humidity arrives with the sea breeze that ventilates the city, while in the three northern cities humidity goes with *more* pollution (fog and haze).
- In this model data, Peshawar's and Islamabad's PM2.5 depends less on local mixing (ρ ≈ −0.10 to −0.15). One plausible reason is that CAMS represents their pollution as more regionally transported; ground sensors might show a stronger local signal. That is a hypothesis, not something this data can confirm.

## backtest
- The gradient-boosting model **beats both baselines at every horizon** on a test period it never saw (August 2025 – September 2026): MAE **6.4, 8.0 and 8.7 µg/m³** at 1, 2 and 3 days, which is **15%, 23% and 25% better than persistence**.
- The gain is largest when it matters most. In the **smog season (November–January)** the improvement grows to **21–31%**; at 3 days ahead the error falls from 18.6 (persistence) to 12.8 µg/m³.
- Persistence is a strong baseline at 1 day because pollution changes slowly. The model's advantage grows with the horizon, because it knows the coming weather (calm, low-mixing days ahead) while persistence does not.
- The seasonal average is weak (MAE ≈ 10.7) because it cannot follow day-to-day swings, and Ridge regression is worse than gradient boosting because the weather effects are non-linear (for example, mixing height only matters once it gets very low).
- On the Lahore 2025–26 chart, the 1-day forecast tracks the build-ups and clean-outs closely, and the 3-day forecast catches most of the big episodes but smooths and sometimes mistimes the sharpest peaks.
- **Important caveat:** in the backtest, the target day's weather comes from observed weather data (ERA5), i.e. a *perfect* weather forecast. Live forecasts use real weather forecasts, so real-world errors will be somewhat larger, especially at 3 days.

## importance
- **Today's PM2.5** is by far the most important input (shuffling it raises the error by about 11 µg/m³): pollution builds up and clears over several days.
- Among weather variables, the **target day's wind** (+1.0), **humidity** (+0.8) and **night-time mixing height** (+0.6) matter most. The model uses the *forecast* conditions of the day being predicted, not just today's, which is why it beats persistence at longer horizons.
- The 7-day average PM2.5 adds context about whether the city is in an episode, and the Lahore indicator captures its much higher baseline.
- Permutation importance measures each input's unique contribution; correlated weather variables share credit, so individual numbers should not be over-interpreted.

## conclusions
**Findings**
1. Lahore's winter smog is severe and systematic: in November–January, 41% of days are "Very unhealthy" or worse even in conservative model data.
2. Calm air and a collapsed night-time boundary layer are the main weather conditions that trap the smog.
3. A gradient-boosting model using recent pollution and the weather forecast beats persistence by 15–25%, rising to 21–31% in smog season, at forecasting 1–3 days ahead.
4. The forecast is published automatically every morning by a GitHub Action, so the project keeps working without manual effort.

**Limitations**
- PM2.5 comes from the CAMS model, not ground monitors; it underestimates extremes, so levels are conservative.
- The backtest assumes perfect weather forecasts; live accuracy will be lower.
- City-wide daily averages hide hotspots and hourly peaks.
- One test period (one smog winter): results could differ in an unusual year.
- Emissions (crop burning, traffic, industry) are not modelled directly, only through their past effect on PM2.5.

**Next steps:** validate against ground monitors (for example the US consulate monitors or Punjab EPA stations), add satellite fire counts from crop burning as a feature, and forecast hourly peaks.
