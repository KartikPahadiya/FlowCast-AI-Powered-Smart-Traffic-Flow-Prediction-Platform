# FlowCast — Final Technical Report

**Meridian Mobility Systems · Data & AI Division · Northline Corridor**
Pipeline version 1.0 · seed 42 · produced by `run_pipeline.py`

---

## 1. What was built

FlowCast is an end-to-end, one-command-reproducible system that turns raw
corridor telemetry into short-horizon forecasts of **traffic volume, travel
time, congestion level and accident risk** for each of the 25 Northline
segments at 30-minute grain:

```
raw tables → M1 ingest/validate → M2 clean/merge → M3 features → M4 EDA
           → M5 classical ML → M6 from-scratch LSTM → M7 Streamlit dashboard
```

Every module is independent, seeded, and testable; every prediction on the
dashboard is traceable to a persisted, carded model.

## 2. Data engineering (M1–M3)

| Step | What happened |
| --- | --- |
| Ingest | 178,469 traffic + 10,872 weather + 151 calendar rows loaded with strict schema/type/range validation |
| Dedup | 1,767 exact duplicate sensor rows dropped (kept most-complete record on key collisions) |
| Outliers | 704 physically impossible values (negative volume, speed > 200 km/h, occupancy > 100 %) ruled to null |
| Missing windows | 4,499 fully-dropped 30-minute windows restored by reindexing each segment onto the full 48-window daily grid; partial gaps time-interpolated, then segment × time-of-day median fallback (27k+ values repaired) |
| Congestion | 31,123 blank labels derived from V/C = volume ÷ (capacity/2) banding (<0.50 Free-flow · 0.50–0.79 Moderate · 0.80–0.99 Heavy · ≥1.00 Severe) |
| Weather | 13 messy labels mapped to Clear/Cloudy/Rain/Fog; DD/MM/YYYY parsed strictly; hourly rows broadcast to both half-hour windows; 278 temperature/visibility gaps station-interpolated |
| Calendar | holidays, events, roadworks joined on date; holiday×peak interaction |
| Features | 25 features: cyclical hour/day, weekend/peak flags, lags t−1/t−2/t−48, rolling means/std (4 & 8 windows), lag-based V/C ratio & headroom, rain/low-visibility flags, vehicle-mix shares |
| Leakage guard | all lags/rollings computed within segment, shifted one window; V/C uses lagged volume only; scalers fit on train only |

Output: 181,200 clean records (25 × 151 × 48). Full detail:
`reports/data_quality_report.md`.

**Split:** time-based on the sorted timeline — train ≤ 2025-04-06 (70 %),
validation ≤ 2025-05-09 (15 %), test 2025-05-09 → 2025-05-31 (15 %), never random.

## 3. Mathematics in the pipeline (PRD §10)

| Concept | Where it lives |
| --- | --- |
| Matrix–vector products | `LinearRegressionFromScratch` predicts Xw; the LSTM layers compute Wx + b |
| Gradient descent | analytic d(MSE)/dw = (2/n)·Xᵀ(Xw − y), batch GD, 2,000 iterations — converges to sklearn-parity (RMSE 71.9 vs 70.9) |
| Probability | accident-risk output is a calibrated ranking probability; congestion outputs are class probabilities |
| Descriptive statistics | median imputation, mode fills, EDA summaries |
| Variance / std | rolling-volatility features, z-score outlier screening, confidence bands |
| Correlation / covariance | EDA redundancy check; feature selection |
| Scaling | z-score for SVM/GD, standardisation inside the LSTM data pipeline |
| Loss functions | MSE (+MAE/MAPE/R²) for regression; cross-entropy for classification; LSTM loss = MSE + 0.5·CE |

## 4. Results (all on the untouched test window)

### 4.1 Volume regression — primary metric RMSE / MAPE

| Model | RMSE (veh) | MAPE | R² | Train (s) |
| --- | --- | --- | --- | --- |
| Linear Regression (sklearn) | 70.91 | 13.55 % | 0.9389 | <1 |
| **Linear Regression (from scratch, NumPy GD)** | 71.88 | 14.18 % | 0.9372 | 47 |
| Decision Tree | 64.05 | 10.16 % | 0.9502 | 4 |
| Random Forest (150 trees) | 57.91 | 9.12 % | 0.9593 | 51 |
| **XGBoost ✅ winner** | **55.20** | **8.80 %** | **0.9630** | 4 |
| LSTM (from scratch) | 58.31 | 10.18 % | 0.9587 | 124 |

**Success criterion MAPE ≤ 12 %: MET (8.80 %).**

### 4.2 Travel-time regression

| Model | RMSE (min) | MAPE | R² |
| --- | --- | --- | --- |
| Linear Regression (both implementations) | 1.90 | 41.3 %* | 0.45 |
| Decision Tree | 1.21 | 10.50 % | 0.777 |
| Random Forest | 1.10 | 9.62 % | 0.817 |
| **XGBoost ✅ winner** | **1.04** | **9.21 %** | **0.836** |

\* LR MAPE is inflated by near-zero travel times in free-flow night windows
(MAPE is undefined-skewed at y≈0); RMSE/R² tell the real story.

### 4.3 Congestion classification (4-class) — macro-F1

| Model | macro-F1 | accuracy | ROC-AUC (OVR) |
| --- | --- | --- | --- |
| Decision Tree | 0.7491 | 0.8640 | 0.968 |
| **Random Forest ✅ winner** | **0.7638** | **0.8753** | 0.974 |
| XGBoost | 0.7586 | 0.8751 | 0.974 |
| SVM (RBF, 25k subsample) | 0.7520 | 0.8670 | 0.954 |
| LSTM (from scratch) | 0.7586 | 0.8734 | — |

**Success criterion macro-F1 ≥ 0.80: NOT MET (0.764).** Error analysis shows
confusion concentrated between **Heavy ↔ Severe** — expected, since the two
bands share feature space and Severe is rare (≈5.5 % of windows). Remedy
roadmapped in §7.

### 4.4 Accident-risk classification — ROC-AUC

| Model | ROC-AUC | note |
| --- | --- | --- |
| Decision Tree | 0.529 | |
| **Random Forest ✅ winner** | **0.622** | best ranking quality |
| XGBoost | 0.566 | |
| SVM | 0.542 | |

**Success criterion ROC-AUC ≥ 0.75: NOT MET.** This is a *data* limitation,
not a modelling one: incidents are ≈0.9 % of windows and near-uniform across
segments. The empirical **Bayes ceiling** — AUC of the true per-group rates
P(accident | weather × congestion × hour), i.e. the best any model could do
with perfect knowledge of the generative groups — is **≈ 0.68**, below the
0.75 target. The model still ranks risk 4× above base rate in its top decile,
which is operationally useful for patrol prioritisation; the target itself is
unattainable on this dataset.

### 4.5 Deep vs classical (head-to-head, PRD §12.4 / §20)

| | XGBoost (best classical) | LSTM (from scratch) |
| --- | --- | --- |
| Volume RMSE | **55.20** | 58.31 |
| Volume MAPE | **8.80 %** | 10.18 % |
| Congestion macro-F1 | **0.764** | 0.759 |

**The LSTM did not earn its added complexity on volume RMSE** — the honest
verdict, recorded in `reports/benchmark_report.md`. Diagnosis: with rich
lag/rolling features the tabular signal is already well captured; the LSTM's
advantage would likely appear with longer horizons, sparser features, or
network-context inputs (roadmap §7). Its regularisation behaved exactly as
designed: validation MSE (0.0396) tracked below training MSE (0.0423) at
early stopping — convergence, not memorisation.

## 5. Confidence & governance (PRD §13.4, FR-11)

- Volume forecasts ship with ±1.96σ bands from validation residuals
  (`volume_lower/upper`); empirical coverage is shown per segment on the
  **Prediction confidence** view.
- Congestion predictions carry the predicted class probability.
- Every model has a versioned card in `reports/model_cards/` recording
  training window, feature set, hyperparameters and test metrics.

## 6. Dashboard (M7)

`streamlit run dashboard/app.py` — all nine PRD views plus upload /
predict / report modules, every number a real model output:

Live prediction · Historical trends · Congestion heatmap · Road comparison ·
Model performance · Feature importance · Forecast visualisation ·
Prediction confidence · Weather vs traffic · Predict (custom) ·
Data upload (validates, cleans and quarantines bad input) · Reports & insights
(range summary export).

## 7. Recommendations / roadmap

1. **Heavy↔Severe confusion:** derive an ordinal regression head or tune
   per-class thresholds on validation to push macro-F1 toward 0.80.
2. **Accident risk:** renegotiate the 0.75 AUC target against the measured
   ≈0.68 Bayes ceiling, or enrich the target with incident severity/near-miss
   data.
3. **Sequence modelling:** re-visit the LSTM at +60…+120 min horizons, where
   trajectory information should dominate instantaneous features.
4. PRD §19 items: streaming ingestion, graph models over network topology,
   probabilistic (quantile) forecasting, drift monitoring.

## 8. Reproduction

```bash
pip install -r requirements.txt
python run_pipeline.py            # end-to-end, ~8 min on a workstation
streamlit run dashboard/app.py    # dashboard
```

Stages (`data`, `classical` or `m5a–m5d`, `dl`, `finalize`) are individually
runnable and resumable. Everything is seeded (seed 42).
