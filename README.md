# FlowCast — AI-Powered Smart Traffic Flow Prediction Platform

An end-to-end AI system that forecasts **traffic volume, congestion level,
travel time and accident risk** for a 25-segment urban arterial corridor
(the *Northline Corridor*) at 30-minute granularity, built to the FlowCast
PRD v1.0 (Meridian Mobility Systems).

Raw sensor telemetry + weather + calendar → validated/cleaned/merged dataset
→ engineered features → classical ML family **and a from-scratch LSTM** →
interactive Streamlit dashboard with confidence-banded forecasts.

## Highlights

| Task | Best model | Test result | PRD target |
| --- | --- | --- | --- |
| Volume forecast | XGBoost | **MAPE 8.80 %** (RMSE 55.2, R² 0.963) | ≤ 12 % ✅ |
| Travel time | XGBoost | RMSE 1.04 min (R² 0.836) | — |
| Congestion (4-class) | Random Forest | macro-F1 0.764 | ≥ 0.80 (near miss, see report) |
| Accident risk | Random Forest | ROC-AUC 0.622 | ≥ 0.75 (data ceiling ≈ 0.68, see report) |
| DL vs ML benchmark | — | LSTM RMSE 58.31 vs XGB 55.20 → classical wins honestly | — |

- **Linear regression implemented from scratch** in NumPy (analytic gradient,
  batch gradient descent) — matches sklearn to within 1.5 % RMSE.
- **LSTM built and trained from scratch** in PyTorch (2-layer, dropout,
  multi-task volume + congestion heads, early stopping, LR-on-plateau).
- **Zero-leakage protocol:** time-based train/val/test split, all lags/rollings
  shifted one window within segment, scalers fit on train only.
- Deterministic, seeded (42), one-command reproducible.

## Quick start

```bash
pip install -r requirements.txt

# end-to-end pipeline (~8 min): ingest → clean → features → EDA
#   → classical ML → LSTM → predictions/scoreboard/cards
python run_pipeline.py

# or stage by stage (each stage is resumable):
python run_pipeline.py data        # M1-M4: cleaning, features, EDA, DQ report
python run_pipeline.py classical   # M5: all classical models
python run_pipeline.py dl          # M6: from-scratch LSTM
python run_pipeline.py finalize    # scoreboard, model cards, predictions

# dashboard (9 PRD views + upload/predict/report)
streamlit run dashboard/app.py
```

## Repository layout

```
data/raw/                  three source CSVs (delivered dataset)
data/processed/            versioned analysis-ready tables (generated)
src/flowcast/
  ingest.py                M1 — schema validation, typed load, quality log
  clean.py                 M2 — dedup, outliers, imputation, harmonisation, merge
  features.py              M3 — lags, rollings, encodings, targets, time split
  eda.py                   M4 — figures + data-quality report writer
  models_classical.py      M5 — LR(from scratch)/DT/RF/XGB/SVM + metrics
  models_dl.py             M6 — from-scratch LSTM trainer & evaluator
  reports.py               model cards + classical-vs-deep benchmark
  config.py                every seed/split/path in one place
dashboard/app.py           M7 — Streamlit operations dashboard
notebooks/eda.ipynb        executed EDA notebook (build via build_eda_notebook.py)
reports/                   data-quality report · benchmark · final report · model cards · figures
models/artifacts/          persisted models, scoreboard, LSTM weights (generated)
run_pipeline.py            one-command orchestrator with resumable stages
```

## Data & methodology (summary)

- **Grain:** segment × 30-minute window · 25 segments · 2025-01-01 → 2025-05-31 · 181,200 clean records.
- **Cleaning:** 1,767 duplicates removed, 704 impossible values repaired, 4,499 dropped sensor windows restored onto the full daily grid, 31,123 congestion labels derived from the V/C banding, hourly weather broadcast to both half-hour windows, 13 weather label variants harmonised, mixed date formats parsed strictly.
- **Features:** 25 engineered features (cyclical time, lags t−1/t−2/t−48, rolling stats, lag-based V/C, weather flags, calendar interactions).
- **Validation:** earliest 70 % trains, next 15 % validates, final 15 % tests — never random; SVM baseline trained on a seeded 25k subsample (cubic cost), documented on its model card.
- **Honest reporting:** two PRD success criteria are not met, with quantified evidence — the congestion macro-F1 gap (0.764 vs 0.80, Heavy↔Severe confusion) and the accident-risk AUC ceiling (Bayes-optimal ≈ 0.68 < 0.75 target). Details in `reports/final_report.md`.

## Reports

- [reports/data_quality_report.md](reports/data_quality_report.md) — every defect found and its resolution
- [reports/benchmark_report.md](reports/benchmark_report.md) — classical vs deep head-to-head
- [reports/final_report.md](reports/final_report.md) — mathematics, methods, results, recommendations
- [reports/model_cards/](reports/model_cards/) — per-model lineage and test metrics
- [notebooks/eda.ipynb](notebooks/eda.ipynb) — executed EDA narrative

## Stack

Python 3.11+ · NumPy · Pandas · scikit-learn · XGBoost · PyTorch ·
Streamlit · Plotly · Matplotlib/Seaborn · Jupyter (pinned in `requirements.txt`)

---

*Built as a single-engineer four-week capstone sprint against FlowCast PRD v1.0.*
