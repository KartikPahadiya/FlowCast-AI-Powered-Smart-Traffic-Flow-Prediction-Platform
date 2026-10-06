# Model Card — XGBoost (traffic_volume)

| Field | Value |
| --- | --- |
| Model | `XGBoost` |
| Task / target | regression → `traffic_volume` |
| Training window | first 70 % of the timeline (2025-01-01 → 2025-04-06) |
| Validation window | next 15 % (2025-04-06 → 2025-05-09) — used for early stopping / tuning |
| Test window | final 15 % (2025-05-09 → 2025-05-31), never touched during training |
| Split | time-based on the 30-minute timeline, never random (PRD §13.1) |
| Features | 25 engineered features: cyclical time encodings, lags (t-1, t-2, t-48), rolling means/std (4, 8 windows), V/C ratio, weather flags, calendar interactions |
| Primary metric | **RMSE = 55.2** |

## Test metrics

{
  "MAE": 37.362614304875756,
  "MAPE": 8.796050255596002,
  "R2": 0.9629757391770912,
  "train_seconds": 3.9
}

## Confusion matrix (test)

"n/a"



_Card generated 2026-10-06 · seed 42 · pipeline version FlowCast AI-Powered Smart Traffic Flow Prediction Platform_
