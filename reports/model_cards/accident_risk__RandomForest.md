# Model Card — RandomForest (accident_risk)

| Field | Value |
| --- | --- |
| Model | `RandomForest` |
| Task / target | classification → `accident_risk` |
| Training window | first 70 % of the timeline (2025-01-01 → 2025-04-06) |
| Validation window | next 15 % (2025-04-06 → 2025-05-09) — used for early stopping / tuning |
| Test window | final 15 % (2025-05-09 → 2025-05-31), never touched during training |
| Split | time-based on the 30-minute timeline, never random (PRD §13.1) |
| Features | 25 engineered features: cyclical time encodings, lags (t-1, t-2, t-48), rolling means/std (4, 8 windows), V/C ratio, weather flags, calendar interactions |
| Primary metric | **ROC_AUC = 0.6219** |

## Test metrics

{
  "macro_F1": 0.4976067643826942,
  "accuracy": 0.9904726598702502,
  "macro_precision": 0.4952363299351251,
  "macro_recall": 0.5,
  "train_seconds": 13.2
}

## Confusion matrix (test)

[[26718, 0], [257, 0]]



_Card generated 2026-10-06 · seed 42 · pipeline version FlowCast AI-Powered Smart Traffic Flow Prediction Platform_
