# Model Card — RandomForest (congestion)

| Field | Value |
| --- | --- |
| Model | `RandomForest` |
| Task / target | classification → `congestion` |
| Training window | first 70 % of the timeline (2025-01-01 → 2025-04-06) |
| Validation window | next 15 % (2025-04-06 → 2025-05-09) — used for early stopping / tuning |
| Test window | final 15 % (2025-05-09 → 2025-05-31), never touched during training |
| Split | time-based on the 30-minute timeline, never random (PRD §13.1) |
| Features | 25 engineered features: cyclical time encodings, lags (t-1, t-2, t-48), rolling means/std (4, 8 windows), V/C ratio, weather flags, calendar interactions |
| Primary metric | **macro_F1 = 0.7638** |

## Test metrics

{
  "accuracy": 0.8753290083410565,
  "macro_precision": 0.7753591156604503,
  "macro_recall": 0.7563557995076869,
  "ROC_AUC": 0.9743941226690231,
  "train_seconds": 11.6
}

## Confusion matrix (test)

[[15524, 750, 0, 0], [604, 5469, 423, 16], [0, 578, 1717, 337], [0, 12, 643, 902]]



_Card generated 2026-10-06 · seed 42 · pipeline version FlowCast AI-Powered Smart Traffic Flow Prediction Platform_
