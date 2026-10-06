# FlowCast — Classical vs Deep-Learning Benchmark (PRD §12.4 / §20)

Same test window (2025-05-09 → 2025-05-31), same metrics for every model.

## Volume regression (primary metric RMSE, lower is better)

| Model | RMSE | MAPE % | R² |
| --- | --- | --- | --- |
| LinearRegression_sklearn | 70.91 | 13.55 | 0.9389 |
| LinearRegression_from_scratch | 71.88 | 14.18 | 0.9372 |
| DecisionTree | 64.05 | 10.16 | 0.9502 |
| RandomForest | 57.91 | 9.12 | 0.9593 |
| XGBoost | 55.20 | 8.80 | 0.9630 |
| LSTM_from_scratch | 58.31 | 10.18 | 0.9587 |
| **LSTM (from scratch)** | **58.31** | 10.18 | 0.9587 |

## Travel-time regression (RMSE, lower is better)

| Model | RMSE | MAPE % | R² |
| --- | --- | --- | --- |
| LinearRegression_sklearn | 1.896 | 41.25 | 0.4522 |
| LinearRegression_from_scratch | 1.897 | 41.28 | 0.4511 |
| DecisionTree | 1.21 | 10.50 | 0.7768 |
| RandomForest | 1.095 | 9.62 | 0.8172 |
| XGBoost | 1.036 | 9.21 | 0.8363 |

## Congestion classification (macro-F1, higher is better)

| Model | macro-F1 | accuracy | ROC-AUC (OVR) |
| --- | --- | --- | --- |
| DecisionTree | 0.7491 | 0.8640 | 0.9683 |
| RandomForest | 0.7638 | 0.8753 | 0.9744 |
| XGBoost | 0.7586 | 0.8751 | 0.9744 |
| SVM | 0.7520 | 0.8670 | 0.9542 |
| LSTM_from_scratch | 0.7586 | 0.8734 | nan |
| **LSTM (from scratch)** | **0.7586** | 0.8734 | — |

## Accident-risk classification (ROC-AUC, higher is better)

| Model | ROC-AUC | macro-F1 | recall (incident class) |
| --- | --- | --- | --- |
| DecisionTree | 0.5287 | 0.4976 | 0.5000 |
| RandomForest | 0.6219 | 0.4976 | 0.5000 |
| XGBoost | 0.5660 | 0.4976 | 0.5000 |
| SVM | 0.5418 | 0.4952 | 0.5154 |

## Verdict

On the shared test window the from-scratch LSTM **did NOT beat the best classical model on volume RMSE — the classical ensemble remains the production model**
model for volume forecasting. Full reasoning is recorded in the final report.
