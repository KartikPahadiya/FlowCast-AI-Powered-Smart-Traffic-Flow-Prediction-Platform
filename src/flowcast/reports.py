"""Model cards and the classical-vs-deep benchmark report (PRD §13.4, §20)."""
import json
import logging
from datetime import date

from . import config

log = logging.getLogger("flowcast.reports")

REG_TARGETS = ["traffic_volume", "travel_time"]
CLF_TARGETS = ["congestion", "accident_risk"]
LABELS = {"congestion": ["Free-flow", "Moderate", "Heavy", "Severe"],
          "accident_risk": ["no incident", "incident"]}


def write_model_card(target, model_name, metrics, split_desc, extra=""):
    primary = ("RMSE" if target in REG_TARGETS else
               "macro_F1" if target == "congestion" else "ROC_AUC")
    others = {k: v for k, v in metrics.items()
              if k not in (primary, "confusion_matrix", "note")}
    md = f"""# Model Card — {model_name} ({target})

| Field | Value |
| --- | --- |
| Model | `{model_name}` |
| Task / target | {"regression" if target in REG_TARGETS else "classification"} → `{target}` |
| Training window | first 70 % of the timeline (2025-01-01 → 2025-04-06) |
| Validation window | next 15 % (2025-04-06 → 2025-05-09) — used for early stopping / tuning |
| Test window | final 15 % (2025-05-09 → 2025-05-31), never touched during training |
| Split | time-based on the 30-minute timeline, never random (PRD §13.1) |
| Features | 25 engineered features: cyclical time encodings, lags (t-1, t-2, t-48), rolling means/std (4, 8 windows), V/C ratio, weather flags, calendar interactions |
| Primary metric | **{primary} = {metrics[primary]:.4g}** |

## Test metrics

{json.dumps(others, indent=2)}

## Confusion matrix (test)

{json.dumps(metrics.get("confusion_matrix", "n/a"))}

{extra}

_Card generated {date.today().isoformat()} · seed {config.RANDOM_SEED} · pipeline version {config.PROJECT_ROOT.name}_
"""
    fname = config.MODEL_CARDS_DIR / f"{target}__{model_name.replace(' ', '_')}.md"
    fname.write_text(md, encoding="utf-8")
    return fname


def write_benchmark(scoreboard, lstm_vol, lstm_cong, dl_beat_classical, path):
    def row(tgt, metric):
        lines = [f"| {tgt} | {m} | {v.get(metric, '—'):.4g} |"
                 for m, v in scoreboard.get(tgt, {}).items() if isinstance(v, dict)]
        return "\n".join(lines)

    md = f"""# FlowCast — Classical vs Deep-Learning Benchmark (PRD §12.4 / §20)

Same test window (2025-05-09 → 2025-05-31), same metrics for every model.

## Volume regression (primary metric RMSE, lower is better)

| Model | RMSE | MAPE % | R² |
| --- | --- | --- | --- |
"""
    for m, v in scoreboard["traffic_volume"].items():
        md += f"| {m} | {v['RMSE']:.2f} | {v['MAPE']:.2f} | {v['R2']:.4f} |\n"
    md += f"""| **LSTM (from scratch)** | **{lstm_vol['RMSE']:.2f}** | {lstm_vol['MAPE']:.2f} | {lstm_vol['R2']:.4f} |

## Travel-time regression (RMSE, lower is better)

| Model | RMSE | MAPE % | R² |
| --- | --- | --- | --- |
"""
    for m, v in scoreboard["travel_time"].items():
        md += f"| {m} | {v['RMSE']:.4g} | {v['MAPE']:.2f} | {v['R2']:.4f} |\n"
    md += """
## Congestion classification (macro-F1, higher is better)

| Model | macro-F1 | accuracy | ROC-AUC (OVR) |
| --- | --- | --- | --- |
"""
    for m, v in scoreboard["congestion"].items():
        md += (f"| {m} | {v['macro_F1']:.4f} | {v['accuracy']:.4f} | "
               f"{v.get('ROC_AUC') or float('nan'):.4f} |\n")
    md += (f"| **LSTM (from scratch)** | **{lstm_cong['macro_F1']:.4f}** | "
           f"{lstm_cong['accuracy']:.4f} | — |\n")
    md += """
## Accident-risk classification (ROC-AUC, higher is better)

| Model | ROC-AUC | macro-F1 | recall (incident class) |
| --- | --- | --- | --- |
"""
    for m, v in scoreboard["accident_risk"].items():
        md += (f"| {m} | {v['ROC_AUC']:.4f} | {v['macro_F1']:.4f} | "
               f"{v['macro_recall']:.4f} |\n")
    verdict = ("earned its added complexity — it beats the best classical "
               if dl_beat_classical else
               "did NOT beat the best classical model on volume RMSE — the "
               "classical ensemble remains the production model")
    md += f"""
## Verdict

On the shared test window the from-scratch LSTM **{verdict}**
model for volume forecasting. Full reasoning is recorded in the final report.
"""
    path.write_text(md, encoding="utf-8")
    log.info("benchmark report written to %s", path)
