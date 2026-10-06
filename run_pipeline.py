"""One-command pipeline (PRD §3.3 / §18).

    python run_pipeline.py            # everything, in order
    python run_pipeline.py data       # M1-M4: ingest, clean, features, EDA
    python run_pipeline.py classical  # M5: all classical models
    python run_pipeline.py dl         # M6: from-scratch LSTM
    python run_pipeline.py finalize   # scoreboard merge, cards, predictions artifact

Stages are resumable: each persists its outputs, so a failed or long
stage can be rerun on its own.
"""
import argparse
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd

from src.flowcast import (clean, config, eda, features, ingest,
                          models_classical, models_dl, reports)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("flowcast.pipeline")

TRAINER_PKL = config.MODELS_DIR / "trainer.pkl"
LSTM_METRICS = config.MODELS_DIR / "lstm_metrics.json"


def stage_data():
    t0 = time.time()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cal, wth, trf = ingest.load_all()
    qlog = ingest.QualityLog()
    cleaned = clean.run_cleaning(cal, wth, trf, qlog)
    cleaned.to_parquet(config.PROCESSED_PARQUET, index=False)
    qlog.to_frame().to_csv(config.REPORTS_DIR / "_quality_log.csv", index=False)
    eda.write_quality_report(qlog.to_frame(), cleaned, config.QUALITY_REPORT)

    feat = features.build_features(cleaned)
    feat.to_parquet(config.FEATURES_PARQUET, index=False)
    train, val, test = features.time_split(feat)
    log.info("splits — train %d / val %d / test %d",
             len(train), len(val), len(test))
    eda.run_eda(cleaned)
    log.info("stage=data done in %.0fs", time.time() - t0)


def stage_classical():
    t0 = time.time()
    feat = pd.read_parquet(config.FEATURES_PARQUET)
    train, val, test = features.time_split(feat)
    trainer = models_classical.Trainer(train, val, test)
    trainer.train_regressors("traffic_volume")
    trainer.train_classifiers("congestion", "congestion_code", [0, 1, 2, 3])
    pos = int((train["accident_risk"] == 0).sum())
    trainer.train_classifiers("accident_risk", "accident_risk", [0, 1],
                              pos_weight=max(1, pos // max(1, int(train["accident_risk"].sum()))))
    trainer.train_regressors("travel_time")
    models_classical.save_artifacts(trainer)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trainer, TRAINER_PKL)
    log.info("stage=classical done in %.0fs", time.time() - t0)


def _get_trainer():
    if TRAINER_PKL.exists():
        return joblib.load(TRAINER_PKL)
    feat = pd.read_parquet(config.FEATURES_PARQUET)
    train, val, test = features.time_split(feat)
    return models_classical.Trainer(train, val, test)


def _save_trainer(trainer):
    models_classical.save_artifacts(trainer)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trainer, TRAINER_PKL)


def stage_m5a():  # volume regression
    trainer = _get_trainer()
    trainer.train_regressors("traffic_volume")
    _save_trainer(trainer)


def stage_m5b():  # congestion classification
    trainer = _get_trainer()
    trainer.train_classifiers("congestion", "congestion_code", [0, 1, 2, 3])
    _save_trainer(trainer)


def stage_m5c():  # accident-risk classification
    from src.flowcast.features import RISK_FEATURE_COLS
    trainer = _get_trainer()
    trainer.train_classifiers("accident_risk", "accident_risk", [0, 1],
                              pos_weight=3, feature_cols=RISK_FEATURE_COLS)
    _save_trainer(trainer)


def stage_m5d():  # travel-time regression
    trainer = _get_trainer()
    trainer.train_regressors("travel_time")
    _save_trainer(trainer)
    log.info("classical substages complete")


def stage_dl():
    t0 = time.time()
    feat = pd.read_parquet(config.FEATURES_PARQUET)
    lstm, lstm_data, history = models_dl.train_lstm(feat)
    lstm_vol, lstm_cong, lstm_res = models_dl.evaluate_lstm(lstm, lstm_data)
    models_dl.save_lstm(lstm, history, config.MODELS_DIR / "lstm.pt")
    json.dump(history, open(config.MODELS_DIR / "lstm_history.json", "w"))
    lstm_res.to_parquet(config.MODELS_DIR / "lstm_test_predictions.parquet",
                        index=False)
    json.dump({"volume": lstm_vol, "congestion": lstm_cong},
              open(LSTM_METRICS, "w"), indent=2)
    log.info("LSTM volume RMSE=%.2f MAPE=%.2f%% | congestion macroF1=%.4f",
             lstm_vol["RMSE"], lstm_vol["MAPE"], lstm_cong["macro_F1"])
    log.info("stage=dl done in %.0fs", time.time() - t0)


def stage_finalize():
    t0 = time.time()
    trainer = joblib.load(TRAINER_PKL)
    lstm_m = json.load(open(LSTM_METRICS))
    trainer.results["traffic_volume"]["LSTM_from_scratch"] = lstm_m["volume"]
    trainer.results["congestion"]["LSTM_from_scratch"] = lstm_m["congestion"]
    with open(config.SCOREBOARD_JSON, "w") as f:
        json.dump(trainer.results, f, indent=2)

    best_clf = min(v["RMSE"] for k, v in trainer.results["traffic_volume"].items()
                   if k != "LSTM_from_scratch")
    dl_wins = lstm_m["volume"]["RMSE"] < best_clf
    reports.write_benchmark(trainer.results, lstm_m["volume"],
                            lstm_m["congestion"], dl_wins,
                            config.REPORTS_DIR / "benchmark_report.md")

    for target in ("traffic_volume", "travel_time"):
        win = max(trainer.results[target],
                  key=lambda k: trainer.results[target][k]["R2"])
        reports.write_model_card(target, win,
                                 trainer.results[target][win], "")
    cong_win = max(trainer.results["congestion"],
                   key=lambda k: trainer.results["congestion"][k]["macro_F1"])
    reports.write_model_card("congestion", cong_win,
                             trainer.results["congestion"][cong_win], "")
    risk_win = max(trainer.results["accident_risk"],
                   key=lambda k: trainer.results["accident_risk"][k]["ROC_AUC"])
    reports.write_model_card("accident_risk", risk_win,
                             trainer.results["accident_risk"][risk_win], "")

    # -------- corridor prediction artifact (dashboard input) --------
    feat = pd.read_parquet(config.FEATURES_PARQUET)
    _, val, _ = features.time_split(feat)
    vol_win = max(trainer.results["traffic_volume"],
                  key=lambda k: trainer.results["traffic_volume"][k]["R2"])
    tt_win = max(trainer.results["travel_time"],
                 key=lambda k: trainer.results["travel_time"][k]["R2"])
    vol_model = trainer.fitted[f"traffic_volume__{vol_win}"]
    tt_model = trainer.fitted[f"travel_time__{tt_win}"]
    cong_model = trainer.fitted[f"congestion__{cong_win}"]
    risk_model = trainer.fitted[f"accident_risk__{risk_win}"]

    X = feat[features.FEATURE_COLS].to_numpy(np.float32)
    from src.flowcast.features import RISK_FEATURE_COLS
    Xrisk = feat[RISK_FEATURE_COLS].to_numpy(np.float32)
    out = feat[["road_id", "road_name", "timestamp", "traffic_volume",
                "avg_speed", "travel_time", "congestion_level",
                "weather_condition", "rainfall", "visibility",
                "public_holiday", "event_flag"]].copy()
    out["pred_volume"] = vol_model.predict(X)
    out["pred_travel_time"] = tt_model.predict(X)
    out["pred_congestion_code"] = cong_model.predict(X).astype(int)
    out["pred_congestion"] = pd.Categorical.from_codes(
        out["pred_congestion_code"],
        categories=["Free-flow", "Moderate", "Heavy", "Severe"])
    out["risk_probability"] = risk_model.predict_proba(Xrisk)[:, 1]

    resid = val["traffic_volume"].to_numpy() - vol_model.predict(
        val[features.FEATURE_COLS].to_numpy(np.float32))
    sigma = float(np.std(resid))
    out["volume_lower"] = np.maximum(0, out["pred_volume"] - 1.96 * sigma)
    out["volume_upper"] = out["pred_volume"] + 1.96 * sigma
    out["confidence_width"] = out["volume_upper"] - out["volume_lower"]
    if hasattr(cong_model, "predict_proba"):
        out["congestion_confidence"] = cong_model.predict_proba(X).max(1)

    stamps = np.sort(feat["timestamp"].unique())
    test_start = stamps[int(len(stamps) * 0.85)]
    out["is_test"] = out["timestamp"] > test_start
    out.to_parquet(config.PREDICTIONS_PARQUET, index=False)
    log.info("predictions: %s (%d rows)", config.PREDICTIONS_PARQUET, len(out))
    log.info("stage=finalize done in %.0fs", time.time() - t0)


STAGES = {"data": stage_data, "classical": stage_classical,
          "m5a": stage_m5a, "m5b": stage_m5b, "m5c": stage_m5c, "m5d": stage_m5d,
          "dl": stage_dl, "finalize": stage_finalize}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all"] + list(STAGES))
    args = ap.parse_args()
    order = (["data", "classical", "dl", "finalize"]
             if args.stage == "all" else [args.stage])
    for name in order:
        log.info("==== stage %s ====", name)
        STAGES[name]()


if __name__ == "__main__":
    main()
