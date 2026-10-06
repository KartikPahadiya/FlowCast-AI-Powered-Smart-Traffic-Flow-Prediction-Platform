"""Central configuration for the FlowCast pipeline.

All paths, seeds and split fractions live here so a reviewer can rerun
the entire pipeline deterministically from a single place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_CARDS_DIR = REPORTS_DIR / "model_cards"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TRAFFIC_CSV = DATA_RAW / "traffic_sensor_log.csv"
WEATHER_CSV = DATA_RAW / "weather_observations.csv"
CALENDAR_CSV = DATA_RAW / "calendar_events.csv"

PROCESSED_PARQUET = DATA_PROCESSED / "flowcast_processed.parquet"
FEATURES_PARQUET = DATA_PROCESSED / "flowcast_features.parquet"
PREDICTIONS_PARQUET = DATA_PROCESSED / "flowcast_predictions.parquet"
SCOREBOARD_JSON = MODELS_DIR / "scoreboard.json"
QUALITY_REPORT = REPORTS_DIR / "data_quality_report.md"

# ---------------------------------------------------------------- seeds
RANDOM_SEED = 42

# ------------------------------------------------- time-aware splitting
# Applied on the sorted unique timeline of 30-minute timestamps.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder (0.15) is the untouched test window

# ------------------------------------------------------------- capacity
WINDOWS_PER_HOUR = 2  # 30-minute aggregation

# congestion derivation bands on V/C = volume / (capacity / 2)
CONGESTION_BANDS = [
    (0.50, "Free-flow"),
    (0.80, "Moderate"),
    (1.00, "Heavy"),
    (float("inf"), "Severe"),
]
CONGESTION_CLASSES = ["Free-flow", "Moderate", "Heavy", "Severe"]

# controlled vocabulary for weather_condition
WEATHER_VOCAB = {"clear": "Clear", "cloudy": "Cloudy", "overcast": "Cloudy",
                 "rain": "Rain", "rainy": "Rain", "fog": "Fog", "foggy": "Fog"}

# LSTM hyper-parameters (searched ranges documented in the model card)
LSTM_CONFIG = {
    "seq_len": 24,          # 12 hours of 30-minute steps
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 512,
    "max_epochs": 10,
    "patience": 3,
    "weight_decay": 1e-4,
}

# SVM is O(n^2)-ish in training samples; the PRD allows a baseline, so the
# margin model is trained on a seeded stratified subsample of the train window.
SVM_TRAIN_SUBSAMPLE = 25_000
