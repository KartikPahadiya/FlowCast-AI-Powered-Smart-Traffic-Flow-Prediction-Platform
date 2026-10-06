"""M3 — Feature engineering.

Lag and rolling features are computed strictly within each segment in
time order and shifted by one window, so no feature can see the future
relative to the prediction time (PRD Section 9.2 / leakage guard).
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger("flowcast.features")

FEATURE_COLS = [
    # temporal cyclical encodings
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_peak",
    # lag features
    "vol_lag1", "vol_lag2", "vol_lag48",
    "speed_lag1", "speed_lag48",
    # rolling statistics (past-only, shifted by 1)
    "vol_roll4_mean", "vol_roll4_std", "vol_roll8_mean", "vol_roll8_std",
    "speed_roll4_mean", "speed_roll8_mean",
    # capacity
    "vc_ratio", "capacity_headroom",
    # weather
    "rain_flag", "low_vis_flag", "temperature", "rainfall", "visibility",
    "weather_encoded",
    # calendar
    "public_holiday", "event_flag", "roadwork_flag", "holiday_peak",
    # segment statics
    "signal_timing", "road_capacity",
    "share_tw", "share_car", "share_lcv", "share_hcv",
]

TARGET_COLS = ["traffic_volume", "travel_time", "congestion_level",
               "accident_risk"]

# extra features legal only for the accident-risk target: concurrent
# window occupancy and one-hot weather are observable *now*, and the
# risk target is a same-window probability (PRD Table 18)
RISK_EXTRA_COLS = ["occupancy", "wx_clear", "wx_cloudy", "wx_rain", "wx_fog",
                   "cong_lag1"]
RISK_FEATURE_COLS = FEATURE_COLS + RISK_EXTRA_COLS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["road_id", "timestamp"]).copy()
    g = df.groupby("road_id")

    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_peak"] = hour.isin([7, 7.5, 8, 8.5, 9, 9.5,
                               17, 17.5, 18, 18.5, 19, 19.5]).astype(int)

    # --- lag features (shifted: value at t-k predicts t) -------------
    df["vol_lag1"] = g["traffic_volume"].shift(1)
    df["vol_lag2"] = g["traffic_volume"].shift(2)
    df["vol_lag48"] = g["traffic_volume"].shift(48)
    df["speed_lag1"] = g["avg_speed"].shift(1)
    df["speed_lag48"] = g["avg_speed"].shift(48)

    # --- rolling statistics (exclude the current window) -------------
    shifted_vol = g["traffic_volume"].shift(1)
    shifted_spd = g["avg_speed"].shift(1)
    roll = shifted_vol.groupby(df["road_id"])
    roll_s = shifted_spd.groupby(df["road_id"])
    df["vol_roll4_mean"] = roll.transform(lambda s: s.rolling(4, min_periods=2).mean())
    df["vol_roll4_std"] = roll.transform(lambda s: s.rolling(4, min_periods=2).std())
    df["vol_roll8_mean"] = roll.transform(lambda s: s.rolling(8, min_periods=2).mean())
    df["vol_roll8_std"] = roll.transform(lambda s: s.rolling(8, min_periods=2).std())
    df["speed_roll4_mean"] = roll_s.transform(lambda s: s.rolling(4, min_periods=2).mean())
    df["speed_roll8_mean"] = roll_s.transform(lambda s: s.rolling(8, min_periods=2).mean())

    # --- capacity (computed on lagged volume: no target leakage) ------
    df["vc_ratio"] = df["vol_lag1"] / (df["road_capacity"] / 2)
    df["capacity_headroom"] = df["road_capacity"] / 2 - df["vol_lag1"]

    # --- weather flags -------------------------------------------------
    df["rain_flag"] = (df["rainfall"] > 0).astype(int)
    df["low_vis_flag"] = (df["visibility"] < 1000).astype(int)
    df["weather_encoded"] = pd.Categorical(
        df["weather_condition"], categories=["Clear", "Cloudy", "Rain", "Fog"]
    ).codes.astype(float)

    # --- calendar interactions -----------------------------------------
    df["holiday_peak"] = df["public_holiday"] * df["is_peak"]

    # --- risk-only extras (concurrent-but-observable signals) ----------
    for w in ["Clear", "Cloudy", "Rain", "Fog"]:
        df[f"wx_{w.lower()}"] = (df["weather_condition"] == w).astype(float)

    # --- targets ---------------------------------------------------------
    df["accident_risk"] = (df["accident_count"] > 0).astype(int)
    df["congestion_code"] = pd.Categorical(
        df["congestion_level"], categories=["Free-flow", "Moderate", "Heavy", "Severe"]
    ).codes
    df["cong_lag1"] = df.groupby("road_id")["congestion_code"].shift(1)

    # rows without full lag history cannot be modelled honestly -> drop
    n0 = len(df)
    df = df.dropna(subset=["vol_lag1", "vol_lag48", "vol_roll8_mean"])
    log.info("dropped %d rows without full 48-window lag history", n0 - len(df))
    df["vol_roll4_std"] = df["vol_roll4_std"].fillna(0)
    df["vol_roll8_std"] = df["vol_roll8_std"].fillna(0)
    return df.reset_index(drop=True)


def time_split(df: pd.DataFrame):
    """Time-based train/val/test split on the sorted unique timeline."""
    stamps = np.sort(df["timestamp"].unique())
    n = len(stamps)
    tr_end = stamps[int(n * 0.70)]
    va_end = stamps[int(n * 0.85)]
    train = df[df["timestamp"] <= tr_end]
    val = df[(df["timestamp"] > tr_end) & (df["timestamp"] <= va_end)]
    test = df[df["timestamp"] > va_end]
    return train, val, test
