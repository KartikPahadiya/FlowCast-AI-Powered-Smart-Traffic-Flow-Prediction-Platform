"""M2 — Cleaning, wrangling & merge.

Deterministic ordered pipeline (PRD Section 9):
  1. schema/type validation (done in M1)
  2. duplicate removal (exact + road_id/timestamp key)
  3. outlier detection & repair (rule-based + z-score/IQR)
  4. missing-value handling (time interpolation -> segment/window median)
  5. categorical harmonisation (weather labels, congestion derivation)
  6. temporal alignment & merge (hourly weather -> 30-min grain)
  7. final validation & quality assertions
"""
import logging

import numpy as np
import pandas as pd

from . import config
from .ingest import QualityLog

log = logging.getLogger("flowcast.clean")

SENSOR_COLS = ["traffic_volume", "avg_speed", "occupancy"]


# ------------------------------------------------------------- helpers
def _derive_congestion(volume: pd.Series, capacity: pd.Series) -> pd.Series:
    """Band V/C = volume / (capacity/2) into the four congestion classes."""
    vc = volume / (capacity / config.WINDOWS_PER_HOUR)
    out = pd.Series(np.nan, index=volume.index, dtype="object")
    for thresh, label in config.CONGESTION_BANDS:
        out = out.where(out.notna() | (vc >= thresh), label)
    # rows with no volume get Free-flow by convention
    out = out.fillna("Free-flow")
    return out


def harmonise_weather_labels(series: pd.Series) -> pd.Series:
    key = series.fillna("").str.strip().str.lower()
    mapped = key.map(config.WEATHER_VOCAB)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].dropna().unique())
        raise ValueError(f"weather labels outside controlled vocabulary: {bad}")
    return mapped


# ------------------------------------------------------------ traffic
def clean_traffic(traffic: pd.DataFrame, qlog: QualityLog) -> pd.DataFrame:
    df = traffic.copy()
    df["timestamp"] = df["date"] + pd.to_timedelta(df["time"] + ":00")

    # --- duplicates -------------------------------------------------
    n0 = len(df)
    df = df.drop_duplicates()
    qlog.add("dedup", "traffic exact duplicate rows removed", n0 - len(df))
    n0 = len(df)
    df["_n_null"] = df.isna().sum(axis=1)
    df = (df.sort_values("_n_null")
            .drop_duplicates(subset=["road_id", "timestamp"], keep="first")
            .drop(columns="_n_null").sort_index())
    qlog.add("dedup", "traffic key duplicates (road_id+timestamp) removed",
             n0 - len(df))

    # --- impossible values by rule ----------------------------------
    for col, rule, desc in [
        ("traffic_volume", df["traffic_volume"] < 0, "negative traffic_volume nulled"),
        ("avg_speed", df["avg_speed"] > 200, "avg_speed > 200 km/h nulled"),
        ("occupancy", df["occupancy"] > 100, "occupancy > 100% nulled"),
    ]:
        n = int(rule.sum())
        df.loc[rule, col] = np.nan
        qlog.add("outliers", desc, n)

    # --- statistical extremes (z-score / IQR cap) --------------------
    for col in SENSOR_COLS:
        s = df[col].dropna()
        z = (s - s.mean()) / s.std()
        iqr = s.quantile(0.75) - s.quantile(0.25)
        cap_hi = s.quantile(0.75) + 3 * iqr
        mask = ((z.abs() > 6) & (df[col] > cap_hi)) | (df[col] > cap_hi * 1.5)
        n = int(mask.sum())
        df.loc[mask, col] = np.nan
        qlog.add("outliers", f"{col}: statistical extremes (z>6 & IQR cap) nulled", n)

    # --- restore missing sensor windows ----------------------------
    # Reindex each segment onto the full 48-window daily grid so lag
    # features later align to true clock time.
    all_times = sorted(df["time"].unique())  # keep the source's "HH:MM" strings
    full_idx = []
    for road, g in df.groupby("road_id"):
        days = sorted(g["date"].unique())
        grid = pd.MultiIndex.from_product([days, all_times],
                                          names=["date", "time"])
        g = g.set_index(["date", "time"]).reindex(grid)
        g["road_id"] = road
        for c in ["road_name", "latitude", "longitude",
                  "weather_station_id", "road_capacity"]:
            g[c] = g[c].ffill().bfill()
        full_idx.append(g.reset_index())
    n_before = len(df)
    df = pd.concat(full_idx, ignore_index=True)
    qlog.add("missing", "entire sensor windows restored for dropout gaps",
             len(df) - n_before)
    df["timestamp"] = df["date"] + pd.to_timedelta(
        df["time"].astype(str).str[:5] + ":00")

    # --- imputation ---------------------------------------------------
    df = df.sort_values(["road_id", "timestamp"])
    for col in SENSOR_COLS:
        n_null = int(df[col].isna().sum())
        df[col] = df.groupby("road_id")[col] \
                   .transform(lambda s: s.interpolate(limit_direction="both"))
        still = int(df[col].isna().sum())
        if still:
            seg_med = df.groupby(["road_id", df["timestamp"].dt.time])[col] \
                        .transform("median")
            df[col] = df[col].fillna(seg_med)
        qlog.add("imputation", f"{col}: time-interpolated / median-filled",
                 n_null - int(df[col].isna().sum()))

    # travel_time has no nulls in the raw file, but restored windows do
    df["travel_time"] = df["travel_time"].fillna(
        df.groupby("road_id")["travel_time"].transform("median"))
    df["accident_count"] = df["accident_count"].fillna(0).astype(int)
    df["vehicle_count"] = df["vehicle_count"].fillna(df["traffic_volume"])
    df["signal_timing"] = df["signal_timing"].fillna(
        df.groupby("road_id")["signal_timing"].transform("median"))

    # --- congestion harmonisation & derivation -----------------------
    canon = {c.lower(): c for c in config.CONGESTION_CLASSES}
    df["congestion_level"] = (df["congestion_level"].str.strip()
                              .str.lower().map(canon))
    n_blank = int(df["congestion_level"].isna().sum())
    derived = _derive_congestion(df["traffic_volume"], df["road_capacity"])
    df["congestion_level"] = df["congestion_level"].fillna(derived)
    qlog.add("harmonisation", "congestion_level blanks derived from V/C banding",
             n_blank)

    # --- nested JSON vehicle mix --------------------------------------
    mix = pd.json_normalize(df["vehicle_type_dist"].apply(
        lambda s: {} if pd.isna(s) else __import__("json").loads(s)))
    for cls, name in [("2W", "tw"), ("Car", "car"), ("LCV", "lcv"), ("HCV", "hcv")]:
        df[f"share_{name}"] = pd.to_numeric(mix.get(cls), errors="coerce")
    mix_cols = [c for c in df.columns if c.startswith("share_")]
    df[mix_cols] = df.groupby("road_id")[mix_cols].transform(
        lambda s: s.ffill().bfill())
    df = df.drop(columns=["vehicle_type_dist"])

    return df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)


# ------------------------------------------------------------- weather
def clean_weather(weather: pd.DataFrame, qlog: QualityLog) -> pd.DataFrame:
    df = weather.copy()
    df["weather_condition"] = harmonise_weather_labels(df["weather_condition"])
    n0 = len(df)
    df = df.drop_duplicates()
    qlog.add("dedup", "weather exact duplicates removed", n0 - len(df))
    # hourly grain key
    df["weather_hour"] = df["date"] + pd.to_timedelta(df["time"] + ":00")
    df = df.drop(columns=["date", "time"]).rename(columns={"station_id": "weather_station_id"})
    for col in ["temperature", "visibility", "rainfall"]:
        n = int(df[col].isna().sum())
        df[col] = df.groupby("weather_station_id")[col] \
                    .transform(lambda s: s.interpolate(limit_direction="both"))
        df[col] = df[col].fillna(df[col].median())
        qlog.add("imputation", f"{col}: station-interpolated / median-filled", n)
    return df


# -------------------------------------------------------------- merge
def merge_sources(traffic: pd.DataFrame, weather: pd.DataFrame,
                  calendar: pd.DataFrame, qlog: QualityLog) -> pd.DataFrame:
    df = traffic.copy()
    # broadcast each hour's weather to both of its 30-minute windows
    df["weather_hour"] = df["timestamp"].dt.floor("h")
    n0 = len(df)
    df = df.merge(weather, on=["weather_station_id", "weather_hour"], how="left")
    df = df.drop(columns=["weather_hour"])
    qlog.add("merge", "rows after weather join (hourly -> 30-min broadcast)", n0)

    n0 = len(df)
    df = df.merge(calendar, on="date", how="left")
    for c in ["public_holiday", "event_flag", "roadwork_flag"]:
        df[c] = df[c].fillna(0).astype(int)
    df["holiday_name"] = df["holiday_name"].fillna("")
    df["event_name"] = df["event_name"].fillna("")
    qlog.add("merge", "rows after calendar join", n0)
    return df


# ---------------------------------------------------------- validation
def validate_final(df: pd.DataFrame, qlog: QualityLog) -> pd.DataFrame:
    required = ["road_id", "timestamp", "traffic_volume", "avg_speed",
                "occupancy", "congestion_level", "travel_time",
                "weather_condition", "temperature", "rainfall", "visibility"]
    for col in required:
        assert not df[col].isna().any(), f"nulls remain in modelling column {col}"
        n_fixed = int(df[col].isna().sum())
        if n_fixed:
            qlog.add("validation", f"{col}: residual nulls (must be 0)", n_fixed)
    assert (df["traffic_volume"] >= 0).all(), "negative volume survived cleaning"
    assert (df["avg_speed"] <= 200).all(), "impossible speed survived cleaning"
    assert (df["occupancy"] <= 100).all(), "occupancy>100 survived cleaning"
    assert not df.duplicated(subset=["road_id", "timestamp"]).any(), \
        "key duplicates survived cleaning"
    qlog.add("validation", "final row count", len(df))
    qlog.add("validation", "segments", df["road_id"].nunique())
    qlog.add("validation", "time span (days)",
             (df["timestamp"].max() - df["timestamp"].min()).days + 1)
    return df


def run_cleaning(calendar, weather, traffic, qlog: QualityLog) -> pd.DataFrame:
    log.info("cleaning traffic table ...")
    traffic = clean_traffic(traffic, qlog)
    log.info("cleaning weather table ...")
    weather = clean_weather(weather, qlog)
    log.info("merging sources ...")
    df = merge_sources(traffic, weather, calendar, qlog)
    df = validate_final(df, qlog)
    return df
