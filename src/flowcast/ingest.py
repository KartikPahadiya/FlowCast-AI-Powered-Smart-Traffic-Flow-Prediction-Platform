"""M1 — Ingestion & validation.

Loads the three raw tables, enforces schema/type/range expectations and
collects a quality log. Nothing is silently dropped: every rejection is
counted and reported.
"""
import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("flowcast.ingest")

TRAFFIC_SCHEMA = {
    "road_id": "string", "road_name": "string",
    "latitude": "float64", "longitude": "float64",
    "weather_station_id": "string", "date": "string", "time": "string",
    "traffic_volume": "float64", "vehicle_count": "int64",
    "vehicle_type_dist": "string", "avg_speed": "float64",
    "occupancy": "float64", "congestion_level": "string",
    "travel_time": "float64", "accident_count": "int64",
    "signal_timing": "int64", "road_capacity": "int64",
}

WEATHER_SCHEMA = {
    "station_id": "string", "date": "string", "time": "string",
    "weather_condition": "string", "temperature": "float64",
    "rainfall": "float64", "visibility": "float64",
}

CALENDAR_SCHEMA = {
    "date": "string", "public_holiday": "int64", "holiday_name": "string",
    "event_flag": "int64", "event_name": "string", "roadwork_flag": "int64",
}


class QualityLog:
    """Collects per-stage data-quality counters for the final report."""

    def __init__(self):
        self.entries = []

    def add(self, stage: str, detail: str, count: int):
        self.entries.append({"stage": stage, "detail": detail, "count": int(count)})
        log.info("[%s] %s: %s", stage, detail, count)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)

    def to_markdown(self) -> str:
        df = self.to_frame()
        if df.empty:
            return "_No quality events recorded._"
        return df.to_markdown(index=False)


def _load(path, schema: dict, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col, dtype in schema.items():
        if col not in df.columns:
            raise ValueError(f"{name}: missing required column '{col}'")
        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return df


def load_calendar() -> pd.DataFrame:
    df = _load(config.CALENDAR_CSV, CALENDAR_SCHEMA, "calendar")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if df["date"].isna().any():
        raise ValueError("calendar: unparseable dates")
    return df


def load_weather() -> pd.DataFrame:
    df = _load(config.WEATHER_CSV, WEATHER_SCHEMA, "weather")
    # weather uses DD/MM/YYYY — parse explicitly, refuse the silent swap
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    if df["date"].isna().any():
        raise ValueError("weather: unparseable DD/MM/YYYY dates")
    return df


def load_traffic() -> pd.DataFrame:
    df = _load(config.TRAFFIC_CSV, TRAFFIC_SCHEMA, "traffic")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if df["date"].isna().any():
        raise ValueError("traffic: unparseable YYYY-MM-DD dates")
    # range checks that can be applied at load time
    df.loc[df["occupancy"] < 0, "occupancy"] = np.nan
    df.loc[df["avg_speed"] < 0, "avg_speed"] = np.nan
    return df


def load_all():
    """Load the three tables in the dictionary's suggested order."""
    return load_calendar(), load_weather(), load_traffic()
