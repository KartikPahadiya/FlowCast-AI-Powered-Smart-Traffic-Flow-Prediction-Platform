"""M4 — EDA & reporting.

Produces the figures used by the notebook and dashboard and writes the
data-quality report (raw -> processed), documenting every defect found
and how it was resolved.
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

log = logging.getLogger("flowcast.eda")
sns.set_theme(style="whitegrid")
CONG_ORDER = ["Free-flow", "Moderate", "Heavy", "Severe"]
CONG_COLORS = {"Free-flow": "#2ecc71", "Moderate": "#f1c40f",
               "Heavy": "#e67e22", "Severe": "#e74c3c"}


def fig_volume_by_hour(df, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    df.assign(hour=df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60) \
      .groupby("hour")["traffic_volume"].mean().plot(ax=ax)
    ax.set_title("Corridor mean traffic volume by time of day")
    ax.set_ylabel("vehicles / 30-min window")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_congestion_distribution(df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df["congestion_level"].value_counts().reindex(CONG_ORDER)
    vc.plot.bar(ax=ax, color=[CONG_COLORS[c] for c in CONG_ORDER])
    ax.set_title("Congestion level distribution"); ax.set_ylabel("windows")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_volume_by_congestion(df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="congestion_level", y="traffic_volume",
                order=CONG_ORDER, palette=CONG_COLORS, ax=ax, showfliers=False)
    ax.set_title("Volume distribution per congestion class")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_weather_impact(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x="weather_condition", y="traffic_volume",
                order=["Clear", "Cloudy", "Rain", "Fog"], ax=axes[0],
                showfliers=False)
    axes[0].set_title("Volume by weather condition")
    rain = df.assign(rain=pd.cut(df["rainfall"], [-0.1, 0, 2, 5, 50],
                                 labels=["0", "0-2", "2-5", "5+"])) \
             .groupby("rain", observed=True)["avg_speed"].mean()
    rain.plot.bar(ax=axes[1], color="#3498db")
    axes[1].set_title("Mean speed by rainfall band (mm/h)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_correlation_heatmap(df, cols, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[cols].corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=False, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_weekday_pattern(df, path):
    pivot = df.assign(dow=df["timestamp"].dt.dayofweek,
                      hour=df["timestamp"].dt.hour) \
              .pivot_table(index="dow", columns="hour",
                           values="traffic_volume", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    ax.set_title("Mean volume: day-of-week x hour")
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                       rotation=0)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Dependency-free markdown table (avoids requiring `tabulate`)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(" --- " for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_quality_report(qlog_df, cleaned, path):
    dup_rows = qlog_df[qlog_df.stage == "dedup"]["count"].sum()
    out_rows = qlog_df[qlog_df.stage == "outliers"]["count"].sum()
    imputed = qlog_df[qlog_df.stage == "imputation"]["count"].sum()
    derived = qlog_df[(qlog_df.stage == "harmonisation")]["count"].iloc[0]
    md = f"""# FlowCast Data-Quality Report

**Raw sources:** `traffic_sensor_log.csv` ({178469:,} rows), `weather_observations.csv`
({10872:,} rows), `calendar_events.csv` ({151:,} rows)
**Processed output:** {len(cleaned):,} segment × 30-minute records ·
{cleaned['road_id'].nunique()} segments · 2025-01-01 → 2025-05-31

## Defects found and how each was resolved

| Defect (PRD §8.3 / Data Dictionary §5) | Resolution (pipeline stage) |
| --- | --- |
| Exact duplicate log rows from detector retries | Dropped after load — {dup_rows:,} rows |
| Physically impossible outliers: negative volume, speed > 200 km/h, occupancy > 100 % | Ruled to null, then time-interpolated — {out_rows} values |
| Missing 30-minute windows (full sensor dropout) | Segment reindexed onto the full 48-window daily grid — {int(qlog_df[(qlog_df.stage=='missing')]['count'].iloc[0]):,} windows restored, flagged via interpolation |
| Null volume / speed / occupancy in partial windows | Per-segment time interpolation, then segment × time-of-day median fallback — {imputed:,} values repaired |
| Blank congestion_level (~15 % of rows) | Derived from V/C = volume ÷ (capacity/2) banding (<0.50 Free-flow, 0.50–0.79 Moderate, 0.80–0.99 Heavy, ≥1.00 Severe) — {derived:,} rows |
| Inconsistent weather labels ('RAIN', 'rainy', 'rain ', 'foggy', …) | Mapped to controlled vocabulary Clear / Cloudy / Rain / Fog (case- and whitespace-normalised) |
| Mixed date formats (weather DD/MM/YYYY vs traffic YYYY-MM-DD) | Parsed explicitly per table with format-strict `to_datetime`, joined on a common timestamp |
| Hourly weather vs half-hourly traffic grain | Each hour's observation broadcast to both of its 30-minute windows via floor-to-hour join |
| Missing temperature / visibility in weather feed | Station-level interpolation, then median fallback |
| Nested `vehicle_type_dist` JSON | Parsed into four share columns (`share_tw/car/lcv/hcv`) |

## Final validation assertions (all must pass)

- No nulls in any modelling column
- `traffic_volume ≥ 0`, `avg_speed ≤ 200`, `occupancy ≤ 100` everywhere
- No `(road_id, timestamp)` duplicates
- Final grain: {len(cleaned):,} rows = 25 segments × 151 days × 48 windows

## Per-stage quality log

{ _df_to_markdown(qlog_df) }

_Generated by `src/flowcast/clean.py` via `run_pipeline.py` — deterministic and reproducible (seed {config.RANDOM_SEED})._
"""
    path.write_text(md, encoding="utf-8")
    log.info("data-quality report written to %s", path)


def run_eda(cleaned):
    """Generate all EDA figures into reports/figures/."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_volume_by_hour(cleaned, config.FIGURES_DIR / "volume_by_hour.png")
    fig_congestion_distribution(cleaned, config.FIGURES_DIR / "congestion_dist.png")
    fig_volume_by_congestion(cleaned, config.FIGURES_DIR / "volume_by_congestion.png")
    fig_weather_impact(cleaned, config.FIGURES_DIR / "weather_impact.png")
    fig_weekday_pattern(cleaned, config.FIGURES_DIR / "weekday_pattern.png")
    num_cols = ["traffic_volume", "avg_speed", "occupancy", "travel_time",
                "rainfall", "visibility", "temperature", "signal_timing",
                "road_capacity", "accident_count"]
    fig_correlation_heatmap(cleaned, num_cols,
                            config.FIGURES_DIR / "correlation_heatmap.png")
    log.info("EDA figures written to %s", config.FIGURES_DIR)
