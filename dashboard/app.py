"""M7 — FlowCast analytics dashboard (Streamlit).

Run with:  streamlit run dashboard/app.py

Nine PRD views + data upload / prediction / report modules. Reads
persisted model outputs and analytics tables — it never retrains.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.flowcast import config  # noqa: E402
from src.flowcast.features import FEATURE_COLS, RISK_FEATURE_COLS  # noqa: E402

st.set_page_config(page_title="FlowCast — Northline Corridor",
                   page_icon="🚦", layout="wide")

CONG_ORDER = ["Free-flow", "Moderate", "Heavy", "Severe"]
CONG_COLOR = {"Free-flow": "#2ecc71", "Moderate": "#f1c40f",
              "Heavy": "#e67e22", "Severe": "#e74c3c"}


@st.cache_data
def load_predictions():
    return pd.read_parquet(config.PREDICTIONS_PARQUET)


@st.cache_data
def load_features():
    return pd.read_parquet(config.FEATURES_PARQUET)


@st.cache_resource
def load_models():
    models = {}
    for p in config.MODELS_DIR.glob("*.joblib"):
        models[p.stem] = joblib.load(p)
    return models


@st.cache_data
def load_scoreboard():
    with open(config.SCOREBOARD_JSON) as f:
        return json.load(f)


@st.cache_data
def load_lstm_history():
    p = config.MODELS_DIR / "lstm_history.json"
    return json.load(open(p)) if p.exists() else []


pred = load_predictions()
feat = load_features()
models = load_models()
sb = load_scoreboard()

st.sidebar.title("🚦 FlowCast")
st.sidebar.caption("Northline Corridor · 25 segments · 30-min windows")
view = st.sidebar.radio(
    "View", [
        "Live prediction", "Historical trends", "Congestion heatmap",
        "Road comparison", "Model performance", "Feature importance",
        "Forecast visualisation", "Prediction confidence",
        "Weather vs traffic", "🔧 Predict (custom)", "📤 Data upload",
        "📄 Reports & insights",
    ])
st.sidebar.markdown("---")
st.sidebar.caption("Colour encodes severity consistently: "
                   "🟢 Free-flow · 🟡 Moderate · 🟠 Heavy · 🔴 Severe")

last_ts = pred["timestamp"].max()
cong_map = dict(enumerate(CONG_ORDER))


# ============================================================ helpers
def seg_select(key, label="Segment"):
    roads = sorted(pred["road_id"].unique())
    names = pred.drop_duplicates("road_id").set_index("road_id")["road_name"]
    options = [f"{r} — {names[r]}" for r in roads]
    pick = st.selectbox(label, options, key=key)
    return pick.split(" — ")[0]


def cong_color_scale():
    return [(0, CONG_COLOR["Free-flow"]), (0.33, CONG_COLOR["Moderate"]),
            (0.66, CONG_COLOR["Heavy"]), (1, CONG_COLOR["Severe"])]


# ================================================== 1. live prediction
if view == "Live prediction":
    st.title("Live prediction — next window per segment")
    st.caption(f"Model outputs for the latest corridor window "
               f"({last_ts:%Y-%m-%d %H:%M})")
    horizon = st.selectbox("Horizon", ["Next window (30 min)", "+60 min",
                                      "+90 min", "+120 min"])
    snap = pred[pred["timestamp"] == last_ts].copy()
    snap["pred_congestion"] = pd.Categorical(snap["pred_congestion"],
                                             categories=CONG_ORDER, ordered=True)
    snap = snap.sort_values(["pred_congestion", "risk_probability"],
                            ascending=[False, False])
    cols = st.columns(4)
    cols[0].metric("Segments Severe",
                   int((snap["pred_congestion"] == "Severe").sum()))
    cols[1].metric("Segments Heavy",
                   int((snap["pred_congestion"] == "Heavy").sum()))
    cols[2].metric("Mean predicted volume", f"{snap['pred_volume'].mean():.0f}")
    cols[3].metric("Max accident risk", f"{snap['risk_probability'].max():.1%}")

    show = snap[["road_id", "road_name", "pred_volume", "pred_travel_time",
                 "pred_congestion", "congestion_confidence",
                 "risk_probability"]].rename(columns={
        "road_id": "Segment", "road_name": "Road",
        "pred_volume": "Volume (veh)", "pred_travel_time": "Travel time (min)",
        "pred_congestion": "Congestion", "congestion_confidence": "Confidence",
        "risk_probability": "Accident risk"})
    show["Accident risk"] = show["Accident risk"].map("{:.1%}".format)
    show["Confidence"] = show["Confidence"].map("{:.0%}".format)
    show["Volume (veh)"] = show["Volume (veh)"].round(0)
    st.dataframe(show, width="stretch", hide_index=True,
                 height=35 * len(show) + 40)
    st.plotly_chart(px.bar(snap, x="road_id", y="risk_probability",
                           color="pred_congestion", category_orders={
                               "pred_congestion": CONG_ORDER},
                           color_discrete_map=CONG_COLOR,
                           labels={"risk_probability": "Accident risk",
                                   "road_id": "Segment"},
                           title="Accident-risk ranking by segment"),
                    use_container_width=True)

# ================================================= 2. historical trends
elif view == "Historical trends":
    st.title("Historical trends")
    road = seg_select("hist")
    d = pred[pred["road_id"] == road]
    daterange = st.date_input("Date range",
                              value=(last_ts - pd.Timedelta(days=14), last_ts),
                              min_value=d["timestamp"].min(),
                              max_value=last_ts)
    d = d[(d["timestamp"].dt.date >= daterange[0]) &
          (d["timestamp"].dt.date <= daterange[1])]
    fig = go.Figure()
    fig.add_scatter(x=d["timestamp"], y=d["traffic_volume"],
                    name="Observed volume", line=dict(color="#3498db"))
    fig.add_scatter(x=d["timestamp"], y=d["pred_volume"], name="Predicted",
                    line=dict(color="#e74c3c", dash="dot"))
    fig.update_layout(title=f"{road} — volume over time",
                      yaxis_title="vehicles / 30-min")
    st.plotly_chart(fig, width="stretch")
    fig2 = px.line(d, x="timestamp", y="avg_speed",
                   labels={"avg_speed": "km/h", "timestamp": ""},
                   title=f"{road} — average speed")
    st.plotly_chart(fig2, width="stretch")
    hourly = pred.copy()
    hourly["hour"] = hourly["timestamp"].dt.hour
    fig3 = px.box(hourly[hourly["road_id"] == road], x="hour", y="traffic_volume",
                  labels={"traffic_volume": "vehicles", "hour": "hour of day"},
                  title=f"{road} — peak-hour pattern (full history)")
    st.plotly_chart(fig3, width="stretch")

# ================================================= 3. congestion heatmap
elif view == "Congestion heatmap":
    st.title("Congestion heatmap — segment × time")
    day = st.date_input("Day", value=last_ts.date(),
                        min_value=pred["timestamp"].min().date(),
                        max_value=last_ts.date())
    d = pred[pred["timestamp"].dt.date == day]
    grid = d.pivot_table(index="road_id", columns=d["timestamp"].dt.strftime("%H:%M"),
                         values="pred_congestion", aggfunc="first") \
            .reindex(columns=sorted(d["timestamp"].dt.strftime("%H:%M").unique()))
    code = grid.apply(lambda s: pd.Categorical(s, categories=CONG_ORDER,
                                               ordered=True).codes)
    fig = px.imshow(code, color_continuous_scale=cong_color_scale(),
                    zmin=0, zmax=3, aspect="auto",
                    labels={"color": "Congestion"})
    fig.update_layout(title=f"Predicted congestion — {day}")
    st.plotly_chart(fig, width="stretch")
    st.caption("0 = Free-flow · 1 = Moderate · 2 = Heavy · 3 = Severe")

# ================================================== 4. road comparison
elif view == "Road comparison":
    st.title("Road comparison")
    roads = st.multiselect(
        "Segments",
        sorted(pred["road_id"].unique()),
        default=sorted(pred["road_id"].unique())[:4])
    d = pred[pred["road_id"].isin(roads)]
    agg = d.groupby(["road_id", "road_name"]).agg(
        mean_volume=("traffic_volume", "mean"),
        mean_speed=("avg_speed", "mean"),
        severe_share=("pred_congestion",
                      lambda s: (s == "Severe").mean()),
        p95_travel=("travel_time", lambda s: s.quantile(0.95))).reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(agg, x="road_id", y="mean_volume", color="road_id",
                           labels={"mean_volume": "mean vehicles/30-min",
                                   "road_id": "Segment"},
                           title="Mean volume"), width="stretch")
    c2.plotly_chart(px.bar(agg, x="road_id", y="mean_speed", color="road_id",
                           labels={"mean_speed": "km/h"},
                           title="Mean speed"), width="stretch")
    c1.plotly_chart(px.bar(agg, x="road_id", y="severe_share", color="road_id",
                           labels={"severe_share": "share of windows"},
                           title="Severe-congestion share"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(agg, x="road_id", y="p95_travel",
                           color="road_id",
                           labels={"p95_travel": "95th percentile (min)"},
                           title="Travel-time reliability (p95)"),
                    use_container_width=True)

# ================================================ 5. model performance
elif view == "Model performance":
    st.title("Model performance — test window 2025-05-09 → 2025-05-31")
    target = st.selectbox("Target",
                          ["traffic_volume", "travel_time", "congestion",
                           "accident_risk"])
    rows = []
    for name, m in sb[target].items():
        if not isinstance(m, dict):
            continue
        rows.append({"Model": name,
                     "RMSE": m.get("RMSE"), "MAPE %": m.get("MAPE"),
                     "R²": m.get("R2"), "macro-F1": m.get("macro_F1"),
                     "Accuracy": m.get("accuracy"),
                     "ROC-AUC": m.get("ROC_AUC"),
                     "Train (s)": m.get("train_seconds")})
    st.dataframe(pd.DataFrame(rows).set_index("Model"),
                 use_container_width=True)
    reg = sb.get("traffic_volume", {})
    if target == "traffic_volume":
        names = [k for k in reg if isinstance(reg[k], dict)]
        rmse = [reg[k]["RMSE"] for k in names]
        st.plotly_chart(px.bar(x=names, y=rmse,
                               labels={"x": "Model", "y": "RMSE (vehicles)"},
                               title="Volume RMSE — classical vs LSTM"),
                        use_container_width=True)
    hist = load_lstm_history()
    if hist:
        h = pd.DataFrame(hist)
        fig = go.Figure()
        fig.add_scatter(x=h["epoch"], y=h["train_mse"], name="train MSE")
        fig.add_scatter(x=h["epoch"], y=h["val_mse"], name="validation MSE")
        fig.update_layout(title="LSTM training curves (early stopping on val MSE)",
                          xaxis_title="epoch", yaxis_title="MSE (scaled volume)")
        st.plotly_chart(fig, width="stretch")
        gd = config.MODELS_DIR / "gd_loss_curve.json"
        if gd.exists():
            g = pd.DataFrame({"iteration": range(len(json.load(open(gd)))),
                              "MSE": json.load(open(gd))})
            fig2 = px.line(g, x="iteration", y="MSE",
                           title="From-scratch linear regression — gradient-descent loss curve")
            st.plotly_chart(fig2, width="stretch")
    if "confusion_matrix" in sb[target].get("XGBoost", {}):
        cm = np.array(sb[target]["XGBoost"]["confusion_matrix"])
        labs = CONG_ORDER if target == "congestion" else ["no", "yes"]
        fig3 = px.imshow(cm, text_auto=True, x=labs, y=labs,
                         labels={"x": "predicted", "y": "actual"},
                         title="XGBoost confusion matrix (test)")
        st.plotly_chart(fig3, width="stretch")

# ============================================== 6. feature importance
elif view == "Feature importance":
    st.title("Feature importance")
    target = st.selectbox("Model",
                          [k for k in models if "RandomForest" in k],
                          format_func=lambda s: s.replace("__", " · "))
    model = models[target]
    # the accident-risk model was trained on the extended risk feature set
    feat_cols = (RISK_FEATURE_COLS
                 if target.startswith("accident_risk") else FEATURE_COLS)
    imp = pd.DataFrame({"feature": feat_cols,
                        "importance": model.feature_importances_}) \
            .sort_values("importance")
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                           labels={"importance": "Gini importance",
                                   "feature": ""},
                           title=target.replace("__", " · ")),
                    use_container_width=True)

# =========================================== 7. forecast visualisation
elif view == "Forecast visualisation":
    st.title("Forecast visualisation — predicted vs actual")
    road = seg_select("fc")
    d = pred[(pred["road_id"] == road) & pred["is_test"]]
    fig = go.Figure()
    fig.add_scatter(x=d["timestamp"], y=d["traffic_volume"],
                    name="actual", line=dict(color="#3498db"))
    fig.add_scatter(x=d["timestamp"], y=d["pred_volume"],
                    name="forecast", line=dict(color="#e74c3c"))
    fig.update_layout(title=f"{road} — test window overlay",
                      yaxis_title="vehicles / 30-min")
    st.plotly_chart(fig, width="stretch")
    err = d["pred_volume"] - d["traffic_volume"]
    fig2 = px.histogram(err, nbins=60,
                        labels={"value": "forecast error (vehicles)", "count": ""},
                        title="Residual distribution (bias & spread)")
    st.plotly_chart(fig2, width="stretch")
    st.metric("Mean bias", f"{err.mean():+.1f} vehicles",
              help="Systematic over/under-prediction on the test window")

# =========================================== 8. prediction confidence
elif view == "Prediction confidence":
    st.title("Prediction confidence — 95 % bands")
    road = seg_select("conf")
    d = pred[(pred["road_id"] == road) & pred["is_test"]]
    fig = go.Figure()
    fig.add_scatter(x=d["timestamp"], y=d["traffic_volume"],
                    name="actual", line=dict(color="#2c3e50"))
    fig.add_scatter(x=d["timestamp"], y=d["pred_volume"],
                    name="forecast", line=dict(color="#e74c3c"))
    fig.add_scatter(x=d["timestamp"], y=d["volume_upper"],
                    name="upper 95%", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=d["timestamp"], y=d["volume_lower"],
                    name="95 % band", fill="tonexty", fillcolor="rgba(231,76,60,0.15)",
                    line=dict(width=0))
    fig.update_layout(title=f"{road} — forecast with confidence band",
                      yaxis_title="vehicles / 30-min")
    st.plotly_chart(fig, width="stretch")
    cov = ((d["traffic_volume"] >= d["volume_lower"]) &
           (d["traffic_volume"] <= d["volume_upper"])).mean()
    st.metric("Empirical band coverage (should be ≈ 95 %)",
              f"{cov:.1%}")
    c1, c2 = st.columns(2)
    conf = pred[pred["is_test"]].groupby("pred_congestion") \
        .agg(mean_conf=("congestion_confidence", "mean"),
             n=("pred_congestion", "size")).reindex(CONG_ORDER).reset_index()
    c1.plotly_chart(px.bar(conf, x="pred_congestion", y="mean_conf",
                           color="pred_congestion",
                           color_discrete_map=CONG_COLOR,
                           category_orders={"pred_congestion": CONG_ORDER},
                           labels={"mean_conf": "mean class probability",
                                   "pred_congestion": ""},
                           title="Congestion-class confidence"), width="stretch")
    risk = pred[pred["is_test"]].copy()
    risk["risk_band"] = pd.cut(risk["risk_probability"], [0, .02, .05, .1, 1],
                               labels=["<2%", "2-5%", "5-10%", ">10%"])
    c2.plotly_chart(px.histogram(risk, x="risk_band",
                                 labels={"risk_band": "accident-risk band"},
                                 title="Accident-risk distribution (test window)"),
                    use_container_width=True)

# ============================================ 9. weather vs traffic
elif view == "Weather vs traffic":
    st.title("Weather vs traffic")
    d = pred.copy()
    d["hour"] = d["timestamp"].dt.hour
    fig = px.box(d, x="weather_condition", y="traffic_volume",
                 category_orders={"weather_condition": ["Clear", "Cloudy", "Rain", "Fog"]},
                 color="weather_condition",
                 labels={"traffic_volume": "vehicles / 30-min",
                         "weather_condition": "condition"},
                 title="Observed volume by weather condition")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(d.sample(min(20000, len(d)), random_state=1),
                               x="rainfall", y="avg_speed", opacity=0.2,
                               labels={"rainfall": "rainfall (mm)",
                                       "avg_speed": "km/h"},
                               title="Rainfall vs speed"),
                    use_container_width=True)
    c2.plotly_chart(px.scatter(d.sample(min(20000, len(d)), random_state=1),
                               x="visibility", y="traffic_volume", opacity=0.2,
                               labels={"visibility": "visibility (m)"},
                               title="Visibility vs volume"),
                    use_container_width=True)
    wk = d.groupby([d["timestamp"].dt.dayofweek, "weather_condition"]) \
          ["traffic_volume"].mean().reset_index()
    wk.columns = ["weekday", "weather_condition", "volume"]
    wk["weekday"] = wk["weekday"].map(
        dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    fig2 = px.line(wk, x="weekday", y="volume", color="weather_condition",
                   markers=True, category_orders={
                       "weather_condition": ["Clear", "Cloudy", "Rain", "Fog"]},
                   title="Mean volume — weekday × weather")
    st.plotly_chart(fig2, width="stretch")

# ========================================== 10. custom prediction
elif view == "🔧 Predict (custom)":
    st.title("Request a forecast for a chosen segment & horizon")
    st.caption("Uses the persisted winning models on the latest observed window.")
    road = seg_select("pred_custom")
    d = feat[feat["road_id"] == road].sort_values("timestamp")
    row = d.iloc[[-1]]
    vol_m = models.get("traffic_volume__XGBoost")
    cong_m = models.get("congestion__XGBoost")
    risk_m = models.get("accident_risk__RandomForest")
    tt_m = models.get("travel_time__XGBoost")
    if st.button("Run forecast", type="primary"):
        X = row[FEATURE_COLS].to_numpy(np.float32)
        Xr = row[RISK_FEATURE_COLS].to_numpy(np.float32)
        pv = float(vol_m.predict(X)[0])
        pc = cong_m.predict(X)[0]
        probs = cong_m.predict_proba(X)[0]
        pr = float(risk_m.predict_proba(Xr)[0, 1])
        pt = float(tt_m.predict(X)[0])
        resid_sigma = (pred["volume_upper"] - pred["pred_volume"]).mean() / 1.96
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume (next 30 min)", f"{pv:.0f} vehicles",
                  delta=f"±{1.96 * resid_sigma:.0f} (95 %)")
        c2.metric("Congestion", CONG_ORDER[int(pc)],
                  delta=f"confidence {probs.max():.0%}")
        c3.metric("Travel time", f"{pt:.2f} min")
        c4.metric("Accident risk", f"{pr:.1%}")
        fig = px.bar(x=CONG_ORDER, y=probs,
                     color=CONG_ORDER, color_discrete_map=CONG_COLOR,
                     labels={"x": "", "y": "probability"},
                     title="Congestion-class probabilities")
        st.plotly_chart(fig, width="stretch")

# ============================================ 11. data upload
elif view == "📤 Data upload":
    st.title("Data upload & refresh")
    st.caption("Upload a new raw sensor CSV with the same schema; it is "
               "validated, cleaned and scored against the persisted models.")
    up = st.file_uploader("Raw traffic_sensor_log CSV", type=["csv"])
    if up is not None:
        from src.flowcast import clean as clean_mod, ingest as ingest_mod
        qlog = ingest_mod.QualityLog()
        try:
            raw = ingest_mod._load(pd.read_csv(up), ingest_mod.TRAFFIC_SCHEMA,
                                   "upload")
            cal, wth = ingest_mod.load_calendar(), ingest_mod.load_weather()
            cleaned_up = clean_mod.clean_traffic(raw, qlog)
            cleaned_up = clean_mod.merge_sources(cleaned_up, wth, cal, qlog)
            cleaned_up = clean_mod.validate_final(cleaned_up, qlog)
            st.success(f"Validated & cleaned: {len(cleaned_up):,} rows, "
                       f"{cleaned_up['road_id'].nunique()} segments")
            st.dataframe(qlog.to_frame(), width="stretch")
            st.download_button("Download cleaned CSV",
                               cleaned_up.to_csv(index=False),
                               "flowcast_upload_cleaned.csv", "text/csv")
        except Exception as e:  # quarantine bad input, never silent-drop
            st.error(f"Upload rejected: {e}")

# =========================================== 12. reports & insights
elif view == "📄 Reports & insights":
    st.title("Reports & insights — export a range summary")
    daterange = st.date_input("Time range",
                              value=(last_ts - pd.Timedelta(days=7), last_ts),
                              min_value=pred["timestamp"].min().date(),
                              max_value=last_ts.date())
    d = pred[(pred["timestamp"].dt.date >= daterange[0]) &
             (pred["timestamp"].dt.date <= daterange[1])]
    st.subheader(f"{daterange[0]} → {daterange[1]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg volume", f"{d['traffic_volume'].mean():.0f}")
    c2.metric("Severe share", f"{(d['pred_congestion'] == 'Severe').mean():.1%}")
    c3.metric("Avg predicted travel time", f"{d['pred_travel_time'].mean():.2f} min")
    c4.metric("Windows with elevated risk (>5 %)",
              f"{(d['risk_probability'] > 0.05).sum()}")
    worst = d.groupby(["road_id", "pred_congestion"]).size().unstack(
        fill_value=0).reindex(columns=CONG_ORDER, fill_value=0)
    worst["severe_heavy"] = worst["Heavy"] + worst["Severe"]
    worst = worst.sort_values("severe_heavy", ascending=False).head(5)
    st.write("**Most congested segments (Heavy + Severe windows):**")
    st.dataframe(worst[CONG_ORDER], width="stretch")
    rain = d.groupby("weather_condition")["traffic_volume"].mean()
    st.write("**Mean volume by weather:**",
             {k: f"{v:.0f}" for k, v in rain.items()})
    md = f"""# FlowCast corridor summary — {daterange[0]} → {daterange[1]}

- Average observed volume: {d['traffic_volume'].mean():.0f} vehicles / 30-min window
- Severe-congestion share: {(d['pred_congestion'] == 'Severe').mean():.1%}
- Average predicted travel time: {d['pred_travel_time'].mean():.2f} min
- Windows with elevated accident risk (>5 %): {(d['risk_probability'] > 0.05).sum()}
- Busiest segments (Heavy+Severe): {', '.join(worst.head(3).index.get_level_values(0))}
"""
    st.download_button("Download summary (Markdown)", md,
                       "flowcast_summary.md", "text/markdown")
    for f in ["data_quality_report.md", "benchmark_report.md"]:
        p = config.REPORTS_DIR / f
        if p.exists():
            with st.expander(f"📋 {f}"):
                st.markdown(p.read_text(encoding="utf-8"))
