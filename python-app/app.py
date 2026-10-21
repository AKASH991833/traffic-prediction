"""FlowCast - interstate traffic volume prediction (Streamlit).
Run: streamlit run app.py
Model: XGBoost on UCI Metro Interstate Traffic Volume (I-94, 2012-2018).
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="FlowCast - Traffic Prediction", page_icon="🚦", layout="wide")

@st.cache_resource
def load_model():
    m = XGBRegressor(); m.load_model(os.path.join(HERE, "artifacts", "model.json"))
    return m

@st.cache_data
def load_data():
    a = os.path.join(HERE, "artifacts")
    metrics = json.load(open(os.path.join(a, "metrics.json")))
    hist = pd.read_csv(os.path.join(a, "history_sample.csv"), parse_dates=["date_time"])
    prof = pd.read_csv(os.path.join(a, "hourly_profile.csv"))
    return metrics, hist, prof

model = load_model()
metrics, hist, prof = load_data()
FEATS = metrics["features"]
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

st.title("🚦 FlowCast")
st.caption("Hourly traffic-volume prediction for I-94 westbound (Minneapolis-St Paul) - "
           "XGBoost trained on 48,204 real hourly observations from 2012-2018.")

with st.sidebar:
    st.header("Conditions")
    day = st.selectbox("Day", DAYS, index=2)
    hour = st.slider("Hour of day", 0, 23, 8, format="%d:00")
    month = st.slider("Month", 1, 12, 6)
    holiday = st.checkbox("Public holiday")
    st.subheader("Weather")
    temp_c = st.slider("Temperature (°C)", -30, 40, 20)
    rain = st.slider("Rain last hour (mm)", 0.0, 60.0, 0.0, 0.5)
    snow = st.slider("Snow last hour (mm)", 0.0, 30.0, 0.0, 0.5)
    clouds = st.slider("Cloud cover (%)", 0, 100, 40)
    weather = st.selectbox("Weather type", ["Clear","Clouds","Drizzle","Fog","Haze","Mist",
                                            "Rain","Smoke","Snow","Squall","Thunderstorm"])
    st.divider()
    st.subheader("Honest limits")
    st.caption(f"Time-based holdout: MAE {metrics['mae_veh_per_h']:,} vehicles/h, R² {metrics['r2']}. "
               "One freeway only (I-94 MN, 2012-2018); it learns recurring congestion, "
               "not live incidents, roadworks or today's actual conditions.")

def predict(day_name, hour, month, holiday, temp_c, rain, snow, clouds, weather):
    dow = DAYS.index(day_name)
    row = {f: 0 for f in FEATS}
    row.update({"hour":hour,"dow":dow,"month":month,"is_weekend":int(dow>=5),
                "is_holiday":int(holiday),"temp_c":temp_c,"rain_1h":rain,
                "snow_1h":snow,"clouds_all":clouds})
    wcol = f"weather_{weather}"
    if wcol in row: row[wcol] = 1
    return float(model.predict(pd.DataFrame([row])[FEATS])[0])

v = predict(day, hour, month, holiday, temp_c, rain, snow, clouds, weather)
c1, c2, c3 = st.columns(3)
c1.metric("Predicted traffic volume", f"{int(round(v)):,} vehicles/h")
typical = prof[(prof.dow == DAYS.index(day)) & (prof.hour == hour)].traffic_volume
if len(typical):
    delta = v - float(typical.iloc[0])
    c2.metric("Typical for this slot", f"{int(typical.iloc[0]):,} vehicles/h",
              f"{delta:+,.0f} vs prediction")
c3.metric("Congestion level", "Heavy" if v > 4500 else "Moderate" if v > 2500 else "Light")

st.subheader("Predicted volume across the day")
curve = pd.DataFrame({"hour": list(range(24)),
    "vehicles/h": [predict(day, h, month, holiday, temp_c, rain, snow, clouds, weather) for h in range(24)]})
st.line_chart(curve.set_index("hour"))

st.subheader("Model vs. reality (last 30 days of holdout)")
chart = hist.rename(columns={"traffic_volume":"actual","predicted":"predicted"}).set_index("date_time")
st.line_chart(chart)

st.subheader("What the model pays attention to")
imp = pd.DataFrame({"feature": FEATS, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False).head(10)
st.bar_chart(imp.set_index("feature"))
