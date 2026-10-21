"""Train FlowCast's traffic-volume model on the UCI Metro Interstate dataset
(48,204 hourly records, I-94 westbound, 2012-2018).
Artifacts: model.json (XGBoost), metrics.json, history_sample.csv, hourly_profile.csv.
Run: python train.py
"""
import json, os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = ["hour","dow","month","is_weekend","is_holiday","temp_c","rain_1h","snow_1h",
         "clouds_all","weather_Clouds","weather_Drizzle","weather_Fog","weather_Haze",
         "weather_Mist","weather_Rain","weather_Smoke","weather_Snow","weather_Squall",
         "weather_Thunderstorm"]

def load():
    df = pd.read_csv(os.path.join(HERE,"data","Metro_Interstate_Traffic_Volume.csv.gz"))
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").drop_duplicates("date_time").reset_index(drop=True)
    df["hour"] = df.date_time.dt.hour
    df["dow"] = df.date_time.dt.dayofweek
    df["month"] = df.date_time.dt.month
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["is_holiday"] = df.holiday.notna().astype(int)
    df["temp_c"] = (df.temp - 273.15).round(1)
    df.loc[df.temp_c < -40, "temp_c"] = np.nan           # known 0K sensor glitches
    df["temp_c"] = df.temp_c.fillna(df.temp_c.median())
    df.loc[df.rain_1h > 100, "rain_1h"] = 100            # clip sensor outliers
    wx = pd.get_dummies(df.weather_main, prefix="weather")
    df = pd.concat([df, wx], axis=1)
    for c in FEATS:
        if c not in df: df[c] = 0
    return df

def main():
    df = load()
    cut = df.date_time.quantile(0.8)                     # time-based holdout (no leakage)
    tr, te = df[df.date_time <= cut], df[df.date_time > cut]
    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06,
                         subsample=0.9, colsample_bytree=0.9, n_jobs=4, random_state=42)
    model.fit(tr[FEATS], tr.traffic_volume)
    pred = model.predict(te[FEATS])
    mae = float(mean_absolute_error(te.traffic_volume, pred))
    r2 = float(r2_score(te.traffic_volume, pred))
    print(f"train={len(tr)} test={len(te)} MAE={mae:.0f} vehicles/h  R2={r2:.3f}")

    out = os.path.join(HERE,"artifacts"); os.makedirs(out, exist_ok=True)
    model.save_model(os.path.join(out,"model.json"))
    json.dump({"mae_veh_per_h":round(mae),"r2":round(r2,3),"n_train":int(len(tr)),
               "n_test":int(len(te)),"dataset":"UCI Metro Interstate Traffic Volume (I-94, 2012-2018)",
               "features":FEATS}, open(os.path.join(out,"metrics.json"),"w"), indent=2)
    te.assign(predicted=pred.round(0))[["date_time","traffic_volume","predicted"]] \
      .tail(24*30).to_csv(os.path.join(out,"history_sample.csv"), index=False)
    df.groupby(["dow","hour"]).traffic_volume.mean().round(0).reset_index() \
      .to_csv(os.path.join(out,"hourly_profile.csv"), index=False)
    print("artifacts written")

if __name__ == "__main__":
    main()
