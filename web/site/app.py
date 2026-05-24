import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import requests
import json
import os
import copy
import pickle
from pathlib import Path

import joblib

# --- НАСТРОЙКИ СТРАНИЦЫ И CSS ---
st.set_page_config(page_title="NYC Taxi Analytics 3D", layout="wide")

# CSS-хак: превращаем radio-кнопки в красивые текстовые вкладки
st.markdown("""
    <style>
    div[role="radiogroup"] > label > div:first-child { display: none; }
    div[role="radiogroup"] { 
        gap: 30px; 
        padding-bottom: 10px; 
        border-bottom: 1px solid #444; 
        margin-bottom: 20px;
    }
    div[role="radiogroup"] > label { font-weight: bold; font-size: 1.1rem; cursor: pointer; }
    </style>
""", unsafe_allow_html=True)

st.title("🚖 3D Аналитика потоков (Умная навигация)")

# --- 1. ЗАГРУЗКА ДАННЫХ И КАРТЫ ---
@st.cache_data
def load_data():
    used_cols = [
        'tpep_pickup_datetime', 'total_amount', 'trip_distance', 
        'PU_Borough', 'PU_Zone', 'DO_Borough', 'DO_Zone', 
        'PU_lat', 'PU_lon', 'DO_lat', 'DO_lon',
        'passenger_count', 'fare_amount',
        'PULocationID', 'DOLocationID', 'VendorID', 'RatecodeID', 'payment_type',
        'temperature', 'precipitation', 'snowfall', 'weather_code'
    ]
    
    # ИЗМЕНЕНИЕ ИЗ ВАШЕГО ФАЙЛА: Загружаем новый датасет
    df = pd.read_parquet("my_clean_3_with_weather.parquet", columns=used_cols)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek 
    
    df = df.dropna(subset=['PU_lat', 'PU_lon', 'DO_lat', 'DO_lon'])
    df = df[(df['PU_lon'] != 0) & (df['PU_lat'] != 0) & (df['DO_lon'] != 0) & (df['DO_lat'] != 0)]
    df = df[(df['fare_amount'] >= 0) & (df['passenger_count'] >= 0)] 
    
    return df

@st.cache_data
def load_borough_geojson():
    url = "https://raw.githubusercontent.com/ResidentMario/geoplot-data/master/nyc-boroughs.geojson"
    try:
        response = requests.get(url, timeout=15)
        geojson = response.json()
        colors = {
            "Manhattan": [102, 194, 165, 200],  
            "Brooklyn": [252, 141, 98, 200],    
            "Queens": [141, 160, 203, 200],     
            "Bronx": [231, 138, 195, 200],      
            "Staten Island": [166, 216, 84, 200]
        }
        for feature in geojson.get('features', []):
            props = feature.get('properties', {})
            boro = props.get('BoroName') or props.get('boro_name') or ""
            feature['properties']['fill_color'] = colors.get(boro, [200, 200, 200, 100])
        return geojson
    except Exception as e:
        return None

@st.cache_data
def load_taxi_zones_geojson():
    if os.path.exists("taxi_zones.geojson"):
        with open("taxi_zones.geojson", "r", encoding="utf-8") as f:
            return json.load(f)
    url = "https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON"
    try:
        response = requests.get(url, timeout=15)
        return response.json()
    except Exception as e:
        return None

@st.cache_data
def load_clusters():
    if os.path.exists("clusters.csv"):
        return pd.read_csv("clusters.csv")
    else:
        return pd.DataFrame({"PULocationID": np.arange(1, 264), "cluster_id": np.random.randint(0, 8, size=263)})

def colorize_geojson_with_clusters(geojson, clusters_df):
    if not geojson: return None
    cluster_map = {str(k): v for k, v in clusters_df.set_index("PULocationID")["cluster_id"].to_dict().items()}
    cluster_colors = {
        0: [228, 26, 28, 200], 1: [55, 126, 184, 200], 2: [77, 175, 74, 200], 3: [152, 78, 163, 200],
        4: [255, 127, 0, 200], 5: [0, 191, 191, 200], 6: [255, 215, 0, 200], 7: [255, 20, 147, 200],
    }
    colored_geojson = copy.deepcopy(geojson)
    for feature in colored_geojson.get("features", []):
        loc_id = str(feature.get("properties", {}).get("location_id", ""))
        c_id = cluster_map.get(loc_id, -1) 
        feature["properties"]["cluster_id"] = c_id if c_id != -1 else "Неизвестно"
        feature["properties"]["fill_color"] = cluster_colors.get(c_id, [200, 200, 200, 100])
    return colored_geojson


def get_app_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

APP_DIR = get_app_dir()
MODELS_DIR = APP_DIR

LOCAL_MODEL_CANDIDATES = {
    "demand": [
        "demand_model_with_weather.joblib",
        "demand_model.joblib",
    ],
    "revenue": [
        "revenue_model_with_weather.joblib",
        "revenue_model.joblib",
    ],
    "time": [
        "model_3_geo_weather.joblib",
        "time_model.joblib",
        "trip_time_model.joblib",
    ],
    "price": [
        "catboost_price_model.joblib",
        "price_catboost_model.joblib",
        "catboost_model.joblib",
        "price_model.joblib",
        "fare_model.joblib",
    ],
}

DEFAULT_FEATURES = {
    "demand": [
        "PULocationID", "temperature", "precipitation", "snowfall", "weather_code",
        "month", "day", "hour", "dayofweek", "is_weekend",
        "lag_1_demand", "lag_24_demand", "lag_1_revenue", "lag_24_revenue",
    ],
    "revenue": [
        "PULocationID", "temperature", "precipitation", "snowfall", "weather_code",
        "month", "day", "hour", "dayofweek", "is_weekend",
        "lag_1_demand", "lag_24_demand", "lag_1_revenue", "lag_24_revenue",
    ],
    "time": [
        "PULocationID", "DOLocationID", "PU_cluster", "DO_cluster",
        "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend",
    ],
    "price": [
        "trip_distance", "passenger_count", "duration_min", "is_rush_hour", "is_night_tariff",
        "gps_distance", "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend",
        "speed_mph", "temperature", "precipitation", "snowfall", "weather_code",
        "pickup_airport", "dropoff_airport", "same_borough", "interborough_trip",
        "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos",
        "PULocationID", "DOLocationID", "PU_cluster", "DO_cluster",
        "PU_zone_average_price", "DO_zone_average_price", "route_average_price", "ratecode_average_price",
        "RatecodeID", "payment_type", "VendorID", "PU_Borough", "DO_Borough", "distance_group",
    ],
}

PRICE_AIRPORT_ZONE_IDS = {1, 132, 138}


def find_model_file(task):
    for filename in LOCAL_MODEL_CANDIDATES.get(task, []):
        path = MODELS_DIR / filename
        if path.exists():
            return path

    local_joblibs = sorted(MODELS_DIR.glob("*.joblib"))
    if task == "price":
        catboost_candidates = [
            path for path in local_joblibs
            if "catboost" in path.name.lower() or "cat_boost" in path.name.lower()
        ]
        if catboost_candidates:
            return catboost_candidates[0]

    keywords = {
        "demand": ["demand"],
        "revenue": ["revenue"],
        "time": ["time", "duration"],
        "price": ["price", "cost", "fare", "amount"],
    }.get(task, [])
    keyword_candidates = [
        path for path in local_joblibs
        if any(word in path.name.lower() for word in keywords)
    ]
    return keyword_candidates[0] if keyword_candidates else None

def unwrap_model_object(model_object):
    if isinstance(model_object, dict):
        features = None
        for key in ["features", "feature_names", "feature_columns", "model_features", "columns"]:
            if key in model_object:
                features = list(model_object[key])
                break
        for key in ["model", "best_model", "catboost_model", "pipeline", "estimator", "boosting_model"]:
            if key in model_object:
                return model_object[key], features
    if isinstance(model_object, (tuple, list)) and model_object:
        model = model_object[0]
        features = None
        for item in model_object[1:]:
            if isinstance(item, (list, tuple, pd.Index)):
                features = list(item)
                break
        return model, features
    return model_object, None


def load_model_from_path(path):
    if path.suffix.lower() == ".cbm":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(path))
        return model, None

    try:
        model_object = joblib.load(path)
    except Exception:
        with open(path, "rb") as file:
            model_object = pickle.load(file)
    return unwrap_model_object(model_object)


def get_model_feature_names(model, saved_features=None, fallback=None):
    if saved_features:
        return list(saved_features)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_names_"):
        names = list(model.feature_names_)
        if names:
            return names
    if hasattr(model, "get_feature_names"):
        try:
            names = list(model.get_feature_names())
            if names:
                return names
        except Exception:
            pass
    return list(fallback) if fallback else None


def inspect_ml_predictor_files():
    predictors = {}
    for task in ["demand", "revenue", "time", "price"]:
        path = find_model_file(task)
        if path is None:
            predictors[task] = {"path": None, "error": "Файл модели не найден"}
        else:
            predictors[task] = {"path": path, "error": None}
    return predictors


@st.cache_resource(show_spinner=False)
def load_ml_predictors():
    predictors = {}
    shared_features = None
    features_path = MODELS_DIR / "model_features_with_weather.joblib"
    if features_path.exists():
        try:
            shared_features = list(joblib.load(features_path))
        except Exception:
            shared_features = None

    for task in ["demand", "revenue", "time", "price"]:
        path = find_model_file(task)
        if path is None:
            predictors[task] = {"model": None, "features": DEFAULT_FEATURES.get(task), "path": None, "error": "Файл модели не найден"}
            continue
        try:
            model, saved_features = load_model_from_path(path)
            if task in {"demand", "revenue"} and shared_features is not None:
                saved_features = shared_features
            features = get_model_feature_names(model, saved_features, DEFAULT_FEATURES.get(task))
            predictors[task] = {"model": model, "features": features, "path": path, "error": None}
        except Exception as exc:
            predictors[task] = {"model": None, "features": DEFAULT_FEATURES.get(task), "path": path, "error": str(exc)}
    return predictors


def haversine_miles(lon1, lat1, lon2, lat2):
    radius_miles = 3958.8
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(radius_miles * c)


@st.cache_data(show_spinner=False)
def build_zone_reference_from_columns(pu_ids, pu_zones, pu_boroughs, pu_lons, pu_lats, do_ids, do_zones, do_boroughs, do_lons, do_lats):
    pu = pd.DataFrame({
        "LocationID": pu_ids,
        "Zone": pu_zones,
        "Borough": pu_boroughs,
        "lon": pu_lons,
        "lat": pu_lats,
    }).dropna(subset=["LocationID", "Zone", "Borough", "lon", "lat"]).drop_duplicates("LocationID")
    do = pd.DataFrame({
        "LocationID": do_ids,
        "Zone": do_zones,
        "Borough": do_boroughs,
        "lon": do_lons,
        "lat": do_lats,
    }).dropna(subset=["LocationID", "Zone", "Borough", "lon", "lat"]).drop_duplicates("LocationID")
    zones = pd.concat([pu, do], ignore_index=True).drop_duplicates("LocationID")
    zones["LocationID"] = zones["LocationID"].astype(int)
    zones = zones.sort_values(["Borough", "Zone"]).reset_index(drop=True)
    zones["label"] = zones.apply(lambda row: f"{int(row['LocationID'])} — {row['Zone']} ({row['Borough']})", axis=1)
    return zones


def build_zone_reference(df):
    pu_cols = ["PULocationID", "PU_Zone", "PU_Borough", "PU_lon", "PU_lat"]
    do_cols = ["DOLocationID", "DO_Zone", "DO_Borough", "DO_lon", "DO_lat"]
    if not set(pu_cols).issubset(df.columns) or not set(do_cols).issubset(df.columns):
        return pd.DataFrame()
    pu_unique = df[pu_cols].dropna(subset=pu_cols).drop_duplicates("PULocationID")
    do_unique = df[do_cols].dropna(subset=do_cols).drop_duplicates("DOLocationID")
    return build_zone_reference_from_columns(
        tuple(pu_unique["PULocationID"].tolist()),
        tuple(pu_unique["PU_Zone"].tolist()),
        tuple(pu_unique["PU_Borough"].tolist()),
        tuple(pu_unique["PU_lon"].tolist()),
        tuple(pu_unique["PU_lat"].tolist()),
        tuple(do_unique["DOLocationID"].tolist()),
        tuple(do_unique["DO_Zone"].tolist()),
        tuple(do_unique["DO_Borough"].tolist()),
        tuple(do_unique["DO_lon"].tolist()),
        tuple(do_unique["DO_lat"].tolist()),
    )


def build_cluster_lookup():
    clusters_df = load_clusters()
    if "PULocationID" not in clusters_df.columns or "cluster_id" not in clusters_df.columns:
        return {}
    return clusters_df.set_index("PULocationID")["cluster_id"].to_dict()


def estimate_zone_hour_stats(df, pu_location_id, hour):
    if not {"PULocationID", "pickup_hour", "tpep_pickup_datetime", "total_amount"}.issubset(df.columns):
        return 1.0, 25.0
    zone_hour = df[(df["PULocationID"] == pu_location_id) & (df["pickup_hour"] == hour)]
    if zone_hour.empty:
        zone_hour = df[df["PULocationID"] == pu_location_id]
    if zone_hour.empty:
        return 1.0, 25.0
    date_hours = zone_hour["tpep_pickup_datetime"].dt.floor("h").nunique()
    date_hours = max(1, int(date_hours))
    return float(len(zone_hour) / date_hours), float(zone_hour["total_amount"].sum() / date_hours)


def build_price_stats(df, pu_location_id, do_location_id, ratecode_id):
    if "total_amount" not in df.columns:
        return {
            "global_average_price": 25.0,
            "PU_zone_average_price": 25.0,
            "DO_zone_average_price": 25.0,
            "route_average_price": 25.0,
            "ratecode_average_price": 25.0,
        }
    global_avg = float(df["total_amount"].mean()) if not df.empty else 25.0
    result = {
        "global_average_price": global_avg,
        "PU_zone_average_price": global_avg,
        "DO_zone_average_price": global_avg,
        "route_average_price": global_avg,
        "ratecode_average_price": global_avg,
    }
    if "PULocationID" in df.columns:
        values = df.loc[df["PULocationID"] == pu_location_id, "total_amount"]
        if not values.empty:
            result["PU_zone_average_price"] = float(values.mean())
    if "DOLocationID" in df.columns:
        values = df.loc[df["DOLocationID"] == do_location_id, "total_amount"]
        if not values.empty:
            result["DO_zone_average_price"] = float(values.mean())
    if {"PULocationID", "DOLocationID"}.issubset(df.columns):
        values = df.loc[(df["PULocationID"] == pu_location_id) & (df["DOLocationID"] == do_location_id), "total_amount"]
        if not values.empty:
            result["route_average_price"] = float(values.mean())
    if "RatecodeID" in df.columns:
        values = df.loc[df["RatecodeID"] == ratecode_id, "total_amount"]
        if not values.empty:
            result["ratecode_average_price"] = float(values.mean())
    return result


def distance_group(distance):
    if distance < 1:
        return "very_short"
    if distance < 3:
        return "short"
    if distance < 8:
        return "medium"
    if distance < 15:
        return "long"
    return "very_long"


def add_time_features(values, pickup_datetime):
    hour = int(pickup_datetime.hour)
    dayofweek = int(pickup_datetime.dayofweek)
    month = int(pickup_datetime.month)
    day = int(pickup_datetime.day)
    values.update({
        "hour": hour,
        "pickup_hour": hour,
        "day": day,
        "dayofweek": dayofweek,
        "pickup_dayofweek": dayofweek,
        "pickup_day_of_week": dayofweek,
        "month": month,
        "pickup_month": month,
        "is_weekend": int(dayofweek >= 5),
        "is_rush_hour": int(hour in {7, 8, 9, 16, 17, 18, 19}),
        "is_night_tariff": int(hour >= 20 or hour < 6),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dayofweek_sin": np.sin(2 * np.pi * dayofweek / 7),
        "dayofweek_cos": np.cos(2 * np.pi * dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    })
    return values


def feature_value(feature, values):
    if feature in values:
        return values[feature]
    lower_values = {str(key).lower(): value for key, value in values.items()}
    lower = str(feature).lower()
    if lower in lower_values:
        return lower_values[lower]
    aliases = {
        "pickup_day_of_week": "pickup_dayofweek",
        "weekday": "dayofweek",
        "trip_duration_min": "duration_min",
        "duration_minutes": "duration_min",
        "duration_min": "duration_min",
        "trip_time": "duration_min",
        "trip_time_min": "duration_min",
        "gps_distance_miles": "gps_distance",
        "route_avg_price": "route_average_price",
        "pu_zone_avg_price": "PU_zone_average_price",
        "do_zone_avg_price": "DO_zone_average_price",
        "ratecode_avg_price": "ratecode_average_price",
    }
    if lower in aliases and aliases[lower] in values:
        return values[aliases[lower]]
    if "temperature" in lower or lower == "temp":
        return values.get("temperature", 0.0)
    if "precip" in lower or "rain" in lower:
        return values.get("precipitation", 0.0)
    if "snow" in lower:
        return values.get("snowfall", 0.0)
    if "weather" in lower:
        return values.get("weather_code", 0)
    if "airport" in lower and "pickup" in lower:
        return values.get("pickup_airport", 0)
    if "airport" in lower and ("dropoff" in lower or lower.startswith("do_")):
        return values.get("dropoff_airport", 0)
    if "borough" in lower and ("pu" in lower or "pickup" in lower):
        return values.get("PU_Borough", "Unknown")
    if "borough" in lower and ("do" in lower or "dropoff" in lower):
        return values.get("DO_Borough", "Unknown")
    if "zone" in lower and "average" in lower and "pu" in lower:
        return values.get("PU_zone_average_price", values.get("global_average_price", 25.0))
    if "zone" in lower and "average" in lower and "do" in lower:
        return values.get("DO_zone_average_price", values.get("global_average_price", 25.0))
    if "route" in lower and "average" in lower:
        return values.get("route_average_price", values.get("global_average_price", 25.0))
    if "ratecode" in lower and "average" in lower:
        return values.get("ratecode_average_price", values.get("global_average_price", 25.0))
    if "cluster" in lower and "pu" in lower:
        return values.get("PU_cluster", -1)
    if "cluster" in lower and "do" in lower:
        return values.get("DO_cluster", -1)
    if "distance_group" in lower:
        return values.get("distance_group", "medium")
    if "lag_1_demand" in lower:
        return values.get("lag_1_demand", 1.0)
    if "lag_24_demand" in lower:
        return values.get("lag_24_demand", 1.0)
    if "lag_1_revenue" in lower:
        return values.get("lag_1_revenue", 25.0)
    if "lag_24_revenue" in lower:
        return values.get("lag_24_revenue", 25.0)
    if "speed" in lower:
        return values.get("speed_mph", 0.0)
    if "same_borough" in lower:
        return values.get("same_borough", 0)
    if "interborough" in lower:
        return values.get("interborough_trip", 0)
    return 0


def prepare_model_input(task, values, features):
    columns = features if features else DEFAULT_FEATURES.get(task, list(values.keys()))
    row = {feature: feature_value(feature, values) for feature in columns}
    frame = pd.DataFrame([row], columns=columns)
    categorical_candidates = {
        "PULocationID", "DOLocationID", "PU_cluster", "DO_cluster", "RatecodeID", "payment_type", "VendorID",
        "PU_Borough", "DO_Borough", "PU_Zone", "DO_Zone", "distance_group", "store_and_fwd_flag",
    }
    if task == "price":
        for col in frame.columns:
            if col in categorical_candidates or frame[col].dtype == object:
                frame[col] = frame[col].astype("category")
    else:
        for col in frame.columns:
            if frame[col].dtype == object:
                frame[col] = frame[col].astype("category")
    return frame


def predict_one(task, predictor, values):
    model = predictor.get("model")
    if model is None:
        return None, predictor.get("error") or "Модель не загружена"
    features = predictor.get("features")
    frame = prepare_model_input(task, values, features)
    try:
        prediction = model.predict(frame)
    except Exception as first_error:
        numeric_frame = frame.copy()
        for col in numeric_frame.columns:
            if str(numeric_frame[col].dtype) == "category" or numeric_frame[col].dtype == object:
                numeric_frame[col] = numeric_frame[col].astype("category").cat.codes
        try:
            prediction = model.predict(numeric_frame)
        except Exception as second_error:
            return None, f"{first_error}; fallback: {second_error}"
    return float(np.ravel(prediction)[0]), None



@st.cache_data(show_spinner=False)
def build_zone_reference_from_geojson_cached():
    zones_geojson = load_taxi_zones_geojson()
    if zones_geojson is None:
        return pd.DataFrame()

    rows = []
    for feature in zones_geojson.get("features", []):
        props = feature.get("properties", {}) or {}
        loc_id = props.get("location_id") or props.get("LocationID") or props.get("LocationId") or props.get("OBJECTID")
        zone = props.get("zone") or props.get("Zone") or props.get("zone_name") or props.get("Name") or "Unknown"
        borough = props.get("borough") or props.get("Borough") or props.get("boro_name") or props.get("BoroName") or "Unknown"

        try:
            loc_id = int(loc_id)
        except Exception:
            continue

        points = []

        def collect_points(obj):
            if isinstance(obj, (list, tuple)):
                if len(obj) >= 2 and isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float)):
                    lon = float(obj[0])
                    lat = float(obj[1])
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        points.append((lon, lat))
                else:
                    for item in obj:
                        collect_points(item)

        geometry = feature.get("geometry") or {}
        collect_points(geometry.get("coordinates", []))

        if points:
            lon = float(np.mean([point[0] for point in points]))
            lat = float(np.mean([point[1] for point in points]))
        else:
            lon = 0.0
            lat = 0.0

        rows.append({
            "LocationID": loc_id,
            "Zone": str(zone),
            "Borough": str(borough),
            "lon": lon,
            "lat": lat,
        })

    if not rows:
        return pd.DataFrame()

    zones = pd.DataFrame(rows).drop_duplicates("LocationID")
    zones = zones.sort_values(["Borough", "Zone"]).reset_index(drop=True)
    zones["label"] = zones.apply(lambda row: f"{int(row['LocationID'])} — {row['Zone']} ({row['Borough']})", axis=1)
    return zones


def simple_price_stats(trip_distance, ratecode_id):
    rough_price = max(8.0, 5.0 + float(trip_distance) * 3.2)
    return {
        "global_average_price": rough_price,
        "PU_zone_average_price": rough_price,
        "DO_zone_average_price": rough_price,
        "route_average_price": rough_price,
        "ratecode_average_price": rough_price if int(ratecode_id) == 1 else rough_price * 1.2,
    }

def render_predictor_status(predictors, loaded=False):
    names = {
        "demand": "Спрос",
        "price": "Цена поездки",
        "time": "Время поездки",
        "revenue": "Выручка",
    }
    title = "Статус загруженных моделей" if loaded else "Статус файлов моделей"
    with st.expander(title):
        for task, model_title in names.items():
            predictor = predictors.get(task, {})
            path = predictor.get("path")
            error = predictor.get("error")
            if loaded and predictor.get("model") is not None:
                model_file_name = path.name if path is not None else "неизвестный файл"
                st.success(f"{model_title}: загружена `{model_file_name}`")
            elif loaded and path is not None and error:
                st.error(f"{model_title}: файл найден, но не удалось загрузить `{path.name}`. Ошибка: {error}")
            elif path is not None:
                st.info(f"{model_title}: найден файл `{path.name}`")
            else:
                st.warning(f"{model_title}: {error}")


def render_ml_predictions_tab(df):
    st.markdown("### 🤖 ML-прогноз поездки")
    st.markdown(
        "Введите параметры поездки. В этой версии вкладка не сканирует большой parquet-датасет: зоны берутся из `taxi_zones.geojson`, а модели загружаются только после нажатия кнопки прогноза."
    )

    predictor_files = inspect_ml_predictor_files()
    render_predictor_status(predictor_files, loaded=False)

    zones = build_zone_reference_from_geojson_cached()
    use_zone_dropdowns = not zones.empty

    col1, col2, col3 = st.columns(3)
    with col1:
        pickup_date = st.date_input("Дата посадки", value=pd.Timestamp("2024-06-15").date())
    with col2:
        pickup_hour = st.slider("Час посадки", 0, 23, 18)
    with col3:
        passenger_count = st.number_input("Пассажиров", min_value=1, max_value=6, value=1, step=1)

    if use_zone_dropdowns:
        zone_by_label = zones.set_index("label").to_dict(orient="index")
        labels = list(zone_by_label.keys())
        default_pu = next((i for i, label in enumerate(labels) if "Midtown" in label or "Times Sq" in label), 0)
        default_do = next((i for i, label in enumerate(labels) if "JFK" in label or "Airport" in label), min(1, len(labels) - 1))

        col4, col5 = st.columns(2)
        with col4:
            pu_label = st.selectbox("Зона посадки", labels, index=default_pu)
        with col5:
            do_label = st.selectbox("Зона высадки", labels, index=default_do)

        pu = zone_by_label[pu_label]
        do = zone_by_label[do_label]
        pu_location_id = int(pu["LocationID"])
        do_location_id = int(do["LocationID"])
        pu_zone = str(pu["Zone"])
        do_zone = str(do["Zone"])
        pu_borough = str(pu["Borough"])
        do_borough = str(do["Borough"])
        pu_lon = float(pu["lon"])
        pu_lat = float(pu["lat"])
        do_lon = float(do["lon"])
        do_lat = float(do["lat"])
        gps_distance = haversine_miles(pu_lon, pu_lat, do_lon, do_lat) if all([pu_lon, pu_lat, do_lon, do_lat]) else 1.0
    else:
        st.warning("`taxi_zones.geojson` не найден или не прочитан. Включен запасной режим: введите ID зон вручную.")
        col4, col5 = st.columns(2)
        with col4:
            pu_location_id = st.number_input("PULocationID", min_value=1, max_value=263, value=237, step=1)
        with col5:
            do_location_id = st.number_input("DOLocationID", min_value=1, max_value=263, value=132, step=1)
        pu_zone = "Unknown"
        do_zone = "Unknown"
        pu_borough = "Unknown"
        do_borough = "Unknown"
        pu_lon = 0.0
        pu_lat = 0.0
        do_lon = 0.0
        do_lat = 0.0
        gps_distance = 1.0

    col6, col7, col8, col9 = st.columns(4)
    with col6:
        trip_distance = st.number_input("Дистанция поездки, мили", min_value=0.1, max_value=100.0, value=max(0.1, round(gps_distance * 1.25, 2)), step=0.1)
    with col7:
        vendor_id = st.selectbox("VendorID", [1, 2], index=1)
    with col8:
        ratecode_id = st.selectbox("RatecodeID", [1, 2, 3, 4, 5, 6], index=0)
    with col9:
        payment_type = st.selectbox("Тип оплаты", [1, 2, 3, 4], index=0)

    col10, col11, col12, col13 = st.columns(4)
    with col10:
        temperature = st.number_input("Температура, °C", value=15.0, step=0.5)
    with col11:
        precipitation = st.number_input("Осадки, мм", min_value=0.0, value=0.0, step=0.1)
    with col12:
        snowfall = st.number_input("Снег, мм", min_value=0.0, value=0.0, step=0.1)
    with col13:
        weather_code = st.number_input("Код погоды", min_value=0, value=1, step=1)

    with st.expander("Дополнительные признаки для моделей спроса и выручки"):
        st.caption("Чтобы вкладка не падала на сервере, значения по умолчанию здесь не считаются по всему parquet-файлу, а задаются вручную.")
        col14, col15, col16, col17 = st.columns(4)
        with col14:
            lag_1_demand = st.number_input("Спрос час назад", min_value=0.0, value=1.0, step=1.0)
        with col15:
            lag_24_demand = st.number_input("Спрос сутки назад", min_value=0.0, value=1.0, step=1.0)
        with col16:
            lag_1_revenue = st.number_input("Выручка час назад, $", min_value=0.0, value=25.0, step=10.0)
        with col17:
            lag_24_revenue = st.number_input("Выручка сутки назад, $", min_value=0.0, value=25.0, step=10.0)

    pickup_datetime = pd.Timestamp(pickup_date) + pd.Timedelta(hours=int(pickup_hour))
    cluster_lookup = build_cluster_lookup()
    pu_cluster = int(cluster_lookup.get(pu_location_id, -1))
    do_cluster = int(cluster_lookup.get(do_location_id, -1))
    same_borough = int(pu_borough == do_borough)
    interborough_trip = int(not same_borough)
    price_stats = simple_price_stats(trip_distance, ratecode_id)

    base_values = {
        "PULocationID": int(pu_location_id),
        "DOLocationID": int(do_location_id),
        "PU_cluster": pu_cluster,
        "DO_cluster": do_cluster,
        "PU_Borough": pu_borough,
        "DO_Borough": do_borough,
        "PU_Zone": pu_zone,
        "DO_Zone": do_zone,
        "PU_lon": pu_lon,
        "PU_lat": pu_lat,
        "DO_lon": do_lon,
        "DO_lat": do_lat,
        "VendorID": int(vendor_id),
        "RatecodeID": int(ratecode_id),
        "payment_type": int(payment_type),
        "passenger_count": int(passenger_count),
        "trip_distance": float(trip_distance),
        "gps_distance": float(gps_distance),
        "temperature": float(temperature),
        "precipitation": float(precipitation),
        "snowfall": float(snowfall),
        "weather_code": int(weather_code),
        "pickup_airport": int(int(pu_location_id) in PRICE_AIRPORT_ZONE_IDS),
        "dropoff_airport": int(int(do_location_id) in PRICE_AIRPORT_ZONE_IDS),
        "same_borough": same_borough,
        "interborough_trip": interborough_trip,
        "distance_group": distance_group(float(trip_distance)),
        "lag_1_demand": float(lag_1_demand),
        "lag_24_demand": float(lag_24_demand),
        "lag_1_revenue": float(lag_1_revenue),
        "lag_24_revenue": float(lag_24_revenue),
        **price_stats,
    }
    base_values = add_time_features(base_values, pickup_datetime)

    if st.button("Получить прогноз", type="primary"):
        with st.spinner("Загружаю модели и считаю прогноз..."):
            predictors = load_ml_predictors()
        render_predictor_status(predictors, loaded=True)

        time_prediction, time_error = predict_one("time", predictors["time"], base_values)
        if time_prediction is not None:
            duration_min = max(0.1, float(time_prediction))
        else:
            duration_min = max(1.0, float(trip_distance) / 12.0 * 60.0)

        base_values["duration_min"] = duration_min
        base_values["trip_duration_min"] = duration_min
        base_values["speed_mph"] = float(trip_distance) / max(duration_min / 60.0, 1 / 60)

        demand_prediction, demand_error = predict_one("demand", predictors["demand"], base_values)
        revenue_prediction, revenue_error = predict_one("revenue", predictors["revenue"], base_values)
        price_prediction, price_error = predict_one("price", predictors["price"], base_values)

        st.markdown("#### Результаты")
        res1, res2, res3, res4 = st.columns(4)
        with res1:
            if demand_prediction is None:
                st.metric("Спрос", "—")
                st.caption(demand_error)
            else:
                st.metric("Спрос", f"{max(0, demand_prediction):.1f} поездок/час")
        with res2:
            if price_prediction is None:
                st.metric("Цена", "—")
                st.caption(price_error)
            else:
                st.metric("Цена", f"${max(0, price_prediction):.2f}")
        with res3:
            if time_prediction is None:
                st.metric("Время", f"~{duration_min:.1f} мин")
                st.caption("Модель времени недоступна, показана грубая оценка по дистанции.")
            else:
                st.metric("Время", f"{duration_min:.1f} мин")
        with res4:
            if revenue_prediction is None:
                st.metric("Выручка", "—")
                st.caption(revenue_error)
            else:
                st.metric("Выручка", f"${max(0, revenue_prediction):.2f}/час")

        st.markdown("#### Входная строка для моделей")
        shown_values = {
            "pickup_datetime": pickup_datetime,
            "PULocationID": pu_location_id,
            "DOLocationID": do_location_id,
            "PU_cluster": pu_cluster,
            "DO_cluster": do_cluster,
            "trip_distance": trip_distance,
            "duration_min": duration_min,
            "temperature": temperature,
            "precipitation": precipitation,
            "snowfall": snowfall,
            "weather_code": weather_code,
        }
        st.dataframe(pd.DataFrame([shown_values]), width="stretch")

with st.spinner("Загрузка и подготовка данных..."):
    data = load_data()
    geojson_data = load_borough_geojson()

borough_labels = pd.DataFrame([
    {"name": "MANHATTAN", "lon": -73.97, "lat": 40.78, "align": "middle", "base": "center"},
    {"name": "BROOKLYN", "lon": -73.94, "lat": 40.64, "align": "middle", "base": "center"},
    {"name": "QUEENS", "lon": -73.83, "lat": 40.72, "align": "middle", "base": "center"},
    {"name": "BRONX", "lon": -73.88, "lat": 40.84, "align": "middle", "base": "center"},
    {"name": "STATEN ISLAND", "lon": -74.15, "lat": 40.58, "align": "middle", "base": "center"}
])

# --- 2. НАВИГАЦИЯ (УМНЫЕ ВКЛАДКИ) ---
tabs = ["🗺️ 3D Карта потоков", "🌡️ Активность (Тепловая)", "📊 Бизнес-аналитика", "🤖 ML-прогноз", "🗄️ Данные", "🧩 Кластеры зон"]
selected_tab = st.radio("Навигация:", tabs, horizontal=True, label_visibility="collapsed")

# --- 3. БОКОВАЯ ПАНЕЛЬ И ФИЛЬТРЫ ---
# Задаем значения "По умолчанию" (чтобы на других вкладках данные показывались целиком)
max_pass = max(1, int(data['passenger_count'].max()) if not data.empty and data['passenger_count'].max() > 0 else 1)

# ИЗМЕНЕНИЕ ИЗ ВАШЕГО ФАЙЛА: считаем лимиты по total_amount
max_fare = max(1.0, float(data['total_amount'].quantile(0.99)) if not data.empty and data['total_amount'].max() > 0 else 10.0)

hour_range = (0, 23)
selected_pu_boroughs = []
selected_do_boroughs = []
# ИЗМЕНЕНИЕ ИЗ ВАШЕГО ФАЙЛА: Пассажиры от 1
pass_range = (1, max_pass)
fare_range = (0.0, max_fare)

# ОСНОВНАЯ ЛОГИКА СКРЫТИЯ САЙДБАРА
if selected_tab == "🗺️ 3D Карта потоков":
    st.sidebar.header("Фильтры потоков")
    hour_range = st.sidebar.slider("Час посадки", 0, 23, (0, 23))
    all_boroughs = sorted(data['PU_Borough'].dropna().unique())
    selected_pu_boroughs = st.sidebar.multiselect("Район выезда (Откуда)", all_boroughs, placeholder="Все районы")
    selected_do_boroughs = st.sidebar.multiselect("Район въезда (Куда)", all_boroughs, placeholder="Все районы")
    
    # ИЗМЕНЕНИЕ ИЗ ВАШЕГО ФАЙЛА: Пассажиры от 1
    pass_range = st.sidebar.slider("Количество пассажиров", 1, max_pass, (1, max_pass))
    # ИЗМЕНЕНИЕ ИЗ ВАШЕГО ФАЙЛА: Название и логика фильтра цены
    fare_range = st.sidebar.slider("Общая цена поездки ($)", 0.0, max_fare, (0.0, max_fare))
else:
    st.sidebar.info("💡 **Панель фильтров скрыта.** \n\nОна доступна только на вкладке «3D Карта потоков».\n\nСейчас вы анализируете данные по всему городу целиком.")

# Применяем фильтры к данным
filtered_data = data[
    (data['pickup_hour'] >= hour_range[0]) & 
    (data['pickup_hour'] <= hour_range[1]) &
    (data['passenger_count'] >= pass_range[0]) & 
    (data['passenger_count'] <= pass_range[1]) &
    (data['total_amount'] >= fare_range[0]) & # ИЗМЕНЕНИЕ: Фильтруем по total_amount
    (data['total_amount'] <= fare_range[1])
]

if selected_pu_boroughs:
    filtered_data = filtered_data[filtered_data['PU_Borough'].isin(selected_pu_boroughs)]
if selected_do_boroughs:
    filtered_data = filtered_data[filtered_data['DO_Borough'].isin(selected_do_boroughs)]

# --- АГРЕГАЦИЯ ДЛЯ КАРТЫ ---
global_pu_centers = data.groupby('PU_Zone')[['PU_lon', 'PU_lat']].mean().reset_index()
global_do_centers = data.groupby('DO_Zone')[['DO_lon', 'DO_lat']].mean().reset_index()

agg_flows = filtered_data.groupby(['PU_Zone', 'DO_Zone']).agg(
    trip_count=('total_amount', 'count'),       
    avg_fare=('total_amount', 'mean'),          
    avg_dist=('trip_distance', 'mean')
).reset_index()

agg_flows = agg_flows.merge(global_pu_centers, on='PU_Zone', how='left')
agg_flows = agg_flows.merge(global_do_centers, on='DO_Zone', how='left')
agg_flows = agg_flows[agg_flows['PU_Zone'] != agg_flows['DO_Zone']]
agg_flows['avg_fare'] = agg_flows['avg_fare'].round(2)
agg_flows['avg_dist'] = agg_flows['avg_dist'].round(2)
agg_flows = agg_flows.dropna(subset=['PU_lon', 'PU_lat', 'DO_lon', 'DO_lat'])

if not agg_flows.empty:
    max_trips = agg_flows['trip_count'].max()
    agg_flows['line_width'] = (agg_flows['trip_count'] / max_trips) * 11 + 1
else:
    agg_flows['line_width'] = 1

max_trips_in_data = int(agg_flows['trip_count'].max()) if not agg_flows.empty else 1
slider_max = max(2, int(max_trips_in_data / 2)) 
default_min = max(1, int(max_trips_in_data * 0.05)) 

# Ползунок скрытия мелких маршрутов тоже прячем, если мы не на карте
if selected_tab == "🗺️ 3D Карта потоков":
    min_trips = st.sidebar.slider("Скрыть редкие маршруты", min_value=1, max_value=slider_max, value=default_min)
else:
    min_trips = default_min

map_data = agg_flows[agg_flows['trip_count'] >= min_trips]
view_state = pdk.ViewState(latitude=40.71, longitude=-73.95, zoom=10, pitch=45, bearing=15)

text_layer = pdk.Layer(
    "TextLayer", data=borough_labels, get_position="[lon, lat, 500]", get_text="name",
    get_size=22, get_color=[0, 0, 0, 255], get_text_anchor="align", get_alignment_baseline="base"
)

# --- 4. ОТРИСОВКА ИНТЕРФЕЙСА (Зависит от выбранной вкладки) ---

if selected_tab == "🗺️ 3D Карта потоков":
    layers_tab1 = []
    if geojson_data:
        base_layer = pdk.Layer("GeoJsonLayer", data=geojson_data, opacity=0.8, stroked=True, filled=True, extruded=False, get_fill_color="properties.fill_color", get_line_color=[255, 255, 255, 150], pickable=False)
        layers_tab1.append(base_layer)
        
    layers_tab1.append(text_layer)
        
    if not map_data.empty:
        arc_layer = pdk.Layer("ArcLayer", data=map_data, get_source_position="[PU_lon, PU_lat]", get_target_position="[DO_lon, DO_lat]", get_source_color=[0, 102, 204, 255], get_target_color=[255, 128, 0, 255], get_width="line_width", auto_highlight=True, pickable=True)
        layers_tab1.append(arc_layer)

    if len(layers_tab1) <= 2 and map_data.empty:
        st.warning("Нет данных по заданным фильтрам.")
    else:
        tooltip = {"html": "<b>{PU_Zone} ➡️ {DO_Zone}</b><br/>Поездок: {trip_count}<br/>Чек: ${avg_fare}<br/>Дистанция: {avg_dist} миль", "style": {"backgroundColor": "white", "color": "black", "padding": "10px", "borderRadius": "5px"}}
        st.pydeck_chart(pdk.Deck(layers=layers_tab1, initial_view_state=view_state, map_style="light", tooltip=tooltip))

elif selected_tab == "🌡️ Активность (Тепловая)":
    heatmap_type = st.radio("Что анализируем?", ["Посадки (Спрос)", "Высадки (Притяжение)"], horizontal=True)
    if "Посадки" in heatmap_type:
        zone_heat = filtered_data.groupby('PU_Zone').size().reset_index(name='trip_count')
        zone_heat = zone_heat.merge(global_pu_centers, on='PU_Zone', how='inner')
        lon_col, lat_col, tooltip_zone = "PU_lon", "PU_lat", "PU_Zone"
    else:
        zone_heat = filtered_data.groupby('DO_Zone').size().reset_index(name='trip_count')
        zone_heat = zone_heat.merge(global_do_centers, on='DO_Zone', how='inner')
        lon_col, lat_col, tooltip_zone = "DO_lon", "DO_lat", "DO_Zone"
        
    layers_tab2 = []
    if geojson_data:
        flat_base_layer = pdk.Layer("GeoJsonLayer", data=geojson_data, opacity=0.3, stroked=True, filled=True, extruded=False, get_fill_color="properties.fill_color", get_line_color=[255, 255, 255, 100], pickable=False)
        layers_tab2.append(flat_base_layer)
        
    layers_tab2.append(text_layer)
        
    if not zone_heat.empty:
        max_trips_heat = zone_heat['trip_count'].max() if zone_heat['trip_count'].max() > 0 else 1
        zone_heat['color_g'] = (255 * (1 - (zone_heat['trip_count'] / max_trips_heat))).astype(int)
        heatmap_layer = pdk.Layer("HeatmapLayer", data=zone_heat, get_position=f"[{lon_col}, {lat_col}]", get_weight="trip_count", opacity=0.6, radiusPixels=40)
        scatter_layer = pdk.Layer("ScatterplotLayer", data=zone_heat, get_position=f"[{lon_col}, {lat_col}]", get_radius="trip_count", radius_scale=1500 / max_trips_heat if max_trips_heat > 0 else 1, radius_min_pixels=5, radius_max_pixels=40, get_fill_color="[255, color_g, 0, 200]", get_line_color="[255, 255, 255, 255]", pickable=True, auto_highlight=True)
        layers_tab2.extend([heatmap_layer, scatter_layer])
        
    tooltip_heat = {"html": f"<b>{{{tooltip_zone}}}</b><br/><h2 style='margin:0;color:red;'>{{trip_count}}</h2><small>заказов</small>", "style": {"backgroundColor": "white", "color": "black", "padding": "10px", "borderRadius": "5px"}}
    st.pydeck_chart(pdk.Deck(layers=layers_tab2, initial_view_state=pdk.ViewState(latitude=40.71, longitude=-73.95, zoom=10, pitch=0), map_style="light", tooltip=tooltip_heat))

elif selected_tab == "📊 Бизнес-аналитика":
    col1, col2 = st.columns(2)
    with col1:
        top_trips = map_data.sort_values('trip_count', ascending=False).head(10)
        if not top_trips.empty:
            top_trips['route'] = top_trips['PU_Zone'] + " ➡️ " + top_trips['DO_Zone']
            st.plotly_chart(px.bar(top_trips, x='trip_count', y='route', orientation='h', title="Самые частые маршруты").update_layout(yaxis={'categoryorder':'total ascending'}), width="stretch")

    with col2:
        popular_routes = map_data[map_data['trip_count'] > 5]
        if not popular_routes.empty:
            top_revenue = popular_routes.sort_values('avg_fare', ascending=False).head(10)
            top_revenue['route'] = top_revenue['PU_Zone'] + " ➡️ " + top_revenue['DO_Zone']
            st.plotly_chart(px.bar(top_revenue, x='avg_fare', y='route', orientation='h', title="Самый высокий средний чек ($)", color_discrete_sequence=['#2ca02c']).update_layout(yaxis={'categoryorder':'total ascending'}), width="stretch")
            
    col3, col4 = st.columns(2)
    with col3:
        days_map = {0: '1. Пн', 1: '2. Вт', 2: '3. Ср', 3: '4. Чт', 4: '5. Пт', 5: '6. Сб', 6: '7. Вс'}
        heatmap_df = filtered_data.groupby(['pickup_day_of_week', 'pickup_hour']).size().reset_index(name='trips')
        heatmap_df['day_name'] = heatmap_df['pickup_day_of_week'].map(days_map)
        fig_heat = px.density_heatmap(heatmap_df, x="pickup_hour", y="day_name", z="trips", title="Матрица нагрузки (День недели / Час)", labels={'pickup_hour': 'Час посадки', 'day_name': 'День недели', 'trips': 'Заказы'}, color_continuous_scale="Reds").update_yaxes(categoryorder='category descending')
        st.plotly_chart(fig_heat, width="stretch")

    with col4:
        hourly_demand = filtered_data.groupby('pickup_hour').size().reset_index(name='trips')
        st.plotly_chart(px.line(hourly_demand, x='pickup_hour', y='trips', markers=True, title="Общий спрос по часам (Загруженность)", labels={'pickup_hour': 'Час дня', 'trips': 'Поездок'}), width="stretch")

    col5, col6 = st.columns(2)
    with col5:
        fpm_df = filtered_data[filtered_data['trip_distance'] > 0.5].groupby('PU_Borough').agg(total_fare=('fare_amount', 'sum'), total_dist=('trip_distance', 'sum')).reset_index()
        fpm_df['fare_per_mile'] = fpm_df['total_fare'] / fpm_df['total_dist']
        st.plotly_chart(px.bar(fpm_df.sort_values('fare_per_mile', ascending=False), x='PU_Borough', y='fare_per_mile', title="Рентабельность ($ за милю) по районам", labels={'PU_Borough': 'Район', 'fare_per_mile': '$ за милю'}, color='fare_per_mile', color_continuous_scale="Greens"), width="stretch")

    with col6:
        sample_size = min(2000, len(filtered_data))
        scatter_sample = filtered_data.sample(sample_size) if not filtered_data.empty else filtered_data
        st.plotly_chart(px.scatter(scatter_sample, x='trip_distance', y='fare_amount', color='PU_Borough', title="Стоимость vs Дистанция (Выборка 2000 поездок)", labels={'trip_distance': 'Дистанция (миль)', 'fare_amount': 'Тариф ($)'}, opacity=0.6), width="stretch")

    col7, col8 = st.columns(2)
    with col7:
        pass_df = filtered_data['passenger_count'].value_counts().reset_index()
        pass_df.columns = ['Пассажиры', 'Количество']
        pass_df = pass_df[pass_df['Пассажиры'] > 0]
        pass_df['Пассажиры'] = pass_df['Пассажиры'].astype(int).astype(str) + " чел."
        st.plotly_chart(px.pie(pass_df, values='Количество', names='Пассажиры', hole=0.4, title="Структура поездок по числу пассажиров"), width="stretch")

    with col8:
        max_dist = min(25, filtered_data['trip_distance'].max()) 
        counts, bins = np.histogram(filtered_data[(filtered_data['trip_distance'] > 0) & (filtered_data['trip_distance'] <= max_dist)]['trip_distance'], bins=25)
        bins_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
        dist_hist_df = pd.DataFrame({'Дистанция (мили)': bins_labels, 'Количество': counts})
        st.plotly_chart(px.bar(dist_hist_df, x='Дистанция (мили)', y='Количество', title="Какую дистанцию чаще всего ездят?", color_discrete_sequence=['#1f77b4']), width="stretch")

elif selected_tab == "🤖 ML-прогноз":
    render_ml_predictions_tab(data)

elif selected_tab == "🗄️ Данные":
    st.dataframe(map_data.sort_values(by='trip_count', ascending=False).drop(columns=['PU_lon', 'PU_lat', 'DO_lon', 'DO_lat', 'line_width', 'color_g'], errors='ignore'))

elif selected_tab == "🧩 Кластеры зон":
    st.markdown("### 🧩 Кластеризация округов (NYC Taxi Zones)")
    st.markdown("Модель машинного обучения разделила географические зоны Нью-Йорка на 8 уникальных кластеров. Каждому кластеру присвоен свой цвет.")
    
    clusters_df = load_clusters()
    zones_geojson = load_taxi_zones_geojson()
    
    if zones_geojson is None:
        st.error("⚠️ Не удалось загрузить геометрию мелких зон. Если официальный портал блокирует ваш IP, скачайте файл `taxi_zones.geojson` и положите его в папку с проектом.")
    else:
        colored_zones = colorize_geojson_with_clusters(zones_geojson, clusters_df)
        cluster_layer = pdk.Layer("GeoJsonLayer", data=colored_zones, opacity=0.8, stroked=True, filled=True, extruded=False, get_fill_color="properties.fill_color", get_line_color=[255, 255, 255, 100], pickable=True, auto_highlight=True)
        tooltip_cluster = {"html": "<b>Зона:</b> {zone} ({borough})<br/><b>Location ID:</b> {location_id}<hr style='margin: 5px 0; border: 0.5px solid #ccc;'/><h3 style='margin: 0; color: #ff4b4b;'>Кластер: {cluster_id}</h3>", "style": {"backgroundColor": "white", "color": "black", "padding": "10px", "borderRadius": "5px"}}
        
        st.pydeck_chart(pdk.Deck(layers=[cluster_layer], initial_view_state=pdk.ViewState(latitude=40.71, longitude=-73.95, zoom=10, pitch=0), map_style="light", tooltip=tooltip_cluster))
        
        st.markdown("---")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if not os.path.exists("clusters.csv"):
                st.warning("⚠️ Файл `clusters.csv` не найден! Показаны случайные ДЕМО-ДАННЫЕ. **Сохраните ваш датафрейм со скриншота в файл `clusters.csv`** (с колонками PULocationID и cluster_id), чтобы увидеть реальную кластеризацию.")
            st.markdown("#### 📂 Датасет кластеров (Срез)")
            st.dataframe(clusters_df)
            
        with col_c2:
            st.markdown("#### 📊 Распределение зон по кластерам")
            plot_df = clusters_df.copy()
            plot_df['cluster_str'] = 'Кластер ' + plot_df['cluster_id'].astype(str)
            cluster_counts = plot_df['cluster_str'].value_counts().reset_index()
            cluster_counts.columns = ['Кластер', 'Количество зон']
            cluster_counts = cluster_counts.sort_values('Кластер')
            st.plotly_chart(px.bar(cluster_counts, x="Кластер", y="Количество зон", color="Кластер", color_discrete_sequence=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#00bfbf', '#ffd700', '#ff1493']), width="stretch")