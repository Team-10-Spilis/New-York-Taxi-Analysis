import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import requests
import json
import os
import copy

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
        'passenger_count', 'fare_amount'
    ]
    
    # ИЗМЕНЕНИЕ ИЗ ВАШЕГО ФАЙЛА: Загружаем новый датасет
    df = pd.read_parquet("data.parquet", columns=used_cols)
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
tabs = ["🗺️ 3D Карта потоков", "🌡️ Активность (Тепловая)", "📊 Бизнес-аналитика", "🗄️ Данные", "🧩 Кластеры зон"]
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