import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import requests

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="NYC Taxi Analytics 3D", layout="wide")
st.title("🚖 3D Аналитика потоков (Цветные Районы + Подписи)")

# --- 1. ЗАГРУЗКА ДАННЫХ И КАРТЫ ---
@st.cache_data
def load_data():
    used_cols = [
        'tpep_pickup_datetime', 'total_amount', 'trip_distance', 
        'PU_Borough', 'PU_Zone', 'DO_Borough', 'DO_Zone', 
        'PU_lat', 'PU_lon', 'DO_lat', 'DO_lon',
        'passenger_count', 'fare_amount'
    ]
    
    df = pd.read_parquet("my_clean_3_with_weather.parquet", columns=used_cols) # Надо менять название файла
    
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek 
    
    df = df.dropna(subset=['PU_lat', 'PU_lon', 'DO_lat', 'DO_lon'])
    df = df[(df['PU_lon'] != 0) & (df['PU_lat'] != 0) & (df['DO_lon'] != 0) & (df['DO_lat'] != 0)]
    df = df[(df['fare_amount'] >= 0) & (df['passenger_count'] >= 0)] 
    
    return df

@st.cache_data
def load_borough_geojson():
    """Загружает границы 5 районов Нью-Йорка и красит их"""
    url = "https://raw.githubusercontent.com/ResidentMario/geoplot-data/master/nyc-boroughs.geojson"
    try:
        response = requests.get(url, timeout=15)
        geojson = response.json()
        
        # Насыщенная пастельная палитра для 5 районов
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

with st.spinner("Загрузка и подготовка данных..."):
    data = load_data()
    geojson_data = load_borough_geojson()
    if geojson_data is None:
        st.error("⚠️ Не удалось загрузить границы районов (ошибка сети). Карта будет плоской.")

# ❗️ ЖЕЛЕЗОБЕТОННЫЙ ДАТАФРЕЙМ С ПОДПИСЯМИ ❗️
# align и base прописаны здесь, чтобы обойти баги парсера PyDeck
borough_labels = pd.DataFrame([
    {"name": "MANHATTAN", "lon": -73.97, "lat": 40.78, "align": "middle", "base": "center"},
    {"name": "BROOKLYN", "lon": -73.94, "lat": 40.64, "align": "middle", "base": "center"},
    {"name": "QUEENS", "lon": -73.83, "lat": 40.72, "align": "middle", "base": "center"},
    {"name": "BRONX", "lon": -73.88, "lat": 40.84, "align": "middle", "base": "center"},
    {"name": "STATEN ISLAND", "lon": -74.15, "lat": 40.58, "align": "middle", "base": "center"}
])

# --- 2. БОКОВАЯ ПАНЕЛЬ: ФИЛЬТРЫ ---
st.sidebar.header("Фильтры поездок")

hour_range = st.sidebar.slider("Час посадки", 0, 23, (0, 23))
all_boroughs = sorted(data['PU_Borough'].dropna().unique())
selected_pu_boroughs = st.sidebar.multiselect("Район въезда (Откуда)", all_boroughs, placeholder="Все районы")
selected_do_boroughs = st.sidebar.multiselect("Район выезда (Куда)", all_boroughs, placeholder="Все районы")

max_pass = int(data['passenger_count'].max()) if not data.empty and data['passenger_count'].max() > 0 else 1
max_pass = max(1, max_pass)
pass_range = st.sidebar.slider("Количество пассажиров", 1, max_pass, (1, max_pass))

if not data.empty and data['total_amount'].max() > 0:
    max_fare = float(data['total_amount'].quantile(0.99))
else:
    max_fare = 10.0

max_fare = max(1.0, max_fare)
fare_range = st.sidebar.slider("Общая цена поездки ($)", 0.0, max_fare, (0.0, max_fare))

filtered_data = data[
    (data['pickup_hour'] >= hour_range[0]) & 
    (data['pickup_hour'] <= hour_range[1]) &
    (data['passenger_count'] >= pass_range[0]) & 
    (data['passenger_count'] <= pass_range[1]) &
    (data['total_amount'] >= fare_range[0]) &
    (data['total_amount'] <= fare_range[1])
]

if selected_pu_boroughs:
    filtered_data = filtered_data[filtered_data['PU_Borough'].isin(selected_pu_boroughs)]
if selected_do_boroughs:
    filtered_data = filtered_data[filtered_data['DO_Borough'].isin(selected_do_boroughs)]

# --- 3. АГРЕГАЦИЯ ПОТОКОВ ---
with st.spinner("Анализ направлений..."):
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

min_trips = st.sidebar.slider("Скрыть редкие маршруты", min_value=1, max_value=slider_max, value=default_min)
map_data = agg_flows[agg_flows['trip_count'] >= min_trips]

view_state = pdk.ViewState(latitude=40.71, longitude=-73.95, zoom=10, pitch=45, bearing=15)

# --- ❗️ ИСПРАВЛЕННЫЙ СЛОЙ ПОДПИСЕЙ ❗️ ---
text_layer = pdk.Layer(
    "TextLayer",
    data=borough_labels,
    # Поднимаем текст на 500 метров вверх (Z-ось), чтобы он "парил" и не перекрывался цветом районов!
    get_position="[lon, lat, 500]", 
    get_text="name",
    get_size=22, # Сделал шрифт крупнее
    get_color=[0, 0, 0, 255], # 100% черный и непрозрачный
    get_text_anchor="align",
    get_alignment_baseline="base"
)

# --- 4. ИНТЕРФЕЙС ---
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 3D Карта потоков", "🌡️ Активность (Тепловая)", "📊 Бизнес-аналитика", "🗄️ Данные"])

# =========== ВКЛАДКА 1: 3D КАРТА ===========
with tab1:
    layers_tab1 = []
    
    # 1. СЛОЙ 2D РАЙОНОВ (без 3D экструзии)
    if geojson_data:
        base_layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson_data,
            opacity=0.8, 
            stroked=True,
            filled=True,
            extruded=False, 
            get_fill_color="properties.fill_color",
            get_line_color=[255, 255, 255, 150],
            pickable=False
        )
        layers_tab1.append(base_layer)
        
    # 2. СЛОЙ ПОДПИСЕЙ РАЙОНОВ
    layers_tab1.append(text_layer)
        
    # 3. СЛОЙ ДУГ (Потоки такси)
    if not map_data.empty:
        arc_layer = pdk.Layer(
            "ArcLayer", data=map_data,
            get_source_position="[PU_lon, PU_lat]", get_target_position="[DO_lon, DO_lat]",
            get_source_color=[0, 102, 204, 255], get_target_color=[255, 128, 0, 255],
            get_width="line_width", auto_highlight=True, pickable=True,
        )
        layers_tab1.append(arc_layer)

    if len(layers_tab1) <= 2 and map_data.empty:
        st.warning("Нет данных по заданным фильтрам.")
    else:
        tooltip = {
            "html": "<b>{PU_Zone} ➡️ {DO_Zone}</b><br/>Поездок: {trip_count}<br/>Чек: ${avg_fare}<br/>Дистанция: {avg_dist} миль",
            "style": {"backgroundColor": "white", "color": "black", "padding": "10px", "borderRadius": "5px"}
        }
        st.pydeck_chart(pdk.Deck(layers=layers_tab1, initial_view_state=view_state, map_style="light", tooltip=tooltip))

# =========== ВКЛАДКА 2: ТЕПЛОВАЯ КАРТА ===========
with tab2:
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
        flat_base_layer = pdk.Layer(
            "GeoJsonLayer", data=geojson_data, opacity=0.3, stroked=True, filled=True,
            extruded=False, 
            get_fill_color="properties.fill_color", get_line_color=[255, 255, 255, 100], pickable=False
        )
        layers_tab2.append(flat_base_layer)
        
    # Добавляем подписи и на тепловую карту тоже
    layers_tab2.append(text_layer)
        
    if not zone_heat.empty:
        max_trips_heat = zone_heat['trip_count'].max() if zone_heat['trip_count'].max() > 0 else 1
        zone_heat['color_g'] = (255 * (1 - (zone_heat['trip_count'] / max_trips_heat))).astype(int)
        
        heatmap_layer = pdk.Layer("HeatmapLayer", data=zone_heat, get_position=f"[{lon_col}, {lat_col}]", get_weight="trip_count", opacity=0.6, radiusPixels=40)
        scatter_layer = pdk.Layer(
            "ScatterplotLayer", data=zone_heat, get_position=f"[{lon_col}, {lat_col}]",
            get_radius="trip_count", radius_scale=1500 / max_trips_heat if max_trips_heat > 0 else 1, radius_min_pixels=5, radius_max_pixels=40,
            get_fill_color="[255, color_g, 0, 200]", get_line_color="[255, 255, 255, 255]", pickable=True, auto_highlight=True
        )
        layers_tab2.extend([heatmap_layer, scatter_layer])
        
    tooltip_heat = {"html": f"<b>{{{tooltip_zone}}}</b><br/><h2 style='margin:0;color:red;'>{{trip_count}}</h2><small>заказов</small>", "style": {"backgroundColor": "white", "color": "black", "padding": "10px", "borderRadius": "5px"}}
    
    st.pydeck_chart(pdk.Deck(layers=layers_tab2, initial_view_state=pdk.ViewState(latitude=40.71, longitude=-73.95, zoom=10, pitch=0), map_style="light", tooltip=tooltip_heat))

# =========== ВКЛАДКА 3: БИЗНЕС-АНАЛИТИКА ===========
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        top_trips = map_data.sort_values('trip_count', ascending=False).head(10)
        if not top_trips.empty:
            top_trips['route'] = top_trips['PU_Zone'] + " ➡️ " + top_trips['DO_Zone']
            fig1 = px.bar(top_trips, x='trip_count', y='route', orientation='h', title="Самые частые маршруты")
            fig1.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, width="stretch")

    with col2:
        popular_routes = map_data[map_data['trip_count'] > 5]
        if not popular_routes.empty:
            top_revenue = popular_routes.sort_values('avg_fare', ascending=False).head(10)
            top_revenue['route'] = top_revenue['PU_Zone'] + " ➡️ " + top_revenue['DO_Zone']
            fig2 = px.bar(top_revenue, x='avg_fare', y='route', orientation='h', title="Самый высокий средний чек ($)", color_discrete_sequence=['#2ca02c'])
            fig2.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, width="stretch")
            
    col3, col4 = st.columns(2)
    with col3:
        days_map = {0: '1. Пн', 1: '2. Вт', 2: '3. Ср', 3: '4. Чт', 4: '5. Пт', 5: '6. Сб', 6: '7. Вс'}
        heatmap_df = filtered_data.groupby(['pickup_day_of_week', 'pickup_hour']).size().reset_index(name='trips')
        heatmap_df['day_name'] = heatmap_df['pickup_day_of_week'].map(days_map)
        
        fig_heat = px.density_heatmap(
            heatmap_df, x="pickup_hour", y="day_name", z="trips", 
            title="Матрица нагрузки (День недели / Час)",
            labels={'pickup_hour': 'Час посадки', 'day_name': 'День недели', 'trips': 'Заказы'},
            color_continuous_scale="Reds"
        )
        fig_heat.update_yaxes(categoryorder='category descending') 
        st.plotly_chart(fig_heat, width="stretch")

    with col4:
        hourly_demand = filtered_data.groupby('pickup_hour').size().reset_index(name='trips')
        fig3 = px.line(hourly_demand, x='pickup_hour', y='trips', markers=True, title="Общий спрос по часам (Загруженность)", labels={'pickup_hour': 'Час дня', 'trips': 'Поездок'})
        st.plotly_chart(fig3, width="stretch")

    col5, col6 = st.columns(2)
    with col5:
        fpm_df = filtered_data[filtered_data['trip_distance'] > 0.5].groupby('PU_Borough').agg(
            total_fare=('fare_amount', 'sum'),
            total_dist=('trip_distance', 'sum')
        ).reset_index()
        fpm_df['fare_per_mile'] = fpm_df['total_fare'] / fpm_df['total_dist']
        
        fig_fpm = px.bar(
            fpm_df.sort_values('fare_per_mile', ascending=False), 
            x='PU_Borough', y='fare_per_mile', 
            title="Рентабельность ($ за милю пути) по районам посадки",
            labels={'PU_Borough': 'Район', 'fare_per_mile': '$ за милю'},
            color='fare_per_mile', color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_fpm, width="stretch")

    with col6:
        sample_size = min(2000, len(filtered_data))
        scatter_sample = filtered_data.sample(sample_size) if not filtered_data.empty else filtered_data
        fig4 = px.scatter(scatter_sample, x='trip_distance', y='fare_amount', color='PU_Borough', title="Стоимость vs Дистанция (Выборка 2000 поездок)", labels={'trip_distance': 'Дистанция (миль)', 'fare_amount': 'Тариф ($)'}, opacity=0.6)
        st.plotly_chart(fig4, width="stretch")

    col7, col8 = st.columns(2)
    with col7:
        pass_df = filtered_data['passenger_count'].value_counts().reset_index()
        pass_df.columns = ['Пассажиры', 'Количество']
        pass_df = pass_df[pass_df['Пассажиры'] > 0]
        pass_df['Пассажиры'] = pass_df['Пассажиры'].astype(int).astype(str) + " чел."
        
        fig_pie = px.pie(
            pass_df, values='Количество', names='Пассажиры', hole=0.4,
            title="Структура поездок по числу пассажиров"
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col8:
        max_dist = min(25, filtered_data['trip_distance'].max()) 
        counts, bins = np.histogram(filtered_data[(filtered_data['trip_distance'] > 0) & (filtered_data['trip_distance'] <= max_dist)]['trip_distance'], bins=25)
        
        bins_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
        dist_hist_df = pd.DataFrame({'Дистанция (мили)': bins_labels, 'Количество': counts})
        
        fig_hist = px.bar(
            dist_hist_df, x='Дистанция (мили)', y='Количество', 
            title="Какую дистанцию чаще всего ездят?",
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_hist, width="stretch")

# =========== ВКЛАДКА 4: СЫРЫЕ ДАННЫЕ ===========
with tab4:
    st.dataframe(map_data.sort_values(by='trip_count', ascending=False).drop(columns=['PU_lon', 'PU_lat', 'DO_lon', 'DO_lat', 'line_width', 'color_g'], errors='ignore'))
