import pandas as pd


# Функции для чтения файлов погоды с учетом пропуска метаданных
def load_hourly_weather(file_path):
    # skiprows=3 пропускает первые 3 строчки (координаты, таймзону и пустую строку)
    df_weather = pd.read_csv(file_path, skiprows=3)

    # Переименуем колонки в удобный формат без спецсимволов
    rename_dict = {
        'time': 'weather_time',
        'temperature_2m (°C)': 'temperature',
        'relative_humidity_2m (%)': 'humidity',
        'precipitation (mm)': 'precipitation',
        'rain (mm)': 'rain',
        'snowfall (cm)': 'snowfall',
        'weather_code (wmo code)': 'weather_code',
        'cloud_cover (%)': 'cloud_cover',
        'wind_speed_10m (km/h)': 'wind_speed'
    }
    # Переименовываем только те колонки, которые есть в датасете
    df_weather = df_weather.rename(columns={k: v for k, v in rename_dict.items() if k in df_weather.columns})

    # Переводим колонку времени в формат datetime
    df_weather['weather_time'] = pd.to_datetime(df_weather['weather_time'])

    return df_weather


# Загружаем оба года (замените названия файлов на ваши реальные, если они отличаются)
try:
    weather_2023 = load_hourly_weather('weather_2023.csv')  # Укажите ваше имя файла за 2023
    weather_2024 = load_hourly_weather('weather_2024.csv')  # Укажите ваше имя файла за 2024

    # Объединяем 2023 и 2024 годы в один датафрейм
    full_weather = pd.concat([weather_2023, weather_2024], ignore_index=True)
    print(f"Погода загружена! Всего строк: {full_weather.shape[0]}")
    print(full_weather[['weather_time', 'temperature', 'precipitation', 'snowfall']].head(3))

except FileNotFoundError as e:
    print(f"Ошибка: Не удалось найти файл. Проверьте имя. Технический текст: {e}")

# 1. Загружаем ваш очищенный датасет такси
df_taxi = pd.read_parquet('my_clean_2.parquet')

# 2. Приводим время поездки к формату datetime
df_taxi['tpep_pickup_datetime'] = pd.to_datetime(df_taxi['tpep_pickup_datetime'])

# 3. Создаем колонку 'pickup_hour', округляя время поездки вниз до ближайшего часа
# Например: 2024-01-01 00:24:15 превратится в 2024-01-01 00:00:00
# Это позволит нам сопоставить поездку с точным часом в погодном файле
df_taxi['pickup_hour'] = df_taxi['tpep_pickup_datetime'].dt.floor('h')

# 4. Выбираем, какие именно погодные признаки мы хотим добавить.
# Рекомендую взять температуру, осадки, снегопад и код погоды (он кодирует ясно/туман/ливень)
weather_features = ['weather_time', 'temperature', 'precipitation', 'snowfall', 'weather_code']

# 5. Делаем merge (слияние) по созданным часовым колонкам
df_res = df_taxi.merge(
    full_weather[weather_features],
    left_on='pickup_hour',
    right_on='weather_time',
    how='left'  # Используем left, чтобы не потерять поездки, если по какому-то часу нет погоды
)

# 6. Удаляем технические колонки, которые использовались для склейки, чтобы не дублировать данные
df_res = df_res.drop(columns=['pickup_hour', 'weather_time'])

print(f"Слияние завершено! Размер итогового датасета: {df_res.shape}")
print("\nПроверяем новые столбцы в данных такси:")
print(df_res[['tpep_pickup_datetime', 'temperature', 'precipitation', 'weather_code']].head())

# 7. Сохраняем обновленный датасет
df_res.to_parquet('my_clean_3_with_weather.parquet', index=False)
print("\nФайл сохранен как 'my_clean_3_with_weather.parquet'")