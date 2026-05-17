import numpy as np
import pandas as pd


def audit_taxi_anomalies(dataframe):
    # Делаем копию, чтобы случайно не испортить исходный датасет
    df_check = dataframe.copy()

    print("=== ЗАПУСК КОМПЛЕКСНОГО АУДИТА АНОМАЛИЙ ===\n")
    total_rows = len(df_check)

    # 0. Подготовка дат
    df_check["tpep_pickup_datetime"] = pd.to_datetime(
        df_check["tpep_pickup_datetime"]
    )
    df_check["tpep_dropoff_datetime"] = pd.to_datetime(
        df_check["tpep_dropoff_datetime"]
    )

    # Вычисляем вспомогательные метрики: длительность в минутах и скорость в милях/час
    df_check["duration_min"] = (
        df_check["tpep_dropoff_datetime"] - df_check["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    # Избегаем деления на ноль при расчете скорости
    df_check["speed_mph"] = np.where(
        df_check["duration_min"] > 0,
        df_check["trip_distance"] / (df_check["duration_min"] / 60),
        0,
    )

    # Словарь для хранения результатов
    anomalies = {}

    # 1. ВРЕМЕННЫЕ АНОМАЛИИ
    # Машина времени: высадка произошла РАНЬШЕ посадки
    anomalies["Высадка раньше посадки (назад в будущее)"] = (
        df_check["duration_min"] < 0
    ).sum()
    # Нулевая длительность при ненулевом расстоянии
    anomalies["Мгновенные поездки (время = 0 мин, но дистанция > 0)"] = (
        (df_check["duration_min"] == 0) & (df_check["trip_distance"] > 0)
    ).sum()
    # Слишком долгие поездки (например, больше 12 часов в пределах города)
    anomalies["Поездки дольше 12 часов (возможно, забыли выключить счетчик)"] = (
        df_check["duration_min"] > 720
    ).sum()

    # 2. ГЕОГРАФИЧЕСКИЕ И СКОРОСТНЫЕ АНОМАЛИИ
    # Космическая скорость (в Нью-Йорке ограничение ~25-50 mph, все что выше 100 mph — явный баг GPS)
    anomalies["Сверхзвуковая скорость (> 100 миль/час)"] = (
        df_check["speed_mph"] > 100
    ).sum()
    # Телепортация: дистанция = 0, но деньги за поездку сняты
    anomalies["Телепортация (дистанция = 0, но стоимость > 5$)"] = (
        (df_check["trip_distance"] <= 0) & (df_check["fare_amount"] > 5)
    ).sum()
    # Слишком длинная дистанция (больше 150 миль — это уже выезд далеко за пределы штата)
    anomalies["Поездки на сверхдальние дистанции (> 150 миль)"] = (
        df_check["trip_distance"] > 150
    ).sum()

    # 3. ФИНАНСОВЫЕ АНОМАЛИИ
    # Отрицательные стоимости (иногда так кодируют возвраты, но для анализа это мусор)
    anomalies["Отрицательный тариф (fare_amount < 0)"] = (
        df_check["fare_amount"] < 0
    ).sum()
    # Бесплатный сыр: дистанция приличная, а тариф нулевой
    anomalies["Бесплатные поездки (тариф = 0 при дистанции > 1 мили)"] = (
        (df_check["fare_amount"] <= 0) & (df_check["trip_distance"] > 1)
    ).sum()
    # Подозрительно щедрые чаевые (чаевые превышают стоимость самой поездки)
    anomalies["Аномальные чаевые (tip_amount > total_amount * 0.5)"] = (
        df_check["tip_amount"] > (df_check["total_amount"] * 0.5)
    ).sum()

    # Проверка математики: total_amount должен быть равен сумме всех составляющих
    # Допускаем погрешность в 1 цент из-за округлений округления float
    expected_total = (
        df_check["fare_amount"]
        + df_check["extra"]
        + df_check["mta_tax"]
        + df_check["tip_amount"]
        + df_check["tolls_amount"]
        + df_check["improvement_surcharge"]
        + df_check.get("congestion_surcharge", 0)
        + df_check.get("airport_fee", 0)
    )

    anomalies["Ошибка калькуляции (total_amount не равен сумме сборов)"] = (
        (df_check["total_amount"] - expected_total).abs() > 0.02
    ).sum()

    # 4. ФИЗИЧЕСКИЕ АНОМАЛИИ
    # Ноль пассажиров (такси ехало пустое, но включило счетчик)
    anomalies["Поездки с 0 пассажиров"] = (
        df_check["passenger_count"] == 0
    ).sum()
    # Автобус вместо желтого седана (в обычное такси Нью-Йорка нельзя сажать больше 6 человек по закону)
    anomalies["Перегруз (пассажиров > 6)"] = (
        df_check["passenger_count"] > 6
    ).sum()

    # ВЫВОД ОТЧЕТА
    print(f"Всего строк на проверке: {total_rows:,}\n")
    print(f"{'Тип найденной аномалии':<60} | {'Кол-во строк':<12} | {'Процент':<8}")
    print("-" * 85)

    total_anomalous_flags = 0
    for name, count in anomalies.items():
        percentage = (count / total_rows) * 100
        print(f"{name:<60} | {count:<12,} | {percentage:.4f}%")
        total_anomalous_flags += count

    print("-" * 85)
    print(f"Аудит завершен.")


df2 = pd.read_parquet('my_clean_2.parquet')
audit_taxi_anomalies(df2)