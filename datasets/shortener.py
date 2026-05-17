from pathlib import Path
import pyarrow.parquet as pq


ROWS_LIMIT = 100000
FILE_NAME = "my_clean_3_with_weather.parquet"

script_dir = Path(__file__).resolve().parent
input_path = script_dir / FILE_NAME
output_path = script_dir / f"short_{FILE_NAME}"

if not input_path.exists():
    raise FileNotFoundError(f"Файл не найден: {input_path}")

table = pq.read_table(input_path)
sample_table = table.slice(0, ROWS_LIMIT)
pq.write_table(sample_table, output_path)

print(f"Сохранено: {output_path.name}")
print(f"Строк: {sample_table.num_rows}")