from pathlib import Path
import ast
import json
import re

ROOT = Path.cwd()

TEXT_EXTS = {
    ".py", ".ipynb", ".md", ".txt", ".yaml", ".yml", ".json"
}

DATA_EXTS = {
    ".csv", ".parquet", ".xlsx", ".xls", ".json", ".geojson",
    ".shp", ".pkl", ".joblib", ".png", ".jpg", ".jpeg", ".html"
}

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", ".idea"
}

PATH_RE = re.compile(
    r"""["']([^"']+\.(?:csv|parquet|xlsx|xls|json|geojson|shp|pkl|joblib|png|jpg|jpeg|html))["']"""
)


def skipped(path):
    return any(part in SKIP_DIRS for part in path.parts)


def is_probable_path(s):
    if s.startswith(("http://", "https://", "s3://")):
        return False
    if "{" in s or "}" in s:
        return False
    return Path(s).suffix.lower() in DATA_EXTS


def exists_relative_to_anywhere(ref, source_file):
    p = Path(ref)

    if p.is_absolute():
        return p.exists()

    variants = [
        ROOT / p,
        source_file.parent / p,
    ]

    return any(v.exists() for v in variants)


def extract_from_python(code):
    refs = set()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        refs.update(PATH_RE.findall(code))
        return refs

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if is_probable_path(node.value):
                refs.add(node.value)

    refs.update(PATH_RE.findall(code))
    return refs


def extract_from_ipynb(path):
    refs = set()

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return refs

    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)

        if cell.get("cell_type") == "code":
            refs.update(extract_from_python(source))
        else:
            refs.update(PATH_RE.findall(source))

    return refs


def extract_from_text(path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1251", errors="ignore")

    if path.suffix == ".py":
        return extract_from_python(text)

    return set(PATH_RE.findall(text))


broken = []

for path in ROOT.rglob("*"):
    if skipped(path):
        continue
    if not path.is_file():
        continue
    if path.suffix.lower() not in TEXT_EXTS:
        continue

    if path.suffix.lower() == ".ipynb":
        refs = extract_from_ipynb(path)
    else:
        refs = extract_from_text(path)

    for ref in sorted(refs):
        if not exists_relative_to_anywhere(ref, path):
            broken.append((path.relative_to(ROOT), ref))

if not broken:
    print("Битых ссылок на файлы не найдено")
else:
    print("Файлы, где есть ссылки на несуществующие пути:\n")
    for source, ref in broken:
        print(f"{source} -> {ref}")