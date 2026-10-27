import csv
import os
import tempfile


ROI_HEADER = ["Arquivo", "Classe", "Canto Superior Esquerdo Fígado", "Canto Superior Esquerdo Rim", "HI"]
TEXTURE_HEADER = ["Arquivo", "Classe", "Coarseness", "Contrast", "Periodicity", "Roughness"] + [
    f"glcm_distance_{distance}_{feature}"
    for distance in (1, 2, 4, 8)
    for feature in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "entropy")
]


def ensure_directory_exists(directory):
    if directory:
        os.makedirs(directory, exist_ok=True)


def write_roi_data_to_csv(file_path, data):
    _upsert_csv_row(file_path, data, ROI_HEADER)


def write_texture_features_to_csv(file_path, features):
    _upsert_csv_row(file_path, features, TEXTURE_HEADER)


def _upsert_csv_row(file_path, data, default_header):
    if len(data) != len(default_header):
        raise ValueError(f"Esperadas {len(default_header)} colunas; recebidas {len(data)}.")

    directory = os.path.dirname(os.fspath(file_path)) or "."
    ensure_directory_exists(directory)
    rows = {}
    header = default_header
    if os.path.isfile(file_path):
        with open(file_path, newline="", encoding="utf-8-sig") as source:
            reader = csv.reader(source)
            header = next(reader, default_header)
            for row in reader:
                if row:
                    rows[row[0]] = row

    if len(header) != len(data):
        raise ValueError("O CSV existente tem um número de colunas diferente.")
    rows[data[0]] = data

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=directory, delete=False) as target:
            temp_path = target.name
            writer = csv.writer(target)
            writer.writerow(header)
            writer.writerows(rows.values())
        os.replace(temp_path, file_path)
    except Exception:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise
