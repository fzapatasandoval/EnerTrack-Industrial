"""Ejecuta consumo, imagen y búsqueda de reportes y devuelve un solo JSON.

No reentrena modelos ni llama a Gemini. Las métricas comparativas se leen de
los archivos ya calculados y no se vuelven a evaluar.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from defect_models import ACADEMIC_WARNING, DATASET_ORIGIN
from forecast_series import clean_series, hw_one_step, load_rows, project_root
from maintenance_search import MODEL_NAME, load_corpus, load_encoder, search
from predict_consumption import WINDOW, load_checkpoint, predict_lstm
from predict_defect import predict

UMBRAL_REVISION_PCT = 25.0
HISTORICAL_NOTE = "Evaluación histórica ya verificada. No es una nueva medición de esta consulta."


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Tipo no serializable: {type(value).__name__}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def deviation(actual: float, expected: float) -> dict:
    if expected <= 0:
        return {
            "consumo_esperado_kwh": expected,
            "desviacion_pct": None,
            "umbral_revision_pct": UMBRAL_REVISION_PCT,
            "revisar": None,
            "nota": "La desviación no se calcula porque el consumo esperado no es positivo.",
        }
    percent = (actual - expected) / expected * 100.0
    return {
        "consumo_esperado_kwh": expected,
        "desviacion_pct": percent,
        "umbral_revision_pct": UMBRAL_REVISION_PCT,
        "revisar": abs(percent) > UMBRAL_REVISION_PCT,
    }


def historical_forecasts(root: Path) -> dict:
    rows = read_csv_rows(root / "results" / "forecast_comparison.csv")
    config = json.loads((root / "results" / "forecast_config.json").read_text(encoding="utf-8"))
    return {
        "ok": True,
        "alcance": HISTORICAL_NOTE,
        "periodo": {
            "inicio": config["fecha_inicio_prueba"],
            "fin": config["fecha_fin_prueba"],
            "dias": config["prueba_dias"],
        },
        "archivo": "results/forecast_comparison.csv",
        "modelos": [
            {
                "modelo": row["modelo"],
                "mae": as_float(row["mae"]),
                "rmse": as_float(row["rmse"]),
                "inferencia_ms_por_pronostico": as_float(row["inferencia_ms_por_pronostico"]),
            }
            for row in rows
        ],
    }


def analyze_series_from_rows(rows: list[dict], columns: dict, root: Path, origen: str) -> dict:
    series = clean_series(rows, columns)
    values = [float(value) for value in series["values"]]
    config = json.loads((root / "results" / "forecast_config.json").read_text(encoding="utf-8"))
    parameters = config["holt_winters"]
    forecasts = hw_one_step(values, parameters["alpha"], parameters["beta"], parameters["gamma"])
    last_actual = values[-1]
    checkpoint = load_checkpoint(root / "models" / "lstm_consumo_best.pt")
    array = np.asarray(values, dtype=np.float64)
    lstm_last = predict_lstm(array, len(values) - 1, checkpoint)
    lstm_next = predict_lstm(array, len(values), checkpoint)
    return {
        "ok": True,
        "archivo": origen,
        "caracter": "simulado",
        "fuente": (
            "Serie recibida en la petición. Se trata como simulación académica, no como medición de una planta real."
            if origen == "peticion"
            else "Serie académica Historico_90_dias. No es una medición de una planta real."
        ),
        "fecha_ultima": series["fecha_fin"],
        "consumo_real_ultimo_kwh": last_actual,
        "n_dias": len(values),
        "parametros_holt_winters_congelados": {
            "alpha": parameters["alpha"],
            "beta": parameters["beta"],
            "gamma": parameters["gamma"],
            "estacionalidad_dias": parameters["estacionalidad_dias"],
            "nota": "Parámetros de la evaluación anterior. No se volvieron a elegir con este archivo.",
        },
        "holt_winters": {
            "pronostico_siguiente_dia_kwh": forecasts[-1],
            **deviation(last_actual, forecasts[-2]),
        },
        "lstm": {
            "pronostico_siguiente_dia_kwh": lstm_next,
            "ventana_dias": WINDOW,
            **deviation(last_actual, lstm_last),
        },
        "evaluacion_historica": historical_forecasts(root),
    }


def analyze_series(csv_path: Path, root: Path) -> dict:
    rows, columns = load_rows(csv_path)
    return analyze_series_from_rows(rows, columns, root, str(csv_path))


def historical_images(root: Path) -> dict:
    rows = read_csv_rows(root / "results" / "model_comparison.csv")
    return {
        "ok": True,
        "alcance": HISTORICAL_NOTE + " Corresponde a las 270 imágenes de prueba, no a la imagen enviada ahora.",
        "archivo": "results/model_comparison.csv",
        "modelos": [
            {
                "modelo": row["model"],
                "accuracy": as_float(row["test_accuracy"]),
                "macro_f1": as_float(row["test_macro_f1"]),
                "imagenes_prueba": int(row["test_images"]),
            }
            for row in rows
        ],
    }


def analyze_image(image_path: Path, root: Path, nombre: str | None = None) -> dict:
    if not image_path.is_file():
        raise FileNotFoundError("No existe la imagen indicada.")
    current = predict(image_path, "mobilenet_v3_small")
    return {
        "ok": True,
        "archivo": nombre or image_path.name,
        "modelo": current["modelo"],
        "clase_predicha": current["clase_predicha"],
        "confianza": current["confianza"],
        "top_3_clases": current["top_3_clases"],
        "origen_dataset": DATASET_ORIGIN,
        "advertencia_clases": ACADEMIC_WARNING,
        "evaluacion_historica": historical_images(root),
    }


def historical_search(root: Path) -> dict:
    rows = read_csv_rows(root / "results" / "maintenance_search_metrics.csv")
    models = []
    for row in rows:
        models.append(
            {
                "metodo": row["metodo"],
                "modelo": row["modelo"] or None,
                "consultas": int(row["consultas"]),
                "recall_at_3": as_float(row["recall_at_3"]),
                "mrr": as_float(row["mrr"]),
            }
        )
    return {
        "ok": True,
        "alcance": HISTORICAL_NOTE + " Compara MiniLM y TF-IDF en las 16 consultas fijadas de antemano.",
        "archivo": "results/maintenance_search_metrics.csv",
        "modelos": models,
    }


def analyze_text(query: str, root: Path) -> dict:
    corpus = load_corpus()
    model, device = load_encoder()
    found = search(query, "semantico", corpus["reportes"], model=model)
    found.pop("ranking_completo", None)
    documents = []
    for item in found["resultados"]:
        documents.append(
            {
                "id": item["id"],
                "texto": item["texto"],
                "puntaje": item["puntaje"],
                "tipo_puntaje": item["tipo_puntaje"],
                "fuente": item["fuente"],
                "caracter": item["caracter"],
                "cita": item["cita"],
            }
        )
    return {
        "ok": True,
        "consulta": query,
        "modelo": MODEL_NAME,
        "dispositivo": device,
        "caracter_corpus": corpus["caracter"],
        "fuente_corpus": corpus["fuente"],
        "documentos": documents,
        "evaluacion_historica": historical_search(root),
    }


def run_part(name: str, function) -> tuple[dict, str | None, float]:
    started = time.perf_counter()
    try:
        result = function()
    except Exception as error:
        elapsed = (time.perf_counter() - started) * 1000.0
        return (
            {"ok": False, "error": f"{type(error).__name__}: {error}"},
            f"{name}: {type(error).__name__}: {error}",
            elapsed,
        )
    elapsed = (time.perf_counter() - started) * 1000.0
    result["tiempo_ms"] = elapsed
    return result, None, elapsed


def analyze(
    csv_path: Path | None,
    image_path: Path,
    query: str,
    registros: list[dict] | None = None,
    nombre_imagen: str | None = None,
) -> dict:
    root = project_root()
    errors = []
    if registros is None:
        if csv_path is None:
            raise ValueError("Falta la serie de consumo.")
        series_call = lambda: analyze_series(csv_path, root)
    else:
        columns = {"fecha": "fecha", "consumo": "consumo_kwh", "archivo": "peticion"}
        series_call = lambda: analyze_series_from_rows(registros, columns, root, "peticion")
    series, series_error, series_ms = run_part("serie_temporal", series_call)
    image, image_error, image_ms = run_part(
        "imagenes",
        lambda: analyze_image(image_path, root, nombre_imagen),
    )
    text, text_error, text_ms = run_part("nlp", lambda: analyze_text(query, root))
    for message in (series_error, image_error, text_error):
        if message:
            errors.append(message)
    payload = {
        "ok": not errors,
        "serie_temporal": series,
        "imagenes": image,
        "nlp": text,
        "fuentes_y_limitaciones": {
            "consumos": "Sintéticos. Pertenecen a la simulación académica Historico_90_dias.",
            "reportes": "Sintéticos. No son órdenes reales de una empresa.",
            "imagenes": "NEU Surface Defect Database, pública y real, de Kechen Song y Yunhui Yan.",
            "relacion_entre_fuentes": "Las tres fuentes no describen un mismo evento industrial ni una misma falla.",
            "gemini": "Este script no llama a Gemini. La redacción posterior sigue en n8n.",
            "clases_de_imagen": "Las seis clases del modelo son defectos. No existe clase normal.",
        },
        "tiempos_ms": {
            "serie_temporal": series_ms,
            "imagenes": image_ms,
            "nlp": text_ms,
            "total": series_ms + image_ms + text_ms,
        },
        "errores": errors,
    }
    safe = json_safe(payload)
    encoded = json.dumps(safe, ensure_ascii=False)
    json.loads(encoded)
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description="Análisis integrado de EnerTrack")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--imagen", required=True, type=Path)
    parser.add_argument("--consulta", required=True)
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()
    csv_path = args.csv if args.csv.is_absolute() else Path.cwd() / args.csv
    image_path = args.imagen if args.imagen.is_absolute() else Path.cwd() / args.imagen
    payload = analyze(csv_path, image_path, args.consulta)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.salida:
        destination = args.salida if args.salida.is_absolute() else Path.cwd() / args.salida
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
