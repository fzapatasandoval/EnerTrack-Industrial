"""Búsqueda de reportes de mantenimiento simulados.

La búsqueda semántica usa un Sentence Transformer multilingüe ya preentrenado,
sin fine-tuning. La alternativa cuenta palabras mediante TF-IDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
TOKEN_PATTERN = r"(?u)\b[\wáéíóúüñÁÉÍÓÚÜÑ0-9]+(?:[-./][\wáéíóúüñÁÉÍÓÚÜÑ0-9]+)*\b"
SPANISH_STOPWORDS = [
    "a", "al", "algo", "ante", "con", "de", "del", "el", "ella", "en", "es",
    "esa", "ese", "eso", "esta", "este", "ha", "hay", "la", "las", "le", "lo",
    "los", "me", "mi", "muy", "no", "o", "para", "por", "que", "se", "si",
    "sin", "su", "sus", "un", "una", "y", "ya", "como", "mas", "más", "pero",
    "fue", "ser", "son", "del", "qué", "cuál", "necesito", "busco",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def corpus_path() -> Path:
    return project_root() / "data" / "maintenance_reports" / "reportes.json"


def queries_path() -> Path:
    return project_root() / "data" / "maintenance_reports" / "consultas.json"


def load_corpus() -> dict:
    payload = json.loads(corpus_path().read_text(encoding="utf-8"))
    reports = payload["reportes"]
    if not 40 <= len(reports) <= 60:
        raise RuntimeError(f"El corpus debe tener entre 40 y 60 reportes y tiene {len(reports)}.")
    if payload.get("caracter") != "simulado":
        raise RuntimeError("El corpus no está marcado como simulado.")
    identifiers = [report["id"] for report in reports]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Hay identificadores de reporte repetidos.")
    for report in reports:
        if report.get("caracter", payload["caracter"]) != "simulado":
            raise RuntimeError(f"{report['id']} no está marcado como simulado.")
        report["caracter"] = "simulado"
        report["fuente"] = payload["fuente"]
        if report["cita"] != report["id"]:
            raise RuntimeError(f"La cita de {report['id']} no coincide con su identificador.")
    return payload


def load_queries() -> list[dict]:
    payload = json.loads(queries_path().read_text(encoding="utf-8"))
    reports = {report["id"] for report in load_corpus()["reportes"]}
    queries = payload["consultas"]
    if len(queries) < 12:
        raise RuntimeError("Hacen falta al menos 12 consultas de evaluación.")
    for query in queries:
        missing = [item for item in query["relevantes"] if item not in reports]
        if not query["relevantes"] or missing:
            raise RuntimeError(f"Relevantes inválidos en {query['id']}: {missing}")
    return queries


def select_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(device: str | None = None):
    from sentence_transformers import SentenceTransformer

    chosen = device or select_device()
    try:
        model = SentenceTransformer(MODEL_NAME, device=chosen)
        model.encode(["prueba de carga"], show_progress_bar=False)
    except Exception:
        if chosen == "cpu":
            raise
        chosen = "cpu"
        model = SentenceTransformer(MODEL_NAME, device=chosen)
    return model, chosen


def semantic_scores(model, query: str, reports: list[dict]) -> np.ndarray:
    texts = [report["texto"] for report in reports]
    document_embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(document_embeddings @ query_embedding[0], dtype=np.float64)


def keyword_scores(query: str, reports: list[dict]) -> np.ndarray:
    texts = [report["texto"] for report in reports]
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN, stop_words=SPANISH_STOPWORDS)
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    return cosine_similarity(query_vector, matrix).ravel()


def rank_reports(reports: list[dict], scores: np.ndarray) -> list[dict]:
    order = sorted(range(len(reports)), key=lambda index: (-float(scores[index]), reports[index]["id"]))
    ranked = []
    for index in order:
        report = reports[index]
        ranked.append(
            {
                "id": report["id"],
                "texto": report["texto"],
                "puntaje": float(scores[index]),
                "fuente": report["fuente"],
                "caracter": report["caracter"],
                "cita": report["cita"],
            }
        )
    return ranked


def search(query: str, method: str, reports: list[dict], model=None) -> dict:
    if method == "semantico":
        if model is None:
            model, _device = load_encoder()
        scores = semantic_scores(model, query, reports)
        score_type = "similitud_coseno"
        model_name = MODEL_NAME
    elif method == "palabras":
        scores = keyword_scores(query, reports)
        score_type = "coseno_tfidf"
        model_name = None
    else:
        raise ValueError(method)
    ranked = rank_reports(reports, scores)
    for item in ranked:
        item["tipo_puntaje"] = score_type
    return {
        "consulta": query,
        "metodo": method,
        "modelo": model_name,
        "caracter_corpus": "simulado",
        "resultados": ranked[:TOP_K],
        "ranking_completo": [item["id"] for item in ranked],
    }
