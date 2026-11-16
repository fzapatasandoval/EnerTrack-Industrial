"""Evalúa la búsqueda semántica y la búsqueda por palabras con las mismas consultas.

Recall@3 es la fracción de reportes relevantes que aparecen entre los tres primeros.
MRR usa el puesto del primer reporte relevante en el ranking completo; si no aparece, aporta 0.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from maintenance_search import MODEL_NAME, load_corpus, load_encoder, load_queries, search


def recall_at_3(relevant: list[str], ranking: list[str]) -> float:
    found = set(relevant).intersection(ranking[:3])
    return len(found) / len(relevant)


def reciprocal_rank(relevant: list[str], ranking: list[str]) -> float:
    relevant_ids = set(relevant)
    for position, identifier in enumerate(ranking, start=1):
        if identifier in relevant_ids:
            return 1.0 / position
    return 0.0


def main() -> int:
    corpus = load_corpus()
    queries = load_queries()
    reports = corpus["reportes"]
    model, device = load_encoder()
    rows = []
    for query in queries:
        for method in ("semantico", "palabras"):
            found = search(query["texto"], method, reports, model=model)
            ranking = found["ranking_completo"]
            top = found["resultados"]
            rows.append(
                {
                    "consulta_id": query["id"],
                    "consulta": query["texto"],
                    "metodo": method,
                    "relevantes": query["relevantes"],
                    "recall_at_3": recall_at_3(query["relevantes"], ranking),
                    "reciprocal_rank": reciprocal_rank(query["relevantes"], ranking),
                    "top3": [
                        {
                            "id": item["id"],
                            "puntaje": item["puntaje"],
                            "es_relevante": item["id"] in query["relevantes"],
                        }
                        for item in top
                    ],
                    "relevantes_ausentes_en_top3": [
                        identifier for identifier in query["relevantes"] if identifier not in ranking[:3]
                    ],
                }
            )
    summary = []
    for method in ("semantico", "palabras"):
        subset = [row for row in rows if row["metodo"] == method]
        summary.append(
            {
                "metodo": method,
                "modelo": MODEL_NAME if method == "semantico" else "",
                "consultas": len(subset),
                "recall_at_3": sum(row["recall_at_3"] for row in subset) / len(subset),
                "mrr": sum(row["reciprocal_rank"] for row in subset) / len(subset),
                "dispositivo": device if method == "semantico" else "cpu",
                "fine_tuning": False,
            }
        )
    results = project_results(corpus, device, summary, rows)
    write_outputs(results)
    for item in summary:
        print(
            f"{item['metodo']} recall_at_3={item['recall_at_3']:.4f} mrr={item['mrr']:.4f}",
            flush=True,
        )
    return 0


def project_results(corpus: dict, device: str, summary: list[dict], rows: list[dict]) -> dict:
    return {
        "corpus_id": corpus["corpus_id"],
        "caracter": corpus["caracter"],
        "fuente": corpus["fuente"],
        "reportes": len(corpus["reportes"]),
        "modelo": MODEL_NAME,
        "dispositivo": device,
        "fine_tuning": False,
        "definicion_recall_at_3": "reportes relevantes recuperados en el top 3, dividido por los relevantes de la consulta",
        "definicion_mrr": "promedio de 1/puesto del primer relevante en el ranking completo",
        "resumen": summary,
        "consultas": rows,
    }


def write_outputs(results: dict) -> None:
    root = Path(__file__).resolve().parents[1] / "results"
    root.mkdir(parents=True, exist_ok=True)
    (root / "maintenance_search_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (root / "maintenance_search_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metodo", "consultas", "recall_at_3", "mrr", "modelo", "dispositivo"])
        writer.writeheader()
        for item in results["resumen"]:
            writer.writerow(
                {
                    "metodo": item["metodo"],
                    "consultas": item["consultas"],
                    "recall_at_3": item["recall_at_3"],
                    "mrr": item["mrr"],
                    "modelo": item["modelo"],
                    "dispositivo": item["dispositivo"],
                }
            )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
