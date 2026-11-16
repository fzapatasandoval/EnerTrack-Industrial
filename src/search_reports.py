"""Recupera los tres reportes más cercanos a una consulta.

Ejemplo:
    python src/search_reports.py "vibración alta en la sopladora" --metodo semantico
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from maintenance_search import load_corpus, load_encoder, search


def main() -> int:
    parser = argparse.ArgumentParser(description="Búsqueda de reportes de mantenimiento simulados")
    parser.add_argument("consulta")
    parser.add_argument("--metodo", choices=("semantico", "palabras"), default="semantico")
    args = parser.parse_args()
    reports = load_corpus()["reportes"]
    model = None
    if args.metodo == "semantico":
        model, _device = load_encoder()
    found = search(args.consulta, args.metodo, reports, model=model)
    found.pop("ranking_completo", None)
    print(json.dumps(found, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
