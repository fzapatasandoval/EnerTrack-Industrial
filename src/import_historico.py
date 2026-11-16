"""Valida el CSV exportado de la hoja Historico_90_dias.

Si el archivo no está, indica la ruta y las columnas esperadas y termina.
No inventa consumos ni rellena la serie con valores aproximados.
"""

from __future__ import annotations

import sys
from pathlib import Path

from forecast_series import clean_series, csv_path, load_rows, missing_csv_message, save_json


def main() -> int:
    if not csv_path().is_file():
        print(missing_csv_message(), file=sys.stderr)
        return 1
    rows, columns = load_rows()
    series = clean_series(rows, columns)
    report = {key: value for key, value in series.items() if key not in {"values", "dates"}}
    report["n_serie"] = len(series["values"])
    report["fecha_inicio"] = series["fecha_inicio"]
    report["fecha_fin"] = series["fecha_fin"]
    destination = Path(__file__).resolve().parents[1] / "results" / "forecast_data_quality.json"
    save_json(destination, report)
    print(
        f"filas={report['n_filas_leidas']} validos={report['n_validos']} "
        f"descartados={report['n_descartados']} duplicados={report['n_duplicados']} "
        f"interpolados={report['dias_interpolados']} serie={report['n_serie']}"
    )
    print(f"desde={report['fecha_inicio']} hasta={report['fecha_fin']}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
