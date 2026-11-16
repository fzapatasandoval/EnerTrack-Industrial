"""Serie diaria de Historico_90_dias y Holt-Winters del nodo original.

El Holt-Winters es aditivo, con estacionalidad de 7 días y pronóstico a un paso.
La grilla de alpha, beta y gamma y la ventana de 21 días salen del nodo Python
de n8n. No se usa statsmodels, porque su inicialización no es la misma.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

VENTANA_EVALUACION = 21
ESTACION = 7
MIN_REGISTROS = 35
MAX_DIAS_INTERPOLADOS = 9
MAX_HISTORIA = 90
ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7)
BETAS = (0.0, 0.05, 0.1, 0.2)
GAMMAS = (0.05, 0.1, 0.2, 0.3, 0.5)
CSV_PATH = Path("data/consumo/historico_90_dias.csv")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def csv_path() -> Path:
    return project_root() / CSV_PATH


def missing_csv_message() -> str:
    return (
        "No está el CSV de consumo.\n"
        "Exporta desde Google Sheets la hoja Historico_90_dias, en formato CSV.\n"
        "La primera fila debe incluir las columnas fecha y consumo_kwh "
        "(observacion es opcional).\n"
        "Coloca el archivo en:\n"
        f"  {csv_path()}\n"
        "No completes la serie a mano: este proyecto no inventa consumos."
    )


def days_from_civil(year: int, month: int, day: int) -> int:
    year = year - (1 if month <= 2 else 0)
    era = (year if year >= 0 else year - 399) // 400
    year_of_era = year - era * 400
    day_of_year = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146097 + day_of_era - 719468


def valid_ymd(year: int, month: int, day: int) -> bool:
    if year < 2000 or year > 2100 or month < 1 or month > 12 or day < 1:
        return False
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return day <= days[month - 1]


def parse_fecha(value) -> tuple[int, str] | None:
    if value is None:
        return None
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    separator = "-" if "-" in text else ("/" if "/" in text else None)
    if separator is None:
        return None
    parts = text.split(separator)
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    first, second, third = (int(part) for part in parts)
    if len(parts[0]) == 4:
        year, month, day = first, second, third
    elif len(parts[2]) == 4:
        day, month, year = first, second, third
    else:
        return None
    if not valid_ymd(year, month, day):
        return None
    return days_from_civil(year, month, day), f"{year:04d}-{month:02d}-{day:02d}"


def parse_consumo(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().lower().replace(" ", "").replace("kwh", "")
        if text == "":
            return None
        if "," in text and "." in text:
            text = text.replace(".", "").replace(",", ".")
        elif "," in text:
            text = text.replace(",", ".")
        try:
            number = float(text)
        except ValueError:
            return None
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        return None
    return number


def _column(fieldnames: list[str], exact: set[str], partial: tuple[str, ...]) -> str | None:
    normalized = {name: name.strip().lower().replace(" ", "_") for name in fieldnames}
    for name, key in normalized.items():
        if key in exact:
            return name
    for name, key in normalized.items():
        if any(piece in key for piece in partial):
            return name
    return None


def load_rows(path: Path | None = None) -> tuple[list[dict], dict]:
    source = path or csv_path()
    if not source.is_file():
        raise FileNotFoundError(missing_csv_message())
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError("El CSV no tiene encabezados.")
        fecha_key = _column(list(reader.fieldnames), {"fecha", "date", "dia", "día"}, ("fecha", "date"))
        consumo_key = _column(
            list(reader.fieldnames),
            {"consumo_kwh", "consumo", "kwh", "energia_kwh"},
            ("consumo", "kwh"),
        )
        if fecha_key is None or consumo_key is None:
            raise RuntimeError("El CSV debe incluir columnas de fecha y consumo_kwh.")
        rows = list(reader)
    return rows, {"fecha": fecha_key, "consumo": consumo_key, "archivo": str(source)}


def clean_series(rows: list[dict], columns: dict) -> dict:
    by_day: dict[int, float] = {}
    iso_by_day: dict[int, str] = {}
    discarded = 0
    duplicates = 0
    duplicate_dates = []
    for row in rows:
        parsed_date = parse_fecha(row.get(columns["fecha"]))
        consumption = parse_consumo(row.get(columns["consumo"]))
        if parsed_date is None or consumption is None:
            discarded += 1
            continue
        day, iso = parsed_date
        if day in by_day:
            duplicates += 1
            duplicate_dates.append(iso)
            continue
        by_day[day] = consumption
        iso_by_day[day] = iso
    if len(by_day) < MIN_REGISTROS:
        raise RuntimeError(f"Datos insuficientes: {len(by_day)} registros válidos.")
    days = sorted(by_day)
    kept = [day for day in days if day > days[-1] - MAX_HISTORIA]
    dropped_by_window = len(days) - len(kept)
    values = []
    dates = []
    interpolated = []
    gaps = []
    for index, day in enumerate(kept):
        if index > 0:
            gap = day - kept[index - 1]
            if gap > 1:
                gaps.append({"desde": iso_by_day[kept[index - 1]], "hasta": iso_by_day[day], "dias_faltantes": gap - 1})
            previous = by_day[kept[index - 1]]
            current = by_day[day]
            for step in range(1, gap):
                filled = previous + (current - previous) * step / gap
                values.append(filled)
                # La fecha civil se reconstruye solo para documentar el hueco interpolado.
                dates.append(f"interpolado-{iso_by_day[kept[index - 1]]}-mas-{step}")
                interpolated.append(dates[-1])
        values.append(by_day[day])
        dates.append(iso_by_day[day])
    if len(interpolated) > MAX_DIAS_INTERPOLADOS:
        raise RuntimeError(
            f"Demasiados días faltantes: {len(interpolated)} interpolados, máximo {MAX_DIAS_INTERPOLADOS}."
        )
    return {
        "values": values,
        "dates": dates,
        "n_filas_leidas": len(rows),
        "n_descartados": discarded,
        "n_duplicados": duplicates,
        "fechas_duplicadas": duplicate_dates,
        "n_validos": len(by_day),
        "n_fuera_de_90_dias": dropped_by_window,
        "dias_interpolados": len(interpolated),
        "huecos": gaps,
        "fecha_inicio": dates[0],
        "fecha_fin": dates[-1],
        "columnas": columns,
    }


def hw_one_step(values: list[float], alpha: float, beta: float, gamma: float, season_length: int = ESTACION) -> list[float | None]:
    count = len(values)
    level = sum(values[:season_length]) / season_length
    trend = (sum(values[season_length: 2 * season_length]) - sum(values[:season_length])) / (season_length * season_length)
    season = [values[index] - level for index in range(season_length)]
    forecasts: list[float | None] = [None] * (count + 1)
    for time in range(season_length, count + 1):
        seasonal = season[time % season_length]
        forecasts[time] = level + trend + seasonal
        if time == count:
            break
        observed = values[time]
        new_level = alpha * (observed - seasonal) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        season[time % season_length] = gamma * (observed - new_level) + (1 - gamma) * seasonal
        level = new_level
    return forecasts


def select_hw_parameters(values: list[float], test_start: int) -> dict:
    best = None
    for alpha in ALPHAS:
        for beta in BETAS:
            for gamma in GAMMAS:
                forecasts = hw_one_step(values, alpha, beta, gamma)
                sse = 0.0
                for time in range(2 * ESTACION, test_start):
                    sse += (values[time] - forecasts[time]) ** 2
                if best is None or sse < best["sse"]:
                    best = {"sse": sse, "alpha": alpha, "beta": beta, "gamma": gamma}
    return best


def regression_errors(actual: list[float], predicted: list[float]) -> dict:
    residuals = [observed - forecast for observed, forecast in zip(actual, predicted)]
    mae = sum(abs(error) for error in residuals) / len(residuals)
    rmse = (sum(error * error for error in residuals) / len(residuals)) ** 0.5
    return {"mae": mae, "rmse": rmse}


def test_start_index(count: int) -> int:
    start = count - VENTANA_EVALUACION
    if start < 2 * ESTACION + 7:
        raise RuntimeError("La serie es demasiado corta para separar los últimos 21 días.")
    return start


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
