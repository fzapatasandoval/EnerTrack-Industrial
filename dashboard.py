"""Dashboard académico de EnerTrack Industrial.

Lee resultados ya calculados y, si hay una cuenta de servicio, la hoja
Registro_alertas. No ejecuta los modelos ni inventa diagnósticos de Gemini.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SPREADSHEET_ID = "1V2im7Ar9qfM5oQL_WF3URSqJKj4c2DdptzpYy67FDh4"
SHEET_ALERTS = "Registro_alertas"
SHEET_DECISIONS = "Registro_decisiones"
ALERT_COLUMNS = (
    "fecha_ejecucion",
    "fecha_dato",
    "consumo_real_kwh",
    "consumo_esperado_kwh",
    "desviacion_pct",
    "mae_holt_winters",
    "mae_ingenuo",
    "revisar",
    "datos_simulados",
    "prioridad",
    "diagnostico",
    "recomendacion",
    "justificacion",
    "estado",
)
DECISION_COLUMNS = (
    "fecha_decision",
    "fecha_dato",
    "decision_supervisor",
    "accion_autorizada",
    "prioridad",
)
ACCOUNT_FIELDS = ("type", "private_key", "client_email", "token_uri")
STATUS_TEXT = {
    "ok": "conectada",
    "vacia": "conectada, sin filas",
    "sin_credenciales": "Google Sheets no está conectado",
    "credenciales_invalidas": "la credencial no es válida",
    "sin_acceso": "la cuenta no puede leer el documento",
    "hoja_ausente": "la hoja no existe en el documento",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_local_history() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "data" / "consumo" / "historico_90_dias.csv")
    frame["fecha"] = pd.to_datetime(frame["fecha"])
    frame["consumo_kwh"] = pd.to_numeric(frame["consumo_kwh"])
    expected = pd.read_csv(ROOT / "results" / "forecast_predictions.csv")
    expected["fecha"] = pd.to_datetime(expected["fecha"])
    expected = expected.rename(columns={"holt_winters": "esperado_holt_winters_kwh"})
    merged = frame.merge(expected[["fecha", "esperado_holt_winters_kwh"]], on="fecha", how="left")
    return merged.sort_values("fecha")


def load_sample() -> dict:
    return read_json(ROOT / "results" / "integrated_sample.json")


def metric_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast = pd.read_csv(ROOT / "results" / "forecast_comparison.csv")
    images = pd.read_csv(ROOT / "results" / "model_comparison.csv")
    nlp = pd.read_csv(ROOT / "results" / "maintenance_search_metrics.csv")
    return forecast, images, nlp


def secret_value(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        return None
    return None


def spreadsheet_id() -> str:
    value = secret_value("spreadsheet_id") or os.environ.get("ENERTRACK_SPREADSHEET_ID")
    return str(value or SPREADSHEET_ID)


def service_account_info() -> tuple[dict | None, str | None]:
    nested = None
    try:
        if "gcp_service_account" in st.secrets:
            nested = dict(st.secrets["gcp_service_account"])
    except Exception:
        nested = None
    info = nested
    if info is None:
        path = os.environ.get("ENERTRACK_SHEETS_CREDENTIALS", "").strip()
        if not path:
            return None, "sin_credenciales"
        file = Path(path)
        if not file.is_file():
            return None, "credenciales_invalidas"
        try:
            info = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, "credenciales_invalidas"
    if not isinstance(info, dict) or info.get("type") != "service_account":
        return None, "credenciales_invalidas"
    if any(not str(info.get(field, "")).strip() for field in ACCOUNT_FIELDS):
        return None, "credenciales_invalidas"
    return info, None


def read_workflow_sheets() -> dict[str, tuple[pd.DataFrame | None, str]]:
    info, problem = service_account_info()
    names = (SHEET_ALERTS, SHEET_DECISIONS)
    if problem:
        return {name: (None, problem) for name in names}
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        credentials = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        book = gspread.authorize(credentials).open_by_key(spreadsheet_id())
    except Exception as error:
        status = sheet_error(error)
        return {name: (None, status) for name in names}
    return {name: read_worksheet(book, name) for name in names}


def sheet_error(error: Exception) -> str:
    status = getattr(getattr(error, "response", None), "status_code", None)
    if type(error).__name__ == "SpreadsheetNotFound" or status in (401, 403, 404):
        return "sin_acceso"
    return "credenciales_invalidas"


def read_worksheet(book, name: str) -> tuple[pd.DataFrame | None, str]:
    try:
        records = book.worksheet(name).get_all_records()
    except Exception as error:
        if type(error).__name__ == "WorksheetNotFound":
            return None, "hoja_ausente"
        return None, sheet_error(error)
    if not records:
        return pd.DataFrame(), "vacia"
    return pd.DataFrame(records), "ok"


def find_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def image_path(sample: dict) -> Path | None:
    raw = str(sample.get("imagenes", {}).get("archivo", ""))
    direct = Path(raw)
    if direct.is_file():
        return direct
    marker = "neu_surface_defects/"
    normalized = raw.replace("\\", "/")
    if marker in normalized:
        candidate = ROOT / "data" / "neu_surface_defects" / normalized.split(marker, 1)[1]
        if candidate.is_file():
            return candidate
    return None


def selection_bounds(selected, minimum, maximum):
    values = []
    pending = [selected]
    while pending:
        item = pending.pop()
        if isinstance(item, tuple):
            pending.extend(item)
        elif hasattr(item, "year"):
            values.append(item)
    if len(values) >= 2:
        return min(values), max(values)
    if len(values) == 1:
        return values[0], values[0]
    return minimum, maximum


def filter_by_date(frame: pd.DataFrame, column: str, start, end) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    dates = pd.to_datetime(frame[column], errors="coerce")
    mask = dates.dt.date.between(start, end)
    return frame.loc[mask].copy()


def numeric(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "").replace("kWh", "").replace(" ", "")
        if value.count(",") == 1 and "." not in value:
            value = value.replace(",", ".")
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def format_kwh(value) -> str:
    number = numeric(value)
    return "Sin dato" if number is None else f"{number:.0f} kWh"


def format_percent(value) -> str:
    number = numeric(value)
    return "Sin dato" if number is None else f"{number:.2f} %"


def text_value(row: pd.Series, column: str | None) -> str:
    if column is None or column not in row.index:
        return "Sin dato"
    value = row[column]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Sin dato"
    text = str(value).strip()
    return text or "Sin dato"


def ordered_columns(frame: pd.DataFrame, preferred: tuple[str, ...]) -> pd.DataFrame:
    columns = [column for name in preferred if (column := find_column(frame, name))]
    remaining = [column for column in frame.columns if column not in columns]
    return frame[columns + remaining]


def latest_row(frame: pd.DataFrame | None, *date_names: str) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    column = find_column(frame, *date_names)
    if column is None:
        return frame.iloc[-1]
    ordered = frame.copy()
    ordered["_orden"] = pd.to_datetime(ordered[column], errors="coerce")
    ordered["_pos"] = range(len(ordered))
    return ordered.sort_values(["_orden", "_pos"], na_position="first").iloc[-1]


def extend_bounds(frame: pd.DataFrame | None, minimum, maximum, *date_names: str):
    if frame is None or frame.empty:
        return minimum, maximum
    column = find_column(frame, *date_names)
    if column is None:
        return minimum, maximum
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return minimum, maximum
    return min(minimum, dates.min().date()), max(maximum, dates.max().date())


def show_sheet_table(frame, status, sheet_name, preferred, date_names, start, end, caption) -> None:
    if status in {"sin_credenciales", "credenciales_invalidas", "sin_acceso", "hoja_ausente"}:
        st.info(f"{sheet_name}: {STATUS_TEXT[status]}.")
        return
    if frame is None or frame.empty:
        st.info(f"{sheet_name} está conectada y no tiene filas.")
        return
    column = find_column(frame, *date_names)
    visible = filter_by_date(frame, column, start, end) if column else frame
    if visible.empty:
        st.caption(f"No hay filas de {sheet_name} entre {start} y {end}.")
        st.caption(caption)
        return
    display = ordered_columns(visible, preferred).copy()
    for numeric_name in ("consumo_real_kwh", "consumo_esperado_kwh", "desviacion_pct", "mae_holt_winters", "mae_ingenuo"):
        numeric_column = find_column(display, numeric_name)
        if numeric_column:
            display[numeric_column] = pd.to_numeric(display[numeric_column], errors="coerce")
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(caption)


def main() -> None:
    st.set_page_config(page_title="EnerTrack Industrial", layout="wide")
    st.title("EnerTrack Industrial")
    st.caption(
        "Prototipo académico. Los consumos y los reportes son sintéticos. "
        "Las imágenes NEU son públicas y reales. No describen una misma falla."
    )

    history = load_local_history()
    sample = load_sample()
    forecast_metrics, image_metrics, nlp_metrics = metric_frames()
    sheets = read_workflow_sheets()
    alerts, alerts_status = sheets[SHEET_ALERTS]
    decisions, decisions_status = sheets[SHEET_DECISIONS]
    connected = alerts_status not in {"sin_credenciales", "credenciales_invalidas", "sin_acceso"}
    series = sample["serie_temporal"]
    example_date = series["fecha_ultima"]

    minimum = history["fecha"].min().date()
    maximum = history["fecha"].max().date()
    if connected:
        minimum, maximum = extend_bounds(alerts, minimum, maximum, "fecha_ejecucion", "fecha_dato")
        minimum, maximum = extend_bounds(decisions, minimum, maximum, "fecha_decision", "fecha_dato")
    selected = st.sidebar.date_input(
        "Fechas a explorar",
        value=(minimum, maximum),
        min_value=minimum,
        max_value=maximum,
    )
    start, end = selection_bounds(selected, minimum, maximum)
    visible_history = filter_by_date(history, "fecha", start, end)
    current_alert = latest_row(alerts, "fecha_ejecucion", "fecha_dato") if alerts_status == "ok" else None

    if alerts_status == "ok" and decisions_status == "ok":
        connection_text = "Registro_alertas y Registro_decisiones conectadas."
    elif alerts_status in {"sin_credenciales", "credenciales_invalidas", "sin_acceso"}:
        connection_text = f"{STATUS_TEXT[alerts_status]}. Los ejemplos locales no son información en tiempo real."
    else:
        connection_text = (
            f"Registro_alertas: {STATUS_TEXT.get(alerts_status, alerts_status)}. "
            f"Registro_decisiones: {STATUS_TEXT.get(decisions_status, decisions_status)}."
        )
    st.sidebar.caption(connection_text)

    st.subheader("Último análisis")
    if current_alert is not None:
        executed = text_value(current_alert, find_column(alerts, "fecha_ejecucion"))
        observed = text_value(current_alert, find_column(alerts, "fecha_dato"))
        first, second, third, fourth, fifth = st.columns(5)
        first.metric("Consumo real", format_kwh(current_alert.get(find_column(alerts, "consumo_real_kwh") or "")))
        second.metric("Consumo esperado", format_kwh(current_alert.get(find_column(alerts, "consumo_esperado_kwh") or "")))
        third.metric("Desviación", format_percent(current_alert.get(find_column(alerts, "desviacion_pct") or "")))
        fourth.metric("Prioridad", text_value(current_alert, find_column(alerts, "prioridad")))
        fifth.metric("Revisión", text_value(current_alert, find_column(alerts, "revisar")))
        st.caption(
            f"Una sola fila de Registro_alertas: ejecución {executed}, dato {observed}. "
            "No se combina con ejecuciones anteriores."
        )
        st.write("**Diagnóstico:**", text_value(current_alert, find_column(alerts, "diagnostico")))
        st.write("**Recomendación:**", text_value(current_alert, find_column(alerts, "recomendacion")))
        st.write("**Justificación:**", text_value(current_alert, find_column(alerts, "justificacion")))
        st.caption("Texto guardado por el workflow. Este dashboard no llama a Gemini.")
    elif not connected:
        holt = series["holt_winters"]
        first, second, third, fourth, fifth = st.columns(5)
        first.metric("Consumo real", format_kwh(series["consumo_real_ultimo_kwh"]))
        second.metric("Consumo esperado", format_kwh(holt["consumo_esperado_kwh"]))
        third.metric("Desviación", format_percent(holt["desviacion_pct"]))
        fourth.metric("Prioridad", "Sin dato")
        fifth.metric("Revisión", "Sí" if holt["revisar"] else "No")
        st.caption(
            f"Ejemplo histórico local del {example_date}, guardado en results/integrated_sample.json. "
            "Google Sheets no está conectado. No es el último análisis del workflow ni información en tiempo real."
        )
        st.info("El diagnóstico y la recomendación de Gemini están en Registro_alertas. Esta muestra local no los contiene.")
    elif alerts_status == "vacia":
        st.info("Registro_alertas está conectada y no tiene filas. No hay un último análisis que mostrar.")
    else:
        st.info(f"No se pudo leer Registro_alertas: {STATUS_TEXT.get(alerts_status, alerts_status)}.")

    st.subheader("Consumo diario")
    chart = visible_history.rename(
        columns={"consumo_kwh": "Real (kWh)", "esperado_holt_winters_kwh": "Esperado Holt-Winters (kWh)"}
    )
    st.line_chart(chart.set_index("fecha")[["Real (kWh)", "Esperado Holt-Winters (kWh)"]], height=320)
    st.caption(
        "La serie completa sale de Historico_90_dias. "
        "El valor esperado solo existe en los 21 días de la evaluación ya hecha, del 2026-10-19 al 2026-11-08."
    )

    st.subheader("Métricas históricas de evaluación")
    st.caption(
        "Estas cifras salen de las evaluaciones ya guardadas. "
        "No son el último análisis ejecutado por el workflow."
    )
    left, middle, right = st.columns(3)
    with left:
        st.markdown("**Serie temporal**")
        st.dataframe(
            forecast_metrics.rename(columns={"inferencia_ms_por_pronostico": "ms por pronóstico"}),
            hide_index=True,
            width="stretch",
        )
        st.caption("MAE y RMSE en kWh. Evaluación de 21 días, no se recalcula aquí.")
    with middle:
        st.markdown("**Imágenes**")
        st.dataframe(
            image_metrics[["model", "test_accuracy", "test_macro_f1", "test_images"]].rename(
                columns={"model": "modelo", "test_accuracy": "accuracy", "test_macro_f1": "macro F1", "test_images": "prueba"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("270 imágenes de prueba NEU. Todas las clases son defectos.")
    with right:
        st.markdown("**Búsqueda de reportes**")
        st.dataframe(
            nlp_metrics[["metodo", "recall_at_3", "mrr", "consultas"]],
            hide_index=True,
            width="stretch",
        )
        st.caption("16 consultas fijadas antes de ejecutar los modelos.")

    st.subheader("Última clasificación visual")
    images = sample["imagenes"]
    photo, text = st.columns([1, 2])
    path = image_path(sample)
    with photo:
        if path is not None:
            st.image(str(path), caption=path.name)
        else:
            st.info("La fotografía NEU de la muestra local no está en este equipo.")
    with text:
        st.metric("Clase predicha", images["clase_predicha"])
        st.metric("Confianza", f"{images['confianza']:.1%}")
        st.write("Modelo:", images["modelo"])
        st.write("Tres clases principales:")
        st.dataframe(pd.DataFrame(images["top_3_clases"]), hide_index=True, width="stretch")
        st.caption(images["advertencia_clases"])
        st.caption(
            f"Ejemplo histórico local asociado al dato del {example_date}. "
            "No es el último análisis del workflow ni información en tiempo real."
        )

    st.subheader("Reportes recuperados")
    st.caption(
        f"Consulta de la muestra local: {sample['nlp']['consulta']}. "
        "Los textos son simulados y sus citas son SIM-xxx. "
        f"Es el ejemplo histórico local del {example_date}, no una búsqueda en tiempo real."
    )
    for document in sample["nlp"]["documentos"]:
        st.markdown(f"**{document['cita']}** · similitud {document['puntaje']:.3f}")
        st.write(document["texto"])

    st.subheader("Ejecuciones guardadas")
    show_sheet_table(
        alerts,
        alerts_status,
        SHEET_ALERTS,
        ALERT_COLUMNS,
        ("fecha_dato", "fecha_ejecucion"),
        start,
        end,
        "Cada fila es una ejecución distinta. El bloque Último análisis usa solo la más reciente.",
    )

    st.subheader("Decisiones humanas")
    show_sheet_table(
        decisions,
        decisions_status,
        SHEET_DECISIONS,
        DECISION_COLUMNS,
        ("fecha_decision", "fecha_dato"),
        start,
        end,
        "Fecha y resultado registrados por el supervisor. No se inventan decisiones.",
    )


if __name__ == "__main__":
    main()
