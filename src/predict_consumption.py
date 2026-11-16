"""Pronóstico a un día del consumo, con salida JSON.

Por defecto pronostica el día siguiente al último registro, usando solo la
historia disponible. Con --fecha se hace un pronóstico de un paso para una
fecha ya presente en la serie, sin usar el consumo de ese día.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from forecast_series import clean_series, hw_one_step, load_rows, project_root
from train_lstm_forecast import WINDOW, SmallLSTM


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def predict_lstm(values: np.ndarray, target_index: int, checkpoint: dict) -> float:
    window = values[target_index - WINDOW:target_index]
    scaled = ((window - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    model = SmallLSTM()
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(scaled[None, :, None]).to(device)
        predicted = float(model(tensor).detach().cpu())
    return predicted * checkpoint["std"] + checkpoint["mean"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pronóstico diario de consumo")
    parser.add_argument("--modelo", choices=("lstm", "holt_winters", "ingenuo"))
    parser.add_argument("--fecha", help="Fecha YYYY-MM-DD dentro de la serie. Si se omite, se pronostica el día siguiente.")
    args = parser.parse_args()
    root = project_root()
    config = json.loads((root / "results" / "forecast_config.json").read_text(encoding="utf-8"))
    model_name = args.modelo or config["modelo_recomendado"]
    rows, columns = load_rows()
    series = clean_series(rows, columns)
    values = np.asarray(series["values"], dtype=np.float64)
    dates = series["dates"]
    if args.fecha:
        if args.fecha not in dates:
            raise SystemExit(f"La fecha {args.fecha} no está en la serie limpia.")
        target_index = dates.index(args.fecha)
        if target_index < WINDOW:
            raise SystemExit("No hay 14 días previos para pronosticar esa fecha.")
        target_date = args.fecha
    else:
        target_index = len(values)
        target_date = "dia_siguiente_al_ultimo_registro"
    if model_name == "lstm":
        checkpoint = load_checkpoint(root / "models" / "lstm_consumo_best.pt")
        prediction = predict_lstm(values, target_index, checkpoint)
        detail = "LSTM de una capa, ventana de 14 días"
    elif model_name == "holt_winters":
        hw = config["holt_winters"]
        forecasts = hw_one_step(values.tolist(), hw["alpha"], hw["beta"], hw["gamma"])
        prediction = forecasts[target_index]
        detail = "Holt-Winters aditivo semanal, equivalente al nodo original"
    else:
        prediction = float(values[target_index - 1])
        detail = "Consumo del día anterior"
    payload = {
        "modelo": model_name,
        "detalle": detail,
        "fecha_objetivo": target_date,
        "consumo_pronosticado_kwh": prediction,
        "evaluacion": "one_step_con_historia_real_hasta_el_dia_anterior",
        "ventana_dias": WINDOW if model_name == "lstm" else None,
        "fuente": "Hoja Historico_90_dias. Serie sintética del prototipo EnerTrack.",
        "caracter": "simulado",
        "advertencia": "Prototipo académico. Los consumos son simulados y no son mediciones de una planta real.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
