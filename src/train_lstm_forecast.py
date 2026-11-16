"""Entrena una LSTM de una capa y la compara con Holt-Winters y el modelo ingenuo.

La prueba son los últimos 21 días. El normalizador y la parada temprana no ven
ese tramo. Dentro de la prueba, cada pronóstico a un día usa los consumos reales
anteriores: es evaluación rolling de un paso.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forecast_series import (
    ESTACION,
    VENTANA_EVALUACION,
    clean_series,
    hw_one_step,
    load_rows,
    regression_errors,
    save_json,
    select_hw_parameters,
    test_start_index,
)

SEED = 42
WINDOW = 14
VALIDATION_DAYS = 14
HIDDEN = 16
MAX_EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 1e-3
BATCH_SIZE = 16


class SmallLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _state = self.lstm(inputs)
        return self.head(output[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def windows(values: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for index in range(start, end):
        features.append(values[index - WINDOW:index])
        targets.append(values[index])
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std


def train_lstm(values: np.ndarray, train_end: int, test_start: int, device: torch.device) -> dict:
    mean = float(values[:train_end].mean())
    std = float(values[:train_end].std())
    if std == 0:
        raise RuntimeError("La desviación del tramo de entrenamiento es cero.")
    scaled = scale(values, mean, std)
    train_x, train_y = windows(scaled, WINDOW, train_end)
    val_x, val_y = windows(scaled, train_end, test_start)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x[:, :, None]), torch.from_numpy(train_y)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    model = SmallLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    best_state = None
    best_mae = None
    best_epoch = 0
    wait = 0
    history = []
    val_inputs = torch.from_numpy(val_x[:, :, None]).to(device)
    val_targets = val_y * std + mean
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_y)
            seen += len(batch_y)
        model.eval()
        with torch.inference_mode():
            predicted = model(val_inputs).detach().cpu().numpy() * std + mean
        mae = float(np.mean(np.abs(val_targets - predicted)))
        improved = best_mae is None or mae < best_mae
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "val_mae": mae, "improved": int(improved)})
        print(f"lstm epoca={epoch} train_loss={total_loss / seen:.4f} val_mae={mae:.4f}", flush=True)
        if improved:
            best_mae = mae
            best_epoch = epoch
            wait = 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "epochs_ran": history[-1]["epoch"],
        "history": history,
        "state_dict": best_state,
    }


def predict_scaled(model: nn.Module, scaled: np.ndarray, start: int, end: int, device: torch.device, std: float, mean: float) -> np.ndarray:
    features, _targets = windows(scaled, start, end)
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(features[:, :, None]).to(device)
        predicted = model(tensor).detach().cpu().numpy()
    return predicted * std + mean


def time_lstm(model, scaled, test_start, count, device, std, mean) -> float:
    model.eval()
    indexes = range(test_start, count)
    with torch.inference_mode():
        for index in list(indexes)[:5]:
            tensor = torch.from_numpy(scaled[index - WINDOW:index].astype(np.float32)[None, :, None]).to(device)
            model(tensor)
        synchronize(device)
        started = time.perf_counter()
        for index in indexes:
            tensor = torch.from_numpy(scaled[index - WINDOW:index].astype(np.float32)[None, :, None]).to(device)
            model(tensor)
            synchronize(device)
        elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / (count - test_start)


def time_call(function, repeats: int = 1) -> float:
    function()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - started) * 1000.0 / repeats


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    set_seed(SEED)
    device = select_device()
    rows, columns = load_rows()
    series = clean_series(rows, columns)
    values = np.asarray(series["values"], dtype=np.float64)
    count = len(values)
    test_start = test_start_index(count)
    train_end = test_start - VALIDATION_DAYS
    if train_end <= WINDOW + 5:
        raise RuntimeError("No queda suficiente historia para entrenar con una ventana de 14 días.")
    hw_params = select_hw_parameters(values.tolist(), test_start)
    hw_forecasts = hw_one_step(values.tolist(), hw_params["alpha"], hw_params["beta"], hw_params["gamma"])
    actual = values[test_start:count]
    hw_pred = np.asarray(hw_forecasts[test_start:count], dtype=np.float64)
    naive_pred = values[test_start - 1:count - 1]
    trained = train_lstm(values, train_end, test_start, device)
    scaled = (values - trained["mean"]) / trained["std"]
    lstm_pred = predict_scaled(trained["model"], scaled.astype(np.float32), test_start, count, device, trained["std"], trained["mean"])
    hw_time = time_call(lambda: hw_one_step(values.tolist(), hw_params["alpha"], hw_params["beta"], hw_params["gamma"])) / VENTANA_EVALUACION
    naive_time = time_call(lambda: values[test_start - 1:count - 1].copy()) / VENTANA_EVALUACION
    lstm_time = time_lstm(trained["model"], scaled.astype(np.float32), test_start, count, device, trained["std"], trained["mean"])
    metrics = {
        "holt_winters": regression_errors(actual.tolist(), hw_pred.tolist()),
        "lstm": regression_errors(actual.tolist(), lstm_pred.tolist()),
        "ingenuo": regression_errors(actual.tolist(), naive_pred.tolist()),
    }
    root = Path(__file__).resolve().parents[1]
    checkpoint = {
        "state_dict": trained["state_dict"],
        "window": WINDOW,
        "hidden": HIDDEN,
        "layers": 1,
        "mean": trained["mean"],
        "std": trained["std"],
        "best_epoch": trained["best_epoch"],
        "best_val_mae": trained["best_val_mae"],
        "seed": SEED,
    }
    model_path = root / "models" / "lstm_consumo_best.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    ranking = sorted(metrics, key=lambda name: (metrics[name]["mae"], metrics[name]["rmse"]))
    save_json(
        root / "results" / "forecast_data_quality.json",
        {key: value for key, value in series.items() if key not in {"values", "dates"}},
    )
    save_json(
        root / "results" / "forecast_config.json",
        {
            "seed": SEED,
            "device": device.type,
            "ventana_entrada_dias": WINDOW,
            "justificacion_ventana": "La estacionalidad original es semanal. Catorce días cubren dos ciclos y permiten comparar la semana previa.",
            "validacion_dias": VALIDATION_DAYS,
            "prueba_dias": VENTANA_EVALUACION,
            "train_end_index": train_end,
            "test_start_index": test_start,
            "fecha_inicio_prueba": series["dates"][test_start],
            "fecha_fin_prueba": series["dates"][count - 1],
            "evaluacion": "rolling_one_step",
            "lstm": {
                "capas": 1,
                "hidden": HIDDEN,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "best_epoch": trained["best_epoch"],
                "epochs_ran": trained["epochs_ran"],
                "val_mae": trained["best_val_mae"],
                "normalizador": "media y desviación del tramo de entrenamiento",
            },
            "holt_winters": {
                "alpha": hw_params["alpha"],
                "beta": hw_params["beta"],
                "gamma": hw_params["gamma"],
                "estacionalidad_dias": ESTACION,
                "seleccion": "SSE de un paso desde el día 14 hasta el día anterior a la prueba, con la grilla del nodo original",
                "equivalencia": "Misma inicialización, mismas ecuaciones aditivas y misma grilla que el nodo Python de n8n. La LSTM reserva 14 días de validación que Holt-Winters sí incluye en su SSE, porque el nodo original no separa validación.",
            },
            "modelo_recomendado": ranking[0],
        },
    )
    write_csv(
        root / "results" / "lstm_training_log.csv",
        trained["history"],
    )
    write_csv(
        root / "results" / "forecast_comparison.csv",
        [
            {
                "modelo": name,
                "mae": metrics[name]["mae"],
                "rmse": metrics[name]["rmse"],
                "inferencia_ms_por_pronostico": {"holt_winters": hw_time, "lstm": lstm_time, "ingenuo": naive_time}[name],
                "fechas_prueba": VENTANA_EVALUACION,
            }
            for name in ("holt_winters", "lstm", "ingenuo")
        ],
    )
    write_csv(
        root / "results" / "forecast_predictions.csv",
        [
            {
                "fecha": series["dates"][test_start + offset],
                "consumo_kwh": actual[offset],
                "holt_winters": hw_pred[offset],
                "lstm": lstm_pred[offset],
                "ingenuo": naive_pred[offset],
            }
            for offset in range(VENTANA_EVALUACION)
        ],
    )
    for name in ("holt_winters", "lstm", "ingenuo"):
        print(
            f"RESULT {name} mae={metrics[name]['mae']:.6f} rmse={metrics[name]['rmse']:.6f}",
            flush=True,
        )
    print(f"device={device.type} recomendado={ranking[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
