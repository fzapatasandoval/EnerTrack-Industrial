"""Predice el defecto de una imagen con un checkpoint ya entrenado.

Ejemplo:
    python src/predict_defect.py data/neu_surface_defects/crazing/crazing_1.jpg --model resnet18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from defect_models import (
    ACADEMIC_WARNING,
    DATASET_ORIGIN,
    MODEL_SPECS,
    build_model,
    eval_transform,
    load_label_map,
    open_rgb,
    project_root,
)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(image_path: Path, model_name: str) -> dict:
    checkpoint_path = project_root() / MODEL_SPECS[model_name]["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    names = checkpoint["index_to_label"]
    if names != load_label_map():
        raise RuntimeError("El checkpoint y models/label_map.json no coinciden.")
    model = build_model(model_name, len(names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    device = select_device()
    model.to(device)
    model.eval()
    tensor = eval_transform()(open_rgb(image_path)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu()
    ranking = torch.argsort(probabilities, descending=True)
    top = ranking[:3]
    return {
        "modelo": model_name,
        "clase_predicha": names[int(ranking[0])],
        "confianza": float(probabilities[int(ranking[0])]),
        "top_3_clases": [
            {"clase": names[int(index)], "confianza": float(probabilities[int(index)])}
            for index in top
        ],
        "origen_dataset": DATASET_ORIGIN,
        "advertencia": ACADEMIC_WARNING,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inferencia de defectos NEU")
    parser.add_argument("imagen", type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="resnet18")
    args = parser.parse_args()
    image_path = args.imagen if args.imagen.is_absolute() else Path.cwd() / args.imagen
    if not image_path.is_file():
        print(json.dumps({"error": f"No existe la imagen: {image_path}"}, ensure_ascii=False))
        return 1
    print(json.dumps(predict(image_path, args.model), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
