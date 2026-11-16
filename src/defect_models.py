"""Construcción de ResNet18 y MobileNetV3 Small para seis defectos NEU.

La partición se lee de results/neu_surface_defects_split.csv y no se modifica.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

SEED = 42
SPLIT_CSV = "results/neu_surface_defects_split.csv"
LABEL_MAP_PATH = "models/label_map.json"
EXPECTED_COUNTS = {"train": 1261, "val": 269, "test": 270}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_SPECS = {
    "resnet18": {
        "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "checkpoint": "models/resnet18_best.pt",
    },
    "mobilenet_v3_small": {
        "weights_enum": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "checkpoint": "models/mobilenet_v3_small_best.pt",
    },
}

DATASET_ORIGIN = (
    "NEU Surface Defect Database, Kechen Song y Yunhui Yan, "
    "Northeastern University. Copia pública de Kaggle "
    "kaustubhdikshit/neu-surface-defect-database."
)
ACADEMIC_WARNING = (
    "Prototipo académico. Todas las clases son tipos de defecto superficial; "
    "no existe una clase normal. Estas imágenes no provienen de la planta "
    "ficticia EnerTrack y el resultado no sustituye una inspección industrial."
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split_rows() -> list[dict[str, str]]:
    csv_path = project_root() / SPLIT_CSV
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[str, int] = {}
    labels: set[str] = set()
    paths: set[str] = set()
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
        labels.add(row["label"])
        if row["relative_path"] in paths:
            raise RuntimeError(f"Ruta repetida en el CSV: {row['relative_path']}")
        paths.add(row["relative_path"])
        image_path = project_root() / row["relative_path"]
        if not image_path.is_file():
            raise RuntimeError(f"No existe la imagen del CSV: {row['relative_path']}")
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"La partición cambió: {counts}")
    if len(labels) != 6:
        raise RuntimeError(f"Se esperaban seis clases y hay {sorted(labels)}")
    return rows


def label_names(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["label"] for row in rows})


def rows_for(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def weights_for(model_name: str):
    if model_name == "resnet18":
        return ResNet18_Weights.IMAGENET1K_V1
    if model_name == "mobilenet_v3_small":
        return MobileNet_V3_Small_Weights.IMAGENET1K_V1
    raise ValueError(model_name)


def build_model(model_name: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    weights = weights_for(model_name) if pretrained else None
    if model_name == "resnet18":
        model = resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model
    raise ValueError(model_name)


def freeze_except_final_classifier(model: torch.nn.Module, model_name: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model_name == "resnet18":
        head = model.fc
    elif model_name == "mobilenet_v3_small":
        head = model.classifier[-1]
    else:
        raise ValueError(model_name)
    for parameter in head.parameters():
        parameter.requires_grad = True
    return sum(parameter.numel() for parameter in head.parameters())


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def save_label_map(names: list[str]) -> None:
    payload = {
        "index_to_label": {str(index): name for index, name in enumerate(names)},
        "label_to_index": {name: index for index, name in enumerate(names)},
    }
    path = project_root() / LABEL_MAP_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_label_map() -> list[str]:
    payload = json.loads((project_root() / LABEL_MAP_PATH).read_text(encoding="utf-8"))
    mapping = payload["index_to_label"]
    return [mapping[str(index)] for index in range(len(mapping))]
