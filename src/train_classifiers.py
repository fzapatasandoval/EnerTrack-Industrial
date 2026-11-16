"""Entrena ResNet18, MobileNetV3 Small y una referencia de características simples.

Usa solo results/neu_surface_defects_split.csv. No vuelve a descargar el NEU
ni modifica las particiones. El conjunto de prueba se evalúa una sola vez,
después de elegir cada modelo con validación.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from defect_models import (
    MODEL_SPECS,
    SEED,
    build_model,
    eval_transform,
    freeze_except_final_classifier,
    label_names,
    load_split_rows,
    open_rgb,
    project_root,
    rows_for,
    save_label_map,
    train_transform,
    weights_for,
)

MAX_EPOCHS = 8
PATIENCE = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
BASELINE_C_VALUES = (0.01, 0.1, 1.0, 10.0)
BASELINE_NAME = "estadisticos_textura_logreg"


class SplitDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], label_to_index: dict[str, int], transform):
        self.rows = rows
        self.label_to_index = label_to_index
        self.transform = transform
        self.root = project_root()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = open_rgb(self.root / row["relative_path"])
        return self.transform(image), self.label_to_index[row["label"]]


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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(names))),
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    per_class = []
    for index, name in enumerate(names):
        per_class.append(
            {
                "label": name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def save_confusion(path: Path, matrix: np.ndarray, names: list[str]) -> None:
    rows = []
    for index, name in enumerate(names):
        row = {"true_label": name}
        for column, predicted in enumerate(names):
            row[predicted] = int(matrix[index, column])
        rows.append(row)
    write_csv(path, ["true_label", *names], rows)


def image_features(path: Path) -> np.ndarray:
    image = open_rgb(path).convert("L")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    histogram, _ = np.histogram(pixels, bins=16, range=(0.0, 1.0), density=True)
    horizontal = np.abs(np.diff(pixels, axis=1))
    vertical = np.abs(np.diff(pixels, axis=0))
    values = [
        float(pixels.mean()),
        float(pixels.std()),
        float(pixels.min()),
        float(pixels.max()),
        float(np.percentile(pixels, 10)),
        float(np.percentile(pixels, 50)),
        float(np.percentile(pixels, 90)),
        *histogram.astype(float).tolist(),
        float(horizontal.mean()),
        float(vertical.mean()),
        float(horizontal.std()),
        float(vertical.std()),
    ]
    return np.asarray(values, dtype=np.float64)


def feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    root = project_root()
    return np.vstack([image_features(root / row["relative_path"]) for row in rows])


def train_baseline(splits: dict[str, list[dict]], names: list[str]) -> dict:
    label_to_index = {name: index for index, name in enumerate(names)}
    features = {split: feature_matrix(rows) for split, rows in splits.items()}
    targets = {
        split: np.asarray([label_to_index[row["label"]] for row in rows], dtype=np.int64)
        for split, rows in splits.items()
    }
    validation_rows = []
    best = None
    for penalty in BASELINE_C_VALUES:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(C=penalty, max_iter=2000, random_state=SEED),
                ),
            ]
        )
        pipeline.fit(features["train"], targets["train"])
        predicted = pipeline.predict(features["val"])
        score = float(f1_score(targets["val"], predicted, average="macro", zero_division=0))
        accuracy = float(accuracy_score(targets["val"], predicted))
        validation_rows.append(
            {"C": penalty, "val_macro_f1": score, "val_accuracy": accuracy, "selected": 0}
        )
        if best is None or score > best["val_macro_f1"]:
            best = {
                "pipeline": pipeline,
                "C": penalty,
                "val_macro_f1": score,
                "val_accuracy": accuracy,
            }
    for row in validation_rows:
        row["selected"] = int(row["C"] == best["C"])
    write_csv(
        project_root() / "results" / "baseline_validation.csv",
        ["C", "val_macro_f1", "val_accuracy", "selected"],
        validation_rows,
    )

    probabilities = best["pipeline"].predict_proba(features["test"])
    predicted = probabilities.argmax(axis=1)
    metrics = classification_metrics(targets["test"], predicted, names)
    confidence = probabilities[np.arange(len(predicted)), predicted]
    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "C": best["C"],
            "label_to_index": label_to_index,
            "index_to_label": names,
            "val_macro_f1": best["val_macro_f1"],
        },
        project_root() / "models" / "baseline_logreg.joblib",
    )
    return {
        "model": BASELINE_NAME,
        "metrics": metrics,
        "mean_confidence": float(confidence.mean()),
        "inference_ms_per_image": time_baseline(best["pipeline"], splits["test"]),
        "val_macro_f1": best["val_macro_f1"],
        "val_accuracy": best["val_accuracy"],
        "best_epoch": "",
        "epochs_ran": "",
        "pretrained_weights": "",
        "trainable_parameters": int(
            best["pipeline"].named_steps["classifier"].coef_.size
            + best["pipeline"].named_steps["classifier"].intercept_.size
        ),
        "notes": f"LogisticRegression C={best['C']} elegido con macro F1 de validación",
    }


def make_loader(rows, label_to_index, transform, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        SplitDataset(rows, label_to_index, transform),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.eval()
    if training:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = None
    losses = []
    truths = []
    predictions = []
    confidences = []
    criterion = torch.nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                logits = model(images)
                loss = criterion(logits, labels)
        probabilities = torch.softmax(logits.detach(), dim=1)
        predicted = probabilities.argmax(dim=1)
        losses.append(float(loss.detach().cpu()) * len(labels))
        truths.append(labels.detach().cpu().numpy())
        predictions.append(predicted.cpu().numpy())
        chosen = probabilities[torch.arange(len(predicted)), predicted]
        confidences.append(chosen.cpu().numpy())
    return (
        float(sum(losses) / sum(len(item) for item in truths)),
        np.concatenate(truths),
        np.concatenate(predictions),
        np.concatenate(confidences),
    )


def load_imagenet_weights(model_name: str) -> dict:
    weights = weights_for(model_name)
    state = weights.get_state_dict(progress=True, check_hash=True)
    filename = Path(str(weights.url)).name
    cached = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename
    return {
        "model": model_name,
        "weights_enum": MODEL_SPECS[model_name]["weights_enum"],
        "url": str(weights.url),
        "check_hash": True,
        "state_dict_tensors": len(state),
        "cached_file": str(cached),
        "cached_file_exists": cached.is_file(),
        "cached_bytes": cached.stat().st_size if cached.is_file() else 0,
        "source_categories": len(weights.meta.get("categories", [])),
    }


def train_neural(model_name: str, splits, names: list[str], device: torch.device) -> dict:
    set_seed(SEED)
    weight_record = load_imagenet_weights(model_name)
    label_to_index = {name: index for index, name in enumerate(names)}
    model = build_model(model_name, len(names), pretrained=True)
    trainable = freeze_except_final_classifier(model, model_name)
    model.to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
    )
    train_loader = make_loader(splits["train"], label_to_index, train_transform(), shuffle=True)
    val_loader = make_loader(splits["val"], label_to_index, eval_transform(), shuffle=False)
    history = []
    best_score = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, y_true, y_pred, _ = run_epoch(model, val_loader, device)
        val_metrics = classification_metrics(y_true, y_pred, names)
        improved = val_metrics["macro_f1"] > best_score
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "improved": int(improved),
            }
        )
        print(
            f"{model_name} epoca={epoch} train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_accuracy={val_metrics['accuracy']:.4f}",
            flush=True,
        )
        if improved:
            best_score = val_metrics["macro_f1"]
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break
    model.load_state_dict(best_state)
    checkpoint = {
        "model": model_name,
        "state_dict": best_state,
        "index_to_label": names,
        "label_to_index": label_to_index,
        "weights_enum": weight_record["weights_enum"],
        "weights_url": weight_record["url"],
        "best_epoch": best_epoch,
        "val_macro_f1": best_score,
        "frozen_backbone": True,
    }
    checkpoint_path = project_root() / MODEL_SPECS[model_name]["checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    test_loader = make_loader(splits["test"], label_to_index, eval_transform(), shuffle=False)
    _, y_true, y_pred, confidence = run_epoch(model, test_loader, device)
    metrics = classification_metrics(y_true, y_pred, names)
    elapsed = time_single_images(model, splits["test"], device)
    return {
        "model": model_name,
        "metrics": metrics,
        "mean_confidence": float(confidence.mean()),
        "inference_ms_per_image": elapsed,
        "val_macro_f1": best_score,
        "val_accuracy": next(row["val_accuracy"] for row in history if row["epoch"] == best_epoch),
        "best_epoch": best_epoch,
        "epochs_ran": history[-1]["epoch"],
        "pretrained_weights": weight_record["weights_enum"],
        "trainable_parameters": trainable,
        "notes": "Backbone ImageNet congelado; solo se entrenó la capa lineal final",
        "history": history,
        "weight_record": weight_record,
    }


def time_baseline(pipeline, rows: list[dict[str, str]]) -> float:
    root = project_root()
    for row in rows[:10]:
        vector = image_features(root / row["relative_path"]).reshape(1, -1)
        pipeline.predict_proba(vector)
    durations = []
    for row in rows:
        started = time.perf_counter()
        vector = image_features(root / row["relative_path"]).reshape(1, -1)
        pipeline.predict_proba(vector)
        durations.append(time.perf_counter() - started)
    return float(np.mean(durations) * 1000.0)


def time_single_images(model, rows: list[dict[str, str]], device: torch.device) -> float:
    transform = eval_transform()
    root = project_root()
    model.eval()
    durations = []
    with torch.inference_mode():
        for row in rows[:10]:
            tensor = transform(open_rgb(root / row["relative_path"])).unsqueeze(0).to(device)
            model(tensor)
        synchronize(device)
        for row in rows:
            started = time.perf_counter()
            tensor = transform(open_rgb(root / row["relative_path"])).unsqueeze(0).to(device)
            model(tensor)
            synchronize(device)
            durations.append(time.perf_counter() - started)
    return float(np.mean(durations) * 1000.0)


def comparison_row(result: dict) -> dict:
    return {
        "model": result["model"],
        "test_accuracy": result["metrics"]["accuracy"],
        "test_macro_f1": result["metrics"]["macro_f1"],
        "test_images": 270,
        "mean_predicted_class_probability": result["mean_confidence"],
        "inference_ms_per_image": result["inference_ms_per_image"],
        "val_macro_f1": result["val_macro_f1"],
        "val_accuracy": result["val_accuracy"],
        "best_epoch": result["best_epoch"],
        "epochs_ran": result["epochs_ran"],
        "pretrained_weights": result["pretrained_weights"],
        "trainable_parameters": result["trainable_parameters"],
        "notes": result["notes"],
    }


def main() -> int:
    set_seed(SEED)
    device = select_device()
    rows = load_split_rows()
    names = label_names(rows)
    save_label_map(names)
    splits = {split: rows_for(rows, split) for split in ("train", "val", "test")}
    overlap = set(row["relative_path"] for row in splits["test"]) & (
        {row["relative_path"] for row in splits["train"]} | {row["relative_path"] for row in splits["val"]}
    )
    if overlap:
        raise RuntimeError("Hay imágenes de prueba repetidas en entrenamiento o validación.")
    print(f"device={device.type} clases={names}", flush=True)

    results = [train_baseline(splits, names)]
    print(
        f"baseline test_accuracy={results[0]['metrics']['accuracy']:.4f} "
        f"test_macro_f1={results[0]['metrics']['macro_f1']:.4f}",
        flush=True,
    )
    histories = []
    weight_records = []
    for model_name in ("mobilenet_v3_small", "resnet18"):
        result = train_neural(model_name, splits, names, device)
        results.append(result)
        histories.extend(result["history"])
        weight_records.append(result["weight_record"])

    results_dir = project_root() / "results"
    write_csv(
        results_dir / "model_comparison.csv",
        list(comparison_row(results[0]).keys()),
        [comparison_row(result) for result in results],
    )
    per_class = []
    for result in results:
        save_confusion(
            results_dir / f"confusion_matrix_{result['model']}.csv",
            result["metrics"]["confusion_matrix"],
            names,
        )
        for row in result["metrics"]["per_class"]:
            per_class.append({"model": result["model"], **row})
    write_csv(
        results_dir / "per_class_metrics.csv",
        ["model", "label", "precision", "recall", "f1", "support"],
        per_class,
    )
    if histories:
        write_csv(
            results_dir / "neural_training_log.csv",
            ["model", "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "improved"],
            histories,
        )
    (results_dir / "pretrained_weights.json").write_text(
        json.dumps(weight_records, indent=2) + "\n",
        encoding="utf-8",
    )
    config = {
        "seed": SEED,
        "device": device.type,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "selection_metric": "val_macro_f1",
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "frozen_backbone": True,
        "train_augmentations": ["RandomHorizontalFlip p=0.5", "RandomRotation degrees=10"],
        "eval_transform": "Resize 256, CenterCrop 224, ImageNet normalize",
        "baseline_C_grid": list(BASELINE_C_VALUES),
        "test_used_for_training_or_selection": False,
    }
    (results_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    for result in results:
        print(
            f"RESULT {result['model']} accuracy={result['metrics']['accuracy']:.4f} "
            f"macro_f1={result['metrics']['macro_f1']:.4f} "
            f"ms={result['inference_ms_per_image']:.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error={error}", file=sys.stderr)
        raise
