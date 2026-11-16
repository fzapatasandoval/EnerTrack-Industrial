"""Descarga, organiza y verifica NEU Surface Defect Database.

Fuente pública usada aquí:
Kaggle, kaustubhdikshit/neu-surface-defect-database
(NEU Surface Defect Database, Kechen Song y Yunhui Yan, Northeastern University).

El archivo de Kaggle está empaquetado como NEU-DET: imágenes por clase dentro de
carpetas train/validation y anotaciones XML. Este script junta todas las imágenes,
ignora ese reparto y las anotaciones, y crea una partición propia para clasificación.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

DATASET_HANDLE = "kaustubhdikshit/neu-surface-defect-database"
SEED = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}
CSV_NAME = "neu_surface_defects_split.csv"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_source_images(dataset_root: Path) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.parent.parent.name != "images":
            raise RuntimeError(f"Imagen fuera de una carpeta de clase: {path}")
        found.append((path, path.parent.name))
    if not found:
        raise RuntimeError(f"No hay imágenes en {dataset_root}")
    return found


def organize(source_images: list[tuple[Path, str]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source, label in source_images:
        target = destination / label / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) == sha256(source):
            continue
        target.write_bytes(source.read_bytes())


def inspect(image_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(project_root()).as_posix()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
        records.append(
            {
                "relative_path": relative,
                "label": path.parent.name,
                "width": str(width),
                "height": str(height),
                "mode": mode,
                "sha256": sha256(path),
            }
        )
    return records


def assign_splits(records: list[dict[str, str]]) -> None:
    ordered = sorted(records, key=lambda record: record["relative_path"])
    paths = [record["relative_path"] for record in ordered]
    labels = [record["label"] for record in ordered]
    train_paths, temp_paths, _, temp_labels = train_test_split(
        paths,
        labels,
        test_size=VAL_SIZE + TEST_SIZE,
        stratify=labels,
        random_state=SEED,
    )
    val_paths, test_paths, _, _ = train_test_split(
        temp_paths,
        temp_labels,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=temp_labels,
        random_state=SEED,
    )
    split_of = {path: "train" for path in train_paths}
    split_of.update({path: "val" for path in val_paths})
    split_of.update({path: "test" for path in test_paths})
    for record in records:
        record["split"] = split_of[record["relative_path"]]

    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    for group in by_hash.values():
        splits = {record["split"] for record in group}
        if len(splits) == 1:
            continue
        anchor = min(group, key=lambda record: record["relative_path"])
        for record in group:
            record["split"] = anchor["split"]


def write_csv(records: list[dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["relative_path", "label", "split"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["relative_path"]):
            writer.writerow({name: record[name] for name in fieldnames})


def report(records: list[dict[str, str]]) -> None:
    labels = Counter(record["label"] for record in records)
    splits = Counter(record["split"] for record in records)
    sizes = Counter((record["width"], record["height"]) for record in records)
    modes = Counter(record["mode"] for record in records)
    paths = [record["relative_path"] for record in records]
    hashes = [record["sha256"] for record in records]

    print(f"imagenes={len(records)}")
    print("categorias=" + ", ".join(f"{name}:{labels[name]}" for name in sorted(labels)))
    print("tamanos=" + ", ".join(f"{w}x{h}:{count}" for (w, h), count in sorted(sizes.items())))
    print("modos=" + ", ".join(f"{mode}:{count}" for mode, count in sorted(modes.items())))
    print(
        "particiones="
        + ", ".join(f"{name}:{splits[name]}" for name in ("train", "val", "test"))
    )
    for label in sorted(labels):
        per_split = Counter(
            record["split"] for record in records if record["label"] == label
        )
        print(
            f"clase[{label}]="
            + ", ".join(f"{name}:{per_split[name]}" for name in ("train", "val", "test"))
        )

    if len(paths) != len(set(paths)):
        raise RuntimeError("Hay rutas repetidas dentro del conjunto organizado.")
    by_split: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_split[record["split"]].add(record["sha256"])
    shared = set.intersection(*by_split.values()) if len(by_split) == 3 else set()
    if shared:
        raise RuntimeError("Un mismo contenido aparece en más de una partición.")
    duplicate_groups = [
        sorted(record["relative_path"] for record in records if record["sha256"] == digest)
        for digest, count in Counter(hashes).items()
        if count > 1
    ]
    print(f"grupos_contenido_identico={len(duplicate_groups)}")
    for group in duplicate_groups:
        print("contenido_identico=" + " | ".join(group))


def main() -> int:
    import kagglehub

    root = project_root()
    source_root = Path(kagglehub.dataset_download(DATASET_HANDLE))
    destination = root / "data" / "neu_surface_defects"
    source_images = find_source_images(source_root)
    organize(source_images, destination)
    records = inspect(destination)
    assign_splits(records)
    csv_path = root / "results" / CSV_NAME
    write_csv(records, csv_path)
    report(records)
    print(f"csv={csv_path.relative_to(root).as_posix()}")
    print(f"origen_kagglehub={source_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error={error}", file=sys.stderr)
        raise SystemExit(1)
