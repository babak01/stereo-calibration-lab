from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def match_stereo_pairs(left_dir: str | Path, right_dir: str | Path) -> List[Tuple[Path, Path, str]]:
    left_images = list_images(left_dir)
    right_images = list_images(right_dir)

    right_by_name = {p.name: p for p in right_images}
    pairs: List[Tuple[Path, Path, str]] = []

    for left in left_images:
        if left.name in right_by_name:
            pair_id = left.stem
            pairs.append((left, right_by_name[left.name], pair_id))

    if pairs:
        return pairs

    # Fallback: sorted order if names do not match.
    n = min(len(left_images), len(right_images))
    return [(left_images[i], right_images[i], f"pair_{i:06d}") for i in range(n)]


def read_color(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def save_json(path: str | Path, data: Dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: Sequence[Dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_calibration_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def calibration_image_size(calib: Dict[str, np.ndarray]) -> Tuple[int, int]:
    if "image_size" not in calib:
        raise KeyError("Calibration missing image_size")
    s = calib["image_size"]
    return (int(s[0]), int(s[1]))
