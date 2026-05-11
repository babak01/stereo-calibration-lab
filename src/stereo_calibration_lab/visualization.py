from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def draw_epipolar_lines(image: np.ndarray, spacing: int = 60, color: Tuple[int, int, int] = (0, 220, 255)) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for y in range(spacing, h, spacing):
        cv2.line(out, (0, y), (w, y), color, 1)
    return out


def letterbox(image: np.ndarray, width: int, height: int, bg=(18, 18, 18)) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def labeled_panel(image: np.ndarray, label: str, width: int, height: int, bar_color=(80, 220, 120)) -> np.ndarray:
    panel = letterbox(image, width, height)
    cv2.rectangle(panel, (0, 0), (width, 42), (28, 28, 28), -1)
    cv2.rectangle(panel, (0, 0), (8, height), bar_color, -1)
    cv2.putText(panel, label, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
    return panel


def side_by_side(left: np.ndarray, right: np.ndarray, left_label: str, right_label: str, width: int = 1280, height: int = 720) -> np.ndarray:
    cell_w = width // 2
    left_panel = labeled_panel(left, left_label, cell_w, height, (255, 170, 70))
    right_panel = labeled_panel(right, right_label, cell_w, height, (80, 220, 120))
    return np.hstack([left_panel, right_panel])
