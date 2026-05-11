from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .board_models import CheckerboardConfig


@dataclass
class CheckerboardDetection:
    ok: bool
    corners: Optional[np.ndarray]
    num_corners: int
    message: str


def detect_checkerboard(gray: np.ndarray, config: CheckerboardConfig) -> CheckerboardDetection:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, config.pattern_size, flags)

    if not ok or corners is None:
        return CheckerboardDetection(False, None, 0, "checkerboard_not_detected")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-4,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return CheckerboardDetection(True, corners.astype(np.float32), len(corners), "ok")


def draw_checkerboard(image_bgr: np.ndarray, config: CheckerboardConfig, detection: CheckerboardDetection) -> np.ndarray:
    out = image_bgr.copy()
    if detection.corners is not None:
        cv2.drawChessboardCorners(out, config.pattern_size, detection.corners, detection.ok)
    return out
