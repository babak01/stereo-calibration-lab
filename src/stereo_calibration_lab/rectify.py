from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def load_rectification_maps(calib: Dict[str, np.ndarray]):
    required = ["map1x", "map1y", "map2x", "map2y"]
    missing = [k for k in required if k not in calib]
    if missing:
        raise KeyError(f"Calibration missing rectification maps: {missing}")
    return calib["map1x"].astype(np.float32), calib["map1y"].astype(np.float32), calib["map2x"].astype(np.float32), calib["map2y"].astype(np.float32)


def rectify_pair(left_bgr: np.ndarray, right_bgr: np.ndarray, calib: Dict[str, np.ndarray]):
    map1x, map1y, map2x, map2y = load_rectification_maps(calib)
    left_rect = cv2.remap(left_bgr, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_bgr, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect
