from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .board_models import CheckerboardConfig, CharucoConfig, get_charuco_object_corners, create_charuco_board
from .detect_checkerboard import detect_checkerboard
from .detect_charuco import CharucoDetector, common_charuco_points
from .io_utils import match_stereo_pairs, read_color, read_gray, write_csv, save_json, ensure_dir


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def point_area_fraction(points: np.ndarray, image_shape: Tuple[int, int]) -> float:
    if points is None or len(points) < 4:
        return 0.0
    pts = points.reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(pts)
    area = float(cv2.contourArea(hull))
    h, w = image_shape[:2]
    return area / max(float(w * h), 1.0)


def qc_checkerboard_pairs(
    left_dir: str | Path,
    right_dir: str | Path,
    config: CheckerboardConfig,
    output_dir: str | Path,
    min_sharpness: float = 30.0,
    min_area_fraction: float = 0.01,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    overlay_dir = ensure_dir(output_dir / "overlays")
    rows: List[Dict] = []
    selected: List[Dict] = []

    pairs = match_stereo_pairs(left_dir, right_dir)
    objp = config.object_points()

    for left_path, right_path, pair_id in pairs:
        left_gray = read_gray(left_path)
        right_gray = read_gray(right_path)
        left_color = read_color(left_path)
        right_color = read_color(right_path)

        dl = detect_checkerboard(left_gray, config)
        dr = detect_checkerboard(right_gray, config)
        sharp_l = laplacian_sharpness(left_gray)
        sharp_r = laplacian_sharpness(right_gray)
        area_l = point_area_fraction(dl.corners, left_gray.shape) if dl.corners is not None else 0.0
        area_r = point_area_fraction(dr.corners, right_gray.shape) if dr.corners is not None else 0.0

        status = "PASS"
        reason = "ok"
        if not dl.ok or not dr.ok:
            status, reason = "FAIL", "detection_failed"
        elif sharp_l < min_sharpness or sharp_r < min_sharpness:
            status, reason = "FAIL", "blur_low_sharpness"
        elif area_l < min_area_fraction or area_r < min_area_fraction:
            status, reason = "FAIL", "board_too_small"

        row = {
            "pair_id": pair_id,
            "left_path": str(left_path),
            "right_path": str(right_path),
            "detected_left": dl.ok,
            "detected_right": dr.ok,
            "left_corners": dl.num_corners,
            "right_corners": dr.num_corners,
            "left_sharpness": f"{sharp_l:.6f}",
            "right_sharpness": f"{sharp_r:.6f}",
            "left_area_fraction": f"{area_l:.6f}",
            "right_area_fraction": f"{area_r:.6f}",
            "status": status,
            "reason": reason,
        }
        rows.append(row)
        if status == "PASS":
            selected.append(row)

    write_csv(output_dir / "all_pairs.csv", rows)
    write_csv(output_dir / "selected_pairs.csv", selected)
    summary = {
        "target": "checkerboard",
        "total_pairs": len(rows),
        "selected_pairs": len(selected),
        "min_sharpness": min_sharpness,
        "min_area_fraction": min_area_fraction,
        "all_csv": str(output_dir / "all_pairs.csv"),
        "selected_csv": str(output_dir / "selected_pairs.csv"),
    }
    save_json(output_dir / "qc_summary.json", summary)
    return summary


def qc_charuco_pairs(
    left_dir: str | Path,
    right_dir: str | Path,
    config: CharucoConfig,
    output_dir: str | Path,
    min_sharpness: float = 30.0,
    min_area_fraction: float = 0.01,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    rows: List[Dict] = []
    selected: List[Dict] = []

    board, _ = create_charuco_board(config)
    object_corners = get_charuco_object_corners(board)
    detector = CharucoDetector(config)

    pairs = match_stereo_pairs(left_dir, right_dir)
    for left_path, right_path, pair_id in pairs:
        left_gray = read_gray(left_path)
        right_gray = read_gray(right_path)

        dl = detector.detect(left_gray)
        dr = detector.detect(right_gray)
        obj, img_l, img_r, common_ids = common_charuco_points(dl, dr, object_corners)

        sharp_l = laplacian_sharpness(left_gray)
        sharp_r = laplacian_sharpness(right_gray)
        area_l = point_area_fraction(dl.charuco_corners, left_gray.shape) if dl.charuco_corners is not None else 0.0
        area_r = point_area_fraction(dr.charuco_corners, right_gray.shape) if dr.charuco_corners is not None else 0.0
        common_count = len(common_ids)

        status = "PASS"
        reason = "ok"
        if not dl.ok or not dr.ok:
            status, reason = "FAIL", "detection_failed"
        elif common_count < config.min_common_corners:
            status, reason = "FAIL", "not_enough_common_corners"
        elif sharp_l < min_sharpness or sharp_r < min_sharpness:
            status, reason = "FAIL", "blur_low_sharpness"
        elif area_l < min_area_fraction or area_r < min_area_fraction:
            status, reason = "FAIL", "board_too_small"

        row = {
            "pair_id": pair_id,
            "left_path": str(left_path),
            "right_path": str(right_path),
            "left_markers": dl.num_markers,
            "right_markers": dr.num_markers,
            "left_charuco_corners": dl.num_charuco,
            "right_charuco_corners": dr.num_charuco,
            "common_charuco_corners": common_count,
            "left_sharpness": f"{sharp_l:.6f}",
            "right_sharpness": f"{sharp_r:.6f}",
            "left_area_fraction": f"{area_l:.6f}",
            "right_area_fraction": f"{area_r:.6f}",
            "status": status,
            "reason": reason,
        }
        rows.append(row)
        if status == "PASS":
            selected.append(row)

    write_csv(output_dir / "all_pairs.csv", rows)
    write_csv(output_dir / "selected_pairs.csv", selected)
    summary = {
        "target": "charuco",
        "dictionary": config.dictionary,
        "squares_x": config.squares_x,
        "squares_y": config.squares_y,
        "square_length_mm": config.square_length_mm,
        "marker_length_mm": config.marker_length_mm,
        "min_common_corners": config.min_common_corners,
        "total_pairs": len(rows),
        "selected_pairs": len(selected),
        "min_sharpness": min_sharpness,
        "min_area_fraction": min_area_fraction,
        "all_csv": str(output_dir / "all_pairs.csv"),
        "selected_csv": str(output_dir / "selected_pairs.csv"),
    }
    save_json(output_dir / "qc_summary.json", summary)
    return summary
