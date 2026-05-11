from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .board_models import CheckerboardConfig, CharucoConfig, create_charuco_board, get_charuco_object_corners
from .detect_checkerboard import detect_checkerboard
from .detect_charuco import CharucoDetector, common_charuco_points
from .io_utils import ensure_dir, load_calibration_npz, match_stereo_pairs, read_color, read_gray, save_json, write_csv
from .rectify import rectify_pair
from .visualization import draw_epipolar_lines, side_by_side


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def validate_checkerboard_rectification(
    left_dir: str | Path,
    right_dir: str | Path,
    config: CheckerboardConfig,
    calibration_npz: str | Path,
    output_dir: str | Path,
    max_pairs: Optional[int] = None,
    save_overlays: int = 20,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    overlay_dir = ensure_dir(output_dir / "overlays")
    calib = load_calibration_npz(calibration_npz)
    pairs = match_stereo_pairs(left_dir, right_dir)
    if max_pairs:
        pairs = pairs[:max_pairs]

    rows: List[Dict] = []
    all_errors: List[float] = []
    overlays_saved = 0

    for left_path, right_path, pair_id in pairs:
        left = read_color(left_path)
        right = read_color(right_path)
        left_rect, right_rect = rectify_pair(left, right, calib)
        gl = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        dl = detect_checkerboard(gl, config)
        dr = detect_checkerboard(gr, config)

        if not dl.ok or not dr.ok:
            row = {"pair_id": pair_id, "detected_left": dl.ok, "detected_right": dr.ok, "num_corners": 0,
                   "mean_abs_vertical_error_px": "", "median_abs_vertical_error_px": "", "p95_abs_vertical_error_px": "", "max_abs_vertical_error_px": "", "status": "FAIL"}
            rows.append(row)
            continue

        ydiff = np.abs(dl.corners.reshape(-1, 2)[:, 1] - dr.corners.reshape(-1, 2)[:, 1])
        all_errors.extend([float(x) for x in ydiff])
        st = _stats([float(x) for x in ydiff])
        rows.append({
            "pair_id": pair_id,
            "detected_left": True,
            "detected_right": True,
            "num_corners": len(ydiff),
            "mean_abs_vertical_error_px": f"{st['mean']:.6f}",
            "median_abs_vertical_error_px": f"{st['median']:.6f}",
            "p95_abs_vertical_error_px": f"{st['p95']:.6f}",
            "max_abs_vertical_error_px": f"{st['max']:.6f}",
            "status": "PASS",
        })
        if overlays_saved < save_overlays:
            canvas = side_by_side(draw_epipolar_lines(left_rect), draw_epipolar_lines(right_rect), "LEFT RECTIFIED", "RIGHT RECTIFIED")
            cv2.imwrite(str(overlay_dir / f"{pair_id}_rectified_overlay.png"), canvas)
            overlays_saved += 1

    write_csv(output_dir / "rectification_pair_errors.csv", rows)
    st_all = _stats(all_errors)
    passed = sum(1 for r in rows if r["status"] == "PASS")
    summary = {
        "target": "checkerboard",
        "pairs_checked": len(rows),
        "pairs_passed_detection": passed,
        "total_corner_matches": len(all_errors),
        "mean_abs_vertical_error_px": st_all["mean"],
        "median_abs_vertical_error_px": st_all["median"],
        "p95_abs_vertical_error_px": st_all["p95"],
        "max_abs_vertical_error_px": st_all["max"],
        "pair_csv": str(output_dir / "rectification_pair_errors.csv"),
        "overlay_dir": str(overlay_dir),
    }
    save_json(output_dir / "rectification_summary.json", summary)
    return summary


def validate_charuco_rectification(
    left_dir: str | Path,
    right_dir: str | Path,
    config: CharucoConfig,
    calibration_npz: str | Path,
    output_dir: str | Path,
    max_pairs: Optional[int] = None,
    save_overlays: int = 20,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    overlay_dir = ensure_dir(output_dir / "overlays")
    calib = load_calibration_npz(calibration_npz)
    pairs = match_stereo_pairs(left_dir, right_dir)
    if max_pairs:
        pairs = pairs[:max_pairs]

    board, _ = create_charuco_board(config)
    object_corners = get_charuco_object_corners(board)
    detector = CharucoDetector(config)

    rows: List[Dict] = []
    all_errors: List[float] = []
    overlays_saved = 0

    for left_path, right_path, pair_id in pairs:
        left = read_color(left_path)
        right = read_color(right_path)
        left_rect, right_rect = rectify_pair(left, right, calib)
        gl = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        dl = detector.detect(gl)
        dr = detector.detect(gr)
        obj, img_l, img_r, ids = common_charuco_points(dl, dr, object_corners)

        if obj is None or len(ids) < config.min_common_corners:
            rows.append({"pair_id": pair_id, "detected_left": dl.ok, "detected_right": dr.ok, "num_common_corners": len(ids),
                         "mean_abs_vertical_error_px": "", "median_abs_vertical_error_px": "", "p95_abs_vertical_error_px": "", "max_abs_vertical_error_px": "", "status": "FAIL"})
            continue

        ydiff = np.abs(img_l[:, 1] - img_r[:, 1])
        all_errors.extend([float(x) for x in ydiff])
        st = _stats([float(x) for x in ydiff])
        rows.append({
            "pair_id": pair_id,
            "detected_left": True,
            "detected_right": True,
            "num_common_corners": len(ids),
            "mean_abs_vertical_error_px": f"{st['mean']:.6f}",
            "median_abs_vertical_error_px": f"{st['median']:.6f}",
            "p95_abs_vertical_error_px": f"{st['p95']:.6f}",
            "max_abs_vertical_error_px": f"{st['max']:.6f}",
            "status": "PASS",
        })
        if overlays_saved < save_overlays:
            canvas = side_by_side(draw_epipolar_lines(left_rect), draw_epipolar_lines(right_rect), "LEFT RECTIFIED", "RIGHT RECTIFIED")
            cv2.imwrite(str(overlay_dir / f"{pair_id}_rectified_overlay.png"), canvas)
            overlays_saved += 1

    write_csv(output_dir / "rectification_pair_errors.csv", rows)
    st_all = _stats(all_errors)
    passed = sum(1 for r in rows if r["status"] == "PASS")
    summary = {
        "target": "charuco",
        "pairs_checked": len(rows),
        "pairs_passed_detection": passed,
        "total_common_corner_matches": len(all_errors),
        "mean_abs_vertical_error_px": st_all["mean"],
        "median_abs_vertical_error_px": st_all["median"],
        "p95_abs_vertical_error_px": st_all["p95"],
        "max_abs_vertical_error_px": st_all["max"],
        "pair_csv": str(output_dir / "rectification_pair_errors.csv"),
        "overlay_dir": str(overlay_dir),
    }
    save_json(output_dir / "rectification_summary.json", summary)
    return summary
