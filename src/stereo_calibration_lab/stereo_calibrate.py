from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .board_models import CheckerboardConfig, CharucoConfig, create_charuco_board, get_charuco_object_corners
from .detect_checkerboard import detect_checkerboard
from .detect_charuco import CharucoDetector, common_charuco_points
from .io_utils import ensure_dir, match_stereo_pairs, read_color, read_gray, read_csv_rows, save_json


def _filter_pairs_by_qc(pairs, qc_csv: Optional[str | Path]):
    if qc_csv is None:
        return pairs
    rows = read_csv_rows(qc_csv)
    keep = {r["pair_id"] for r in rows if r.get("status") == "PASS"}
    return [p for p in pairs if p[2] in keep]


def _reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, d) -> float:
    total_err = 0.0
    total_n = 0
    for obj, img, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, d)
        proj = proj.reshape(-1, 2)
        img = img.reshape(-1, 2)
        err = np.linalg.norm(proj - img, axis=1).sum()
        total_err += float(err)
        total_n += len(img)
    return total_err / max(total_n, 1)


def stereo_calibrate_checkerboard(
    left_dir: str | Path,
    right_dir: str | Path,
    config: CheckerboardConfig,
    output_dir: str | Path,
    qc_csv: Optional[str | Path] = None,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    pairs = _filter_pairs_by_qc(match_stereo_pairs(left_dir, right_dir), qc_csv)

    objpoints: List[np.ndarray] = []
    imgpoints_l: List[np.ndarray] = []
    imgpoints_r: List[np.ndarray] = []
    used_pairs: List[str] = []
    image_size = None
    objp = config.object_points()

    for left_path, right_path, pair_id in pairs:
        gl = read_gray(left_path)
        gr = read_gray(right_path)
        if image_size is None:
            image_size = (gl.shape[1], gl.shape[0])
        dl = detect_checkerboard(gl, config)
        dr = detect_checkerboard(gr, config)
        if not dl.ok or not dr.ok:
            continue
        objpoints.append(objp.copy())
        imgpoints_l.append(dl.corners.reshape(-1, 2))
        imgpoints_r.append(dr.corners.reshape(-1, 2))
        used_pairs.append(pair_id)

    if len(objpoints) < 5:
        raise RuntimeError(f"Need at least 5 valid stereo pairs. Got {len(objpoints)}")

    flags_single = 0
    ret_l, K1, d1, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, image_size, None, None, flags=flags_single)
    ret_r, K2, d2, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, image_size, None, None, flags=flags_single)

    flags_stereo = cv2.CALIB_FIX_INTRINSIC
    stereo_rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, K1, d1, K2, d2, image_size, flags=flags_stereo
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, d1, K2, d2, image_size, R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, image_size, cv2.CV_32FC1)

    baseline = float(np.linalg.norm(T))
    mean_left = _reprojection_error(objpoints, imgpoints_l, rvecs_l, tvecs_l, K1, d1)
    mean_right = _reprojection_error(objpoints, imgpoints_r, rvecs_r, tvecs_r, K2, d2)

    npz_path = output_dir / "stereo_calibration.npz"
    np.savez_compressed(
        npz_path,
        image_size=np.array(image_size, dtype=np.int32),
        K1=K1, d1=d1, K2=K2, d2=d2, R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        roi1=np.array(roi1), roi2=np.array(roi2),
    )

    summary = {
        "target": "checkerboard",
        "image_size": list(image_size),
        "num_pairs_used": len(objpoints),
        "used_pairs": used_pairs,
        "left_rms": float(ret_l),
        "right_rms": float(ret_r),
        "stereo_rms": float(stereo_rms),
        "baseline_mm": baseline,
        "mean_left_reprojection_error_px": mean_left,
        "mean_right_reprojection_error_px": mean_right,
        "npz_path": str(npz_path),
    }
    save_json(output_dir / "calibration_summary.json", summary)
    return summary


def stereo_calibrate_charuco(
    left_dir: str | Path,
    right_dir: str | Path,
    config: CharucoConfig,
    output_dir: str | Path,
    qc_csv: Optional[str | Path] = None,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    pairs = _filter_pairs_by_qc(match_stereo_pairs(left_dir, right_dir), qc_csv)

    board, _ = create_charuco_board(config)
    object_corners = get_charuco_object_corners(board)
    detector = CharucoDetector(config)

    objpoints: List[np.ndarray] = []
    imgpoints_l: List[np.ndarray] = []
    imgpoints_r: List[np.ndarray] = []
    used_pairs: List[str] = []
    common_counts: List[int] = []
    image_size = None

    for left_path, right_path, pair_id in pairs:
        gl = read_gray(left_path)
        gr = read_gray(right_path)
        if image_size is None:
            image_size = (gl.shape[1], gl.shape[0])
        dl = detector.detect(gl)
        dr = detector.detect(gr)
        obj, img_l, img_r, ids = common_charuco_points(dl, dr, object_corners)
        if obj is None or len(ids) < config.min_common_corners:
            continue
        objpoints.append(obj)
        imgpoints_l.append(img_l)
        imgpoints_r.append(img_r)
        used_pairs.append(pair_id)
        common_counts.append(len(ids))

    if len(objpoints) < 5:
        raise RuntimeError(f"Need at least 5 valid stereo pairs. Got {len(objpoints)}")

    ret_l, K1, d1, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, image_size, None, None)
    ret_r, K2, d2, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, image_size, None, None)

    stereo_rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, K1, d1, K2, d2, image_size, flags=cv2.CALIB_FIX_INTRINSIC
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, d1, K2, d2, image_size, R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, image_size, cv2.CV_32FC1)

    baseline = float(np.linalg.norm(T))
    mean_left = _reprojection_error(objpoints, imgpoints_l, rvecs_l, tvecs_l, K1, d1)
    mean_right = _reprojection_error(objpoints, imgpoints_r, rvecs_r, tvecs_r, K2, d2)

    npz_path = output_dir / "stereo_calibration.npz"
    np.savez_compressed(
        npz_path,
        image_size=np.array(image_size, dtype=np.int32),
        K1=K1, d1=d1, K2=K2, d2=d2, R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        roi1=np.array(roi1), roi2=np.array(roi2),
    )

    summary = {
        "target": "charuco",
        "image_size": list(image_size),
        "num_pairs_used": len(objpoints),
        "used_pairs": used_pairs,
        "min_common_corners_used": int(min(common_counts)),
        "mean_common_corners_used": float(np.mean(common_counts)),
        "max_common_corners_used": int(max(common_counts)),
        "left_rms": float(ret_l),
        "right_rms": float(ret_r),
        "stereo_rms": float(stereo_rms),
        "baseline_mm": baseline,
        "mean_left_reprojection_error_px": mean_left,
        "mean_right_reprojection_error_px": mean_right,
        "npz_path": str(npz_path),
    }
    save_json(output_dir / "calibration_summary.json", summary)
    return summary
