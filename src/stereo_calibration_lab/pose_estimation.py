from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .board_models import CharucoConfig, create_charuco_board, get_charuco_object_corners
from .detect_charuco import CharucoDetector
from .io_utils import ensure_dir, load_calibration_npz, save_json


def rotation_vector_to_euler_deg(rvec: np.ndarray):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def estimate_charuco_pose(frame_bgr, config: CharucoConfig, K, d, min_corners: int = 8):
    board, _ = create_charuco_board(config)
    object_corners = get_charuco_object_corners(board)
    detector = CharucoDetector(config)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    det = detector.detect(gray)
    annotated = frame_bgr.copy()

    if det.marker_ids is not None and det.marker_corners is not None:
        cv2.aruco.drawDetectedMarkers(annotated, det.marker_corners, det.marker_ids)

    if det.charuco_ids is None or det.charuco_corners is None or len(det.charuco_ids) < min_corners:
        return annotated, {"ok": False, "message": f"need_at_least_{min_corners}_charuco_corners", "num_charuco": det.num_charuco, "num_markers": det.num_markers}

    ids = det.charuco_ids.flatten().astype(int)
    img_pts_all = det.charuco_corners.reshape(-1, 2).astype(np.float32)
    obj_pts = []
    img_pts = []
    max_id = len(object_corners) - 1
    for i, cid in enumerate(ids):
        if 0 <= cid <= max_id:
            obj_pts.append(object_corners[cid])
            img_pts.append(img_pts_all[i])
            x, y = img_pts_all[i]
            cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 255), -1)
            cv2.putText(annotated, str(cid), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)

    if len(obj_pts) < min_corners:
        return annotated, {"ok": False, "message": "not_enough_valid_2d3d_matches", "num_charuco": len(obj_pts), "num_markers": det.num_markers}

    obj_pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, d, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return annotated, {"ok": False, "message": "solvepnp_failed", "num_charuco": len(obj_pts), "num_markers": det.num_markers}

    cv2.drawFrameAxes(annotated, K, d, rvec, tvec, float(config.square_length_mm) * 2.0, 3)
    roll, pitch, yaw = rotation_vector_to_euler_deg(rvec)
    t = tvec.reshape(3)
    return annotated, {
        "ok": True,
        "message": "ok",
        "num_charuco": len(obj_pts),
        "num_markers": det.num_markers,
        "tx_mm": float(t[0]),
        "ty_mm": float(t[1]),
        "tz_mm": float(t[2]),
        "distance_mm": float(np.linalg.norm(t)),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
    }
