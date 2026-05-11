from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .board_models import CharucoConfig, create_charuco_board


@dataclass
class CharucoDetection:
    ok: bool
    charuco_corners: Optional[np.ndarray]
    charuco_ids: Optional[np.ndarray]
    marker_corners: Optional[list]
    marker_ids: Optional[np.ndarray]
    num_markers: int
    num_charuco: int
    message: str


class CharucoDetector:
    def __init__(self, config: CharucoConfig):
        self.config = config
        self.board, self.dictionary = create_charuco_board(config)
        self.use_new = hasattr(cv2.aruco, "CharucoDetector")
        if self.use_new:
            self.detector = cv2.aruco.CharucoDetector(self.board)
            self.aruco_detector = None
        else:
            self.detector = None
            if hasattr(cv2.aruco, "ArucoDetector"):
                params = cv2.aruco.DetectorParameters()
                self.aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, params)
            else:
                self.aruco_detector = None

    def detect(self, gray: np.ndarray) -> CharucoDetection:
        if self.use_new:
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        else:
            if self.aruco_detector is not None:
                marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                params = cv2.aruco.DetectorParameters_create()
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=params)
            if marker_ids is None or len(marker_ids) == 0:
                return CharucoDetection(False, None, None, marker_corners, marker_ids, 0, 0, "markers_not_detected")
            count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board
            )
            if charuco_ids is None or count is None or count < 1:
                return CharucoDetection(False, None, None, marker_corners, marker_ids, len(marker_ids), 0, "charuco_not_interpolated")

        num_markers = 0 if marker_ids is None else len(marker_ids)
        num_charuco = 0 if charuco_ids is None else len(charuco_ids)
        ok = num_charuco > 0
        msg = "ok" if ok else "charuco_not_detected"
        return CharucoDetection(ok, charuco_corners, charuco_ids, marker_corners, marker_ids, num_markers, num_charuco, msg)


def common_charuco_points(
    left: CharucoDetection,
    right: CharucoDetection,
    object_corners: np.ndarray,
):
    if left.charuco_ids is None or right.charuco_ids is None:
        return None, None, None, []

    left_ids = left.charuco_ids.flatten().astype(int)
    right_ids = right.charuco_ids.flatten().astype(int)
    left_pts = left.charuco_corners.reshape(-1, 2).astype(np.float32)
    right_pts = right.charuco_corners.reshape(-1, 2).astype(np.float32)

    left_map = {int(cid): left_pts[i] for i, cid in enumerate(left_ids)}
    right_map = {int(cid): right_pts[i] for i, cid in enumerate(right_ids)}
    common_ids = sorted(set(left_map).intersection(right_map))

    obj_pts = []
    img_l = []
    img_r = []
    valid_ids = []
    max_id = len(object_corners) - 1
    for cid in common_ids:
        if 0 <= cid <= max_id:
            obj_pts.append(object_corners[cid])
            img_l.append(left_map[cid])
            img_r.append(right_map[cid])
            valid_ids.append(cid)

    if not obj_pts:
        return None, None, None, []
    return (
        np.asarray(obj_pts, dtype=np.float32).reshape(-1, 3),
        np.asarray(img_l, dtype=np.float32).reshape(-1, 2),
        np.asarray(img_r, dtype=np.float32).reshape(-1, 2),
        valid_ids,
    )


def draw_charuco(image_bgr: np.ndarray, detection: CharucoDetection) -> np.ndarray:
    out = image_bgr.copy()
    if detection.marker_ids is not None and detection.marker_corners is not None:
        cv2.aruco.drawDetectedMarkers(out, detection.marker_corners, detection.marker_ids)
    if detection.charuco_corners is not None and detection.charuco_ids is not None:
        pts = detection.charuco_corners.reshape(-1, 2)
        ids = detection.charuco_ids.flatten().astype(int)
        for pt, cid in zip(pts, ids):
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(out, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(out, str(cid), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    return out
