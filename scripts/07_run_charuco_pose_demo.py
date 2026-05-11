#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo_calibration_lab.board_models import load_charuco_config
from stereo_calibration_lab.io_utils import ensure_dir, load_calibration_npz, save_json
from stereo_calibration_lab.pose_estimation import estimate_charuco_pose


def _camera_matrix_from_calib(calib, use_rectified=False):
    if use_rectified and "P1" in calib:
        P1 = calib["P1"]
        if P1.shape == (3, 4):
            return P1[:, :3].astype(np.float64), np.zeros((5, 1), dtype=np.float64)
    if "K1" in calib and "d1" in calib:
        return calib["K1"].astype(np.float64), calib["d1"].reshape(-1, 1).astype(np.float64)
    if "camera_matrix" in calib and "dist_coeffs" in calib:
        return calib["camera_matrix"].astype(np.float64), calib["dist_coeffs"].reshape(-1, 1).astype(np.float64)
    raise KeyError("Calibration must contain K1/d1 or camera_matrix/dist_coeffs.")


def main():
    parser = argparse.ArgumentParser(description="Run ChArUco 6-DoF pose demo from camera index or video path.")
    parser.add_argument("--source", required=True, help="Camera index, video file, or image file.")
    parser.add_argument("--board", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-corners", type=int, default=8)
    parser.add_argument("--use-rectified-K", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output)
    cfg = load_charuco_config(args.board)
    calib = load_calibration_npz(args.calibration)
    K, d = _camera_matrix_from_calib(calib, args.use_rectified_K)

    # Source can be a camera index or a file.
    try:
        src = int(args.source)
    except ValueError:
        src = args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = out_dir / f"charuco_pose_demo_{ts}.mp4"
    csv_path = out_dir / f"charuco_pose_demo_{ts}.csv"
    writer = None
    rows = []
    frame_idx = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame_index", "pose_ok", "num_markers", "num_charuco", "tx_mm", "ty_mm", "tz_mm", "distance_mm", "roll_deg", "pitch_deg", "yaw_deg", "message"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            annotated, pose = estimate_charuco_pose(frame, cfg, K, d, min_corners=args.min_corners)
            if pose.get("ok"):
                text = f"POSE OK Z={pose['tz_mm']:.1f} mm corners={pose['num_charuco']}"
            else:
                text = f"POSE FAIL {pose.get('message')} corners={pose.get('num_charuco')}"
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (0, 0, 0), -1)
            cv2.putText(annotated, text[:120], (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
            writer.write(annotated)
            cv2.imshow("ChArUco Pose Demo", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                break
            row = {
                "frame_index": frame_idx,
                "pose_ok": pose.get("ok", False),
                "num_markers": pose.get("num_markers", ""),
                "num_charuco": pose.get("num_charuco", ""),
                "tx_mm": pose.get("tx_mm", ""),
                "ty_mm": pose.get("ty_mm", ""),
                "tz_mm": pose.get("tz_mm", ""),
                "distance_mm": pose.get("distance_mm", ""),
                "roll_deg": pose.get("roll_deg", ""),
                "pitch_deg": pose.get("pitch_deg", ""),
                "yaw_deg": pose.get("yaw_deg", ""),
                "message": pose.get("message", ""),
            }
            csv_writer.writerow(row)
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    save_json(out_dir / f"charuco_pose_demo_{ts}.metadata.json", {"video": str(video_path), "csv": str(csv_path), "source": args.source})
    print({"video": str(video_path), "csv": str(csv_path)})


if __name__ == "__main__":
    main()
