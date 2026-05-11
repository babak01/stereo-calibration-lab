from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from .io_utils import ensure_dir, load_calibration_npz, match_stereo_pairs, read_color
from .rectify import rectify_pair
from .visualization import draw_epipolar_lines, side_by_side


def make_rectification_demo(
    left_dir: str | Path,
    right_dir: str | Path,
    calibration_npz: str | Path,
    output_video: str | Path,
    fps: float = 5.0,
    max_pairs: Optional[int] = 100,
) -> Path:
    output_video = Path(output_video)
    ensure_dir(output_video.parent)
    calib = load_calibration_npz(calibration_npz)
    pairs = match_stereo_pairs(left_dir, right_dir)
    if max_pairs:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise RuntimeError("No stereo pairs found.")

    writer = None
    try:
        for left_path, right_path, pair_id in pairs:
            left = read_color(left_path)
            right = read_color(right_path)
            lrect, rrect = rectify_pair(left, right, calib)
            raw = side_by_side(draw_epipolar_lines(left), draw_epipolar_lines(right), "RAW LEFT", "RAW RIGHT", width=1280, height=360)
            rect = side_by_side(draw_epipolar_lines(lrect), draw_epipolar_lines(rrect), "RECTIFIED LEFT", "RECTIFIED RIGHT", width=1280, height=360)
            frame = cv2.vconcat([raw, rect])
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {output_video}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    return output_video
