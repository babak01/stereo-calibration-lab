#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo_calibration_lab.io_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Extract left/right pairs from a side-by-side stereo video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--left-right", choices=["left_right", "right_left"], default="left_right")
    args = parser.parse_args()

    out = Path(args.output)
    left_dir = ensure_dir(out / "left")
    right_dir = ensure_dir(out / "right")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        h, w = frame.shape[:2]
        half = w // 2
        a = frame[:, :half]
        b = frame[:, half:]
        if args.left_right == "left_right":
            left, right = a, b
        else:
            left, right = b, a
        name = f"pair_{saved:06d}.png"
        cv2.imwrite(str(left_dir / name), left)
        cv2.imwrite(str(right_dir / name), right)
        saved += 1
        frame_idx += 1
        if saved >= args.max_pairs:
            break
    cap.release()
    print({"saved_pairs": saved, "left_dir": str(left_dir), "right_dir": str(right_dir)})


if __name__ == "__main__":
    main()
