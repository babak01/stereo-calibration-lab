#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo_calibration_lab.board_models import load_checkerboard_config, load_charuco_config
from stereo_calibration_lab.validate_epipolar import validate_checkerboard_rectification, validate_charuco_rectification


def main():
    parser = argparse.ArgumentParser(description="Validate stereo rectification using target correspondences.")
    parser.add_argument("--target", choices=["checkerboard", "charuco"], required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--save-overlays", type=int, default=20)
    args = parser.parse_args()

    if args.target == "checkerboard":
        cfg = load_checkerboard_config(args.board)
        summary = validate_checkerboard_rectification(args.left, args.right, cfg, args.calibration, args.output, args.max_pairs, args.save_overlays)
    else:
        cfg = load_charuco_config(args.board)
        summary = validate_charuco_rectification(args.left, args.right, cfg, args.calibration, args.output, args.max_pairs, args.save_overlays)
    print(summary)


if __name__ == "__main__":
    main()
