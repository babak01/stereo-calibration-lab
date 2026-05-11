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
from stereo_calibration_lab.stereo_calibrate import stereo_calibrate_checkerboard, stereo_calibrate_charuco


def main():
    parser = argparse.ArgumentParser(description="Run stereo calibration from image pairs.")
    parser.add_argument("--target", choices=["checkerboard", "charuco"], required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qc-csv", default=None)
    args = parser.parse_args()

    if args.target == "checkerboard":
        cfg = load_checkerboard_config(args.board)
        summary = stereo_calibrate_checkerboard(args.left, args.right, cfg, args.output, args.qc_csv)
    else:
        cfg = load_charuco_config(args.board)
        summary = stereo_calibrate_charuco(args.left, args.right, cfg, args.output, args.qc_csv)
    print(summary)


if __name__ == "__main__":
    main()
