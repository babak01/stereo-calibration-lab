#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo_calibration_lab.board_models import load_checkerboard_config
from stereo_calibration_lab.dataset_qc import qc_checkerboard_pairs


def main():
    parser = argparse.ArgumentParser(description="QC a checkerboard stereo dataset.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-sharpness", type=float, default=30.0)
    parser.add_argument("--min-area-fraction", type=float, default=0.01)
    args = parser.parse_args()

    cfg = load_checkerboard_config(args.board)
    summary = qc_checkerboard_pairs(args.left, args.right, cfg, args.output, args.min_sharpness, args.min_area_fraction)
    print(summary)


if __name__ == "__main__":
    main()
