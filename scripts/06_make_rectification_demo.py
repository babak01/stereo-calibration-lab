#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo_calibration_lab.video_demo import make_rectification_demo


def main():
    parser = argparse.ArgumentParser(description="Make raw-vs-rectified stereo demo video.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-pairs", type=int, default=100)
    args = parser.parse_args()

    path = make_rectification_demo(args.left, args.right, args.calibration, args.output, args.fps, args.max_pairs)
    print({"output_video": str(path)})


if __name__ == "__main__":
    main()
