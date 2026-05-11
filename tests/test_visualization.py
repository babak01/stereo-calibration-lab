from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo_calibration_lab.visualization import letterbox


def test_letterbox_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = letterbox(img, 1280, 720)
    assert out.shape == (720, 1280, 3)
