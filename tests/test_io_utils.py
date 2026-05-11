from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo_calibration_lab.io_utils import ensure_dir


def test_ensure_dir(tmp_path):
    p = ensure_dir(tmp_path / "a" / "b")
    assert p.exists()
    assert p.is_dir()
