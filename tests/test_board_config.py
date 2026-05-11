from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo_calibration_lab.board_models import load_checkerboard_config, load_charuco_config


def test_checkerboard_config_loads():
    cfg = load_checkerboard_config(ROOT / "configs" / "board_checkerboard.yaml")
    assert cfg.pattern_size == (7, 5)
    assert cfg.object_points().shape == (35, 3)


def test_charuco_config_loads():
    cfg = load_charuco_config(ROOT / "configs" / "board_charuco.yaml")
    assert cfg.squares_x == 7
    assert cfg.squares_y == 5
    assert cfg.max_corners == 24
