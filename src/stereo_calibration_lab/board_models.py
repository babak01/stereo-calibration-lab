from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class CheckerboardConfig:
    inner_corners_x: int
    inner_corners_y: int
    square_size_mm: float

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return (self.inner_corners_x, self.inner_corners_y)

    def object_points(self) -> np.ndarray:
        obj = np.zeros((self.inner_corners_x * self.inner_corners_y, 3), np.float32)
        grid = np.mgrid[0:self.inner_corners_x, 0:self.inner_corners_y].T.reshape(-1, 2)
        obj[:, :2] = grid * float(self.square_size_mm)
        return obj


@dataclass(frozen=True)
class CharucoConfig:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    min_common_corners: int = 8

    @property
    def max_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkerboard_config(path: str | Path) -> CheckerboardConfig:
    cfg = load_yaml(path)
    return CheckerboardConfig(
        inner_corners_x=int(cfg["inner_corners_x"]),
        inner_corners_y=int(cfg["inner_corners_y"]),
        square_size_mm=float(cfg["square_size_mm"]),
    )


def load_charuco_config(path: str | Path) -> CharucoConfig:
    cfg = load_yaml(path)
    return CharucoConfig(
        dictionary=str(cfg.get("dictionary", "DICT_5X5_100")),
        squares_x=int(cfg["squares_x"]),
        squares_y=int(cfg["squares_y"]),
        square_length_mm=float(cfg["square_length_mm"]),
        marker_length_mm=float(cfg["marker_length_mm"]),
        min_common_corners=int(cfg.get("min_common_corners", 8)),
    )


def get_aruco_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, dictionary_name):
        available = [x for x in dir(cv2.aruco) if x.startswith("DICT_")]
        raise ValueError(f"Unknown dictionary {dictionary_name}. Available: {available}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def create_charuco_board(config: CharucoConfig):
    dictionary = get_aruco_dictionary(config.dictionary)

    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            board = cv2.aruco.CharucoBoard(
                (config.squares_x, config.squares_y),
                config.square_length_mm,
                config.marker_length_mm,
                dictionary,
            )
            return board, dictionary
        except TypeError:
            pass

    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            config.squares_x,
            config.squares_y,
            config.square_length_mm,
            config.marker_length_mm,
            dictionary,
        )
        return board, dictionary

    raise RuntimeError("Could not create ChArUco board with this OpenCV version.")


def get_charuco_object_corners(board) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)
    if hasattr(board, "chessboardCorners"):
        return np.asarray(board.chessboardCorners, dtype=np.float32)
    raise RuntimeError("Cannot access ChArUco board chessboard corners.")
