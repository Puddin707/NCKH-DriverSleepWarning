# -*- coding: utf-8 -*-
"""
mouth_utils.py
--------------
Tiện ích MediaPipe Face Mesh cho viền miệng.
"""

from typing import Sequence
from geom_types import Point

from geom_utils import (                  # <— dùng lại
    pts_from_landmarks as _pts_from_landmarks,
)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

MOUTH_IDX: Sequence[int] = (61, 81, 13, 312, 291, 402, 14, 88)

__all__ = [
    "MOUTH_IDX",
    "pts_from_landmarks",
    "draw_mouth_polyline",
]

# Re-export API cũ
def pts_from_landmarks(landmarks, idxs, w, h):
    return _pts_from_landmarks(landmarks, idxs, w, h)


def draw_mouth_polyline(
    frame,
    pts8: Sequence[Point],
    closed: bool = True,
    color=(255, 0, 0),
    thickness: int = 1,
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) chưa được cài. Cài đặt để dùng hàm vẽ.")

    n = len(pts8)
    if n < 2:
        return

    try:
        import numpy as np  # type: ignore
        cnt = np.array(pts8, dtype="int32").reshape((-1, 1, 2))
        cv2.polylines(frame, [cnt], isClosed=closed, color=color, thickness=thickness)
    except Exception:
        for i in range(n - 1):
            cv2.line(frame, pts8[i], pts8[i + 1], color, thickness)
        if closed and n >= 3:
            cv2.line(frame, pts8[-1], pts8[0], color, thickness)
