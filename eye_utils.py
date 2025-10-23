# -*- coding: utf-8 -*-
"""
eye_utils.py
------------
Tiện ích xử lý mắt cho MediaPipe Face Mesh.
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

LEFT_EYE_IDX: Sequence[int] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX: Sequence[int] = (263, 387, 385, 362, 380, 373)

__all__ = [
    "LEFT_EYE_IDX",
    "RIGHT_EYE_IDX",
    "pts_from_landmarks",
    "draw_eye_polyline",
]

# Re-export API cũ
def pts_from_landmarks(landmarks, idxs, w, h):
    return _pts_from_landmarks(landmarks, idxs, w, h)


def draw_eye_polyline(
    frame,
    pts6: Sequence[Point],
    closed: bool = True,
    color=(0, 255, 0),
    thickness: int = 1,
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) chưa được cài đặt")

    n = len(pts6)
    if n < 2:
        return

    try:
        import numpy as np  # type: ignore
        cnt = np.array(pts6, dtype="int32").reshape((-1, 1, 2))
        cv2.polylines(frame, [cnt], isClosed=closed, color=color, thickness=thickness)
    except Exception:
        for i in range(n - 1):
            cv2.line(frame, pts6[i], pts6[i + 1], color, thickness)
        if closed and n >= 3:
            cv2.line(frame, pts6[-1], pts6[0], color, thickness)
