# -*- coding: utf-8 -*-
"""
head_utils.py
-------------
Tiện ích xử lý "góc đầu" cho MediaPipe Face Mesh.
"""

from typing import Optional
from geom_types import Point
import math

from geom_utils import (                  # <— dùng lại các hàm chung
    pt_from_landmark as _pt_from_landmark,
    pts_from_landmarks as _pts_from_landmarks,
)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

# ====== Landmarks theo MediaPipe Face Mesh ======
FOREHEAD_IDX: int = 10
CHIN_IDX: int = 152
LEFT_EYE_OUTER: int = 33
RIGHT_EYE_OUTER: int = 263

__all__ = [
    "FOREHEAD_IDX",
    "CHIN_IDX",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_OUTER",
    "pt_from_landmark",
    "pts_from_landmarks",
    "pts_from_idxs",
    "draw_roll_viz",
    "draw_pitch_viz",
]

# Re-export để giữ tương thích API cũ
def pt_from_landmark(landmarks, idx, w, h):
    return _pt_from_landmark(landmarks, idx, w, h)

def pts_from_landmarks(landmarks, idxs, w, h):
    return _pts_from_landmarks(landmarks, idxs, w, h)

def pts_from_idxs(landmarks, idxs, w, h):
    return _pts_from_landmarks(landmarks, idxs, w, h)


# ====== Helpers vẽ trực quan ======
def _wrap180(a: float) -> float:
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a


def draw_roll_viz(
    frame,
    left_eye_outer: Point,
    right_eye_outer: Point,
    roll_deg: float,
    color_eye=(0, 255, 0),
    color_ref=(200, 200, 200),
    color_arc=(0, 128, 255),
    color_text=(0, 0, 255),
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) chưa được cài để dùng hàm vẽ.")
    L = left_eye_outer
    R = right_eye_outer
    cx = (L[0] + R[0]) // 2
    cy = (L[1] + R[1]) // 2
    vx = R[0] - L[0]
    vy = R[1] - L[1]
    length = max(1.0, math.hypot(vx, vy))
    half = int(length * 0.5)

    cv2.circle(frame, L, 3, color_eye, -1)
    cv2.circle(frame, R, 3, color_eye, -1)
    cv2.line(frame, L, R, color_eye, 2)

    cv2.line(frame, (cx - half, cy), (cx + half, cy), color_ref, 1)

    theta_deg = math.degrees(math.atan2(vy, vx))
    start = min(0.0, theta_deg)
    end = max(0.0, theta_deg)

    radius = int(length * 0.6)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start, end, color_arc, 2)

    theta = math.radians(theta_deg)
    arrow_len = int(radius * 0.9)
    ex = cx + int(arrow_len * math.cos(theta))
    ey = cy + int(arrow_len * math.sin(theta))
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), color_arc, 2, tipLength=0.2)

    cv2.putText(
        frame,
        f"Roll: {roll_deg:+.1f} deg",
        (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_text,
        2,
    )


def draw_pitch_viz(
    frame,
    forehead: Point,
    chin: Point,
    pitch_deg: Optional[float] = None,
    color_line=(0, 255, 255),
    color_ref=(200, 200, 200),
    color_arc=(255, 0, 0),
    color_text=(0, 0, 255),
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) chưa được cài để dùng hàm vẽ.")

    fx, fy = forehead
    cx_, cy_ = chin
    vx = cx_ - fx
    vy = cy_ - fy
    length = max(1.0, math.hypot(vx, vy))
    half = int(length * 0.5)

    cv2.circle(frame, forehead, 3, color_line, -1)
    cv2.circle(frame, (cx_, cy_), 3, color_line, -1)
    cv2.line(frame, forehead, (cx_, cy_), color_line, 2)

    mx = (fx + cx_) // 2
    my = (fy + cy_) // 2
    cv2.line(frame, (mx, my - half), (mx, my + half), color_ref, 1)

    theta_vec = math.degrees(math.atan2(vy, vx))

    if pitch_deg is None:
        dot = vy
        cosang = max(-1.0, min(1.0, dot / length))
        pitch_deg = math.degrees(math.acos(cosang))

    start = 90.0
    end = theta_vec
    a = _wrap180(end - start)
    end = start + a
    start, end = (end, start) if end < start else (start, end)

    radius = int(length * 0.6)
    cv2.ellipse(frame, (mx, my), (radius, radius), 0, start, end, color_arc, 2)

    theta = math.radians(theta_vec)
    arrow_len = int(radius * 0.9)
    ex = mx + int(arrow_len * math.cos(theta))
    ey = my + int(arrow_len * math.sin(theta))
    cv2.arrowedLine(frame, (mx, my), (ex, ey), color_arc, 2, tipLength=0.2)

    cv2.putText(
        frame,
        f"Pitch: {pitch_deg:5.1f} deg",
        (mx + 10, my - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_text,
        2,
    )
