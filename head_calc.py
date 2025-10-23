# -*- coding: utf-8 -*-
"""
head_calc.py
------------
Tính góc đầu (loại bỏ pitch 2D dựa trên trán→cằm theo trục ảnh).

Giữ lại:
- head_pitch_rel_eyes_deg: pitch 2D tương đối theo đường mắt (roll-invariant)
- head_roll_deg: roll 2D (góc đường mắt với +Ox), trả về (-180..180]
- head_pitch_3d_deg: pitch xấp xỉ từ (x,y,z) của MediaPipe (độ lớn, không dấu)
- MovingAverage: bộ lọc trung bình trượt cho tín hiệu số thực
"""

import math
from collections import deque
from typing import Deque

from geom_types import Point
from geom_utils import clamp, EPS

__all__ = [
    # "head_pitch_deg",  # ĐÃ BỎ theo yêu cầu
    "head_pitch_rel_eyes_deg",
    "head_roll_deg",
    "head_pitch_3d_deg",
    "MovingAverage",
]


def head_pitch_rel_eyes_deg(
    forehead: Point,
    chin: Point,
    left_eye_outer: Point,
    right_eye_outer: Point,
) -> float:
    """
    Pitch (độ) tương đối theo đường mắt (roll-invariant):
    - e = vector mắt trái → mắt phải
    - n = pháp tuyến của e (xoay 90°) ~ trục dọc khuôn mặt trong ảnh
    - v = trán → cằm
    Pitch = góc giữa v và n. Trả về [0..180], không mang dấu.
    """
    ex = right_eye_outer[0] - left_eye_outer[0]
    ey = right_eye_outer[1] - left_eye_outer[1]
    nx, ny = -ey, ex  # pháp tuyến e

    vx = chin[0] - forehead[0]
    vy = chin[1] - forehead[1]

    dot = vx * nx + vy * ny
    nv = math.hypot(vx, vy)
    nn = math.hypot(nx, ny)
    if nv == 0.0 or nn == 0.0:
        return 0.0
    cosang = clamp(dot / (nv * nn), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def head_roll_deg(left_eye_outer: Point, right_eye_outer: Point) -> float:
    """
    Roll (độ) ~ góc giữa đường nối hai khoé mắt và +Ox (trục ngang).
    Trả về (-180..180].
    """
    vx = right_eye_outer[0] - left_eye_outer[0]
    vy = right_eye_outer[1] - left_eye_outer[1]
    return math.degrees(math.atan2(vy, vx))


def head_pitch_3d_deg(forehead_xyz, chin_xyz) -> float:
    """
    Pitch 3D (độ) xấp xỉ từ (x,y,z) của MediaPipe.
    Trả về độ lớn (không dấu): atan2(|dz|, |dy|). Phụ thuộc hệ toạ độ MediaPipe.
    """
    dy = chin_xyz.y - forehead_xyz.y
    dz = chin_xyz.z - forehead_xyz.z
    return math.degrees(math.atan2(abs(dz), abs(dy) + EPS))
