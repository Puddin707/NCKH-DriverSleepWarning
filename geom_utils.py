# -*- coding: utf-8 -*-
"""
geom_utils.py
-------------
Tiện ích hình học & quy đổi landmark dùng chung.
"""

from typing import List, Sequence
import math
from geom_types import Point

EPS: float = 1e-6  # tránh chia cho 0 / mất ổn định số

__all__ = [
    "EPS",
    "clamp",
    "euclidean_dist",
    "angle_between_vecs_deg",
    "pt_from_landmark",
    "pts_from_landmarks",
]


def clamp(x: float, lo: float, hi: float) -> float:
    """Kẹp x vào khoảng [lo, hi]."""
    return max(lo, min(hi, x))


def euclidean_dist(p1: Point, p2: Point) -> float:
    """Khoảng cách Euclid giữa hai điểm pixel."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle_between_vecs_deg(ax: float, ay: float, bx: float, by: float) -> float:
    """
    Góc (độ) giữa hai vector A(ax,ay) và B(bx,by). Trả về [0..180].
    """
    dot = ax * bx + ay * by
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na == 0.0 or nb == 0.0:
        return 0.0
    cosang = clamp(dot / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


# ====== Quy đổi landmark (0..1) -> pixel ======
def pt_from_landmark(landmarks: Sequence, idx: int, w: int, h: int) -> Point:
    """
    Lấy một điểm pixel (x, y) từ landmark chuẩn hoá (x,y in [0..1]).
    Có kẹp biên an toàn theo kích thước (w,h).
    """
    if not (0 <= idx < len(landmarks)):
        raise IndexError(f"landmark idx={idx} out of range (len={len(landmarks)})")
    lm = landmarks[idx]
    x = int(clamp(lm.x * w, 0, max(0, w - 1)))
    y = int(clamp(lm.y * h, 0, max(0, h - 1)))
    return x, y


def pts_from_landmarks(
    landmarks: Sequence,
    idxs: Sequence[int],
    w: int,
    h: int,
) -> List[Point]:
    """
    Lấy danh sách điểm pixel (x, y) theo thứ tự idxs (có kẹp biên).
    """
    pts: List[Point] = []
    for i in idxs:
        if 0 <= i < len(landmarks):
            lm = landmarks[i]
            x = int(clamp(lm.x * w, 0, max(0, w - 1)))
            y = int(clamp(lm.y * h, 0, max(0, h - 1)))
            pts.append((x, y))  # type: ignore[arg-type]
        else:
            # Bỏ qua idx lỗi để không crash pipeline realtime
            # (tuỳ nhu cầu bạn có thể raise thay vì skip)
            continue
    return pts
