# -*- coding: utf-8 -*-
"""
mar_calc.py
-----------
Tính MAR (Mouth Aspect Ratio) từ 8 điểm landmark của miệng.

Công thức:
    MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 * ||p1 - p5||)
Ý nghĩa: MAR cao → miệng mở nhiều.
"""

from typing import Sequence
from geom_types import Point
from geom_utils import euclidean_dist, EPS

__all__ = ["mouth_aspect_ratio"]


def mouth_aspect_ratio(pts8: Sequence[Point]) -> float:
    """
    Tính MAR từ 8 điểm miệng theo thứ tự: [p1, p2, p3, p4, p5, p6, p7, p8].

    Args:
        pts8: dãy 8 điểm (x, y)

    Returns:
        float: giá trị MAR
    """
    if len(pts8) != 8:
        raise ValueError("Cần đúng 8 điểm miệng để tính MAR")

    p1, p2, p3, p4, p5, p6, p7, p8 = pts8

    num = (
        euclidean_dist(p2, p8)
        + euclidean_dist(p3, p7)
        + euclidean_dist(p4, p6)
    )
    den = 2.0 * euclidean_dist(p1, p5) + EPS

    return num / den
