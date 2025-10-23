# -*- coding: utf-8 -*-
"""
ear_calc.py
-----------
Tính EAR (Eye Aspect Ratio) từ 6 điểm landmark của mắt.

Công thức:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
Ý nghĩa: EAR cao → mắt mở; EAR thấp → mắt nhắm.
"""

from typing import Sequence
from geom_types import Point
from geom_utils import euclidean_dist, EPS

__all__ = ["eye_aspect_ratio"]


def eye_aspect_ratio(pts6: Sequence[Point]) -> float:
    """
    Tính EAR từ 6 điểm mắt theo thứ tự: [p1, p2, p3, p4, p5, p6].

    Args:
        pts6: dãy 6 điểm (x, y)

    Returns:
        float: giá trị EAR
    """
    if len(pts6) != 6:
        raise ValueError("Cần đúng 6 điểm mắt để tính EAR")

    p1, p2, p3, p4, p5, p6 = pts6
    vert1 = euclidean_dist(p2, p6)            # mí trên ngoài <-> mí dưới ngoài
    vert2 = euclidean_dist(p3, p5)            # mí trên trong <-> mí dưới trong
    horiz = euclidean_dist(p1, p4) + EPS      # hai khoé mắt

    return (vert1 + vert2) / (2.0 * horiz)
