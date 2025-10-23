# -*- coding: utf-8 -*-
"""
moving_average.py
-----------------
Các bộ lọc smoothing dùng chung:

1) MovingAverage: trung bình trượt cho dữ liệu thực.
2) MovingAverageAngle: trung bình trượt cho dữ liệu góc (độ), ổn định khi wrap
   (-180..180] hay [0..360) nhờ biểu diễn sin/cos và tính góc trung bình.
3) ExponentialMovingAverage (EMA): trung bình luỹ thừa, ít trễ và phản ứng nhanh.

Ví dụ:
    from moving_average import MovingAverage, MovingAverageAngle, ExponentialMovingAverage

    ma = MovingAverage(window=5)
    print(ma.push(1.0))   # -> 1.0
    print(ma.push(2.0))   # -> 1.5

    maa = MovingAverageAngle(window=5, mode="(-180,180]")
    print(maa.push(179))  # -> ~179
    print(maa.push(-179)) # -> ~180 (ổn định qua biên)

    ema = ExponentialMovingAverage(alpha=0.2)
    print(ema.push(10))   # -> 10
    print(ema.push(20))   # -> 12.0
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Literal


class MovingAverage:
    """Bộ lọc trung bình trượt đơn giản cho số thực."""

    def __init__(self, window: int = 5):
        if window <= 0:
            raise ValueError("window phải > 0")
        self.window = int(window)
        self._buf: Deque[float] = deque(maxlen=self.window)
        self._sum: float = 0.0

    def reset(self) -> None:
        self._buf.clear()
        self._sum = 0.0

    def push(self, x: float) -> float:
        if len(self._buf) == self._buf.maxlen:
            self._sum -= self._buf[0]
        self._buf.append(x)
        self._sum += x
        return self.value

    @property
    def value(self) -> float:
        return self._sum / len(self._buf) if self._buf else 0.0

    @property
    def is_full(self) -> bool:
        return len(self._buf) == self._buf.maxlen

    @property
    def size(self) -> int:
        return len(self._buf)


class MovingAverageAngle:
    """
    Trung bình trượt cho **góc độ** tránh lỗi wrap (ví dụ 179° và -179° trung bình khoảng 180°, không phải 0°).

    Cách làm:
        - Lưu cos(theta), sin(theta) vào cửa sổ trượt
        - Trung bình cos/sin rồi lấy atan2 để ra góc trung bình

    Tham số:
        window: kích thước cửa sổ
        mode: "(-180,180]" (mặc định) hoặc "[0,360)"
        deg_in: đầu vào là độ (True) hay radian (False)
        deg_out: đầu ra là độ (True) hay radian (False)
    """

    def __init__(
        self,
        window: int = 5,
        mode: Literal["(-180,180]", "[0,360)"] = "(-180,180]",
        deg_in: bool = True,
        deg_out: bool = True,
    ):
        if window <= 0:
            raise ValueError("window phải > 0")
        if mode not in ("(-180,180]", "[0,360)"):
            raise ValueError("mode phải là '(-180,180]' hoặc '[0,360)'")
        self.window = int(window)
        self.mode = mode
        self.deg_in = deg_in
        self.deg_out = deg_out
        self._buf_cos: Deque[float] = deque(maxlen=self.window)
        self._buf_sin: Deque[float] = deque(maxlen=self.window)
        self._sum_cos: float = 0.0
        self._sum_sin: float = 0.0

    def reset(self) -> None:
        self._buf_cos.clear()
        self._buf_sin.clear()
        self._sum_cos = 0.0
        self._sum_sin = 0.0

    def _wrap_out(self, angle_rad: float) -> float:
        """Chuẩn hoá góc theo mode & đơn vị đầu ra."""
        if self.mode == "(-180,180]":
            # wrap về (-pi, pi]
            while angle_rad <= -math.pi:
                angle_rad += 2 * math.pi
            while angle_rad > math.pi:
                angle_rad -= 2 * math.pi
            if self.deg_out:
                return math.degrees(angle_rad)
            return angle_rad
        else:
            # wrap về [0, 2pi)
            angle_rad = angle_rad % (2 * math.pi)
            if self.deg_out:
                return math.degrees(angle_rad)
            return angle_rad

    def push(self, angle: float) -> float:
        """Thêm 1 góc, trả về góc trung bình hiện tại theo mode."""
        theta = math.radians(angle) if self.deg_in else angle
        c, s = math.cos(theta), math.sin(theta)

        if len(self._buf_cos) == self._buf_cos.maxlen:
            self._sum_cos -= self._buf_cos[0]
            self._sum_sin -= self._buf_sin[0]

        self._buf_cos.append(c)
        self._buf_sin.append(s)
        self._sum_cos += c
        self._sum_sin += s

        if not self._buf_cos:
            return 0.0 if self.deg_out else 0.0

        avg_c = self._sum_cos / len(self._buf_cos)
        avg_s = self._sum_sin / len(self._buf_sin)
        avg_rad = math.atan2(avg_s, avg_c)
        return self._wrap_out(avg_rad)

    @property
    def value(self) -> float:
        """Lấy giá trị trung bình hiện tại (nếu chưa có dữ liệu → 0 theo đơn vị đầu ra)."""
        if not self._buf_cos:
            return 0.0 if self.deg_out else 0.0
        avg_c = self._sum_cos / len(self._buf_cos)
        avg_s = self._sum_sin / len(self._buf_sin)
        avg_rad = math.atan2(avg_s, avg_c)
        return self._wrap_out(avg_rad)

    @property
    def is_full(self) -> bool:
        return len(self._buf_cos) == self._buf_cos.maxlen

    @property
    def size(self) -> int:
        return len(self._buf_cos)


class ExponentialMovingAverage:
    """
    Trung bình luỹ thừa EMA cho số thực.
    alpha trong (0,1]: càng lớn càng nhạy (ít mượt hơn), càng nhỏ càng mượt (nhưng trễ hơn).
    """

    def __init__(self, alpha: float = 0.2):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha phải trong (0, 1]")
        self.alpha = float(alpha)
        self._y: Optional[float] = None

    def reset(self) -> None:
        self._y = None

    def push(self, x: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = self.alpha * x + (1.0 - self.alpha) * self._y
        return self._y

    @property
    def value(self) -> float:
        return 0.0 if self._y is None else self._y
