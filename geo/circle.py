"""Scalar geometry on the unit circle."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .line import Interval as LinearInterval
from .line import Set as LinearSet


class Angle(float):
    """Angle in radians normalized to ``[0, 2π)``."""

    __slots__ = ()

    TWO_PI = 2.0 * math.pi
    MAX_ANGLE = math.nextafter(TWO_PI, -math.inf)

    def __new__(cls, angle: object = 0.0) -> "Angle":
        if isinstance(angle, cls):
            return angle
        return super().__new__(cls, cls._normalize(float(angle)))

    @property
    def value(self) -> float:
        return float(self)

    @staticmethod
    def _normalize(angle: float) -> float:
        normalized = math.fmod(angle, Angle.TWO_PI)
        if normalized < 0.0:
            normalized += Angle.TWO_PI
        if normalized >= Angle.TWO_PI or normalized > Angle.MAX_ANGLE:
            return 0.0
        return normalized

    def __repr__(self) -> str:
        return f"Angle({float(self)})"

    def __str__(self) -> str:
        return f"{float(self)} rad"

    def __add__(self, other: object) -> "Angle":
        return Angle(float(self) + float(other))

    def __sub__(self, other: object) -> "Angle":
        return Angle(float(self) - float(other))

    def __mul__(self, scalar: float) -> "Angle":
        return Angle(float(self) * scalar)

    def __truediv__(self, scalar: float) -> "Angle":
        return Angle(float(self) / scalar)

    def __neg__(self) -> "Angle":
        return Angle(-float(self))

    def __abs__(self) -> "Angle":
        return self

    def to_degrees(self) -> float:
        return math.degrees(self)

    @classmethod
    def from_degrees(cls, degrees: float) -> "Angle":
        return cls(math.radians(degrees))

    def distance_to(self, other: object) -> "Angle":
        diff = abs(float(self) - float(Angle(other)))
        if diff > math.pi:
            diff = Angle.TWO_PI - diff
        return Angle(diff)

    def is_close(self, other: object, abs_tol: float = 1e-12) -> bool:
        return self.distance_to(other) <= abs_tol

    def opposite(self) -> "Angle":
        return Angle(float(self) + math.pi)

    def complement(self) -> "Angle":
        return Angle(math.pi / 2 - float(self))

    def supplement(self) -> "Angle":
        return Angle(math.pi - float(self))


class Point(Angle):
    """Point on the unit circle represented by its angle."""

    __slots__ = ()

    @property
    def angle(self) -> Angle:
        return Angle(self)

    def __repr__(self) -> str:
        return f"Point({float(self)})"

    def __str__(self) -> str:
        return f"Point({float(self)})"

    def rotate(self, rotation_angle: object) -> "Point":
        return Point(float(self) + float(rotation_angle))

    def opposite(self) -> "Point":
        return Point(Angle(self).opposite())

    def to_cartesian(self) -> tuple[float, float]:
        return (math.cos(self), math.sin(self))

    @classmethod
    def from_cartesian(cls, x: float, y: float) -> "Point":
        if x == 0.0 and y == 0.0:
            raise ValueError(f"Point ({x}, {y}) is zero")
        return cls(math.atan2(y, x))

    @property
    def x(self) -> float:
        return math.cos(self)

    @property
    def y(self) -> float:
        return math.sin(self)


def _as_angle(value: object) -> Angle:
    return Angle(value)


def _previous_angle(value: object) -> float:
    previous = math.nextafter(float(_as_angle(value)), -math.inf)
    if previous < 0.0:
        return Angle.MAX_ANGLE
    return previous


def _normalize_linear_interval(interval: LinearInterval) -> LinearInterval:
    start = float(_as_angle(interval.start))
    end = float(_as_angle(interval.end))
    if start > end:
        raise ValueError(
            "Linear circle intervals must satisfy start <= end. "
            "Use circle.Interval for wrapped arcs."
        )
    return LinearInterval(start, end)


class Set(LinearSet):
    """Set of points on a unit circle."""

    __slots__ = ()

    def __new__(cls, *args: Any) -> "Set":
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        intervals = []
        for arg in args:
            intervals.extend(cls._coerce_argument(arg))
        return super().__new__(cls, *intervals)

    @classmethod
    def _coerce_argument(cls, arg: Any) -> tuple[LinearInterval, ...]:
        if isinstance(arg, Interval):
            return tuple(LinearInterval.from_tuple(iv) for iv in arg)
        if isinstance(arg, Set):
            return tuple(LinearInterval.from_tuple(iv) for iv in arg)
        if isinstance(arg, LinearInterval):
            return (_normalize_linear_interval(arg),)
        if isinstance(arg, (Angle, Point, float, int)):
            angle = float(_as_angle(arg))
            return (LinearInterval(angle, angle),)
        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
            if len(arg) == 0:
                return ()
            if len(arg) == 2 and all(
                isinstance(item, (Angle, Point, float, int)) for item in arg
            ):
                return tuple(
                    LinearInterval.from_tuple(iv)
                    for iv in Interval(arg[0], arg[1])
                )
            intervals = []
            for item in arg:
                intervals.extend(cls._coerce_argument(item))
            return tuple(intervals)
        raise TypeError(f"Unsupported circle-set argument: {arg!r}")

    @classmethod
    def from_linear_set(cls, circle_set: LinearSet) -> "Set":
        return cls(*(LinearInterval.from_tuple(iv) for iv in circle_set))

    def __repr__(self) -> str:
        intervals_repr = ", ".join(
            repr(LinearInterval.from_tuple(iv)) for iv in self
        )
        return f"{self.__class__.__name__}(({intervals_repr}))"

    def contains(self, point: object) -> bool:
        return super().contains(float(_as_angle(point)))

    def __contains__(self, point: object) -> bool:
        return self.contains(point)

    def union(self, other: Any) -> "Set":
        left = LinearSet(*(LinearInterval.from_tuple(iv) for iv in self))
        right = LinearSet(*(LinearInterval.from_tuple(iv) for iv in Set(other)))
        return Set.from_linear_set(left.union(right))

    def intersection(self, other: Any) -> "Set":
        left = LinearSet(*(LinearInterval.from_tuple(iv) for iv in self))
        right = LinearSet(*(LinearInterval.from_tuple(iv) for iv in Set(other)))
        return Set.from_linear_set(left.intersection(right))

    def difference(self, other: Any) -> "Set":
        left = LinearSet(*(LinearInterval.from_tuple(iv) for iv in self))
        right = LinearSet(*(LinearInterval.from_tuple(iv) for iv in Set(other)))
        return Set.from_linear_set(left.difference(right))

    def symmetric_difference(self, other: Any) -> "Set":
        left = LinearSet(*(LinearInterval.from_tuple(iv) for iv in self))
        right = LinearSet(*(LinearInterval.from_tuple(iv) for iv in Set(other)))
        return Set.from_linear_set(left.symmetric_difference(right))

    def complement(self) -> "Set":
        return FULL_SET.difference(self)

    def __invert__(self) -> "Set":
        return self.complement()

    def is_full(self) -> bool:
        return tuple(self) == tuple(FULL_SET)

    def contains_interval(self, interval: "Interval") -> bool:
        other = Set(interval)
        return self.intersection(other) == other

    @classmethod
    def from_single_interval(cls, start: object, end: object) -> "Set":
        return cls(Interval(start, end))

    @classmethod
    def from_intervals(cls, *intervals: "Interval") -> "Set":
        return cls(*intervals)

    def to_angles(self) -> tuple[tuple[float, float], ...]:
        return tuple(tuple(LinearInterval.from_tuple(iv)) for iv in self)


class Interval(Set):
    """Connected interval (arc) on a unit circle."""

    __slots__ = ()

    def __new__(cls, start: Any = 0.0, end: Any = None) -> "Interval":
        if end is None:
            if isinstance(start, cls):
                return start
            if isinstance(start, LinearInterval):
                start, end = start
            elif (
                isinstance(start, Sequence) and
                not isinstance(start, (str, bytes)) and
                len(start) == 2
            ):
                start, end = start
            else:
                end = start

        raw_start = float(start)
        raw_end = float(end)
        start_angle = float(_as_angle(start))
        end_angle = float(_as_angle(end))

        if raw_end < 0.0 and start_angle == 0.0:
            return super().__new__(cls, LinearInterval(0.0, Angle.MAX_ANGLE))
        if start_angle != end_angle and end_angle == _previous_angle(start_angle):
            return super().__new__(cls, LinearInterval(0.0, Angle.MAX_ANGLE))
        if start_angle <= end_angle:
            return super().__new__(cls, LinearInterval(start_angle, end_angle))
        return super().__new__(
            cls,
            LinearInterval(0.0, end_angle),
            LinearInterval(start_angle, Angle.MAX_ANGLE),
        )

    @property
    def start(self) -> Angle:
        if self.is_wrapped():
            return Angle(self[-1][0])
        return Angle(self[0][0])

    @property
    def end(self) -> Angle:
        return Angle(self[0][1])

    def __repr__(self) -> str:
        return f"Interval({float(self.start)}, {float(self.end)})"

    def __str__(self) -> str:
        if self.is_full():
            return "S1"
        return f"[{float(self.start)}, {float(self.end)}]"

    def is_empty(self) -> bool:
        return False

    def is_full_circle(self) -> bool:
        return self.is_full()

    def is_wrapped(self) -> bool:
        return len(self) == 2

    def is_point(self) -> bool:
        return len(self) == 1 and self[0][0] == self[0][1]

    def length(self) -> Angle:
        total = sum(LinearInterval.from_tuple(iv).length() for iv in self)
        return Angle(total)

    def contains_interval(self, other: "Interval") -> bool:
        other_set = Set(other)
        return self.intersection(other_set) == other_set

    def to_tuple(self) -> tuple[float, float]:
        return (float(self.start), float(self.end))


FULL_INTERVAL = Interval(0.0, Angle.MAX_ANGLE)
FULL_SET = Set(LinearInterval(0.0, Angle.MAX_ANGLE))


__all__ = [
    "Angle",
    "Point",
    "Interval",
    "Set",
    "FULL_INTERVAL",
    "FULL_SET",
]
