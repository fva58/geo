"""Angles and sets on a unit circle.

This module models circle subsets on the discrete lattice of Python ``float``
values. The ambient circle is the closed interval ``[0, 2π)`` where the value
just below ``2π`` is stored as ``FloatAngle.MAX_ANGLE``.

Intervals crossing zero are represented internally as two linear intervals:
one from ``0`` to the interval end and one from the interval start to
``MAX_ANGLE``.
"""

import math
from collections.abc import Sequence
from typing import Any, Tuple, Union

from .floatset import FloatInterval, FloatSet


CircleScalar = Union["FloatAngle", "FloatCirclePoint", float, int]


class FloatAngle(float):
    """Angle in radians normalized to ``[0, 2π)``."""

    __slots__ = ()

    TWO_PI = 2.0 * math.pi
    MAX_ANGLE = math.nextafter(TWO_PI, -math.inf)

    def __new__(cls, angle: CircleScalar = 0.0) -> "FloatAngle":
        """Create a normalized angle."""
        if isinstance(angle, cls):
            return angle
        return super().__new__(cls, cls._normalize(float(angle)))

    @property
    def value(self) -> float:
        """Return the underlying float value."""
        return float(self)

    @staticmethod
    def _normalize(angle: float) -> float:
        """Normalize an angle to the circle domain."""
        normalized = math.fmod(angle, FloatAngle.TWO_PI)
        if normalized < 0.0:
            normalized += FloatAngle.TWO_PI
        if normalized >= FloatAngle.TWO_PI:
            return 0.0
        if normalized > FloatAngle.MAX_ANGLE:
            return 0.0
        return normalized

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"FloatAngle({float(self)})"

    def __str__(self) -> str:
        """Return a human-readable representation."""
        return f"{float(self)} rad"

    def __add__(self, other: CircleScalar) -> "FloatAngle":
        """Add another angle."""
        return FloatAngle(float(self) + float(other))

    def __sub__(self, other: CircleScalar) -> "FloatAngle":
        """Subtract another angle."""
        return FloatAngle(float(self) - float(other))

    def __mul__(self, scalar: float) -> "FloatAngle":
        """Multiply by a scalar."""
        return FloatAngle(float(self) * scalar)

    def __truediv__(self, scalar: float) -> "FloatAngle":
        """Divide by a scalar."""
        return FloatAngle(float(self) / scalar)

    def __neg__(self) -> "FloatAngle":
        """Return the additive inverse."""
        return FloatAngle(-float(self))

    def __abs__(self) -> "FloatAngle":
        """Return the absolute value."""
        return self

    def to_degrees(self) -> float:
        """Convert the angle to degrees."""
        return math.degrees(self)

    @classmethod
    def from_degrees(cls, degrees: float) -> "FloatAngle":
        """Build an angle from degrees."""
        return cls(math.radians(degrees))

    def distance_to(self, other: CircleScalar) -> "FloatAngle":
        """Return the shortest angular distance to another angle."""
        diff = abs(float(self) - float(FloatAngle(other)))
        if diff > math.pi:
            diff = FloatAngle.TWO_PI - diff
        return FloatAngle(diff)

    def is_close(self, other: CircleScalar, abs_tol: float = 1e-12) -> bool:
        """Check closeness on the circle."""
        return self.distance_to(other) <= abs_tol

    def opposite(self) -> "FloatAngle":
        """Return the opposite angle."""
        return FloatAngle(float(self) + math.pi)

    def complement(self) -> "FloatAngle":
        """Return the complement to ``π / 2``."""
        return FloatAngle(math.pi / 2 - float(self))

    def supplement(self) -> "FloatAngle":
        """Return the supplement to ``π``."""
        return FloatAngle(math.pi - float(self))


class FloatCirclePoint(FloatAngle):
    """Point on the unit circle represented by its angle."""

    __slots__ = ()

    @property
    def angle(self) -> FloatAngle:
        """Return the point angle."""
        return FloatAngle(self)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"FloatCirclePoint({float(self)})"

    def __str__(self) -> str:
        """Return a human-readable representation."""
        return f"Point({float(self)})"

    def rotate(self, rotation_angle: CircleScalar) -> "FloatCirclePoint":
        """Rotate the point counterclockwise."""
        return FloatCirclePoint(float(self) + float(rotation_angle))

    def opposite(self) -> "FloatCirclePoint":
        """Return the opposite point."""
        return FloatCirclePoint(FloatAngle(self).opposite())

    def to_cartesian(self) -> Tuple[float, float]:
        """Return the point Cartesian coordinates."""
        return (math.cos(self), math.sin(self))

    @classmethod
    def from_cartesian(cls, x: float, y: float) -> "FloatCirclePoint":
        """Create a point from Cartesian coordinates."""
        if x == 0.0 and y == 0.0:
            raise ValueError(f"Point ({x}, {y}) is zero")
        return cls(math.atan2(y, x))

    @property
    def x(self) -> float:
        """Return the x-coordinate."""
        return math.cos(self)

    @property
    def y(self) -> float:
        """Return the y-coordinate."""
        return math.sin(self)


def _as_angle(value: CircleScalar) -> FloatAngle:
    """Convert a scalar-like value to ``FloatAngle``."""
    return FloatAngle(value)


def _previous_angle(value: CircleScalar) -> float:
    """Return the previous representable angle on the circle."""
    previous = math.nextafter(float(_as_angle(value)), -math.inf)
    if previous < 0.0:
        return FloatAngle.MAX_ANGLE
    return previous


def _normalize_linear_interval(interval: FloatInterval) -> FloatInterval:
    """Normalize a linear interval inside the circle domain."""
    left = float(_as_angle(interval.left))
    right = float(_as_angle(interval.right))
    if left > right:
        raise ValueError(
            "Linear circle intervals must satisfy left <= right. "
            "Use FloatCircleInterval for wrapped arcs."
        )
    return FloatInterval(left, right)


class FloatCircleSet(FloatSet):
    """Set of points on a unit circle.

    The internal representation is a normalized ``FloatSet`` over
    ``[0, FloatAngle.MAX_ANGLE]``.
    """

    __slots__ = ()

    def __new__(cls, *args: Any) -> "FloatCircleSet":
        """Create a circle set from intervals, arcs, points, or sequences."""
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]

        intervals = []
        for arg in args:
            intervals.extend(cls._coerce_argument(arg))
        return super().__new__(cls, *intervals)

    @classmethod
    def _coerce_argument(cls, arg: Any) -> Tuple[FloatInterval, ...]:
        """Translate constructor input to linear ``FloatInterval`` objects."""
        if isinstance(arg, FloatCircleInterval):
            return tuple(FloatInterval.from_tuple(iv) for iv in arg)

        if isinstance(arg, FloatCircleSet):
            return tuple(FloatInterval.from_tuple(iv) for iv in arg)

        if isinstance(arg, FloatInterval):
            return (_normalize_linear_interval(arg),)

        if isinstance(arg, (FloatAngle, FloatCirclePoint, float, int)):
            angle = float(_as_angle(arg))
            return (FloatInterval(angle, angle),)

        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
            if len(arg) == 0:
                return ()
            if (len(arg) == 2 and
                    all(isinstance(item, (FloatAngle, FloatCirclePoint,
                                          float, int)) for item in arg)):
                return tuple(FloatInterval.from_tuple(iv)
                             for iv in FloatCircleInterval(arg[0], arg[1]))

            intervals = []
            for item in arg:
                intervals.extend(cls._coerce_argument(item))
            return tuple(intervals)

        raise TypeError(f"Unsupported circle-set argument: {arg!r}")

    @classmethod
    def from_floatset(cls, circle_set: FloatSet) -> "FloatCircleSet":
        """Build a circle set from linear intervals already in circle space."""
        return cls(*(FloatInterval.from_tuple(iv) for iv in circle_set))

    def __repr__(self) -> str:
        """Return a debug representation."""
        intervals_repr = ", ".join(repr(FloatInterval.from_tuple(iv))
                                    for iv in self)
        return f"{self.__class__.__name__}(({intervals_repr}))"

    def contains(self, point: CircleScalar) -> bool:
        """Check if a point belongs to the set."""
        return super().contains(float(_as_angle(point)))

    def __contains__(self, point: CircleScalar) -> bool:
        """Check if a point belongs to the set."""
        return self.contains(point)

    def union(self, other: Any) -> "FloatCircleSet":
        """Return the union with another circle set."""
        left = FloatSet(*(FloatInterval.from_tuple(iv) for iv in self))
        right = FloatSet(*(FloatInterval.from_tuple(iv)
                           for iv in FloatCircleSet(other)))
        return FloatCircleSet.from_floatset(left.union(right))

    def intersection(self, other: Any) -> "FloatCircleSet":
        """Return the intersection with another circle set."""
        left = FloatSet(*(FloatInterval.from_tuple(iv) for iv in self))
        right = FloatSet(*(FloatInterval.from_tuple(iv)
                           for iv in FloatCircleSet(other)))
        return FloatCircleSet.from_floatset(left.intersection(right))

    def difference(self, other: Any) -> "FloatCircleSet":
        """Return the set difference with another circle set."""
        left = FloatSet(*(FloatInterval.from_tuple(iv) for iv in self))
        right = FloatSet(*(FloatInterval.from_tuple(iv)
                           for iv in FloatCircleSet(other)))
        return FloatCircleSet.from_floatset(left.difference(right))

    def symmetric_difference(self, other: Any) -> "FloatCircleSet":
        """Return the symmetric difference with another circle set."""
        left = FloatSet(*(FloatInterval.from_tuple(iv) for iv in self))
        right = FloatSet(*(FloatInterval.from_tuple(iv)
                           for iv in FloatCircleSet(other)))
        return FloatCircleSet.from_floatset(left.symmetric_difference(right))

    def complement(self) -> "FloatCircleSet":
        """Return the complement with respect to the full circle."""
        return FULL_FLOAT_CIRCLE_SET.difference(self)

    def __invert__(self) -> "FloatCircleSet":
        """Return the complement with respect to the full circle."""
        return self.complement()

    def is_full(self) -> bool:
        """Check if the set covers the whole circle."""
        return tuple(self) == tuple(FULL_FLOAT_CIRCLE_SET)

    def contains_interval(self, interval: "FloatCircleInterval") -> bool:
        """Check if the set contains a whole circle interval."""
        other = FloatCircleSet(interval)
        return self.intersection(other) == other

    @classmethod
    def from_single_interval(cls, start: CircleScalar,
                             end: CircleScalar) -> "FloatCircleSet":
        """Create a set from one arc."""
        return cls(FloatCircleInterval(start, end))

    @classmethod
    def from_intervals(cls, *intervals: "FloatCircleInterval") -> "FloatCircleSet":
        """Create a set from several arcs."""
        return cls(*intervals)

    def to_angles(self) -> Tuple[Tuple[float, float], ...]:
        """Return the stored linear intervals as angle pairs."""
        return tuple(tuple(FloatInterval.from_tuple(iv)) for iv in self)


class FloatCircleInterval(FloatCircleSet):
    """Connected interval (arc) on a unit circle."""

    __slots__ = ()

    def __new__(cls, start: Any = 0.0, end: Any = None) -> "FloatCircleInterval":
        """Create an arc from ``start`` to ``end`` counterclockwise."""
        if end is None:
            if isinstance(start, cls):
                return start
            if isinstance(start, FloatInterval):
                start, end = start
            elif (isinstance(start, Sequence) and
                  not isinstance(start, (str, bytes)) and len(start) == 2):
                start, end = start
            else:
                end = start

        raw_start = float(start)
        raw_end = float(end)
        start_angle = float(_as_angle(start))
        end_angle = float(_as_angle(end))

        if raw_end < 0.0 and start_angle == 0.0:
            return super().__new__(cls, FloatInterval(0.0, FloatAngle.MAX_ANGLE))

        if start_angle != end_angle and end_angle == _previous_angle(start_angle):
            return super().__new__(cls, FloatInterval(0.0, FloatAngle.MAX_ANGLE))

        if start_angle <= end_angle:
            return super().__new__(cls, FloatInterval(start_angle, end_angle))

        return super().__new__(
            cls,
            FloatInterval(0.0, end_angle),
            FloatInterval(start_angle, FloatAngle.MAX_ANGLE),
        )

    @property
    def start(self) -> FloatAngle:
        """Return the arc start."""
        if self.is_wrapped():
            return FloatAngle(self[-1][0])
        return FloatAngle(self[0][0])

    @property
    def end(self) -> FloatAngle:
        """Return the arc end."""
        return FloatAngle(self[0][1])

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"FloatCircleInterval({float(self.start)}, {float(self.end)})"

    def __str__(self) -> str:
        """Return a human-readable representation."""
        if self.is_full():
            return "S1"
        return f"[{float(self.start)}, {float(self.end)}]"

    def is_empty(self) -> bool:
        """Circle intervals are never empty."""
        return False

    def is_full_circle(self) -> bool:
        """Compatibility alias for ``is_full``."""
        return self.is_full()

    def is_wrapped(self) -> bool:
        """Check if the arc crosses zero."""
        return len(self) == 2

    def is_point(self) -> bool:
        """Check if the arc consists of a single point."""
        return len(self) == 1 and self[0][0] == self[0][1]

    def length(self) -> FloatAngle:
        """Return the arc length."""
        total = sum(FloatInterval.from_tuple(iv).length() for iv in self)
        return FloatAngle(total)

    def contains(self, point: CircleScalar) -> bool:
        """Check if a point belongs to the arc."""
        return super().contains(point)

    def contains_interval(self, other: "FloatCircleInterval") -> bool:
        """Check if this arc contains another arc."""
        other_set = FloatCircleSet(other)
        return self.intersection(other_set) == other_set

    def to_tuple(self) -> Tuple[float, float]:
        """Return the circular interval endpoints."""
        return (float(self.start), float(self.end))


FULL_FLOAT_CIRCLE_INTERVAL = FloatCircleInterval(
    0.0,
    FloatAngle.MAX_ANGLE,
)
FULL_FLOAT_CIRCLE_SET = FloatCircleSet(
    FloatInterval(0.0, FloatAngle.MAX_ANGLE)
)

__all__ = [
    "FloatAngle",
    "FloatCirclePoint",
    "FloatCircleInterval",
    "FloatCircleSet",
    "FULL_FLOAT_CIRCLE_INTERVAL",
    "FULL_FLOAT_CIRCLE_SET",
]
