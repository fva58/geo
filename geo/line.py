"""Scalar geometry on the real line."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import math

from .utils import ValidTuple


def _previous_float(value: float) -> float:
    """Return the previous representable float."""
    return math.nextafter(value, -math.inf)


def _intervals_are_separated(left: "Interval", right: "Interval") -> bool:
    """Return whether two ordered intervals are separated on
    the float lattice."""
    return left.right < _previous_float(right.left)


def _is_scalar(value: object) -> bool:
    """Return whether a value should be treated as one line point."""
    return isinstance(value, (float, int, Point))


class Point(float):
    """Point on the real line."""

    __slots__ = ()

    def __new__(cls, value: object = 0.0) -> "Point":
        if isinstance(value, cls):
            return value
        return super().__new__(cls, float(value))

    @property
    def value(self) -> float:
        """Return the underlying scalar value."""
        return float(self)

    def distance_to(self, other: object) -> float:
        """Return the distance to another real-line point."""
        return abs(float(self) - float(Point(other)))

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"Point({float(self)})"


class Interval(tuple):
    """Interval of real numbers on the explicit float lattice."""

    __slots__ = ()

    def __new__(cls, left: object, right: object | None = None) -> "Interval":
        if right is None:
            if isinstance(left, cls):
                return left
            if isinstance(left, ValidTuple) and len(left) == 2:
                right = left[1]
                left = left[0]
            else:
                right = left
        return super().__new__(cls, (float(left), float(right)))

    @property
    def left(self) -> float:
        return self[0]

    @property
    def right(self) -> float:
        return self[1]

    def __repr__(self) -> str:
        return f"Interval({self.left}, {self.right})"

    def __str__(self) -> str:
        if self.is_empty():
            return "∅"
        if self.left == self.right:
            return str(self.left)
        return f"[{self.left}, {self.right}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        if self.is_empty() and other.is_empty():
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        if self.is_empty():
            return hash((float("inf"), float("-inf")))
        return super().__hash__()

    def __bool__(self) -> bool:
        """Return whether the interval is non-empty."""
        return not self.is_empty()

    def is_empty(self) -> bool:
        return self.left > self.right

    def is_full(self) -> bool:
        return math.isinf(self.left) and math.isinf(self.right)

    def is_point(self) -> bool:
        return self.left == self.right

    def contains(self, x: float) -> bool:
        if self.is_empty():
            return False
        return self.left <= x <= self.right

    def __contains__(self, x: float) -> bool:
        return self.contains(x)

    def contains_interval(self, other: "Interval") -> bool:
        if self.is_empty():
            return other.is_empty()
        if other.is_empty():
            return True
        return self.left <= other.left and other.right <= self.right

    def is_subset(self, other: "Interval") -> bool:
        return other.contains_interval(self)

    def length(self) -> float:
        if self.is_empty():
            return -math.inf
        return self.right - self.left

    def intersection(self, other: object) -> "Interval":
        other = Interval(other)
        if self.is_empty() or other.is_empty():
            return EMPTY_INTERVAL
        left = max(self.left, other.left)
        right = min(self.right, other.right)
        if left > right:
            return EMPTY_INTERVAL
        return Interval(left, right)

    def union(self, other: object) -> tuple["Interval", ...]:
        other = Interval(other)
        if self.is_empty():
            return () if other.is_empty() else (other,)
        if other.is_empty():
            return (self,)
        if self.left <= other.left and _intervals_are_separated(self, other):
            return (self, other)
        if other.left < self.left and _intervals_are_separated(other, self):
            return (other, self)
        return (Interval(min(self.left, other.left), max(self.right, other.right)),)

    def difference(self, other: object) -> tuple["Interval", ...]:
        other = Interval(other)
        if self.is_empty() or other.is_empty():
            return (self,) if not self.is_empty() else ()
        inter = self.intersection(other)
        if inter.is_empty():
            return (self,)
        result = []
        if self.left <= math.nextafter(inter.left, -math.inf):
            result.append(Interval(self.left, math.nextafter(inter.left, -math.inf)))
        if inter.right <= math.nextafter(self.right, -math.inf):
            result.append(Interval(math.nextafter(inter.right, math.inf), self.right))
        return tuple(result)

    def symmetric_difference(self, other: object) -> tuple["Interval", ...]:
        other = Interval(other)
        return self.difference(other) + other.difference(self)

    def complement(self) -> tuple["Interval", ...]:
        if self.is_empty():
            return (FULL_INTERVAL,)
        result = []
        if not math.isinf(self.left):
            result.append(Interval(-math.inf, math.nextafter(self.left, -math.inf)))
        if not math.isinf(self.right):
            result.append(Interval(math.nextafter(self.right, math.inf), math.inf))
        return tuple(result)

    def __and__(self, other: object) -> "Interval":
        return self.intersection(other)

    def __or__(self, other: object) -> tuple["Interval", ...]:
        return self.union(other)

    def __sub__(self, other: object) -> tuple["Interval", ...]:
        return self.difference(other)

    def __xor__(self, other: object) -> tuple["Interval", ...]:
        return self.symmetric_difference(other)

    def __invert__(self) -> tuple["Interval", ...]:
        return self.complement()

    @classmethod
    def from_tuple(cls, interval: tuple[float, float]) -> "Interval":
        return cls(interval[0], interval[1])


EMPTY_INTERVAL = Interval(math.inf, -math.inf)
FULL_INTERVAL = Interval(-math.inf, math.inf)
ALL_REALS_INTERVAL = FULL_INTERVAL


class Set(tuple):
    """Set of real numbers represented as disjoint intervals."""

    __slots__ = ()

    def __new__(cls, *args: Any) -> "Set":
        if len(args) == 1:
            if isinstance(args[0], cls):
                return args[0]
            if isinstance(args[0], ValidTuple):
                return super().__new__(cls, args[0])
        return super().__new__(cls, cls._normalize(cls._translate(args)))

    @property
    def intervals(self) -> tuple[tuple[float, float], ...]:
        return self

    def __repr__(self) -> str:
        intervals_repr = ", ".join(repr(iv) for iv in self)
        return f"{self.__class__.__name__}(({intervals_repr}))"

    def __str__(self) -> str:
        if not self:
            return "∅"
        return " ∪ ".join(str(Interval.from_tuple(iv)) for iv in self)

    @classmethod
    def _is_interval_pair(cls, value: object) -> bool:
        return (
            isinstance(value, Sequence) and
            not isinstance(value, (str, bytes, Interval, Set)) and
            len(value) == 2 and
            all(_is_scalar(item) for item in value)
        )

    @classmethod
    def _translate_value(
        cls,
        value: Any,
        seen: set[int],
    ) -> tuple[tuple[float, float], ...]:
        if isinstance(value, Interval):
            return (tuple(value),)
        if isinstance(value, cls):
            return tuple(value)
        if isinstance(value, ValidTuple):
            return tuple(value)
        if _is_scalar(value):
            point = float(value)
            return ((point, point),)
        if cls._is_interval_pair(value):
            return ((float(value[0]), float(value[1])),)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            marker = id(value)
            if marker in seen:
                raise TypeError("Recursive containers are not supported")
            seen.add(marker)
            try:
                translated = []
                for item in value:
                    translated.extend(cls._translate_value(item, seen))
                return tuple(translated)
            finally:
                seen.remove(marker)
        raise TypeError(f"Unsupported line-set argument: {value!r}")

    @classmethod
    def _translate(cls, args: Sequence[Any]) -> tuple[tuple[float, float], ...]:
        translated = []
        seen: set[int] = set()
        for value in args:
            translated.extend(cls._translate_value(value, seen))
        return tuple(translated)

    @staticmethod
    def _normalize(
        intervals: Sequence[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...]:
        non_empty = [
            iv for iv in intervals
            if not Interval.from_tuple(iv).is_empty()
        ]
        if not non_empty:
            return ()
        non_empty.sort()
        return Set.merge(non_empty)

    @staticmethod
    def merge(
        intervals: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...]:
        if not intervals:
            return ()
        result = []
        current = intervals[0]
        for interval in intervals[1:]:
            if current[1] >= _previous_float(interval[0]):
                current = (current[0], max(current[1], interval[1]))
            else:
                result.append(current)
                current = interval
        result.append(current)
        return tuple(result)

    def union(self, other: object) -> "Set":
        return Set([Interval.from_tuple(iv) for iv in list(self) + list(Set(other))])

    def intersection(self, other: object) -> "Set":
        result_intervals = []
        for iv1 in self:
            left = Interval.from_tuple(iv1)
            for iv2 in Set(other):
                right = Interval.from_tuple(iv2)
                inter = left.intersection(right)
                if not inter.is_empty():
                    result_intervals.append(inter)
        return Set(result_intervals)

    def difference(self, other: object) -> "Set":
        other = Set(other)
        if not self:
            return self
        if not other:
            return self
        result_intervals = []
        for iv1 in self:
            current_parts = [Interval.from_tuple(iv1)]
            for iv2 in other:
                right = Interval.from_tuple(iv2)
                new_parts = []
                for part in current_parts:
                    new_parts.extend(part.difference(right))
                current_parts = new_parts
                if not current_parts:
                    break
            result_intervals.extend(current_parts)
        return Set(result_intervals)

    def symmetric_difference(self, other: object) -> "Set":
        union_set = self.union(other)
        inter_set = self.intersection(other)
        return union_set.difference(inter_set)

    def contains(self, x: float) -> bool:
        return any(Interval.from_tuple(interval).contains(x) for interval in self)

    def __contains__(self, x: float) -> bool:
        return self.contains(x)

    def __or__(self, other: object) -> "Set":
        return self.union(other)

    def __and__(self, other: object) -> "Set":
        return self.intersection(other)

    def __sub__(self, other: object) -> "Set":
        return self.difference(other)

    def __xor__(self, other: object) -> "Set":
        return self.symmetric_difference(other)

    def is_empty(self) -> bool:
        return not self

    def contains_interval(self, interval: Interval) -> bool:
        interval = Interval(interval)
        if interval.is_empty():
            return True
        return any(iv[0] <= interval.left and iv[1] >= interval.right for iv in self)

    @classmethod
    def from_single_interval(cls, left: object, right: object) -> "Set":
        return cls(Interval(left, right))

    @classmethod
    def from_intervals(cls, *intervals: Interval) -> "Set":
        return cls(intervals)

    def to_tuple(self) -> tuple[tuple[float, float], ...]:
        return tuple(self)


__all__ = [
    "Point",
    "Interval",
    "Set",
    "EMPTY_INTERVAL",
    "FULL_INTERVAL",
    "ALL_REALS_INTERVAL",
]
