"""Floating-point sets and intervals.

This module implements set operations on the discrete lattice of Python
``float`` values. Endpoints are therefore adjusted with ``math.nextafter``
when an operation must exclude a boundary point while still returning closed
intervals.
"""

# pylint: disable=multiple-statements

from typing import Tuple, Any, Sequence, List
import math

from .utils import ValidTuple

class FloatInterval ( tuple ) :
    """Interval of real numbers.

    An interval is represented as a tuple of two numbers (left and
    right bounds).  An empty interval has left bound greater than
    right bound.

    Properties:
        left: Left bound of the interval
        right: Right bound of the interval

    """
    __slots__ = ()

    def __new__ ( cls , left , right=None ) :
        """Initialize a FloatInterval.

        Args:
            left: Left bound of the interval or FloatInterval
            right: Right bound of the interval
        """
        if right is None :
            if isinstance ( left , cls ) : return left
            right = left
        return super().__new__ ( cls , (float(left),float(right)) )

    @property
    def left ( self ) -> float : return self[0]

    @property
    def right ( self ) -> float : return self[1]

    def __repr__(self) -> str:
        """Return string representation of the interval."""
        return f"FloatInterval({self.left}, {self.right})"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        if self.is_empty():
            return "∅"
        if self.left == self.right : return str ( self.left )
        return f"[{self.left}, {self.right}]"

    def __eq__(self, other: object) -> bool:
        """Check if two intervals are equal."""
        if not isinstance(other, FloatInterval):
            return NotImplemented
        if self.is_empty() and other.is_empty():
            return True
        return super().__eq__ ( other )

    def __hash__(self) -> int:
        """Return hash of the interval."""
        if self.is_empty():
            return hash((float('inf'), float('-inf')))
        return super().__hash__ ()

    def is_empty(self) -> bool:
        """Check if the interval is empty.

        Returns:
            True if left > right, False otherwise.
        """
        return self.left > self.right

    def is_full(self) -> bool:
        """Check if the interval is full.

        Returns:
            True if interval is full float set, False otherwise.
        """
        return math.isinf(self.left) and math.isinf(self.right)

    def is_point(self) -> bool:
        """Check if the interval is single point.

        Returns:
            True if interval is single point, False otherwise.
        """
        return self.left == self.right

    def contains(self, x: float) -> bool:
        """Check if the interval contains a point.

        Args:
            x: Point to check

        Returns:
            True if x is in the interval, False otherwise.
        """
        if self.is_empty():
            return False
        return self.left <= x <= self.right

    def contains_interval(self, other: 'FloatInterval') -> bool:
        """Check if this interval contains another interval.

        Args:
            other: Another interval

        Returns:
            True if other ⊆ self
        """
        if self.is_empty() : return other.is_empty ()
        if other.is_empty() : return True
        return self.left <= other.left and other.right <= self.right


    def is_subset(self, other: 'FloatInterval') -> bool:
        """Check if this interval is subset of another interval.

        Args:
            other: Another interval

        Returns:
            True if self ⊆ other, False otherwise.
        """
        return other.contains_interval(self)

    def length(self) -> float:
        """Return length of the interval.

        Returns:
            Length of the interval (-inf for empty interval).
        """
        if self.is_empty():
            return -math.inf
        return self.right - self.left

    def intersection(self, other: 'FloatInterval') -> 'FloatInterval':
        """Compute intersection of two intervals.

        Args:
            other: Another interval

        Returns:
            Intersection interval (may be empty).
        """
        if self.is_empty() or other.is_empty() : return EMPTY_FLOAT_INTERVAL

        left = max(self.left, other.left)
        right = min(self.right, other.right)

        if left > right: return EMPTY_FLOAT_INTERVAL
        return FloatInterval(left, right)

    def union(self, other: 'FloatInterval') -> Tuple['FloatInterval', ...]:
        """Compute union of two intervals.

        Args:
            other: Another interval

        Returns: Tuple of intervals representing the union (0, 1 or 2
            intervals ordered).

        """
        if self.is_empty():
            if other.is_empty():
                return ()
            return (other,)
        if other.is_empty():
            return (self,)

        # Check if intervals overlap or touch
        # Use nextafter to handle floating point precision
        # Intervals are disjoint if right of one is less than left of the other
        # Using <= with nextafter to handle exact comparisons
        if (self.right <= math.nextafter(other.left, -math.inf) or
            other.right <= math.nextafter(self.left, -math.inf)):
            # Disjoint intervals
            if self.left < other.left: return (self, other)
            return (other, self)

        # Overlapping or touching intervals
        left = min(self.left, other.left)
        right = max(self.right, other.right)
        return (FloatInterval(left, right),)

    def difference(self, other: 'FloatInterval') -> Tuple['FloatInterval', ...]:
        """Compute difference (self - other).

        Args:
            other: Interval to subtract

        Returns: Tuple of intervals representing the difference (0, 1,
            or 2 intervals ordered).

        """
        if self.is_empty() or other.is_empty():
            return (self,) if not self.is_empty() else ()

        inter = self.intersection(other)
        if inter.is_empty():
            return (self,)

        result = []
        # In the float model we keep closed intervals and exclude removed
        # boundary points by stepping to the neighboring representable float.
        if self.left <= math.nextafter(inter.left, -math.inf):
            result.append(
                FloatInterval(self.left, math.nextafter(inter.left, -math.inf))
            )

        if inter.right <= math.nextafter(self.right, -math.inf):
            result.append(
                FloatInterval(math.nextafter(inter.right, math.inf), self.right)
            )

        return tuple(result)

    def symmetric_difference(self, other: 'FloatInterval'
                             ) -> Tuple['FloatInterval', ...]:
        """Compute symmetric difference (self Δ other).

        Args:
            other: Another interval

        Returns:
            Tuple of intervals representing the symmetric difference.
        """
        union_intervals = self.union(other)
        inter = self.intersection(other)

        if inter.is_empty():
            return union_intervals

        # For overlapping intervals, symmetric difference is union
        # minus intersection
        result = []
        for interval in union_intervals:
            diff = interval.difference(inter)
            result.extend(diff)
        return tuple(result)

    def complement(self) -> Tuple['FloatInterval',...] :
        """Return the complement of the interval (the rest of the set).

        Returns:
            Complement intervals (0, 1 or 2) of the set.
        """
        res = []
        if not math.isinf ( self[0] ) :
            res.append(
                FloatInterval(-math.inf, math.nextafter(self[0], -math.inf))
            )
        if not math.isinf ( self[1] ) :
            res.append(
                FloatInterval(math.nextafter(self[1], math.inf), math.inf)
            )
        return tuple ( res )

    def __and__(self, other: 'FloatInterval') -> 'FloatInterval':
        """Operator for intersection: self & other."""
        return self.intersection(other)

    def __or__(self, other: 'FloatInterval') -> Tuple['FloatInterval', ...]:
        """Operator for union: self | other."""
        return self.union(other)

    def __sub__(self, other: 'FloatInterval') -> Tuple['FloatInterval', ...]:
        """Operator for difference: self - other."""
        return self.difference(other)

    def __xor__(self, other: 'FloatInterval') -> Tuple['FloatInterval', ...]:
        """Operator for symmetric difference: self ^ other."""
        return self.symmetric_difference(other)

    def __contains__(self, x: float) -> bool:
        """Check if point is in interval: x in interval."""
        return self.contains(x)

    def __bool__(self) -> bool:
        """Boolean conversion: True if interval is not empty."""
        return not self.is_empty()

    def __invert__(self) -> Tuple['FloatInterval',...] :
        """Operator for complement: ~self."""
        return self.complement()

    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> 'FloatInterval':
        """Create interval from tuple."""
        return cls(t[0], t[1])

    def to_tuple(self) -> Tuple[float, float]:
        """Convert interval to tuple."""
        return tuple ( self )


EMPTY_FLOAT_INTERVAL = FloatInterval ( math.inf , -math.inf )
assert EMPTY_FLOAT_INTERVAL.is_empty ()
FULL_FLOAT_INTERVAL = FloatInterval ( -math.inf , math.inf )
assert not FULL_FLOAT_INTERVAL.is_empty ()
ALL_FLOATS_INTERVAL = FULL_FLOAT_INTERVAL


class FloatSet ( tuple ) : # Tuple[Tuple[float,float],...]
    """Set of real numbers represented as disjoint intervals.

    A FloatSet is represented as a tuple of sorted, non-overlapping intervals.
    """
    __slots__ = ()

    def __new__ ( cls , *args : Any ) :
        if len(args) == 1 :
            if isinstance ( args[0] , cls ) : return args[0]
            if isinstance ( args[0] , ValidTuple ) :
                return super().__new__ ( cls , args[0] )
        return super().__new__ ( cls , cls._normalize(cls._translate(args)) )

    @property
    def intervals(self) -> Tuple[Tuple[float,float], ...] :
        """Get intervals of the set."""
        return self

    def __repr__(self) -> str:
        """Return string representation of the set."""
        intervals_repr = ", ".join(repr(iv) for iv in self)
        return f"{self.__class__.__name__}(({intervals_repr}))"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        if not self:
            return "∅"
        return " ∪ ".join(str(iv) for iv in self)

    @staticmethod
    def _translate ( args ) :
        for a in args :
            if isinstance ( a , FloatInterval ) : yield tuple(a)
            elif isinstance ( a , float ) : yield (a,a)
            elif isinstance ( a , int ) : yield (float(a),float(a))
            else : yield from FloatSet._translate ( a )

    @staticmethod
    def _normalize ( intervals: Sequence[Tuple[float,float]]
                    ) -> Tuple[Tuple[float,float], ...] :
        """Normalize intervals: sort and merge overlapping intervals.

        Args:
            intervals: Collection of intervals

        Returns:
            Tuple of sorted, non-overlapping intervals.
        """
        # Filter out empty intervals
        non_empty = [ iv for iv in intervals
                      if not FloatInterval.from_tuple(iv).is_empty() ]
        if not non_empty:
            return ()

        # Sort by left bound
        non_empty.sort()
        return FloatSet.merge ( non_empty )

    @staticmethod
    def merge ( intervals : List[Tuple[float,float]]
               ) -> Tuple[Tuple[float,float],...] :
        """Merge sorted overlapping intervals"""
        if not intervals : return ()
        result = []
        current = intervals[0]

        for interval in intervals[1:] :
            # Check if intervals overlap or touch
            if current[1] > math.nextafter ( interval[0] , -math.inf ) :
                # Merge intervals
                current = ( current[0] , max(current[1],interval[1]) )
            else:
                # No overlap, add current to result
                result.append(current)
                current = interval

        result.append(current)
        return tuple(result)

    # pylint: disable=protected-access
    def union(self, other: 'FloatSet') -> 'FloatSet':
        """Compute union of two sets.

        Args:
            other: Another set

        Returns:
            Union of the sets.
        """
        all_intervals = list(self) + list(other)
        return FloatSet([ FloatInterval.from_tuple(i) for i in all_intervals ])

    def intersection(self, other: 'FloatSet') -> 'FloatSet':
        """Compute intersection of two sets.

        Args:
            other: Another set

        Returns:
            Intersection of the sets.
        """
        result_intervals = []
        for iv1 in self:
            iv1 = FloatInterval.from_tuple ( iv1 )
            for iv2 in other:
                iv2 = FloatInterval.from_tuple ( iv2 )
                inter = iv1.intersection(iv2)
                if not inter.is_empty():
                    result_intervals.append(inter)

        return FloatSet ( result_intervals )

    def difference(self, other: 'FloatSet') -> 'FloatSet':
        """Compute difference (self - other).

        Args:
            other: Set to subtract

        Returns:
            Difference of the sets.
        """
        if not self:
            return self
        if not other:
            return self

        result_intervals = []

        for iv1 in self:
            # Start with the current interval
            iv1 = FloatInterval.from_tuple ( iv1 )
            current_parts = [iv1]

            # Subtract each interval from other set
            for iv2 in other:
                iv2 = FloatInterval.from_tuple ( iv2 )
                new_parts = []
                for part in current_parts:
                    diff = part.difference(iv2)
                    new_parts.extend(diff)
                current_parts = new_parts
                if not current_parts:
                    break

            result_intervals.extend(current_parts)
        return FloatSet(result_intervals)

    def symmetric_difference(self, other: 'FloatSet') -> 'FloatSet':
        """Compute symmetric difference (self Δ other).

        Args:
            other: Another set

        Returns:
            Symmetric difference of the sets.
        """
        union_set = self.union(other)
        inter_set = self.intersection(other)
        return union_set.difference(inter_set)

    def contains (self, x: float) -> bool:
        """Check if point is in set: x in set."""
        for interval in self:
            if FloatInterval.from_tuple(interval).contains(x) :
                return True
        return False

    def __contains__(self, x: float) -> bool:
        """Check if point is in set: x in set."""
        return self.contains ( x )

    def __or__(self, other: 'FloatSet') -> 'FloatSet':
        """Operator for union: self | other."""
        return self.union(other)

    def __and__(self, other: 'FloatSet') -> 'FloatSet':
        """Operator for intersection: self & other."""
        return self.intersection(other)

    def __sub__(self, other: 'FloatSet') -> 'FloatSet':
        """Operator for difference: self - other."""
        return self.difference(other)

    def __xor__(self, other: 'FloatSet') -> 'FloatSet':
        """Operator for symmetric difference: self ^ other."""
        return self.symmetric_difference(other)

    def is_empty(self) -> bool:
        """Check if the set is empty.

        Returns:
            True if set has no intervals, False otherwise.
        """
        return not self

    def contains_interval(self, interval: FloatInterval) -> bool:
        """Check if the set contains an entire interval.

        Args:
            interval: Interval to check

        Returns:
            True if interval is subset of the set, False otherwise.
        """
        if interval.is_empty():
            return True

        for iv in self:
            if iv[0] <= interval.left and iv[1] >= interval.right:
                return True
        return False

    @classmethod
    def from_single_interval(cls, left: float, right: float) -> 'FloatSet':
        """Create set from single interval."""
        return cls ( FloatInterval(left, right) )

    @classmethod
    def from_intervals(cls, *intervals: FloatInterval) -> 'FloatSet':
        """Create set from multiple intervals."""
        return cls(intervals)

    def to_tuple(self) -> Tuple[Tuple[float, float], ...]:
        """Convert set to tuple of interval tuples."""
        return tuple(self)
