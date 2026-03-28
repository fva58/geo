"""Angles, points, intervals and sets on a circle.

This module provides classes for working with angles, points, intervals,
and sets on a unit circle.
"""

# pylint: disable=multiple-statements

import math
from typing import Tuple, Any, Optional, Union, Sequence, List

from .floatset import FloatInterval , FloatSet

class FloatAngle (float) :
    """Angle on a circle in radians.

    An angle is represented as a float value in the range [0, 2π),
    where the right bound is not included.

    Attributes:
        value: Angle value in radians, normalized to [0, 2π)
    """

    __slots__ = ()

    TWO_PI = 2.0 * math.pi
    # The largest value less than 2π
    MAX_ANGLE = math.nextafter(TWO_PI, -math.inf)


    def __new__ ( cls , angle = 0. ) :
        """Initialize a FloatAngle.

        Args:
            angle: Angle in radians (any real number)

        The angle is normalized to the range [0, 2π).
        """
        if isinstance ( angle , cls ) : return angle
        return super().__new__ ( cls , cls._normalize(angle) )

    def __repr__(self) -> str:
        """Return string representation of the angle."""
        return f"FloatAngle({float(self)})"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"{float(self)} rad"

    def __add__(self, other: Union['FloatAngle', float]) -> 'FloatAngle':
        """Add another angle or float to this angle."""
        return FloatAngle(float(self) + float(other))

    def __sub__(self, other: Union['FloatAngle', float]) -> 'FloatAngle':
        """Subtract another angle or float from this angle."""
        return FloatAngle(float(self) - float(other))

    def __mul__(self, scalar: float) -> 'FloatAngle':
        """Multiply angle by a scalar."""
        return FloatAngle(float(self) * float(scalar))

    def __truediv__(self, scalar: float) -> 'FloatAngle':
        """Divide angle by a scalar."""
        return FloatAngle(float(self) / float(scalar))

    def __neg__(self) -> 'FloatAngle':
        """Return negative of the angle."""
        return FloatAngle(-float(self))

    def __abs__(self) -> 'FloatAngle':
        """Return absolute value of the angle."""
        return self

    @staticmethod
    def _normalize(angle: float) -> float:
        """Normalize angle to [0, 2π).

        Args:
            angle: Angle in radians

        Returns:
            Angle normalized to [0, 2π)
        """
        # Use math.fmod for better numerical stability
        normalized = math.fmod(float(angle), FloatAngle.TWO_PI)
        if normalized < 0:
            normalized += FloatAngle.TWO_PI

        # Handle edge case: if normalized is exactly 2π, return 0
        # Using nextafter to check if we're at or above 2π
        if normalized >= FloatAngle.TWO_PI:
            return 0.0

        # Ensure we don't exceed MAX_ANGLE
        if normalized > FloatAngle.MAX_ANGLE:
            return 0.0

        return normalized

    def to_degrees(self) -> float:
        """Convert angle to degrees.

        Returns:
            Angle in degrees
        """
        return math.degrees(self)

    @classmethod
    def from_degrees(cls, degrees: float) -> 'FloatAngle':
        """Create FloatAngle from degrees.

        Args:
            degrees: Angle in degrees

        Returns:
            FloatAngle object
        """
        return cls(math.radians(degrees))

    def distance_to(self, other: 'FloatAngle') -> 'FloatAngle':
        """Calculate the shortest distance (angular difference) to
        another angle.

        Args:
            other: Another angle

        Returns:
            Shortest angular distance in [0, π]

        """
        diff = abs(float(self) - float(other))
        if diff > math.pi:
            diff = FloatAngle.TWO_PI - diff
        return FloatAngle(diff)

    def is_close(self, other: 'FloatAngle', abs_tol: float = 1e-12) -> bool:
        """Check if this angle is close to another angle.

        Args:
            other: Another angle
            abs_tol: Absolute tolerance

        Returns:
            True if angles are close within tolerance
        """
        diff = abs(float(self) - float(other))
        # Handle wrap-around
        if diff > math.pi:
            diff = FloatAngle.TWO_PI - diff
        return diff <= abs_tol

    def opposite(self) -> 'FloatAngle':
        """Return the opposite angle (rotated by π).

        Returns:
            Angle opposite to this one
        """
        return FloatAngle(float(self) + math.pi)

    def complement(self) -> 'FloatAngle':
        """Return the complementary angle (to π/2).

        Returns:
            Complementary angle (π/2 - angle)
        """
        return FloatAngle(math.pi / 2 - float(self))

    def supplement(self) -> 'FloatAngle':
        """Return the supplementary angle (to π).

        Returns:
            Supplementary angle (π - angle)
        """
        return FloatAngle(math.pi - float(self))


class FloatCirclePoint ( FloatAngle ) :
    """Point on a unit circle.

    A point is represented by its angle on the circle.

    Attributes:
        angle: Angle of the point on the circle
    """
    __slots__ = ()

    def __new__ ( cls , angle = 0. ) :
        """Initialize a FloatAngle.

        Args:
            angle: Angle in radians (any real number)

        The angle is normalized to the range [0, 2π).
        """
        if isinstance ( angle , cls ) : return angle
        return super().__new__ ( cls , angle )

    def __repr__(self) -> str:
        """Return string representation of the point."""
        return f"FloatCirclePoint({float(self)})"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"Point at {float(self)}"

    def rotate(self, rotation_angle: Union[FloatAngle, float]
               ) -> 'FloatCirclePoint':
        """Rotate the point by given angle.

        Args:
            rotation_angle: Angle to rotate by (positive = counterclockwise)

        Returns:
            New rotated point
        """
        return FloatCirclePoint(float(self)+float(rotation_angle))

    def opposite(self) -> 'FloatCirclePoint':
        """Return the opposite point (rotated by π).

        Returns:
            Point opposite to this one
        """
        return FloatCirclePoint(FloatAngle(self).opposite())

    def to_cartesian(self) -> Tuple[float, float]:
        """Convert point to Cartesian coordinates on unit circle.

        Returns:
            Tuple (x, y) coordinates
        """
        return (math.cos(self), math.sin(self))

    @classmethod
    def from_cartesian(cls, x: float, y: float) -> 'FloatCirclePoint':
        """Create point from Cartesian coordinates.

        Args:
            x: x-coordinate
            y: y-coordinate

        Returns:
            FloatCirclePoint object

        Raises:
            ValueError: If x and y is zero.
        """
        norm_sq = x*x + y*y
        if x == 0. and y == 0. :
            raise ValueError(f"Point ({x}, {y}) is zero")

        angle = math.atan2(y, x)
        return cls(angle)

    @property
    def x(self) -> float:
        """Get x-coordinate of the point."""
        return math.cos(self)

    @property
    def y(self) -> float:
        """Get y-coordinate of the point."""
        return math.sin(self)


class FloatCircleInterval ( FloatInterval ) :
    """Interval (arc) on a unit circle.

    An interval is represented by interval of floats — two points on
    the circle, defining an arc from the first point to the second
    point moving counterclockwise.

    """

    __slots__ = ()

    def __new__ ( cls , start , end=None ) :
        """Initialize a FloatCircleInterval.

        Args:
            start: Starting point or angle of the interval. May be
                   FloatInterval , FloatCircleInterval or just
                   something convertible to float.
            end: Ending point or angle of the interval
        """

        if end is None : # may be FloatInterval , FloatCircleInterval
                         # or just convertible to float
            if isinstance ( start , cls ) return start
            if isinstance ( start , FloatInterval ) : start , end = start
            else : start = end = float(start)

        if abs ( float(end) - float(start) ) >= FloatAngle.MAX_ANGLE :
            return super(FloatInterval).__new__ (
                cls , ( 0. , FloatAngle.MAX_ANGLE ) )
        return super(FloatInterval).__new__ (
            cls , ( FloatAngle(start) , FloatAngle(end) ) )

    def __repr__(self) -> str:
        """Return string representation of the interval."""
        return f"FloatCircleInterval({self[0]}, {self[1]})"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"[{self[0]}, {self[1]}]"


    # __eq__ from FloatInterval
    # __hash__ from FloatInterval

    def is_empty(self) -> bool:
        """Check if the interval is empty.

        Returns:
            False
        """
        return False

    def is_full(self) -> bool:
        """Check if the interval covers the full circle.

        Returns:
            True if interval covers all points.
            Full circle is represented as interval from a to nextafter(a, -inf).
        """
        if self[0] == self[1]:
            return False  # Point interval is not full circle

        # Check if end is nextafter(start, -inf)
        # Calculate the value just before start (wrapping around)
        expected_end = math.nextafter(self[0], -math.inf)
        if expected_end < 0:
            expected_end = FloatAngle.MAX_ANGLE

        # Check if end equals expected_end
        return self[1] == expected_end

    # is_point from FloatInterval

    def contains(self, point: Union[FloatCirclePoint, FloatAngle, float]
                     ) -> bool:
        """Check if a point is in the interval.

        Args:
            point: Point to check

        Returns:
            True if point is in the interval (including boundaries)
        """
        if not isinstance(point, FloatCirclePoint):
            point = FloatCirclePoint(point)

        # Handle full circle interval
        if self.is_full():
            return True

        if self[0] <= self[1] : return self[0] <= point <= self[1]
        return point >= self[0] or point <= self[1]

    def contains_interval(self, other: 'FloatCircleInterval') -> bool:
        """Check if this interval contains another interval.

        Args:
            other: Another interval

        Returns:
            True if other ⊆ self
        """
        if self[0] <= self[1] :
            return self[0] <= other[0] <= other[1] <= self[1]
        return other[0] >= self[0] and (
            other[1] >= other[0] or other[1] <= self[1] ) or \
            other[0] <= other[1] <= self[1]

    # is_subset from FloatInterval
    def length(self) -> FloatAngle:
        """Return length (angular measure) of the interval.

        Returns:
            Length of the interval in radians
        """
        if self[1] >= self[0] :
            # Simple case: doesn't wrap around
            return FloatAngle ( super().length() )
        else:
            # Wraps around 0
            return FloatAngle( FloatAngle.TWO_PI - self[0] + self[1] )

    def intersection(self, other: 'FloatCircleInterval'
                     ) -> Tuple['FloatCircleInterval', ...]:
        """Compute intersection of two intervals.

        Args:
            other: Another interval

        Returns:
            Tuple of intervals representing the intersection (0, 1 or
            2 intervals)
        """

        if self.is_full():
            return other
        if other.is_full():
            return self

        s = self.to_floatset ()
        o = other.to_floatset ()

        r = s.intersection ( o )
        return self.from_floatset ( r )

    def union(self, other: 'FloatCircleInterval'
              ) -> Tuple['FloatCircleInterval', ...]:
        """Compute union of two intervals.

        Args:
            other: Another interval

        Returns:
            Tuple of intervals representing the union (1 or 2 intervals)
        """
        if self.is_full() or other.is_full():
            return ( FULL_FLOAT_CIRCLE_INTERVAL , )

        s = self.to_floatset ()
        o = other.to_floatset ()

        r = s.union ( o )
        return self.from_floatset ( r )

    def difference(self, other: 'FloatCircleInterval'
              ) -> Tuple['FloatCircleInterval', ...]:
        """Compute difference of two intervals.

        Args:
            other: Another interval

        Returns:
            Tuple of intervals representing the difference (0, 1 or 2 intervals)
        """
        s = self.to_floatset ()
        o = other.to_floatset ()

        r = s.difference ( o )
        return self.from_floatset ( r )

    def symmetric_difference(self, other: 'FloatCircleInterval'
              ) -> Tuple['FloatCircleInterval', ...]:
        """Compute symmetric_difference of two intervals.

        Args:
            other: Another interval

        Returns:
            Tuple of intervals representing the symmetric_difference
        """
        s = self.to_floatset ()
        o = other.to_floatset ()

        r = s.symmetric_difference ( o )
        return self.from_floatset ( r )

    def complement(self) -> Optional['FloatCircleInterval'] :
        """Return the complement of the interval (the rest of the circle).

        Returns:
            Complement interval or None if it is a full circle
        """
        if self.is_full(): return None

        start = nextafter ( self[1] , math.inf )
        if start == FloatAngle.TWO_PI : start = 0.
        end = nextafter ( self[0] , -math.inf )
        if end < 0. : end = FloatAngle.MAX_ANGLE
        return FloatCircleInterval(start,end)

    # __and__ from FloatInterval
    # __or__ from FloatInterval
    # __sub__ from FloatInterval
    # __xor__ from FloatInterval
    # __contains__ from  FloatInterval
    # __bool__ from  FloatInterval
    # __invert__ from FloatInterval
    # from_tuple from  FloatInterval
    # to_tuple from  FloatInterval

    def to_floatset ( self ) -> FloatSet :
        """Convert to corresponding FloatSet"""
        if self.is_full () :
            return FloatSet ( FULL_FLOAT_CIRCLE_INTERVAL )
        return FloatSet(self) if ss <= se else FloatSet (
            FloatInterval ( 0. , self[1] ) ,
            FloatInterval ( self[0] , FloatAngle.MAX_ANGLE ) )

    def from_floatset ( self , s : FloatSet
                       ) -> Tuple[FloatCircleInterval,...] :
        """Close loop at end if exists"""
        # closing interval must be at last position
        if len(s) > 1 and s[0][0] == 0. and s[-1][1] == FloatAngle.MAX_ANGLE :
            # close circle
            s = tuple(r[1:-1]) + ( (s[-1][0] , s[0][1]) , )
        return tuple ( FloatCircleInterval(ss,ee) for ss,ee in s )

FULL_FLOAT_CIRCLE_INTERVAL = FloatCircleInterval ( 0. , FloatAngle.MAX_ANGLE )


class FloatCircleSet ( FloatSet ) :
    """Set of points on a unit circle.

    A set is represented as an ordered collection of non-overlapping
    intervals on the circle.

    Is like FloatSet but :
      1. All interval boundaries in range [0, 2π)
      2. Last interval can wrap zero ( start > end )

    """
    __slots__ = ()

    # __new__ from FloatSet
    # intervals from FloatSet

    def __repr__(self) -> str:
        """Return string representation of the set."""
        intervals_repr = ", ".join(repr(iv) for iv in self)
        return f"FloatCircleSet(({intervals_repr}))"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        if not self :
            return "∅"
        return " ∪ ".join(str(iv) for iv in self)

    # _translate from FloatSet

    @staticmethod
    def _normalize(intervals: List[FloatCircleInterval]
                   ) -> Tuple[FloatCircleInterval, ...]:
        """Normalize intervals: sort and merge overlapping intervals.

        Args:
            intervals: Collection of intervals

        Returns:
            Tuple of sorted, non-overlapping intervals
        """
        if not intervals: return ()

        iv = [ ]

        for i in intervals :
            i = FloatCircleInterval.from_tuple ( i )
            iv.extend ( i.to_floatset() )

        iv.sort ()
        res = self.merge ( iv ) # from FloatSet
        if res[0][0] == 0. and res[-1][1] == FloatAngle.MAX_ANGLE :
            res = res[1:-1] + ( res[-1][0] , res[0][1] )
        return res

    def union(self, other: 'FloatCircleSet') -> 'FloatCircleSet':
        """Compute union of two sets.

        Args:
            other: Another set

        Returns:
            Union of the sets
        """
        all_intervals = list(self) + list(other)
        return FloatCircleSet ([ FloatInterval.from_tuple(i)
                                 for i in all_intervals ])

    def intersection(self, other: 'FloatCircleSet') -> 'FloatCircleSet':
        """Compute intersection of two sets.

        Args:
            other: Another set

        Returns:
            Intersection of the sets
        """
        result_intervals = []
        for iv1 in self:
            iv1 = FloatCircleInterval.from_tuple ( iv1 )
            for iv2 in other._intervals:
                iv2 = FloatCircleInterval.from_tuple ( iv2 )
                inters = iv1.intersection(iv2)
                result_intervals.extend(inters)

        return FloatCircleSet ( result_intervals )








    def __eq__(self, other: object) -> bool:
        """Check if two sets are equal."""
        if not isinstance(other, FloatCircleSet):
            return NotImplemented
        return self._intervals == other._intervals

    def __hash__(self) -> int:
        """Return hash of the set."""
        return hash(self._intervals)

    def __contains__(self, point: Union[FloatCirclePoint, FloatAngle, float]
                     ) -> bool:
        """Check if a point is in the set.

        Args:
            point: Point to check

        Returns:
            True if point is in the set
        """
        if not isinstance(point, FloatCirclePoint):
            point = FloatCirclePoint(point)

        for interval in self._intervals:
            if point in interval:
                return True
        return False

    def __bool__(self) -> bool:
        """Boolean conversion: True if set is not empty."""
        return bool(self._intervals)

    def __len__(self) -> int:
        """Return number of intervals in the set."""
        return len(self._intervals)

    def __iter__(self):
        """Iterate over intervals in the set."""
        return iter(self._intervals)

    def __and__(self, other: 'FloatCircleSet') -> 'FloatCircleSet':
        """Operator for intersection: self & other."""
        return self.intersection(other)

    def __or__(self, other: 'FloatCircleSet') -> 'FloatCircleSet':
        """Operator for union: self | other."""
        return self.union(other)

    def __sub__(self, other: 'FloatCircleSet') -> 'FloatCircleSet':
        """Operator for difference: self - other."""
        return self.difference(other)

    def __xor__(self, other: 'FloatCircleSet') -> 'FloatCircleSet':
        """Operator for symmetric difference: self ^ other."""
        return self.symmetric_difference(other)

    def __invert__(self) -> 'FloatCircleSet':
        """Operator for complement: ~self."""
        return self.complement()


    def is_empty(self) -> bool:
        """Check if the set is empty.

        Returns:
            True if set has no intervals, False otherwise
        """
        return not self._intervals

    def is_full(self) -> bool:
        """Check if the set covers the full circle.

        Returns:
            True if set is the full circle, False otherwise
        """
        if len(self._intervals) != 1:
            return False
        return self._intervals[0].is_full()

    def difference(self, other: 'FloatCircleSet') -> 'FloatCircleSet':
        """Compute difference (self - other).

        Args:
            other: Set to subtract

        Returns:
            Difference of the sets
        """
        if not self._intervals:
            return self
        if not other._intervals:
            return self

        result_intervals = []

        for iv1 in self._intervals:
            # Start with the current interval
            current_parts = [iv1]

            # Subtract each interval from other set
            for iv2 in other._intervals:
                new_parts = []
                for part in current_parts:
                    # Compute difference for circular intervals
                    inter = part.intersection(iv2)
                    # Check if intersection is just a point (start == end)
                    if inter.start == inter.end:
                        # Intersection is just a point, keep the whole part
                        new_parts.append(part)
                    elif inter == part:
                        # Complete overlap, remove the part
                        continue
                    else:
                        # Partial overlap, need to compute difference
                        # For circular intervals, difference can be 0, 1, or 2 intervals
                        if part.start in inter and part.end in inter:
                            # Both boundaries in intersection, part is completely covered
                            continue
                        elif part.start in inter:
                            # Start in intersection, end not in intersection
                            new_parts.append(FloatCircleInterval(inter.end, part.end))
                        elif part.end in inter:
                            # End in intersection, start not in intersection
                            new_parts.append(FloatCircleInterval(part.start, inter.start))
                        else:
                            # Neither boundary in intersection, intersection is in the middle
                            # Result is two intervals
                            new_parts.append(FloatCircleInterval(part.start, inter.start))
                            new_parts.append(FloatCircleInterval(inter.end, part.end))

                current_parts = new_parts
                if not current_parts:
                    break

            result_intervals.extend(current_parts)

        return FloatCircleSet(result_intervals)

    def symmetric_difference(self, other: 'FloatCircleSet') -> 'FloatCircleSet':
        """Compute symmetric difference (self Δ other).

        Args:
            other: Another set

        Returns:
            Symmetric difference of the sets
        """
        union_set = self.union(other)
        inter_set = self.intersection(other)
        return union_set.difference(inter_set)

    def complement(self) -> 'FloatCircleSet':
        """Compute complement of the set (the rest of the circle).

        Returns:
            Complement of the set
        """
        if self.is_empty():
            # Complement of empty set is full circle
            return FloatCircleSet(FloatCircleInterval(0.0, 0.0))

        if self.is_full():
            # Complement of full circle is empty set
            return FloatCircleSet()

        # Complement is union of complements of each interval
        result_intervals = []
        for interval in self._intervals:
            complement_interval = interval.complement()
            # Check if complement is not just a point
            if complement_interval.start != complement_interval.end:
                result_intervals.append(complement_interval)

        # Merge overlapping complement intervals
        return FloatCircleSet(result_intervals)

    def contains_interval(self, interval: FloatCircleInterval) -> bool:
        """Check if the set contains an entire interval.

        Args:
            interval: Interval to check

        Returns:
            True if interval is subset of the set, False otherwise
        """
        # Check if interval is contained in any single interval of the set
        for iv in self._intervals:
            if interval.is_subset(iv):
                return True

        # For circular sets, an interval might be split across set boundaries
        # We need to check if the interval can be covered by union of set intervals
        # This is more complex for circular case
        remaining = interval
        for iv in self._intervals:
            inter = remaining.intersection(iv)
            # Check if intersection is not just a point
            if inter.start != inter.end:
                # Subtract the intersection from remaining
                # For circular intervals, we need careful difference computation
                if inter == remaining:
                    return True
                # Update remaining to be the part not yet covered
                # This is simplified - full implementation would need
                # proper circular difference
                if remaining.start in inter:
                    remaining = FloatCircleInterval(inter.end, remaining.end)
                elif remaining.end in inter:
                    remaining = FloatCircleInterval(remaining.start, inter.start)

        # Check if remaining is just a point or effectively covered
        return remaining.start == remaining.end

    @property
    def intervals(self) -> Tuple[FloatCircleInterval, ...]:
        """Get intervals of the set.

        Returns:
            Tuple of intervals in the set
        """
        return self._intervals

    @classmethod
    def from_single_interval(cls, start: Union[FloatCirclePoint, FloatAngle, float],
                             end: Union[FloatCirclePoint, FloatAngle, float]) -> 'FloatCircleSet':
        """Create set from single interval.

        Args:
            start: Start of interval
            end: End of interval

        Returns:
            FloatCircleSet containing the interval
        """
        interval = FloatCircleInterval(start, end)
        return cls(interval)

    @classmethod
    def from_intervals(cls, *intervals: FloatCircleInterval) -> 'FloatCircleSet':
        """Create set from multiple intervals.

        Args:
            intervals: Intervals to include

        Returns:
            FloatCircleSet containing the intervals
        """
        return cls(intervals)

    def to_angles(self) -> Tuple[Tuple[float, float], ...]:
        """Convert set to tuple of angle tuples.

        Returns:
            Tuple of (start_angle, end_angle) tuples
        """
        return tuple((iv.start.angle.value, iv.end.angle.value) for iv in self._intervals)
