#!/usr/bin/env python3
"""Test script for floatcircle module."""

import math
from geo.floatcircle import (
    FloatAngle,
    FloatCirclePoint,
    FloatCircleInterval,
    FloatCircleSet
)


def test_float_angle():
    """Test FloatAngle class."""
    print("Testing FloatAngle...")
    
    # Test normalization
    a1 = FloatAngle(0.0)
    assert a1.value == 0.0, f"Expected 0.0, got {a1.value}"
    
    a2 = FloatAngle(math.pi)
    assert abs(a2.value - math.pi) < 1e-12, f"Expected π, got {a2.value}"
    
    a3 = FloatAngle(3 * math.pi)
    assert abs(a3.value - math.pi) < 1e-12, f"Expected π for 3π, got {a3.value}"
    
    a4 = FloatAngle(-math.pi)
    assert abs(a4.value - math.pi) < 1e-12, f"Expected π for -π, got {a4.value}"
    
    a5 = FloatAngle(2 * math.pi)
    assert abs(a5.value - 0.0) < 1e-12, f"Expected 0 for 2π, got {a5.value}"
    
    a6 = FloatAngle(2 * math.pi + 0.1)
    assert abs(a6.value - 0.1) < 1e-12, f"Expected 0.1 for 2π+0.1, got {a6.value}"
    
    # Test operations
    a7 = FloatAngle(math.pi / 2)
    a8 = FloatAngle(math.pi / 2)
    assert a7 == a8, "Angles should be equal"
    
    a9 = a7 + a8
    assert abs(a9.value - math.pi) < 1e-12, f"Expected π for π/2 + π/2, got {a9.value}"
    
    a10 = a7 - FloatAngle(math.pi / 4)
    assert abs(a10.value - math.pi / 4) < 1e-12, f"Expected π/4, got {a10.value}"
    
    # Test distance
    a11 = FloatAngle(0.0)
    a12 = FloatAngle(math.pi)
    dist = a11.distance_to(a12)
    assert abs(dist.value - math.pi) < 1e-12, f"Expected π distance, got {dist.value}"
    
    a13 = FloatAngle(0.1)
    a14 = FloatAngle(2 * math.pi - 0.1)
    dist2 = a13.distance_to(a14)
    assert abs(dist2.value - 0.2) < 1e-12, f"Expected 0.2 distance, got {dist2.value}"
    
    print("FloatAngle tests passed!")


def test_float_circle_point():
    """Test FloatCirclePoint class."""
    print("\nTesting FloatCirclePoint...")
    
    # Test creation
    p1 = FloatCirclePoint(0.0)
    assert p1.angle.value == 0.0, f"Expected angle 0.0, got {p1.angle.value}"
    
    p2 = FloatCirclePoint(math.pi)
    assert abs(p2.angle.value - math.pi) < 1e-12, f"Expected angle π, got {p2.angle.value}"
    
    # Test from FloatAngle
    angle = FloatAngle(math.pi / 2)
    p3 = FloatCirclePoint(angle)
    assert p3.angle == angle, "Angle should be preserved"
    
    # Test equality
    p4 = FloatCirclePoint(0.5)
    p5 = FloatCirclePoint(0.5)
    assert p4 == p5, "Points should be equal"
    
    # Test rotation
    p6 = FloatCirclePoint(0.0)
    p7 = p6.rotate(math.pi / 2)
    assert abs(p7.angle.value - math.pi / 2) < 1e-12, f"Expected π/2 after rotation, got {p7.angle.value}"
    
    # Test Cartesian coordinates
    p8 = FloatCirclePoint(0.0)
    x, y = p8.to_cartesian()
    assert abs(x - 1.0) < 1e-12, f"Expected x=1.0, got {x}"
    assert abs(y - 0.0) < 1e-12, f"Expected y=0.0, got {y}"
    
    p9 = FloatCirclePoint(math.pi / 2)
    x, y = p9.to_cartesian()
    assert abs(x - 0.0) < 1e-12, f"Expected x=0.0, got {x}"
    assert abs(y - 1.0) < 1e-12, f"Expected y=1.0, got {y}"
    
    # Test from Cartesian
    p10 = FloatCirclePoint.from_cartesian(1.0, 0.0)
    assert abs(p10.angle.value - 0.0) < 1e-12, f"Expected angle 0.0, got {p10.angle.value}"
    
    p11 = FloatCirclePoint.from_cartesian(0.0, 1.0)
    assert abs(p11.angle.value - math.pi / 2) < 1e-12, f"Expected angle π/2, got {p11.angle.value}"
    
    print("FloatCirclePoint tests passed!")


def test_float_circle_interval():
    """Test FloatCircleInterval class."""
    print("\nTesting FloatCircleInterval...")
    
    # Test simple interval
    i1 = FloatCircleInterval(0.0, math.pi / 2)
    assert not i1.is_empty(), "Interval should not be empty"
    assert not i1.is_full_circle(), "Interval should not be full circle"
    assert abs(i1.length().value - math.pi / 2) < 1e-12, f"Expected length π/2, got {i1.length().value}"
    
    # Test point containment
    p1 = FloatCirclePoint(0.0)
    p2 = FloatCirclePoint(math.pi / 4)
    p3 = FloatCirclePoint(math.pi)
    
    assert p1 in i1, "Start point should be in interval"
    assert p2 in i1, "Middle point should be in interval"
    assert p3 not in i1, "Point outside should not be in interval"
    
    # Test point interval (contains only the point)
    i2 = FloatCircleInterval(0.5, 0.5)
    # Point interval is not empty - it contains the point
    assert not i2.is_empty(), "Point interval contains the point, should not be empty"
    assert i2.length().value == 0.0, f"Point interval should have length 0, got {i2.length().value}"
    # Check that the point itself is in the interval
    p_point = FloatCirclePoint(0.5)
    assert p_point in i2, "Point should be in its own interval"
    # Check that other points are not in the interval
    p_other = FloatCirclePoint(0.6)
    assert p_other not in i2, "Other point should not be in point interval"
    
    # Test full circle interval
    # Full circle is from a to nextafter(a, -inf)
    full_circle_end = math.nextafter(0.0, -math.inf)
    if full_circle_end < 0:
        full_circle_end = FloatAngle.MAX_ANGLE
    i3 = FloatCircleInterval(0.0, full_circle_end)
    assert i3.is_full_circle(), f"Interval from 0 to {full_circle_end} should be full circle"
    # Full circle length should be 2π (or very close due to floating point)
    assert i3.length().value >= FloatAngle.MAX_ANGLE, f"Full circle should have length >= MAX_ANGLE, got {i3.length().value}"
    
    # Test interval wrapping around 0
    i4 = FloatCircleInterval(3 * math.pi / 2, math.pi / 2)
    assert not i4.is_empty(), "Wrapping interval should not be empty"
    assert abs(i4.length().value - math.pi) < 1e-12, f"Expected length π, got {i4.length().value}"
    
    # Test point in wrapping interval
    p4 = FloatCirclePoint(0.0)  # 0 is between 3π/2 and π/2 when wrapping
    p5 = FloatCirclePoint(math.pi)  # π is not between 3π/2 and π/2 when wrapping
    
    assert p4 in i4, "Point 0 should be in wrapping interval"
    assert p5 not in i4, "Point π should not be in wrapping interval"
    
    # Test intersection
    i5 = FloatCircleInterval(0.0, math.pi)
    i6 = FloatCircleInterval(math.pi / 2, 3 * math.pi / 2)
    i7 = i5.intersection(i6)
    assert not i7.is_empty(), "Intersection should not be empty"
    assert abs(i7.length().value - math.pi / 2) < 1e-12, f"Expected intersection length π/2, got {i7.length().value}"
    
    # Test union
    i8 = FloatCircleInterval(0.0, math.pi / 2)
    i9 = FloatCircleInterval(math.pi / 4, math.pi)
    union_result = i8.union(i9)
    assert len(union_result) == 1, "Overlapping intervals should merge to one"
    union_interval = union_result[0]
    assert abs(union_interval.length().value - math.pi) < 1e-12, f"Expected union length π, got {union_interval.length().value}"
    
    print("FloatCircleInterval tests passed!")


def test_float_circle_set():
    """Test FloatCircleSet class."""
    print("\nTesting FloatCircleSet...")
    
    # Test empty set
    s1 = FloatCircleSet()
    assert s1.is_empty(), "Empty constructor should create empty set"
    assert not bool(s1), "Empty set should be falsy"
    
    # Test from single interval
    s2 = FloatCircleSet.from_single_interval(0.0, math.pi / 2)
    assert not s2.is_empty(), "Set from interval should not be empty"
    assert len(s2) == 1, "Should have one interval"
    
    # Test point containment
    p1 = FloatCirclePoint(math.pi / 4)
    p2 = FloatCirclePoint(math.pi)
    
    assert p1 in s2, "Point in interval should be in set"
    assert p2 not in s2, "Point not in interval should not be in set"
    
    # Test from multiple intervals
    i1 = FloatCircleInterval(0.0, math.pi / 2)
    i2 = FloatCircleInterval(math.pi, 3 * math.pi / 2)
    print(f"Interval 1: {i1}, length: {i1.length().value}")
    print(f"Interval 2: {i2}, length: {i2.length().value}")
    # Check if they intersect
    intersection = i1.intersection(i2)
    print(f"Intersection: {intersection}, is_empty: {intersection.is_empty()}")
    # Check union
    union_result = i1.union(i2)
    print(f"Union result length: {len(union_result)}")
    s3 = FloatCircleSet(i1, i2)
    print(f"Set intervals: {list(s3)}")
    assert len(s3) == 2, f"Should have two intervals, got {len(s3)}"
    
    # Test merging overlapping intervals
    i3 = FloatCircleInterval(0.0, math.pi / 2)
    i4 = FloatCircleInterval(math.pi / 4, math.pi)
    s4 = FloatCircleSet(i3, i4)
    assert len(s4) == 1, "Overlapping intervals should merge to one"
    
    # Test union
    s5 = FloatCircleSet.from_single_interval(0.0, math.pi / 2)
    s6 = FloatCircleSet.from_single_interval(math.pi / 4, math.pi)
    s7 = s5.union(s6)
    assert len(s7) == 1, "Union of overlapping sets should have one interval"
    
    # Test intersection
    s8 = FloatCircleSet.from_single_interval(0.0, math.pi)
    s9 = FloatCircleSet.from_single_interval(math.pi / 2, 3 * math.pi / 2)
    s10 = s8.intersection(s9)
    assert len(s10) == 1, "Intersection should have one interval"
    interval = list(s10)[0]
    assert abs(interval.length().value - math.pi / 2) < 1e-12, f"Expected intersection length π/2, got {interval.length().value}"
    
    # Test complement
    s11 = FloatCircleSet.from_single_interval(0.0, math.pi / 2)
    s12 = s11.complement()
    assert not s12.is_empty(), "Complement should not be empty"
    assert len(s12) == 1, "Complement of simple interval should have one interval"
    
    # Test full circle complement
    # Create full circle interval
    full_circle_end = math.nextafter(0.0, -math.inf)
    if full_circle_end < 0:
        full_circle_end = FloatAngle.MAX_ANGLE
    s13 = FloatCircleSet.from_single_interval(0.0, full_circle_end)  # Full circle
    s14 = s13.complement()
    assert s14.is_empty(), "Complement of full circle should be empty"
    
    print("FloatCircleSet tests passed!")


def main():
    """Run all tests."""
    print("Running floatcircle tests...")
    print("=" * 50)
    
    try:
        test_float_angle()
        test_float_circle_point()
        test_float_circle_interval()
        test_float_circle_set()
        
        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        raise
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise


if __name__ == "__main__":
    main()