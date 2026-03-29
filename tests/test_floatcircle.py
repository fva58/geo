"""Unit tests for circle geometry built on top of FloatSet."""

import math
import unittest

from geo.floatcircle import (
    FloatAngle,
    FloatCircleInterval,
    FloatCirclePoint,
    FloatCircleSet,
    FULL_FLOAT_CIRCLE_INTERVAL,
    FULL_FLOAT_CIRCLE_SET,
)
from geo.floatset import FloatSet


class TestFloatAngle(unittest.TestCase):
    """Test cases for ``FloatAngle``."""

    def test_normalization(self):
        """Angles should normalize into the circle domain."""
        self.assertEqual(FloatAngle(0.0).value, 0.0)
        self.assertAlmostEqual(FloatAngle(3 * math.pi).value, math.pi)
        self.assertAlmostEqual(FloatAngle(-math.pi).value, math.pi)
        self.assertEqual(FloatAngle(2 * math.pi).value, 0.0)

    def test_distance(self):
        """Shortest angular distance should respect wrap-around."""
        distance = FloatAngle(0.1).distance_to(2 * math.pi - 0.1)
        self.assertAlmostEqual(distance.value, 0.2)


class TestFloatCirclePoint(unittest.TestCase):
    """Test cases for ``FloatCirclePoint``."""

    def test_point_helpers(self):
        """Point helpers should expose angle and Cartesian coordinates."""
        point = FloatCirclePoint(math.pi / 2)
        self.assertAlmostEqual(point.angle.value, math.pi / 2)
        self.assertAlmostEqual(point.x, 0.0, places=12)
        self.assertAlmostEqual(point.y, 1.0, places=12)

        restored = FloatCirclePoint.from_cartesian(0.0, 1.0)
        self.assertAlmostEqual(restored.angle.value, math.pi / 2)


class TestFloatCircleInterval(unittest.TestCase):
    """Test cases for ``FloatCircleInterval``."""

    def test_inheritance_and_simple_interval(self):
        """Circle intervals should inherit from ``FloatSet``."""
        interval = FloatCircleInterval(0.0, math.pi / 2)
        self.assertIsInstance(interval, FloatSet)
        self.assertEqual(len(interval), 1)
        self.assertFalse(interval.is_wrapped())
        self.assertFalse(interval.is_full())
        self.assertAlmostEqual(interval.start, 0.0)
        self.assertAlmostEqual(interval.end, math.pi / 2)
        self.assertAlmostEqual(interval.length().value, math.pi / 2)

    def test_wrapped_interval_is_stored_as_two_linear_intervals(self):
        """An interval crossing zero should be represented by two pieces."""
        interval = FloatCircleInterval(3 * math.pi / 2, math.pi / 2)
        self.assertEqual(len(interval), 2)
        self.assertTrue(interval.is_wrapped())
        self.assertEqual(interval[0][0], 0.0)
        self.assertAlmostEqual(interval[0][1], math.pi / 2)
        self.assertAlmostEqual(interval[1][0], 3 * math.pi / 2)
        self.assertEqual(interval[1][1], FloatAngle.MAX_ANGLE)
        self.assertIn(0.0, interval)
        self.assertIn(7 * math.pi / 4, interval)
        self.assertNotIn(math.pi, interval)
        self.assertAlmostEqual(interval.length().value, math.pi)

    def test_full_circle(self):
        """The special predecessor endpoint should represent the full circle."""
        self.assertTrue(FULL_FLOAT_CIRCLE_INTERVAL.is_full())
        self.assertTrue(FULL_FLOAT_CIRCLE_INTERVAL.is_full_circle())
        self.assertEqual(len(FULL_FLOAT_CIRCLE_INTERVAL), 1)
        self.assertIn(0.0, FULL_FLOAT_CIRCLE_INTERVAL)
        self.assertIn(math.pi, FULL_FLOAT_CIRCLE_INTERVAL)

    def test_interval_complement_returns_circle_set(self):
        """Complements should be expressed as circle sets."""
        interval = FloatCircleInterval(0.0, math.pi / 2)
        complement = interval.complement()
        self.assertIsInstance(complement, FloatCircleSet)
        self.assertNotIn(0.0, complement)
        self.assertIn(math.pi, complement)


class TestFloatCircleSet(unittest.TestCase):
    """Test cases for ``FloatCircleSet``."""

    def test_union_and_intersection(self):
        """Circle-set operations should reuse FloatSet semantics."""
        left = FloatCircleSet.from_single_interval(0.0, math.pi)
        right = FloatCircleSet.from_single_interval(math.pi / 2, 3 * math.pi / 2)

        union = left.union(right)
        intersection = left.intersection(right)

        self.assertIsInstance(union, FloatCircleSet)
        self.assertIsInstance(intersection, FloatCircleSet)
        self.assertIn(0.0, union)
        self.assertIn(3 * math.pi / 2, union)
        self.assertIn(math.pi / 2, intersection)
        self.assertNotIn(0.0, intersection)

    def test_union_merges_adjacent_representable_arcs(self):
        """Circle-set unions should merge arcs with no float-angle gap."""
        boundary = 1.0
        adjacent = math.nextafter(boundary, math.inf)

        union = FloatCircleSet.from_single_interval(0.0, boundary).union(
            FloatCircleSet.from_single_interval(adjacent, 2.0)
        )

        self.assertEqual(
            union,
            FloatCircleSet.from_single_interval(0.0, 2.0),
        )

    def test_difference_and_complement(self):
        """Difference and complement should stay inside circle space."""
        left = FloatCircleSet.from_single_interval(0.0, math.pi)
        right = FloatCircleSet.from_single_interval(math.pi / 2, math.pi)

        difference = left.difference(right)
        self.assertIn(0.0, difference)
        self.assertNotIn(math.pi, difference)

        complement = left.complement()
        self.assertIsInstance(complement, FloatCircleSet)
        self.assertNotIn(0.0, complement)
        self.assertIn(3 * math.pi / 2, complement)

    def test_full_circle_set(self):
        """The full circle set should cover the whole circle."""
        self.assertTrue(FULL_FLOAT_CIRCLE_SET.is_full())
        self.assertIn(0.0, FULL_FLOAT_CIRCLE_SET)
        self.assertIn(math.pi, FULL_FLOAT_CIRCLE_SET)


if __name__ == "__main__":
    unittest.main()
