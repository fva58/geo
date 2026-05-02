"""Unit tests for circle point, interval, and set classes."""

import math
import unittest

from geo.space.circle import Angle, FULL_INTERVAL, FULL_SET, Interval, Point, Set


class TestAngle(unittest.TestCase):
    """Test cases for ``Angle``."""

    def test_normalization(self):
        """Angles should normalize into the circle domain."""
        self.assertEqual(Angle(0.0).value, 0.0)
        self.assertAlmostEqual(Angle(3 * math.pi).value, math.pi)
        self.assertAlmostEqual(Angle(-math.pi).value, math.pi)
        self.assertEqual(Angle(2 * math.pi).value, 0.0)

    def test_distance(self):
        """Shortest angular distance should respect wrap-around."""
        distance = Angle(0.1).distance_to(2 * math.pi - 0.1)
        self.assertAlmostEqual(distance.value, 0.2)


class TestPoint(unittest.TestCase):
    """Test cases for ``Point`` on the circle."""

    def test_point_helpers(self):
        """Point helpers should expose angle and Cartesian coordinates."""
        point = Point(math.pi / 2)
        self.assertAlmostEqual(point.angle.value, math.pi / 2)
        self.assertAlmostEqual(point.x, 0.0, places=12)
        self.assertAlmostEqual(point.y, 1.0, places=12)

        restored = Point.from_cartesian(0.0, 1.0)
        self.assertAlmostEqual(restored.angle.value, math.pi / 2)


class TestInterval(unittest.TestCase):
    """Test cases for ``Interval`` on the circle."""

    def test_inheritance_and_simple_interval(self):
        """Circle intervals should inherit line-set semantics."""
        interval = Interval(0.0, math.pi / 2)
        self.assertTrue(hasattr(interval, "contains"))
        self.assertEqual(len(interval), 1)
        self.assertFalse(interval.is_wrapped())
        self.assertFalse(interval.is_full())
        self.assertAlmostEqual(interval.start, 0.0)
        self.assertAlmostEqual(interval.end, math.pi / 2)
        self.assertAlmostEqual(interval.length().value, math.pi / 2)

    def test_wrapped_interval_is_stored_as_two_linear_intervals(self):
        """An interval crossing zero should be represented by two pieces."""
        interval = Interval(3 * math.pi / 2, math.pi / 2)
        self.assertEqual(len(interval), 2)
        self.assertTrue(interval.is_wrapped())
        self.assertEqual(interval[0][0], 0.0)
        self.assertAlmostEqual(interval[0][1], math.pi / 2)
        self.assertAlmostEqual(interval[1][0], 3 * math.pi / 2)
        self.assertEqual(interval[1][1], Angle.MAX_ANGLE)
        self.assertIn(0.0, interval)
        self.assertIn(7 * math.pi / 4, interval)
        self.assertNotIn(math.pi, interval)
        self.assertAlmostEqual(interval.length().value, math.pi)

    def test_full_circle(self):
        """The special predecessor endpoint should represent the full circle."""
        self.assertTrue(FULL_INTERVAL.is_full())
        self.assertTrue(FULL_INTERVAL.is_full_circle())
        self.assertEqual(len(FULL_INTERVAL), 1)
        self.assertIn(0.0, FULL_INTERVAL)
        self.assertIn(math.pi, FULL_INTERVAL)

    def test_interval_complement_returns_circle_set(self):
        """Complements should be expressed as circle sets."""
        interval = Interval(0.0, math.pi / 2)
        complement = interval.complement()
        self.assertIsInstance(complement, Set)
        self.assertNotIn(0.0, complement)
        self.assertIn(math.pi, complement)


class TestSet(unittest.TestCase):
    """Test cases for ``Set`` on the circle."""

    def test_union_and_intersection(self):
        """Circle-set operations should reuse line-set semantics."""
        left = Set.from_single_interval(0.0, math.pi)
        right = Set.from_single_interval(math.pi / 2, 3 * math.pi / 2)

        union = left.union(right)
        intersection = left.intersection(right)

        self.assertIsInstance(union, Set)
        self.assertIsInstance(intersection, Set)
        self.assertIn(0.0, union)
        self.assertIn(3 * math.pi / 2, union)
        self.assertIn(math.pi / 2, intersection)
        self.assertNotIn(0.0, intersection)

    def test_union_merges_adjacent_representable_arcs(self):
        """Circle-set unions should merge arcs with no float-angle gap."""
        boundary = 1.0
        adjacent = math.nextafter(boundary, math.inf)

        union = Set.from_single_interval(0.0, boundary).union(
            Set.from_single_interval(adjacent, 2.0)
        )

        self.assertEqual(union, Set.from_single_interval(0.0, 2.0))

    def test_difference_and_complement(self):
        """Difference and complement should stay inside circle space."""
        left = Set.from_single_interval(0.0, math.pi)
        right = Set.from_single_interval(math.pi / 2, math.pi)

        difference = left.difference(right)
        self.assertIn(0.0, difference)
        self.assertNotIn(math.pi, difference)

        complement = left.complement()
        self.assertIsInstance(complement, Set)
        self.assertNotIn(0.0, complement)
        self.assertIn(3 * math.pi / 2, complement)

    def test_full_circle_set(self):
        """The full circle set should cover the whole circle."""
        self.assertTrue(FULL_SET.is_full())
        self.assertIn(0.0, FULL_SET)
        self.assertIn(math.pi, FULL_SET)


if __name__ == "__main__":
    unittest.main()
