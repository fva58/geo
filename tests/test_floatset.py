"""Unit tests for line intervals and line sets."""

import math
import unittest

from geo.real import ALL_REALS_INTERVAL, EMPTY_REAL_INTERVAL, real, realset
from geo.space.line import FULL_INTERVAL, Interval, Point, Set


class TestInterval(unittest.TestCase):
    """Test cases for ``Interval``."""

    def test_initialization(self):
        """Test interval initialization."""
        iv = Interval(0.0)
        self.assertEqual(iv.left, 0.0)
        self.assertEqual(iv.right, 0.0)
        self.assertFalse(iv.is_empty())

        iv = Interval(0.0, 1.0)
        self.assertEqual(iv.left, 0.0)
        self.assertEqual(iv.right, 1.0)
        self.assertFalse(iv.is_empty())

        iv = Interval(iv)
        self.assertEqual(iv.left, 0.0)
        self.assertEqual(iv.right, 1.0)
        self.assertFalse(iv.is_empty())

    def test_empty_interval(self):
        """Test empty interval."""
        empty = Interval(1.0, 0.0)
        self.assertTrue(empty.is_empty())
        self.assertEqual(empty.length(), -math.inf)
        self.assertFalse(bool(empty))

    def test_contains(self):
        """Test point containment."""
        iv = Interval(0.0, 1.0)
        self.assertTrue(0.5 in iv)
        self.assertTrue(0.0 in iv)
        self.assertTrue(1.0 in iv)
        self.assertFalse(1.5 in iv)
        self.assertFalse(-0.5 in iv)

    def test_length(self):
        """Test interval length."""
        iv = Interval(0.0, 5.0)
        self.assertEqual(iv.length(), 5.0)

        point = Interval(3.0, 3.0)
        self.assertEqual(point.length(), 0.0)

        empty = Interval(1.0, 0.0)
        self.assertEqual(empty.length(), -math.inf)

    def test_intersection(self):
        """Test interval intersection."""
        a = Interval(0.0, 5.0)
        b = Interval(2.0, 7.0)
        c = Interval(8.0, 10.0)

        inter_ab = a.intersection(b)
        self.assertEqual(inter_ab.left, 2.0)
        self.assertEqual(inter_ab.right, 5.0)

        inter_ac = a.intersection(c)
        self.assertTrue(inter_ac.is_empty())

        empty = Interval(1.0, 0.0)
        self.assertTrue(a.intersection(empty).is_empty())
        self.assertTrue(empty.intersection(a).is_empty())

    def test_union(self):
        """Test interval union."""
        a = Interval(0.0, 3.0)
        b = Interval(2.0, 5.0)
        c = Interval(6.0, 8.0)

        union_ab = a.union(b)
        self.assertEqual(len(union_ab), 1)
        self.assertEqual(union_ab[0].left, 0.0)
        self.assertEqual(union_ab[0].right, 5.0)

        union_ac = a.union(c)
        self.assertEqual(len(union_ac), 2)
        self.assertEqual(union_ac[0].left, 0.0)
        self.assertEqual(union_ac[0].right, 3.0)
        self.assertEqual(union_ac[1].left, 6.0)
        self.assertEqual(union_ac[1].right, 8.0)

        empty = Interval(1.0, 0.0)
        self.assertEqual(a.union(empty), (a,))
        self.assertEqual(empty.union(a), (a,))
        self.assertEqual(empty.union(empty), ())

    def test_union_merges_adjacent_representable_intervals(self):
        """Adjacent float-lattice intervals should merge."""
        boundary = 1.0
        adjacent = math.nextafter(boundary, math.inf)

        union = Interval(0.0, boundary).union(Interval(adjacent, 2.0))
        self.assertEqual(union, (Interval(0.0, 2.0),))

    def test_difference(self):
        """Test interval difference."""
        a = Interval(0.0, 5.0)
        b = Interval(2.0, 3.0)
        c = Interval(6.0, 8.0)

        diff_ab = a.difference(b)
        self.assertEqual(len(diff_ab), 2)
        self.assertEqual(diff_ab[0].left, 0.0)
        self.assertEqual(diff_ab[0].right, math.nextafter(2.0, -math.inf))
        self.assertEqual(diff_ab[1].left, math.nextafter(3.0, math.inf))
        self.assertEqual(diff_ab[1].right, 5.0)
        self.assertFalse(diff_ab[0].contains(2.0))
        self.assertFalse(diff_ab[1].contains(3.0))

        diff_ac = a.difference(c)
        self.assertEqual(len(diff_ac), 1)
        self.assertEqual(diff_ac[0].left, 0.0)
        self.assertEqual(diff_ac[0].right, 5.0)

        empty = Interval(1.0, 0.0)
        self.assertEqual(a.difference(empty), (a,))
        self.assertEqual(empty.difference(a), ())

    def test_symmetric_difference(self):
        """Test symmetric difference."""
        a = Interval(0.0, 3.0)
        b = Interval(2.0, 5.0)

        sym_diff = a.symmetric_difference(b)
        self.assertEqual(len(sym_diff), 2)
        self.assertEqual(sym_diff[0].left, 0.0)
        self.assertEqual(sym_diff[0].right, math.nextafter(2.0, -math.inf))
        self.assertEqual(sym_diff[1].left, math.nextafter(3.0, math.inf))
        self.assertEqual(sym_diff[1].right, 5.0)

    def test_complement_of_empty_interval_is_full(self):
        """The complement of the empty interval should be the whole line."""
        empty = Interval(1.0, 0.0)
        self.assertEqual(empty.complement(), (FULL_INTERVAL,))
        self.assertEqual(~empty, (FULL_INTERVAL,))

    def test_operators(self):
        """Test operator overloads."""
        a = Interval(0.0, 3.0)
        b = Interval(2.0, 5.0)

        inter = a & b
        self.assertEqual(inter.left, 2.0)
        self.assertEqual(inter.right, 3.0)

        union = a | b
        self.assertEqual(len(union), 1)
        self.assertEqual(union[0].left, 0.0)
        self.assertEqual(union[0].right, 5.0)

        diff = a - b
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0].left, 0.0)
        self.assertEqual(diff[0].right, math.nextafter(2.0, -math.inf))

        sym_diff = a ^ b
        self.assertEqual(len(sym_diff), 2)

    def test_is_subset(self):
        """Test subset relation."""
        a = Interval(0.0, 10.0)
        b = Interval(3.0, 7.0)
        c = Interval(5.0, 15.0)

        self.assertTrue(b.is_subset(a))
        self.assertFalse(a.is_subset(b))
        self.assertFalse(c.is_subset(a))
        self.assertFalse(a.is_subset(c))

        empty = Interval(1.0, 0.0)
        self.assertTrue(empty.is_subset(a))
        self.assertFalse(a.is_subset(empty))

    def test_equality_and_hash(self):
        """Test equality and hashing."""
        a = Interval(0.0, 1.0)
        b = Interval(0.0, 1.0)
        c = Interval(0.0, 2.0)

        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(hash(a), hash(b))

        empty1 = Interval(1.0, 0.0)
        empty2 = Interval(2.0, 1.0)
        self.assertEqual(empty1, empty2)

    def test_complement(self):
        """Test complement in the float model."""
        interval = Interval(0.0, 1.0)
        complement = interval.complement()
        self.assertEqual(len(complement), 2)
        self.assertEqual(complement[0].left, -math.inf)
        self.assertEqual(complement[0].right, math.nextafter(0.0, -math.inf))
        self.assertEqual(complement[1].left, math.nextafter(1.0, math.inf))
        self.assertEqual(complement[1].right, math.inf)


class TestSet(unittest.TestCase):
    """Test cases for ``Set``."""

    def test_initialization(self):
        """Test set initialization."""
        intervals = [Interval(0.0, 1.0), Interval(2.0, 3.0)]
        fset = Set(intervals)
        self.assertEqual(len(fset), 2)
        self.assertFalse(fset.is_empty())

        fset = Set(*intervals)
        self.assertEqual(len(fset), 2)
        self.assertFalse(fset.is_empty())

        fset = Set(1, 2)
        self.assertEqual(len(fset), 2)
        self.assertFalse(fset.is_empty())

    def test_single_pair_argument_means_interval(self):
        """A single numeric pair should be parsed as one interval."""
        fset = Set((1.0, 2.0))
        self.assertEqual(fset, Set(Interval(1.0, 2.0)))

    def test_invalid_inputs_raise_type_error(self):
        """Unsupported constructor inputs should fail predictably."""
        with self.assertRaises(TypeError):
            Set("ab")

        with self.assertRaises(TypeError):
            Set(["ab"])

        with self.assertRaises(TypeError):
            Set(object())

    def test_empty_set(self):
        """Test empty set."""
        empty_set = Set()
        self.assertTrue(empty_set.is_empty())
        self.assertEqual(len(empty_set), 0)
        self.assertFalse(bool(empty_set))

    def test_normalization(self):
        """Test interval normalization."""
        intervals = [
            Interval(2.0, 4.0),
            Interval(1.0, 3.0),
            Interval(0.0, 1.0),
        ]
        fset = Set(intervals)
        self.assertEqual(len(fset), 1)
        self.assertEqual(fset.intervals[0][0], 0.0)
        self.assertEqual(fset.intervals[0][1], 4.0)

        intervals = [
            Interval(0.0, 1.0),
            Interval(3.0, 4.0),
            Interval(6.0, 7.0),
        ]
        fset = Set(intervals)
        self.assertEqual(len(fset), 3)

    def test_normalization_merges_adjacent_representable_intervals(self):
        """Normalization should merge intervals with no float between them."""
        boundary = 1.0
        adjacent = math.nextafter(boundary, math.inf)

        fset = Set(Interval(0.0, boundary), Interval(adjacent, 2.0))
        self.assertEqual(fset, Set(Interval(0.0, 2.0)))

    def test_contains(self):
        """Test point containment in set."""
        fset = Set([Interval(0.0, 1.0), Interval(3.0, 4.0)])

        self.assertTrue(0.5 in fset)
        self.assertTrue(0.0 in fset)
        self.assertTrue(1.0 in fset)
        self.assertTrue(3.5 in fset)
        self.assertFalse(1.5 in fset)
        self.assertFalse(2.5 in fset)
        self.assertFalse(4.5 in fset)

    def test_union(self):
        """Test set union."""
        set1 = Set([Interval(0.0, 2.0), Interval(4.0, 6.0)])
        set2 = Set([Interval(1.0, 5.0)])

        union = set1.union(set2)
        self.assertEqual(len(union), 1)
        self.assertEqual(union.intervals[0][0], 0.0)
        self.assertEqual(union.intervals[0][1], 6.0)

        empty = Set()
        self.assertEqual(set1.union(empty), set1)
        self.assertEqual(empty.union(set1), set1)

    def test_intersection(self):
        """Test set intersection."""
        set1 = Set([Interval(0.0, 3.0), Interval(5.0, 8.0)])
        set2 = Set([Interval(2.0, 6.0)])

        inter = set1.intersection(set2)
        self.assertEqual(len(inter), 2)
        self.assertEqual(inter.intervals[0][0], 2.0)
        self.assertEqual(inter.intervals[0][1], 3.0)
        self.assertEqual(inter.intervals[1][0], 5.0)
        self.assertEqual(inter.intervals[1][1], 6.0)

        empty = Set()
        self.assertTrue(set1.intersection(empty).is_empty())
        self.assertTrue(empty.intersection(set1).is_empty())

    def test_difference(self):
        """Test set difference."""
        set1 = Set([Interval(0.0, 5.0)])
        set2 = Set([Interval(1.0, 2.0), Interval(3.0, 4.0)])

        diff = set1.difference(set2)
        self.assertEqual(len(diff), 3)
        intervals = list(diff.intervals)
        self.assertEqual(intervals[0][0], 0.0)
        self.assertEqual(intervals[0][1], math.nextafter(1.0, -math.inf))
        self.assertEqual(intervals[1][0], math.nextafter(2.0, math.inf))
        self.assertEqual(intervals[1][1], math.nextafter(3.0, -math.inf))
        self.assertEqual(intervals[2][0], math.nextafter(4.0, math.inf))
        self.assertEqual(intervals[2][1], 5.0)

    def test_symmetric_difference(self):
        """Test symmetric difference."""
        set1 = Set([Interval(0.0, 2.0), Interval(4.0, 6.0)])
        set2 = Set([Interval(1.0, 5.0)])

        sym_diff = set1.symmetric_difference(set2)
        self.assertEqual(len(sym_diff), 3)

    def test_operators(self):
        """Test operator overloads."""
        set1 = Set([Interval(0.0, 2.0)])
        set2 = Set([Interval(1.0, 3.0)])

        union = set1 | set2
        self.assertEqual(len(union), 1)
        self.assertEqual(union.intervals[0][0], 0.0)
        self.assertEqual(union.intervals[0][1], 3.0)

        inter = set1 & set2
        self.assertEqual(len(inter), 1)
        self.assertEqual(inter.intervals[0][0], 1.0)
        self.assertEqual(inter.intervals[0][1], 2.0)

        diff = set1 - set2
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff.intervals[0][0], 0.0)
        self.assertEqual(
            diff.intervals[0][1],
            math.nextafter(1.0, -math.inf),
        )

        sym_diff = set1 ^ set2
        self.assertEqual(len(sym_diff), 2)

    def test_from_methods(self):
        """Test factory methods."""
        fset1 = Set.from_single_interval(0.0, 1.0)
        self.assertEqual(len(fset1), 1)
        self.assertEqual(fset1.intervals[0][0], 0.0)
        self.assertEqual(fset1.intervals[0][1], 1.0)

        intervals = [Interval(0.0, 1.0), Interval(2.0, 3.0)]
        fset2 = Set.from_intervals(*intervals)
        self.assertEqual(len(fset2), 2)

    def test_to_tuple(self):
        """Test conversion to tuple."""
        fset = Set([Interval(0.0, 1.0), Interval(2.0, 3.0)])
        self.assertEqual(fset.to_tuple(), ((0.0, 1.0), (2.0, 3.0)))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and floating point precision."""

    def test_touching_intervals(self):
        """Test intervals that touch at endpoints."""
        a = Interval(0.0, 1.0)
        b = Interval(1.0, 2.0)

        union = a.union(b)
        self.assertEqual(len(union), 1)
        self.assertEqual(union[0][0], 0.0)
        self.assertEqual(union[0][1], 2.0)

        inter = a.intersection(b)
        self.assertFalse(inter.is_empty())
        self.assertEqual(inter[0], 1.0)
        self.assertEqual(inter[1], 1.0)
        self.assertEqual(inter.length(), 0.0)

    def test_single_point_intervals(self):
        """Test single point intervals."""
        point = Interval(3.0, 3.0)
        self.assertFalse(point.is_empty())
        self.assertEqual(point.length(), 0.0)
        self.assertTrue(3.0 in point)
        self.assertFalse(3.1 in point)

        interval = Interval(2.0, 4.0)
        self.assertTrue(point.is_subset(interval))
        self.assertEqual(interval.intersection(point), point)
        self.assertEqual(interval.union(point), (Interval(2.0, 4.0),))

    def test_point_wrapper(self):
        """Point wrappers should behave like line scalars."""
        point = Point(3.0)
        self.assertEqual(point.value, 3.0)
        self.assertEqual(point.distance_to(5.0), 2.0)

    def test_floating_point_precision(self):
        """Test handling of floating point precision."""
        a = Interval(0.0, 1.0)
        adjacent = math.nextafter(1.0, math.inf)
        b = Interval(math.nextafter(adjacent, math.inf), 2.0)

        union = a.union(b)
        self.assertEqual(len(union), 2)

        inter = a.intersection(b)
        self.assertTrue(inter.is_empty())


class TestRealModule(unittest.TestCase):
    """Test the real-line aliases."""

    def test_real_aliases(self):
        """Test that ``geo.real`` exposes the active line model."""
        self.assertIs(real, Point)
        self.assertIs(realset, Set)
        self.assertTrue(EMPTY_REAL_INTERVAL.is_empty())
        self.assertTrue(ALL_REALS_INTERVAL.is_full())


if __name__ == "__main__":
    unittest.main()
