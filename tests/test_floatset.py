"""Unit tests for FloatInterval and FloatSet classes."""

import unittest
import math
from geo.floatset import FloatInterval, FloatSet
from geo.real import ALL_REALS_INTERVAL, EMPTY_REAL_INTERVAL, real, realset


class TestFloatInterval(unittest.TestCase):
    """Test cases for FloatInterval class."""

    def test_initialization(self):
        """Test interval initialization."""
        iv = FloatInterval(0.0)
        self.assertEqual(iv.left, 0.0)
        self.assertEqual(iv.right, 0.0)
        self.assertFalse(iv.is_empty())

        iv = FloatInterval(0.0, 1.0)
        self.assertEqual(iv.left, 0.0)
        self.assertEqual(iv.right, 1.0)
        self.assertFalse(iv.is_empty())

        iv = FloatInterval(iv)
        self.assertEqual(iv.left, 0.0)
        self.assertEqual(iv.right, 1.0)
        self.assertFalse(iv.is_empty())

    def test_empty_interval(self):
        """Test empty interval."""
        empty = FloatInterval(1.0, 0.0)
        self.assertTrue(empty.is_empty())
        self.assertEqual(empty.length(), -math.inf)
        self.assertFalse(bool(empty))

    def test_contains(self):
        """Test point containment."""
        iv = FloatInterval(0.0, 1.0)
        self.assertTrue(0.5 in iv)
        self.assertTrue(0.0 in iv)
        self.assertTrue(1.0 in iv)
        self.assertFalse(1.5 in iv)
        self.assertFalse(-0.5 in iv)

    def test_length(self):
        """Test interval length."""
        iv = FloatInterval(0.0, 5.0)
        self.assertEqual(iv.length(), 5.0)

        point = FloatInterval(3.0, 3.0)
        self.assertEqual(point.length(), 0.0)

        empty = FloatInterval(1.0, 0.0)
        self.assertEqual(empty.length(), -math.inf)

    def test_intersection(self):
        """Test interval intersection."""
        a = FloatInterval(0.0, 5.0)
        b = FloatInterval(2.0, 7.0)
        c = FloatInterval(8.0, 10.0)

        # Overlapping intervals
        inter_ab = a.intersection(b)
        self.assertEqual(inter_ab.left, 2.0)
        self.assertEqual(inter_ab.right, 5.0)

        # Disjoint intervals
        inter_ac = a.intersection(c)
        self.assertTrue(inter_ac.is_empty())

        # With empty interval
        empty = FloatInterval(1.0, 0.0)
        self.assertTrue(a.intersection(empty).is_empty())
        self.assertTrue(empty.intersection(a).is_empty())

    def test_union(self):
        """Test interval union."""
        a = FloatInterval(0.0, 3.0)
        b = FloatInterval(2.0, 5.0)
        c = FloatInterval(6.0, 8.0)

        # Overlapping intervals
        union_ab = a.union(b)
        self.assertEqual(len(union_ab), 1)
        self.assertEqual(union_ab[0].left, 0.0)
        self.assertEqual(union_ab[0].right, 5.0)

        # Disjoint intervals
        union_ac = a.union(c)
        self.assertEqual(len(union_ac), 2)
        self.assertEqual(union_ac[0].left, 0.0)
        self.assertEqual(union_ac[0].right, 3.0)
        self.assertEqual(union_ac[1].left, 6.0)
        self.assertEqual(union_ac[1].right, 8.0)

        # With empty interval
        empty = FloatInterval(1.0, 0.0)
        self.assertEqual(a.union(empty), (a,))
        self.assertEqual(empty.union(a), (a,))
        self.assertEqual(empty.union(empty), ())

    def test_difference(self):
        """Test interval difference."""
        a = FloatInterval(0.0, 5.0)
        b = FloatInterval(2.0, 3.0)
        c = FloatInterval(6.0, 8.0)

        # b is inside a
        diff_ab = a.difference(b)
        self.assertEqual(len(diff_ab), 2)
        self.assertEqual(diff_ab[0].left, 0.0)
        self.assertEqual(diff_ab[0].right, math.nextafter(2.0, -math.inf))
        self.assertEqual(diff_ab[1].left, math.nextafter(3.0, math.inf))
        self.assertEqual(diff_ab[1].right, 5.0)
        self.assertFalse(diff_ab[0].contains(2.0))
        self.assertFalse(diff_ab[1].contains(3.0))

        # Disjoint intervals
        diff_ac = a.difference(c)
        self.assertEqual(len(diff_ac), 1)
        self.assertEqual(diff_ac[0].left, 0.0)
        self.assertEqual(diff_ac[0].right, 5.0)

        # Empty interval
        empty = FloatInterval(1.0, 0.0)
        self.assertEqual(a.difference(empty), (a,))
        self.assertEqual(empty.difference(a), ())

    def test_symmetric_difference(self):
        """Test symmetric difference."""
        a = FloatInterval(0.0, 3.0)
        b = FloatInterval(2.0, 5.0)

        sym_diff = a.symmetric_difference(b)
        self.assertEqual(len(sym_diff), 2)
        self.assertEqual(sym_diff[0].left, 0.0)
        self.assertEqual(sym_diff[0].right, math.nextafter(2.0, -math.inf))
        self.assertEqual(sym_diff[1].left, math.nextafter(3.0, math.inf))
        self.assertEqual(sym_diff[1].right, 5.0)

    def test_operators(self):
        """Test operator overloads."""
        a = FloatInterval(0.0, 3.0)
        b = FloatInterval(2.0, 5.0)

        # Intersection
        inter = a & b
        self.assertEqual(inter.left, 2.0)
        self.assertEqual(inter.right, 3.0)

        # Union
        union = a | b
        self.assertEqual(len(union), 1)
        self.assertEqual(union[0].left, 0.0)
        self.assertEqual(union[0].right, 5.0)

        # Difference
        diff = a - b
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0].left, 0.0)
        self.assertEqual(diff[0].right, math.nextafter(2.0, -math.inf))

        # Symmetric difference
        sym_diff = a ^ b
        self.assertEqual(len(sym_diff), 2)

    def test_is_subset(self):
        """Test subset relation."""
        a = FloatInterval(0.0, 10.0)
        b = FloatInterval(3.0, 7.0)
        c = FloatInterval(5.0, 15.0)

        self.assertTrue(b.is_subset(a))
        self.assertFalse(a.is_subset(b))
        self.assertFalse(c.is_subset(a))
        self.assertFalse(a.is_subset(c))

        # Empty interval is subset of everything
        empty = FloatInterval(1.0, 0.0)
        self.assertTrue(empty.is_subset(a))
        self.assertFalse(a.is_subset(empty))

    def test_equality_and_hash(self):
        """Test equality and hashing."""
        a = FloatInterval(0.0, 1.0)
        b = FloatInterval(0.0, 1.0)
        c = FloatInterval(0.0, 2.0)

        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(hash(a), hash(b))

        # Empty intervals
        empty1 = FloatInterval(1.0, 0.0)
        empty2 = FloatInterval(2.0, 1.0)
        self.assertEqual(empty1, empty2)

    def test_complement(self):
        """Test complement in the float model."""
        interval = FloatInterval(0.0, 1.0)
        complement = interval.complement()
        self.assertEqual(len(complement), 2)
        self.assertEqual(complement[0].left, -math.inf)
        self.assertEqual(complement[0].right,
                         math.nextafter(0.0, -math.inf))
        self.assertEqual(complement[1].left, math.nextafter(1.0, math.inf))
        self.assertEqual(complement[1].right, math.inf)


class TestFloatSet(unittest.TestCase):
    """Test cases for FloatSet class."""

    def test_initialization(self):
        """Test set initialization."""
        intervals = [FloatInterval(0.0, 1.0), FloatInterval(2.0, 3.0)]
        fset = FloatSet(intervals)
        self.assertEqual(len(fset), 2)
        self.assertFalse(fset.is_empty())
        fset = FloatSet(*intervals)
        self.assertEqual(len(fset), 2)
        self.assertFalse(fset.is_empty())
        fset = FloatSet(1,2)
        self.assertEqual(len(fset), 2)
        self.assertFalse(fset.is_empty())

    def test_empty_set(self):
        """Test empty set."""
        empty_set = FloatSet()
        self.assertTrue(empty_set.is_empty())
        self.assertEqual(len(empty_set), 0)
        self.assertFalse(bool(empty_set))

    def test_normalization(self):
        """Test interval normalization."""
        # Overlapping intervals should be merged
        intervals = [
            FloatInterval(2.0, 4.0),
            FloatInterval(1.0, 3.0),
            FloatInterval(0.0, 1.0)
        ]
        fset = FloatSet(intervals)
        self.assertEqual(len(fset), 1)
        self.assertEqual(fset.intervals[0][0], 0.0)
        self.assertEqual(fset.intervals[0][1], 4.0)

        # Disjoint intervals should stay separate
        intervals = [
            FloatInterval(0.0, 1.0),
            FloatInterval(3.0, 4.0),
            FloatInterval(6.0, 7.0)
        ]
        fset = FloatSet(intervals)
        self.assertEqual(len(fset), 3)

    def test_contains(self):
        """Test point containment in set."""
        fset = FloatSet([FloatInterval(0.0, 1.0), FloatInterval(3.0, 4.0)])

        self.assertTrue(0.5 in fset)
        self.assertTrue(0.0 in fset)
        self.assertTrue(1.0 in fset)
        self.assertTrue(3.5 in fset)
        self.assertFalse(1.5 in fset)
        self.assertFalse(2.5 in fset)
        self.assertFalse(4.5 in fset)

    def test_union(self):
        """Test set union."""
        set1 = FloatSet([FloatInterval(0.0, 2.0), FloatInterval(4.0, 6.0)])
        set2 = FloatSet([FloatInterval(1.0, 5.0)])

        union = set1.union(set2)
        self.assertEqual(len(union), 1)  # Should merge to single interval
        self.assertEqual(union.intervals[0][0], 0.0)
        self.assertEqual(union.intervals[0][1], 6.0)

        # With empty set
        empty = FloatSet()
        self.assertEqual(set1.union(empty), set1)
        self.assertEqual(empty.union(set1), set1)

    def test_intersection(self):
        """Test set intersection."""
        set1 = FloatSet([FloatInterval(0.0, 3.0), FloatInterval(5.0, 8.0)])
        set2 = FloatSet([FloatInterval(2.0, 6.0)])

        inter = set1.intersection(set2)
        self.assertEqual(len(inter), 2)
        self.assertEqual(inter.intervals[0][0], 2.0)
        self.assertEqual(inter.intervals[0][1], 3.0)
        self.assertEqual(inter.intervals[1][0], 5.0)
        self.assertEqual(inter.intervals[1][1], 6.0)

        # With empty set
        empty = FloatSet()
        self.assertTrue(set1.intersection(empty).is_empty())
        self.assertTrue(empty.intersection(set1).is_empty())

    def test_difference(self):
        """Test set difference."""
        set1 = FloatSet([FloatInterval(0.0, 5.0)])
        set2 = FloatSet([FloatInterval(1.0, 2.0), FloatInterval(3.0, 4.0)])

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
        set1 = FloatSet([FloatInterval(0.0, 2.0), FloatInterval(4.0, 6.0)])
        set2 = FloatSet([FloatInterval(1.0, 5.0)])

        sym_diff = set1.symmetric_difference(set2)
        # Should be: [0,1) ∪ (2,4) ∪ (5,6]
        self.assertEqual(len(sym_diff), 3)

    def test_operators(self):
        """Test operator overloads."""
        set1 = FloatSet([FloatInterval(0.0, 2.0)])
        set2 = FloatSet([FloatInterval(1.0, 3.0)])

        # Union
        union = set1 | set2
        self.assertEqual(len(union), 1)
        self.assertEqual(union.intervals[0][0], 0.0)
        self.assertEqual(union.intervals[0][1], 3.0)

        # Intersection
        inter = set1 & set2
        self.assertEqual(len(inter), 1)
        self.assertEqual(inter.intervals[0][0], 1.0)
        self.assertEqual(inter.intervals[0][1], 2.0)

        # Difference
        diff = set1 - set2
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff.intervals[0][0], 0.0)
        self.assertEqual(diff.intervals[0][1],
                         math.nextafter(1.0, -math.inf))

        # Symmetric difference
        sym_diff = set1 ^ set2
        self.assertEqual(len(sym_diff), 2)

    def test_from_methods(self):
        """Test factory methods."""
        # From single interval
        fset1 = FloatSet.from_single_interval(0.0, 1.0)
        self.assertEqual(len(fset1), 1)
        self.assertEqual(fset1.intervals[0][0], 0.0)
        self.assertEqual(fset1.intervals[0][1], 1.0)

        # From multiple intervals
        intervals = [FloatInterval(0.0, 1.0), FloatInterval(2.0, 3.0)]
        fset2 = FloatSet.from_intervals(*intervals)
        self.assertEqual(len(fset2), 2)

    def test_to_tuple(self):
        """Test conversion to tuple."""
        fset = FloatSet([FloatInterval(0.0, 1.0), FloatInterval(2.0, 3.0)])
        tuples = fset.to_tuple()
        self.assertEqual(tuples, ((0.0, 1.0), (2.0, 3.0)))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and floating point precision."""

    def test_touching_intervals(self):
        """Test intervals that touch at endpoints."""
        a = FloatInterval(0.0, 1.0)
        b = FloatInterval(1.0, 2.0)

        # With math.nextafter, intervals that exactly touch should merge
        union = a.union(b)
        self.assertEqual(len(union), 1)
        self.assertEqual(union[0][0], 0.0)
        self.assertEqual(union[0][1], 2.0)

        # Intersection of touching intervals should be a single point
        inter = a.intersection(b)
        self.assertFalse(inter.is_empty())
        self.assertEqual(inter[0], 1.0)
        self.assertEqual(inter[1], 1.0)
        self.assertEqual(inter.length(), 0.0)

    def test_single_point_intervals(self):
        """Test single point intervals."""
        point = FloatInterval(3.0, 3.0)
        self.assertFalse(point.is_empty())
        self.assertEqual(point.length(), 0.0)
        self.assertTrue(3.0 in point)
        self.assertFalse(3.1 in point)

        # Operations with point intervals
        interval = FloatInterval(2.0, 4.0)
        self.assertTrue(point.is_subset(interval))
        self.assertEqual(interval.intersection(point), point)
        self.assertEqual(interval.union(point), (FloatInterval(2.0, 4.0),))

    def test_floating_point_precision(self):
        """Test handling of floating point precision."""
        # Use nextafter to create intervals that are truly disjoint
        a = FloatInterval(0.0, 1.0)
        b = FloatInterval(math.nextafter(1.0, math.inf), 2.0)

        # Should be considered disjoint (b starts just after 1.0)
        union = a.union(b)
        self.assertEqual(len(union), 2)

        # Intersection should be empty
        inter = a.intersection(b)
        self.assertTrue(inter.is_empty())


class TestRealModule(unittest.TestCase):
    """Test the float-based real aliases."""

    def test_real_aliases(self):
        """Test that geo.real exposes the active float model."""
        self.assertIs(real, float)
        self.assertIs(realset, FloatSet)
        self.assertTrue(EMPTY_REAL_INTERVAL.is_empty())
        self.assertTrue(ALL_REALS_INTERVAL.is_full())


if __name__ == "__main__":
    unittest.main()
