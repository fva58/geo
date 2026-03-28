"""Tests for mapping protocols."""

import unittest

from geo import Chart, Diffeomorphism, InvertibleMap, Map


class ShiftByOne:
    """Simple diffeomorphism on the real line."""

    def __call__(self, point: float) -> float:
        """Apply the forward map."""
        return point + 1.0

    def inverse(self, point: float) -> float:
        """Apply the inverse map."""
        return point - 1.0


class NotADiffeomorphism:
    """Object missing the inverse operation."""

    def __call__(self, point: float) -> float:
        """Apply a one-way map."""
        return point


class SquareMap:
    """Simple one-way map."""

    def __call__(self, point: float) -> float:
        """Apply the map."""
        return point * point


class IdentityChart:
    """Simple chart from a line to its coordinate line."""

    def __call__(self, point: float) -> float:
        """Apply the coordinate map."""
        return point

    def inverse(self, point: float) -> float:
        """Apply the inverse coordinate map."""
        return point


class TestMappingProtocols(unittest.TestCase):
    """Test the public mapping protocols."""

    def test_map_protocol(self):
        """Any callable map should satisfy the base protocol."""
        self.assertIsInstance(SquareMap(), Map)
        self.assertIsInstance(ShiftByOne(), Map)

    def test_invertible_map_protocol(self):
        """Maps with an inverse should satisfy the invertible protocol."""
        mapping = ShiftByOne()
        self.assertIsInstance(mapping, InvertibleMap)
        self.assertEqual(mapping.inverse(3.0), 2.0)

    def test_runtime_checkable_protocol(self):
        """Objects with forward and inverse maps should satisfy the protocol."""
        mapping = ShiftByOne()
        self.assertIsInstance(mapping, Diffeomorphism)
        self.assertEqual(mapping(2.0), 3.0)
        self.assertEqual(mapping.inverse(3.0), 2.0)

    def test_missing_inverse_does_not_match_protocol(self):
        """Objects missing the inverse method should not satisfy the protocol."""
        self.assertNotIsInstance(NotADiffeomorphism(), Diffeomorphism)
        self.assertNotIsInstance(NotADiffeomorphism(), InvertibleMap)

    def test_chart_protocol(self):
        """Charts should satisfy the chart protocol and diffeomorphism protocol."""
        chart = IdentityChart()
        self.assertIsInstance(chart, Chart)
        self.assertIsInstance(chart, Diffeomorphism)
        self.assertEqual(chart(1.5), 1.5)
        self.assertEqual(chart.inverse(1.5), 1.5)


if __name__ == "__main__":
    unittest.main()
