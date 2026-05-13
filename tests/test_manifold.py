"""Tests for manifold protocols and local manifold charts."""

import unittest
from dataclasses import dataclass

from geo.diffeomorphism import Chart
from geo.euclidean import EuclideanNeighborhood, FloatPoint
from geo.manifold import Manifold, ManifoldChart


@dataclass(frozen=True)
class LinePoint:
    """Simple point on a one-dimensional manifold."""

    x: float


class OpenIntervalManifold:
    """Simple manifold modeled by an open interval."""

    def __init__(self, left: float, right: float) -> None:
        """Initialize the manifold bounds."""
        self.left = left
        self.right = right
        self._dim = 1

    @property
    def dim(self) -> int:
        """Return the manifold dimension."""
        return self._dim

    def contains(self, point: LinePoint) -> bool:
        """Check point membership."""
        return self.left < point.x < self.right

    def __contains__(self, point: LinePoint) -> bool:
        """Check point membership."""
        return self.contains(point)


class TestManifoldChart(unittest.TestCase):
    """Test cases for manifold abstractions."""

    def test_manifold_protocol(self):
        """A simple manifold object should satisfy the manifold protocol."""
        manifold = OpenIntervalManifold(-1.0, 1.0)
        self.assertIsInstance(manifold, Manifold)
        self.assertIn(LinePoint(0.0), manifold)
        self.assertNotIn(LinePoint(2.0), manifold)

    def test_local_chart_on_custom_point_type(self):
        """A manifold chart should work with non-Euclidean source points."""
        manifold = OpenIntervalManifold(-1.0, 1.0)
        image = EuclideanNeighborhood.box((-1.0, 1.0))
        chart = ManifoldChart(
            lambda point: FloatPoint(point.x),
            lambda coordinates: LinePoint(coordinates[0]),
            dim=1,
            domain_contains=manifold.contains,
            image=image,
        )
        point = LinePoint(0.25)
        coordinates = chart(point)
        self.assertIsInstance(chart, Chart)
        self.assertEqual(coordinates.to_tuple(), (0.25,))
        self.assertEqual(chart.inverse(coordinates), point)

    def test_domain_and_image_restrictions(self):
        """The chart should reject points and coordinates outside its patch."""
        manifold = OpenIntervalManifold(-1.0, 1.0)
        image = EuclideanNeighborhood.box((-1.0, 1.0))
        chart = ManifoldChart(
            lambda point: FloatPoint(point.x),
            lambda coordinates: LinePoint(coordinates[0]),
            dim=1,
            domain_contains=manifold.contains,
            image=image,
        )
        with self.assertRaises(ValueError):
            chart(LinePoint(2.0))
        with self.assertRaises(ValueError):
            chart.inverse(FloatPoint(2.0))





if __name__ == "__main__":
    unittest.main()
