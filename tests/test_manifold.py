"""Tests for manifold protocols and local manifold charts."""

import unittest
from dataclasses import dataclass

from geo import (
    Atlas,
    Chart,
    ChartTransition,
    EuclideanNeighborhood,
    FloatPoint,
    Manifold,
    ManifoldChart,
)


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
            name="line-chart",
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


class TestAtlasAndTransitions(unittest.TestCase):
    """Test chart transitions and finite atlases."""

    def test_chart_transition(self):
        """Transitions should convert coordinates between charts."""
        manifold = OpenIntervalManifold(-10.0, 10.0)
        image_a = EuclideanNeighborhood.box((-10.0, 10.0))
        image_b = EuclideanNeighborhood.box((-9.0, 11.0))
        chart_a = ManifoldChart(
            lambda point: FloatPoint(point.x),
            lambda coordinates: LinePoint(coordinates[0]),
            dim=1,
            domain_contains=manifold.contains,
            image=image_a,
            name="a",
        )
        chart_b = ManifoldChart(
            lambda point: FloatPoint(point.x + 1.0),
            lambda coordinates: LinePoint(coordinates[0] - 1.0),
            dim=1,
            domain_contains=manifold.contains,
            image=image_b,
            name="b",
        )
        transition = ChartTransition(chart_a, chart_b, name="b o a^-1")
        self.assertIsInstance(transition, Chart)
        self.assertEqual(transition(FloatPoint(2.0)).to_tuple(), (3.0,))
        self.assertEqual(transition.inverse(FloatPoint(3.0)).to_tuple(), (2.0,))

    def test_atlas(self):
        """An atlas should hold charts and build transitions between them."""
        manifold = OpenIntervalManifold(-10.0, 10.0)
        chart_a = ManifoldChart(
            lambda point: FloatPoint(point.x),
            lambda coordinates: LinePoint(coordinates[0]),
            dim=1,
            domain_contains=manifold.contains,
            image=EuclideanNeighborhood.box((-10.0, 10.0)),
            name="a",
        )
        chart_b = ManifoldChart(
            lambda point: FloatPoint(point.x + 1.0),
            lambda coordinates: LinePoint(coordinates[0] - 1.0),
            dim=1,
            domain_contains=manifold.contains,
            image=EuclideanNeighborhood.box((-9.0, 11.0)),
            name="b",
        )
        atlas = Atlas(manifold, chart_a, chart_b, name="line")
        self.assertEqual(len(atlas), 2)
        transition = atlas.transition(0, 1)
        self.assertEqual(transition(FloatPoint(0.5)).to_tuple(), (1.5,))


if __name__ == "__main__":
    unittest.main()
