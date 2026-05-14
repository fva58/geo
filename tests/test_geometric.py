"""Tests for cones and geometric objects."""

import unittest
from dataclasses import dataclass

from geo.cone import (
    CircleSphereObject,
    Cone,
    DirectionSetSphereObject,
    EuclideanCone,
    LocalConeModel,
    RadialCone,
    SphereObject,
    SphericalCone,
)
from geo.euclidean import EuclideanNeighborhood, Point, Vector
from geo.gobject import ChartedGeometricObject, GeometricObjectProtocol
from geo.space.base import SpaceChart as ManifoldChart
from geo.space.circle import Set as CircleSet


@dataclass(frozen=True)
class LinePoint:
    """Simple one-dimensional manifold point."""

    x: float


class LineManifold:
    """Simple one-dimensional manifold."""

    dim = 1

    def contains(self, point: LinePoint) -> bool:
        """Every finite line point belongs to the manifold."""
        return isinstance(point, LinePoint)

    def __contains__(self, point: LinePoint) -> bool:
        """Every finite line point belongs to the manifold."""
        return self.contains(point)


class TestEuclideanCone(unittest.TestCase):
    """Test cases for ``EuclideanCone``."""

    def test_cone_protocol_and_membership(self):
        """A Euclidean cone should satisfy the cone protocol."""
        cone = EuclideanCone(
            1,
            contains=lambda point: point[0] >= 0.0,
            neighborhood=EuclideanNeighborhood.box((0.0, 10.0)),
        )
        self.assertIsInstance(cone, Cone)
        self.assertIn(Point(0.0), cone)
        self.assertIn(Point(3.0), cone)
        self.assertNotIn(Point(-1.0), cone)

    def test_whole_cone(self):
        """The whole-space cone should contain every point of matching dimension."""
        cone = EuclideanCone.whole(2)
        self.assertIn(Point(-1.0, 2.0), cone)


class TestRadialCone(unittest.TestCase):
    """Test cases for ``RadialCone``."""

    def test_half_line_from_directions(self):
        """A radial cone should be definable by admissible directions."""
        cone = RadialCone(
            1,
            contains_direction=lambda direction: direction[0] >= 0.0,
            neighborhood=EuclideanNeighborhood.box((-10.0, 10.0)),
        )
        self.assertIn(Point(0.0), cone)
        self.assertIn(Point(3.0), cone)
        self.assertNotIn(Point(-3.0), cone)

    def test_first_quadrant(self):
        """A radial cone can model a sector in higher dimension."""
        cone = RadialCone(
            2,
            contains_direction=lambda direction: (
                direction[0] >= 0.0 and direction[1] >= 0.0
            ),
            neighborhood=EuclideanNeighborhood.box((-10.0, 10.0), (-10.0, 10.0)),
        )
        self.assertIn(Point(1.0, 2.0), cone)
        self.assertIn(Point(0.0, 0.0), cone)
        self.assertNotIn(Point(-1.0, 2.0), cone)
        self.assertTrue(
            isinstance(Point(1.0, 0.0) - Point(0.0, 0.0), Vector)
        )


class TestSphereObjectsAndSphericalCones(unittest.TestCase):
    """Test cases for explicit sphere objects and the cones they induce."""

    def test_direction_set_sphere_object(self):
        """Sphere objects should classify directions on the unit sphere."""
        sphere_object = DirectionSetSphereObject(
            2,
            contains=lambda direction: direction[0] >= 0.0 and direction[1] >= 0.0,
        )
        self.assertIsInstance(sphere_object, SphereObject)
        self.assertIn(Vector(1.0, 0.0), sphere_object)
        self.assertIn(Vector(1.0, 1.0), sphere_object)
        self.assertNotIn(Vector(-1.0, 1.0), sphere_object)
        self.assertNotIn(Vector.zero(2), sphere_object)

    def test_spherical_cone(self):
        """A spherical cone should derive membership from its sphere object."""
        sphere_object = DirectionSetSphereObject(
            2,
            contains=lambda direction: direction[0] >= 0.0 and direction[1] >= 0.0,
        )
        cone = SphericalCone(
            sphere_object,
            neighborhood=EuclideanNeighborhood.box((-10.0, 10.0), (-10.0, 10.0)),
        )
        self.assertIn(Point(2.0, 3.0), cone)
        self.assertIn(Point(0.0, 0.0), cone)
        self.assertNotIn(Point(-2.0, 3.0), cone)

    def test_circle_sphere_object(self):
        """A sphere object in dimension 2 can be defined by a circle set."""
        circle_object = CircleSphereObject(
            CircleSet.from_single_interval(0.0, 1.5707963267948966),
        )
        self.assertIsInstance(circle_object, SphereObject)
        self.assertIn(Vector(1.0, 0.0), circle_object)
        self.assertIn(Vector(1.0, 1.0), circle_object)
        self.assertNotIn(Vector(-1.0, 1.0), circle_object)

    def test_spherical_cone_from_circle_object(self):
        """A spherical cone can use the existing circle geometry as its base."""
        base = CircleSphereObject(
            CircleSet.from_single_interval(0.0, 1.5707963267948966)
        )
        cone = SphericalCone(
            base,
            neighborhood=EuclideanNeighborhood.box((-10.0, 10.0), (-10.0, 10.0)),
        )
        self.assertIn(Point(2.0, 3.0), cone)
        self.assertNotIn(Point(-2.0, 3.0), cone)


class TestChartedGeometricObject(unittest.TestCase):
    """Test cases for geometric objects with local cone models."""

    def test_geometric_object_protocol(self):
        """A charted geometric object should satisfy the public protocol."""
        manifold = LineManifold()
        chart = ManifoldChart(
            lambda point: Point(point.x),
            lambda coordinates: LinePoint(coordinates[0]),
            dim=1,
            domain_contains=manifold.contains,
            image=EuclideanNeighborhood.box((-10.0, 10.0)),
        )

        def local_model(point: LinePoint) -> LocalConeModel[LinePoint]:
            if point.x == 0.0:
                cone = EuclideanCone(
                    1,
                    contains=lambda coordinates: coordinates[0] >= 0.0,
                    neighborhood=EuclideanNeighborhood.box((0.0, 10.0)),
                )
            else:
                cone = EuclideanCone.whole(1)
            return LocalConeModel(chart, cone)

        obj = ChartedGeometricObject(
            manifold,
            contains=lambda point: point.x >= 0.0,
            local_model=local_model,
        )

        self.assertIsInstance(obj, GeometricObjectProtocol)
        self.assertIn(LinePoint(1.0), obj)
        self.assertNotIn(LinePoint(-1.0), obj)

        boundary_model = obj.local_model_at(LinePoint(0.0))
        self.assertIn(Point(1.0), boundary_model.cone)
        self.assertNotIn(Point(-1.0), boundary_model.cone)

        interior_model = obj.local_model_at(LinePoint(2.0))
        self.assertIn(Point(-3.0), interior_model.cone)

    def test_local_model_requires_point_in_object(self):
        """Requesting a local model outside the object should fail."""
        manifold = LineManifold()
        chart = ManifoldChart(
            lambda point: Point(point.x),
            lambda coordinates: LinePoint(coordinates[0]),
            dim=1,
            domain_contains=manifold.contains,
            image=EuclideanNeighborhood.box((-10.0, 10.0)),
        )
        obj = ChartedGeometricObject(
            manifold,
            contains=lambda point: point.x >= 0.0,
            local_model=lambda point: LocalConeModel(chart, EuclideanCone.whole(1)),
        )
        with self.assertRaises(ValueError):
            obj.local_model_at(LinePoint(-1.0))


if __name__ == "__main__":
    unittest.main()
