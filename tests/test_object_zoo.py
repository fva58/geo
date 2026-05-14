"""Tests for concrete geometric objects."""

import math
import unittest

from geo.euclidean import Point
from geo.gobject import GeometricObjectProtocol
from geo import space as space_pkg
from geo.space.circle import Point as CirclePoint, Set as CircleSet


class TestZeroDimensionalObjects(unittest.TestCase):
    """Test singleton geometric objects."""

    def test_real_point_object(self):
        """A real singleton should have a point cone."""
        obj = space_pkg.line.Space().point(2.0)
        self.assertIsInstance(obj, GeometricObjectProtocol)
        self.assertIn(2.0, obj)
        self.assertNotIn(3.0, obj)

        model = obj.local_model_at(2.0)
        self.assertIn(Point(0.0), model.cone)
        self.assertNotIn(Point(1.0), model.cone)

    def test_circle_point_object(self):
        """A circle singleton should have a point cone."""
        obj = space_pkg.circle.Space().point(CirclePoint(math.pi / 4.0))
        self.assertIn(CirclePoint(math.pi / 4.0), obj)
        self.assertNotIn(CirclePoint(math.pi / 2.0), obj)

        model = obj.local_model_at(CirclePoint(math.pi / 4.0))
        self.assertIn(Point(0.0), model.cone)
        self.assertNotIn(Point(-1.0), model.cone)

    def test_euclidean_point_object(self):
        """A Euclidean singleton should have a point cone."""
        obj = space_pkg.euclidean.Space(2).point(Point(1.0, 2.0))
        self.assertIn(Point(1.0, 2.0), obj)
        self.assertNotIn(Point(1.0, 3.0), obj)

        model = obj.local_model_at(Point(1.0, 2.0))
        self.assertIn(Point(0.0, 0.0), model.cone)
        self.assertNotIn(Point(1.0, 0.0), model.cone)


class TestOneDimensionalObjects(unittest.TestCase):
    """Test one-dimensional objects on the line and circle."""

    def test_real_set_object_interval_and_point(self):
        """A real-line set should expose whole, half-line, and point models."""
        obj = space_pkg.line.Space().subset((0.0, 2.0), 5.0)
        self.assertIn(1.0, obj)
        self.assertIn(5.0, obj)
        self.assertNotIn(3.0, obj)

        interior = obj.local_model_at(1.0)
        self.assertIn(Point(-1.0), interior.cone)
        self.assertIn(Point(1.0), interior.cone)

        boundary = obj.local_model_at(0.0)
        self.assertIn(Point(1.0), boundary.cone)
        self.assertNotIn(Point(-1.0), boundary.cone)

        isolated = obj.local_model_at(5.0)
        self.assertIn(Point(0.0), isolated.cone)
        self.assertNotIn(Point(1.0), isolated.cone)

    def test_circle_set_object_interval_and_point(self):
        """A circle set should expose whole, half-line, and point models."""
        obj = space_pkg.circle.Space().subset(
            (
                CircleSet.from_single_interval(0.0, math.pi / 2.0),
                CirclePoint(math.pi),
            )
        )
        self.assertIn(CirclePoint(math.pi / 4.0), obj)
        self.assertIn(CirclePoint(math.pi), obj)
        self.assertNotIn(CirclePoint(3.0 * math.pi / 4.0), obj)

        interior = obj.local_model_at(CirclePoint(math.pi / 4.0))
        self.assertIn(Point(-1.0), interior.cone)
        self.assertIn(Point(1.0), interior.cone)

        boundary = obj.local_model_at(CirclePoint(0.0))
        self.assertIn(Point(1.0), boundary.cone)
        self.assertNotIn(Point(-1.0), boundary.cone)

        isolated = obj.local_model_at(CirclePoint(math.pi))
        self.assertIn(Point(0.0), isolated.cone)
        self.assertNotIn(Point(1.0), isolated.cone)


class TestPlanarObjects(unittest.TestCase):
    """Test concrete geometric objects in the Euclidean plane."""

    def test_whole_plane(self):
        """The whole plane should have the whole-plane local model."""
        plane = space_pkg.euclidean.Space(2).whole_plane()
        self.assertIn(Point(1.0, 2.0), plane)

        model = plane.local_model_at(Point(1.0, 2.0))
        self.assertIn(Point(-3.0, 4.0), model.cone)

    def test_half_plane(self):
        """A half-plane should distinguish interior from boundary points."""
        plane = space_pkg.euclidean.Space(2).half_plane((0.0, 1.0), offset=0.0)
        self.assertIn(Point(2.0, 3.0), plane)
        self.assertIn(Point(0.0, 0.0), plane)
        self.assertNotIn(Point(0.0, -1.0), plane)

        interior = plane.local_model_at(Point(0.0, 2.0))
        self.assertIn(Point(-1.0, -1.0), interior.cone)

        boundary = plane.local_model_at(Point(2.0, 0.0))
        self.assertIn(Point(0.0, 1.0), boundary.cone)
        self.assertNotIn(Point(0.0, -1.0), boundary.cone)

    def test_planar_angle(self):
        """A planar angle should expose apex, boundary, and interior models."""
        angle = space_pkg.euclidean.Space(2).angle(
            Point(0.0, 0.0),
            0.0,
            math.pi / 2.0,
        )
        self.assertIn(Point(2.0, 3.0), angle)
        self.assertIn(Point(0.0, 0.0), angle)
        self.assertNotIn(Point(-1.0, 1.0), angle)

        apex = angle.local_model_at(Point(0.0, 0.0))
        self.assertIn(Point(1.0, 1.0), apex.cone)
        self.assertNotIn(Point(-1.0, 1.0), apex.cone)

        boundary = angle.local_model_at(Point(2.0, 0.0))
        self.assertIn(Point(0.0, 1.0), boundary.cone)
        self.assertNotIn(Point(0.0, -1.0), boundary.cone)

        interior = angle.local_model_at(Point(2.0, 2.0))
        self.assertIn(Point(-1.0, -1.0), interior.cone)


class TestHigherDimensionalEuclideanObjects(unittest.TestCase):
    """Test the extended Euclidean object zoo."""

    def test_whole_space(self):
        """The whole space should contain every matching point."""
        space = space_pkg.euclidean.Space(3).whole()
        self.assertIn(Point(1.0, 2.0, 3.0), space)

        model = space.local_model_at(Point(1.0, 2.0, 3.0))
        self.assertIn(Point(-1.0, 4.0, 0.5), model.cone)

    def test_hyperplane_and_half_space(self):
        """Hyperplanes and half-spaces should expose expected cones."""
        ambient = space_pkg.euclidean.Space(3)
        hyperplane = ambient.hyperplane((0.0, 0.0, 1.0), offset=0.0)
        half_space = ambient.half_space((0.0, 0.0, 1.0), offset=0.0)

        self.assertIn(Point(1.0, 2.0, 0.0), hyperplane)
        self.assertNotIn(Point(1.0, 2.0, 1.0), hyperplane)
        self.assertIn(Point(1.0, 2.0, 3.0), half_space)
        self.assertNotIn(Point(1.0, 2.0, -1.0), half_space)

        hyper_model = hyperplane.local_model_at(Point(1.0, 2.0, 0.0))
        self.assertIn(Point(1.0, 0.0, 0.0), hyper_model.cone)
        self.assertNotIn(Point(0.0, 0.0, 1.0), hyper_model.cone)

        boundary_model = half_space.local_model_at(Point(0.0, 0.0, 0.0))
        self.assertIn(Point(0.0, 0.0, 1.0), boundary_model.cone)
        self.assertNotIn(Point(0.0, 0.0, -1.0), boundary_model.cone)

    def test_sphere_and_ball(self):
        """Spheres and balls should separate surface from interior."""
        ambient = space_pkg.euclidean.Space(3)
        sphere = ambient.sphere(Point(0.0, 0.0, 0.0), 1.0)
        ball = ambient.ball(Point(0.0, 0.0, 0.0), 1.0)

        self.assertIn(Point(1.0, 0.0, 0.0), sphere)
        self.assertNotIn(Point(0.0, 0.0, 0.0), sphere)
        self.assertIn(Point(0.0, 0.0, 0.0), ball)
        self.assertIn(Point(1.0, 0.0, 0.0), ball)
        self.assertNotIn(Point(2.0, 0.0, 0.0), ball)

        sphere_model = sphere.local_model_at(Point(1.0, 0.0, 0.0))
        self.assertIn(Point(0.0, 1.0, 0.0), sphere_model.cone)
        self.assertNotIn(Point(1.0, 0.0, 0.0), sphere_model.cone)

        ball_model = ball.local_model_at(Point(1.0, 0.0, 0.0))
        self.assertIn(Point(-1.0, 0.0, 0.0), ball_model.cone)
        self.assertNotIn(Point(1.0, 0.0, 0.0), ball_model.cone)

    def test_ellipsoid_and_surface(self):
        """Ellipsoids should use affine images of the unit ball."""
        ambient = space_pkg.euclidean.Space(2)
        surface = ambient.ellipsoid_surface(
            Point(0.0, 0.0),
            ((2.0, 0.0), (0.0, 3.0)),
        )
        body = ambient.ellipsoid(
            Point(0.0, 0.0),
            ((2.0, 0.0), (0.0, 3.0)),
        )

        self.assertIn(Point(2.0, 0.0), surface)
        self.assertNotIn(Point(1.0, 0.0), surface)
        self.assertIn(Point(1.0, 0.0), body)
        self.assertNotIn(Point(3.0, 0.0), body)

        surface_model = surface.local_model_at(Point(2.0, 0.0))
        self.assertIn(Point(0.0, 1.0), surface_model.cone)
        self.assertNotIn(Point(1.0, 0.0), surface_model.cone)

        body_model = body.local_model_at(Point(2.0, 0.0))
        self.assertIn(Point(-1.0, 0.0), body_model.cone)
        self.assertNotIn(Point(1.0, 0.0), body_model.cone)

    def test_cube_and_cube_surface(self):
        """Cubes should distinguish body and surface."""
        ambient = space_pkg.euclidean.Space(3)
        surface = ambient.cube_surface(Point(0.0, 0.0, 0.0), 1.0)
        body = ambient.cube(Point(0.0, 0.0, 0.0), 1.0)

        self.assertIn(Point(1.0, 0.0, 0.0), surface)
        self.assertNotIn(Point(0.0, 0.0, 0.0), surface)
        self.assertIn(Point(0.0, 0.0, 0.0), body)
        self.assertIn(Point(1.0, 1.0, 1.0), body)
        self.assertNotIn(Point(2.0, 0.0, 0.0), body)

        surface_model = surface.local_model_at(Point(1.0, 0.0, 0.0))
        self.assertIn(Point(0.0, 1.0, 0.0), surface_model.cone)
        self.assertNotIn(Point(1.0, 0.0, 0.0), surface_model.cone)

        body_model = body.local_model_at(Point(1.0, 1.0, 1.0))
        self.assertIn(Point(-1.0, -1.0, -1.0), body_model.cone)
        self.assertNotIn(Point(1.0, 0.0, 0.0), body_model.cone)

    def test_parallelepiped_and_surface(self):
        """Parallelepipeds should support non-axis-aligned spans."""
        ambient = space_pkg.euclidean.Space(2)
        spanning_vectors = (
            (2.0, 0.0),
            (1.0, 1.0),
        )
        surface = ambient.parallelepiped_surface(
            Point(0.0, 0.0),
            spanning_vectors,
        )
        body = ambient.parallelepiped(Point(0.0, 0.0), spanning_vectors)

        self.assertIn(Point(2.0, 0.0), surface)
        self.assertNotIn(Point(0.0, 0.0), surface)
        self.assertIn(Point(0.0, 0.0), body)
        self.assertIn(Point(1.0, 0.5), body)
        self.assertNotIn(Point(4.0, 0.0), body)

        surface_model = surface.local_model_at(Point(2.0, 0.0))
        self.assertIn(Point(1.0, 1.0), surface_model.cone)
        self.assertNotIn(Point(1.0, 0.0), surface_model.cone)

        body_model = body.local_model_at(Point(2.0, 0.0))
        self.assertIn(Point(-1.0, 0.0), body_model.cone)
        self.assertNotIn(Point(1.0, 0.0), body_model.cone)

if __name__ == "__main__":
    unittest.main()
