"""Tests for the visualization-aware space layer."""

import math
import unittest

from geo.euclidean import FloatPoint
from geo.space import (
    Circle,
    Euclidean,
    Point,
    RealLine,
    Sphere,
    Torus,
    make_euclidean,
    make_sphere,
    make_torus,
)
from geo.space.base import Space
from geo.space.sphere import SpherePoint
from geo.space.torus import TorusPoint


class TestSpaceProtocol(unittest.TestCase):
    """Test the visualization-aware space protocol."""

    def test_standard_spaces_satisfy_space_protocol(self):
        """Standard spaces should satisfy the ``Space`` protocol."""
        self.assertIsInstance(RealLine(), Space)
        self.assertIsInstance(Circle(), Space)
        self.assertIsInstance(Euclidean(2), Space)
        self.assertIsInstance(Sphere(), Space)
        self.assertIsInstance(Torus(), Space)
        self.assertIsInstance(Point(), Space)

    def test_point_space_is_zero_dimensional(self):
        """The point space should contain only one point."""
        space = Point()

        self.assertEqual(space.dim, 0)
        self.assertIn(FloatPoint.origin(0), space)
        self.assertIn((), space)
        self.assertNotIn(0.0, space)
        self.assertEqual(space.distance((), FloatPoint.origin(0)), 0.0)

        neighborhood = space.neighborhood_at(())
        self.assertEqual(neighborhood.diameter(), 0.0)
        self.assertEqual(neighborhood.outer_radius(), 0.0)
        self.assertEqual(neighborhood.probe_points(), (FloatPoint.origin(0),))

    def test_space_factories_dispatch_on_dimension(self):
        """Factories should choose the correct family member by dimension."""
        self.assertIsInstance(make_euclidean(0), Point)
        self.assertIsInstance(make_euclidean(1), RealLine)
        self.assertIsInstance(make_euclidean(2), Euclidean)

        self.assertIsInstance(make_torus(0), Point)
        self.assertIsInstance(make_torus(1), Circle)
        self.assertIsInstance(make_torus(2), Torus)

        self.assertIsInstance(make_sphere(2), Sphere)

    def test_real_line_visualization(self):
        """The real line should still satisfy the space protocol."""
        self.assertIsInstance(RealLine(), Space)

    def test_unit_circle_visualization(self):
        """The unit circle should still satisfy the space protocol."""
        self.assertIsInstance(Circle(), Space)

    def test_euclidean_plane_visualization(self):
        """Euclidean space should still satisfy the space protocol."""
        self.assertIsInstance(Euclidean(2), Space)

    def test_sphere_distance_and_visualization(self):
        """The sphere should use intrinsic distance."""
        space = Sphere(radius=2.0)
        equator_zero = space.point_from_angles(0.0, 0.0)
        equator_quarter = space.point_from_angles(math.pi / 2.0, 0.0)

        self.assertAlmostEqual(
            space.distance(equator_zero, equator_quarter),
            math.pi,
        )

    def test_sphere_point_normalizes_any_nonzero_ambient_vector(self):
        """Sphere points should be represented by nonzero ambient vectors."""
        space = Sphere(radius=2.0)
        point = space.point((10.0, 0.0, 0.0))

        self.assertIsInstance(point, SpherePoint)
        self.assertEqual(point, SpherePoint(1.0, 0.0, 0.0, dim=2, radius=2.0))
        self.assertEqual(point.sphere_dim, 2)
        self.assertAlmostEqual(point.radius, 2.0)
        self.assertIn((3.0, 0.0, 0.0), space)
        self.assertNotIn((0.0, 0.0, 0.0), space)

    def test_sphere_point_object_and_cap(self):
        """Sphere spaces should expose singleton and cap objects."""
        space = Sphere()
        north_pole = space.point_from_angles(0.0, math.pi / 2.0)
        equator = space.point_from_angles(0.0, 0.0)
        south_pole = space.point_from_angles(0.0, -math.pi / 2.0)

        point_object = space.point_object(north_pole)
        hemisphere = space.cap(north_pole, math.pi / 2.0)

        self.assertIn(north_pole, point_object)
        self.assertNotIn(equator, point_object)
        self.assertIn(north_pole, hemisphere)
        self.assertIn(equator, hemisphere)
        self.assertNotIn(south_pole, hemisphere)

        boundary = hemisphere.local_model_at(equator)
        self.assertIn(FloatPoint(0.0, 1.0), boundary.cone)
        self.assertNotIn(FloatPoint(0.0, -1.0), boundary.cone)

    def test_higher_dimensional_sphere_uses_embedded_coordinates(self):
        """Higher-dimensional spheres should accept an explicit dimension."""
        space = Sphere(dim=3, radius=2.0)
        first = SpherePoint(2.0, 0.0, 0.0, 0.0, dim=3, radius=2.0)
        second = SpherePoint(0.0, 2.0, 0.0, 0.0, dim=3, radius=2.0)

        self.assertIn(first, space)
        self.assertIn((3.0, 0.0, 0.0, 0.0), space)
        self.assertNotIn(FloatPoint(2.0, 0.0, 0.0), space)
        self.assertAlmostEqual(space.distance(first, second), math.pi)

    def test_torus_distance_and_visualization(self):
        """The torus should use the flat product metric."""
        space = Torus(major_radius=3.0, minor_radius=1.0)
        first = TorusPoint(0.0, 0.0)
        second = TorusPoint(math.pi, 0.0)

        self.assertAlmostEqual(space.distance(first, second), math.pi)

    def test_torus_point_requires_two_angles(self):
        """A torus point should reject ambiguous one-angle construction."""
        with self.assertRaises(TypeError):
            TorusPoint(1.0)

    def test_torus_patch_local_model(self):
        """Torus patches should expose product boundary cones."""
        space = Torus()
        patch = space.patch((0.0, math.pi / 2.0), (0.0, math.pi / 2.0))

        self.assertIn(TorusPoint(math.pi / 4.0, math.pi / 4.0), patch)
        self.assertNotIn(TorusPoint(math.pi, math.pi / 4.0), patch)

        boundary = patch.local_model_at(TorusPoint(0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0, 1.0), boundary.cone)
        self.assertNotIn(FloatPoint(1.0, -1.0), boundary.cone)

    def test_higher_dimensional_torus_uses_all_angular_axes(self):
        """Higher-dimensional tori should use one circle factor per axis."""
        space = Torus(dim=3, radii=(3.0, 2.0, 1.0))
        first = TorusPoint(0.0, 0.0, 0.0)
        second = TorusPoint(math.pi, 0.0, 0.0)
        patch = space.patch(
            (0.0, math.pi / 2.0),
            (0.0, math.pi),
            (0.0, math.pi / 2.0),
        )

        self.assertIn(first, space)
        self.assertNotIn(TorusPoint(0.0, 0.0), space)
        self.assertAlmostEqual(space.distance(first, second), math.pi)
        self.assertIn(TorusPoint(math.pi / 4.0, math.pi / 2.0, math.pi / 4.0), patch)
        self.assertNotIn(TorusPoint(math.pi, math.pi / 2.0, math.pi / 4.0), patch)

        boundary = patch.local_model_at(TorusPoint(0.0, 0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 1.0, 1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0, 1.0, 1.0), boundary.cone)
