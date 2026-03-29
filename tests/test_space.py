"""Tests for the visualization-aware space layer."""

import math
import unittest

from geo import (
    EuclideanPlaneSpace,
    FloatCirclePoint,
    FloatPoint,
    RealLineSpace,
    Space,
    SphereSpace,
    TorusPoint,
    TorusSpace,
    UnitCircleSpace,
)


class TestSpaceProtocol(unittest.TestCase):
    """Test the visualization-aware space protocol."""

    def test_standard_spaces_satisfy_space_protocol(self):
        """Standard spaces should satisfy the ``Space`` protocol."""
        self.assertIsInstance(RealLineSpace(), Space)
        self.assertIsInstance(UnitCircleSpace(), Space)
        self.assertIsInstance(EuclideanPlaneSpace(), Space)
        self.assertIsInstance(SphereSpace(), Space)
        self.assertIsInstance(TorusSpace(), Space)

    def test_real_line_visualization(self):
        """The real line should embed into 2D and 3D coordinates."""
        space = RealLineSpace()
        self.assertEqual(space.space_kind, "real-line")
        self.assertEqual(space.to_2d(3.5), (3.5, 0.0))
        self.assertEqual(space.to_3d(-2.0), (-2.0, 0.0, 0.0))

    def test_unit_circle_visualization(self):
        """The unit circle should use its standard Euclidean embedding."""
        space = UnitCircleSpace()
        point = FloatCirclePoint(math.pi / 2.0)
        self.assertEqual(space.space_kind, "unit-circle")
        self.assertAlmostEqual(space.to_2d(point)[0], 0.0)
        self.assertAlmostEqual(space.to_2d(point)[1], 1.0)
        self.assertEqual(space.to_3d(0.0), (1.0, 0.0, 0.0))

    def test_euclidean_plane_visualization(self):
        """Euclidean space should use orthographic coordinates."""
        space = EuclideanPlaneSpace()
        self.assertEqual(space.space_kind, "euclidean")
        self.assertEqual(space.to_2d(FloatPoint(1.0, 2.0)), (1.0, 2.0))
        self.assertEqual(
            space.to_3d(FloatPoint(1.0, 2.0)),
            (1.0, 2.0, 0.0),
        )

    def test_sphere_distance_and_visualization(self):
        """The sphere should use intrinsic distance and explicit embeddings."""
        space = SphereSpace(radius=2.0)
        equator_zero = space.point_from_angles(0.0, 0.0)
        equator_quarter = space.point_from_angles(math.pi / 2.0, 0.0)

        self.assertEqual(space.space_kind, "sphere")
        self.assertAlmostEqual(
            space.distance(equator_zero, equator_quarter),
            math.pi,
        )
        self.assertEqual(
            space.to_3d(equator_zero),
            (2.0, 0.0, 0.0),
        )
        longitude, latitude = space.to_2d(
            equator_quarter,
            method="equirectangular",
        )
        self.assertAlmostEqual(longitude, math.pi / 2.0)
        self.assertAlmostEqual(latitude, 0.0)

    def test_sphere_stereographic_projection_rejects_north_pole(self):
        """Stereographic projection should reject the north pole."""
        space = SphereSpace()
        north_pole = FloatPoint(0.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            space.to_2d(north_pole)

    def test_torus_distance_and_visualization(self):
        """The torus should use the flat product metric."""
        space = TorusSpace(major_radius=3.0, minor_radius=1.0)
        first = TorusPoint(0.0, 0.0)
        second = TorusPoint(math.pi, 0.0)

        self.assertEqual(space.space_kind, "torus")
        self.assertAlmostEqual(space.distance(first, second), math.pi)
        self.assertEqual(space.to_2d(first), (0.0, 0.0))
        self.assertEqual(space.to_3d(first), (4.0, 0.0, 0.0))

    def test_torus_point_requires_two_angles(self):
        """A torus point should reject ambiguous one-angle construction."""
        with self.assertRaises(TypeError):
            TorusPoint(1.0)
