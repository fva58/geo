"""Tests for transforms between spaces."""

import math
import unittest

from geo import (
    FloatPoint,
    RealLineSpace,
    SphereSpace,
    TorusSpace,
    Transform,
    UnitCircleSpace,
    identity_transform,
    visualization_transform_2d,
    visualization_transform_3d,
)


class TestTransforms(unittest.TestCase):
    """Test the transform layer."""

    def test_identity_transform_keeps_points(self):
        """The identity transform should keep points unchanged."""
        space = RealLineSpace()
        transform = identity_transform(space)
        self.assertIsInstance(transform, Transform)
        self.assertEqual(transform(2.5), 2.5)
        self.assertIs(transform.source_space, space)
        self.assertIs(transform.target_space, space)

    def test_visualization_transform_2d_for_circle(self):
        """A 2D visualization transform should map into Euclidean points."""
        circle = UnitCircleSpace()
        transform = visualization_transform_2d(circle)
        image_point = transform(math.pi / 2.0)

        self.assertIsInstance(transform, Transform)
        self.assertEqual(transform.target_space.dim, 2)
        self.assertEqual(image_point.dim, 2)
        self.assertAlmostEqual(image_point[0], 0.0)
        self.assertAlmostEqual(image_point[1], 1.0)

    def test_visualization_transform_3d_for_sphere(self):
        """A 3D visualization transform should expose the sphere embedding."""
        sphere = SphereSpace(radius=2.0)
        transform = visualization_transform_3d(sphere)
        image_point = transform(sphere.point_from_angles(0.0, 0.0))

        self.assertEqual(transform.target_space.dim, 3)
        self.assertEqual(image_point, FloatPoint(2.0, 0.0, 0.0))

    def test_visualization_transform_2d_for_torus(self):
        """A 2D visualization transform should expose the flat torus chart."""
        torus = TorusSpace()
        transform = visualization_transform_2d(torus)
        image_point = transform((math.pi, math.pi / 2.0))

        self.assertEqual(transform.target_space.dim, 2)
        self.assertAlmostEqual(image_point[0], math.pi)
        self.assertAlmostEqual(image_point[1], math.pi / 2.0)

    def test_transform_composition(self):
        """Transforms should compose over the same intermediate space."""
        space = RealLineSpace()
        identity = identity_transform(space)
        line_to_plane = visualization_transform_2d(space)
        composed = identity.then(line_to_plane)

        self.assertEqual(composed(3.0), FloatPoint(3.0, 0.0))

    def test_transform_composition_requires_same_intermediate_space(self):
        """Composition should reject different intermediate space instances."""
        first = identity_transform(RealLineSpace())
        second = visualization_transform_2d(RealLineSpace())

        with self.assertRaises(ValueError):
            first.then(second)

    def test_transform_rejects_points_outside_source_space(self):
        """Transforms should validate source-space membership."""
        sphere = SphereSpace()
        transform = visualization_transform_3d(sphere)

        with self.assertRaises(ValueError):
            transform((0.0, 0.0, 0.0))
