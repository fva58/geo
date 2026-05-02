"""Tests for transforms between spaces."""

import unittest

from geo.transform import Transform, identity_transform
from geo.space import RealLine


class TestTransforms(unittest.TestCase):
    """Test the transform layer."""

    def test_identity_transform_keeps_points(self):
        """The identity transform should keep points unchanged."""
        space = RealLine()
        transform = identity_transform(space)
        self.assertIsInstance(transform, Transform)
        self.assertEqual(transform(2.5), 2.5)
        self.assertIs(transform.source_space, space)
        self.assertIs(transform.target_space, space)

    def test_transform_composition_requires_same_intermediate_space(self):
        """Composition should reject different intermediate space instances."""
        first = identity_transform(RealLine())
        second = identity_transform(RealLine())

        with self.assertRaises(ValueError):
            first.then(second)
