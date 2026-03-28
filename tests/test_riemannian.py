"""Tests for Riemannian spaces and geometric objects."""

import math
import unittest

from geo import (
    EuclideanPlaneSpace,
    FloatCirclePoint,
    FloatPoint,
    FloatVector,
    RealLineSpace,
    RiemannianGeometricObject,
    RiemannianSpace,
    UnitCircleSpace,
)


class TestRiemannianSpaces(unittest.TestCase):
    """Test standard Riemannian spaces."""

    def test_real_line_space(self):
        """The real line should satisfy the Riemannian protocol."""
        space = RealLineSpace()
        self.assertIsInstance(space, RiemannianSpace)
        self.assertEqual(space.metric_tensor(0.0), ((1.0,),))
        self.assertEqual(
            space.inner_product(0.0, FloatVector(2.0), FloatVector(3.0)),
            6.0,
        )
        self.assertEqual(space.norm(0.0, FloatVector(4.0)), 4.0)

    def test_unit_circle_space(self):
        """The unit circle should use the angular metric."""
        space = UnitCircleSpace()
        point = FloatCirclePoint(math.pi / 3.0)
        self.assertIsInstance(space, RiemannianSpace)
        self.assertEqual(space.metric_tensor(point), ((1.0,),))
        self.assertEqual(space.norm(point, FloatVector(2.0)), 2.0)

    def test_euclidean_plane_space(self):
        """The Euclidean plane should carry the identity metric."""
        space = EuclideanPlaneSpace()
        point = FloatPoint(1.0, 2.0)
        self.assertEqual(
            space.metric_tensor(point),
            ((1.0, 0.0), (0.0, 1.0)),
        )
        self.assertEqual(
            space.inner_product(
                point,
                FloatVector(1.0, 2.0),
                FloatVector(3.0, 4.0),
            ),
            11.0,
        )
        self.assertEqual(space.norm(point, FloatVector(3.0, 4.0)), 5.0)


class TestRiemannianObjects(unittest.TestCase):
    """Test geometric objects explicitly placed in Riemannian spaces."""

    def test_real_line_subset(self):
        """A real-line subset should keep its local cone models."""
        space = RealLineSpace()
        obj = space.subset((0.0, 2.0), 5.0, name="segment-and-point")
        self.assertIsInstance(obj, RiemannianGeometricObject)
        self.assertEqual(obj.space, space)
        self.assertIn(1.0, obj)
        self.assertIn(5.0, obj)
        self.assertNotIn(3.0, obj)

        boundary = obj.local_model_at(0.0)
        self.assertIn(FloatPoint(1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0), boundary.cone)

    def test_circle_arc(self):
        """An arc should be a geometric object in the unit circle space."""
        space = UnitCircleSpace()
        obj = space.arc(0.0, math.pi / 2.0, name="quarter-arc")
        self.assertIn(FloatCirclePoint(math.pi / 4.0), obj)
        self.assertNotIn(FloatCirclePoint(math.pi), obj)

        boundary = obj.local_model_at(FloatCirclePoint(0.0))
        self.assertIn(FloatPoint(1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0), boundary.cone)

    def test_plane_objects(self):
        """Standard planar objects should live in the Euclidean plane space."""
        space = EuclideanPlaneSpace()
        half_plane = space.half_plane((0.0, 1.0), name="upper")
        angle = space.angle(FloatPoint(0.0, 0.0), 0.0, math.pi / 2.0)

        self.assertIn(FloatPoint(1.0, 1.0), half_plane)
        self.assertNotIn(FloatPoint(1.0, -1.0), half_plane)
        self.assertIn(FloatPoint(1.0, 1.0), angle)
        self.assertNotIn(FloatPoint(-1.0, 1.0), angle)

        apex_model = angle.local_model_at(FloatPoint(0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 1.0), apex_model.cone)
        self.assertNotIn(FloatPoint(-1.0, 1.0), apex_model.cone)


if __name__ == "__main__":
    unittest.main()
