"""Tests for concrete geometric objects."""

import math
import unittest

from geo import (
    CirclePointObject,
    CircleSetObject,
    EuclideanPointObject,
    FloatCirclePoint,
    FloatCircleSet,
    FloatPoint,
    GeometricObject,
    HalfPlane,
    PlanarAngle,
    RealPointObject,
    RealSetObject,
    WholePlane,
)


class TestZeroDimensionalObjects(unittest.TestCase):
    """Test singleton geometric objects."""

    def test_real_point_object(self):
        """A real singleton should have a point cone."""
        obj = RealPointObject(2.0)
        self.assertIsInstance(obj, GeometricObject)
        self.assertIn(2.0, obj)
        self.assertNotIn(3.0, obj)

        model = obj.local_model_at(2.0)
        self.assertIn(FloatPoint(0.0), model.cone)
        self.assertNotIn(FloatPoint(1.0), model.cone)

    def test_circle_point_object(self):
        """A circle singleton should have a point cone."""
        obj = CirclePointObject(FloatCirclePoint(math.pi / 4.0))
        self.assertIn(FloatCirclePoint(math.pi / 4.0), obj)
        self.assertNotIn(FloatCirclePoint(math.pi / 2.0), obj)

        model = obj.local_model_at(FloatCirclePoint(math.pi / 4.0))
        self.assertIn(FloatPoint(0.0), model.cone)
        self.assertNotIn(FloatPoint(-1.0), model.cone)

    def test_euclidean_point_object(self):
        """A Euclidean singleton should have a point cone."""
        obj = EuclideanPointObject(FloatPoint(1.0, 2.0))
        self.assertIn(FloatPoint(1.0, 2.0), obj)
        self.assertNotIn(FloatPoint(1.0, 3.0), obj)

        model = obj.local_model_at(FloatPoint(1.0, 2.0))
        self.assertIn(FloatPoint(0.0, 0.0), model.cone)
        self.assertNotIn(FloatPoint(1.0, 0.0), model.cone)


class TestOneDimensionalObjects(unittest.TestCase):
    """Test one-dimensional objects on the line and circle."""

    def test_real_set_object_interval_and_point(self):
        """A real-line set should expose whole, half-line, and point models."""
        obj = RealSetObject((0.0, 2.0), 5.0)
        self.assertIn(1.0, obj)
        self.assertIn(5.0, obj)
        self.assertNotIn(3.0, obj)

        interior = obj.local_model_at(1.0)
        self.assertIn(FloatPoint(-1.0), interior.cone)
        self.assertIn(FloatPoint(1.0), interior.cone)

        boundary = obj.local_model_at(0.0)
        self.assertIn(FloatPoint(1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0), boundary.cone)

        isolated = obj.local_model_at(5.0)
        self.assertIn(FloatPoint(0.0), isolated.cone)
        self.assertNotIn(FloatPoint(1.0), isolated.cone)

    def test_circle_set_object_interval_and_point(self):
        """A circle set should expose whole, half-line, and point models."""
        obj = CircleSetObject(
            (
                FloatCircleSet.from_single_interval(0.0, math.pi / 2.0),
                FloatCirclePoint(math.pi),
            )
        )
        self.assertIn(FloatCirclePoint(math.pi / 4.0), obj)
        self.assertIn(FloatCirclePoint(math.pi), obj)
        self.assertNotIn(FloatCirclePoint(3.0 * math.pi / 4.0), obj)

        interior = obj.local_model_at(FloatCirclePoint(math.pi / 4.0))
        self.assertIn(FloatPoint(-1.0), interior.cone)
        self.assertIn(FloatPoint(1.0), interior.cone)

        boundary = obj.local_model_at(FloatCirclePoint(0.0))
        self.assertIn(FloatPoint(1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0), boundary.cone)

        isolated = obj.local_model_at(FloatCirclePoint(math.pi))
        self.assertIn(FloatPoint(0.0), isolated.cone)
        self.assertNotIn(FloatPoint(1.0), isolated.cone)


class TestPlanarObjects(unittest.TestCase):
    """Test concrete geometric objects in the Euclidean plane."""

    def test_whole_plane(self):
        """The whole plane should have the whole-plane local model."""
        plane = WholePlane()
        self.assertIn(FloatPoint(1.0, 2.0), plane)

        model = plane.local_model_at(FloatPoint(1.0, 2.0))
        self.assertIn(FloatPoint(-3.0, 4.0), model.cone)

    def test_half_plane(self):
        """A half-plane should distinguish interior from boundary points."""
        plane = HalfPlane((0.0, 1.0), offset=0.0)
        self.assertIn(FloatPoint(2.0, 3.0), plane)
        self.assertIn(FloatPoint(0.0, 0.0), plane)
        self.assertNotIn(FloatPoint(0.0, -1.0), plane)

        interior = plane.local_model_at(FloatPoint(0.0, 2.0))
        self.assertIn(FloatPoint(-1.0, -1.0), interior.cone)

        boundary = plane.local_model_at(FloatPoint(2.0, 0.0))
        self.assertIn(FloatPoint(0.0, 1.0), boundary.cone)
        self.assertNotIn(FloatPoint(0.0, -1.0), boundary.cone)

    def test_planar_angle(self):
        """A planar angle should expose apex, boundary, and interior models."""
        angle = PlanarAngle(
            FloatPoint(0.0, 0.0),
            0.0,
            math.pi / 2.0,
        )
        self.assertIn(FloatPoint(2.0, 3.0), angle)
        self.assertIn(FloatPoint(0.0, 0.0), angle)
        self.assertNotIn(FloatPoint(-1.0, 1.0), angle)

        apex = angle.local_model_at(FloatPoint(0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 1.0), apex.cone)
        self.assertNotIn(FloatPoint(-1.0, 1.0), apex.cone)

        boundary = angle.local_model_at(FloatPoint(2.0, 0.0))
        self.assertIn(FloatPoint(0.0, 1.0), boundary.cone)
        self.assertNotIn(FloatPoint(0.0, -1.0), boundary.cone)

        interior = angle.local_model_at(FloatPoint(2.0, 2.0))
        self.assertIn(FloatPoint(-1.0, -1.0), interior.cone)


if __name__ == "__main__":
    unittest.main()
