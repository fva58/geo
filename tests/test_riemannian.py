"""Tests for Riemannian spaces and geometric objects."""

import math
import unittest

from geo import (
    EuclideanPlaneSpace,
    FloatCirclePoint,
    FloatPoint,
    FloatVector,
    Hyperplane,
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

    def test_real_line_set_operations(self):
        """Set-theoretic operations should work on real-line objects."""
        space = RealLineSpace()
        left = space.subset((0.0, 2.0), name="left")
        right = space.subset((1.0, 3.0), name="right")

        union = left | right
        intersection = left & right
        difference = left - right
        sym_diff = left ^ right

        self.assertIn(0.5, union)
        self.assertIn(2.5, union)
        self.assertIn(1.5, intersection)
        self.assertNotIn(0.5, intersection)
        self.assertIn(0.5, difference)
        self.assertNotIn(1.5, difference)
        self.assertIn(0.5, sym_diff)
        self.assertIn(2.5, sym_diff)
        self.assertNotIn(1.5, sym_diff)

        boundary = difference.local_model_at(0.0)
        self.assertIn(FloatPoint(1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0), boundary.cone)

    def test_circle_set_operations(self):
        """Set-theoretic operations should work on circle objects."""
        space = UnitCircleSpace()
        left = space.arc(0.0, math.pi / 2.0, name="left")
        right = space.arc(math.pi / 4.0, math.pi, name="right")

        union = left.union(right)
        intersection = left.intersection(right)
        difference = left.difference(right)

        self.assertIn(FloatCirclePoint(math.pi / 8.0), union)
        self.assertIn(FloatCirclePoint(3.0 * math.pi / 4.0), union)
        self.assertIn(FloatCirclePoint(math.pi / 3.0), intersection)
        self.assertNotIn(FloatCirclePoint(0.0), intersection)
        self.assertIn(FloatCirclePoint(0.0), difference)
        self.assertNotIn(FloatCirclePoint(math.pi / 3.0), difference)

    def test_plane_set_operations(self):
        """Set-theoretic operations should work on planar objects."""
        space = EuclideanPlaneSpace()
        upper = space.half_plane((0.0, 1.0), name="upper")
        right = space.half_plane((1.0, 0.0), name="right")

        quadrant = upper & right
        union = upper | right

        self.assertIn(FloatPoint(1.0, 1.0), quadrant)
        self.assertNotIn(FloatPoint(-1.0, 1.0), quadrant)
        self.assertIn(FloatPoint(-1.0, 1.0), union)
        self.assertIn(FloatPoint(1.0, -1.0), union)
        self.assertNotIn(FloatPoint(-1.0, -1.0), union)

        apex_model = quadrant.local_model_at(FloatPoint(0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 1.0), apex_model.cone)
        self.assertNotIn(FloatPoint(-1.0, 1.0), apex_model.cone)

    def test_set_operations_require_same_space(self):
        """Set-theoretic operations should reject mixed ambient spaces."""
        real_line = RealLineSpace()
        another_real_line = RealLineSpace()

        with self.assertRaises(ValueError):
            real_line.point(0.0).union(another_real_line.point(0.0))

    def test_parallel_projection_onto_hyperplane(self):
        """Parallel projection should return a new geometric object."""
        space = EuclideanPlaneSpace()
        source_line = RiemannianGeometricObject.from_charted(
            space,
            Hyperplane((0.0, 1.0), offset=1.0),
            name="source-line",
        )
        source_half_line = source_line & space.half_plane((1.0, 0.0), offset=0.0)
        target_line = Hyperplane((0.0, 1.0), offset=0.0)

        projected = source_half_line.project_along_direction_onto(
            Hyperplane((0.0, 1.0), offset=1.0),
            target_line,
            (0.0, -1.0),
            name="parallel-projected-half-line",
        )

        self.assertIsInstance(projected, RiemannianGeometricObject)
        self.assertIn(FloatPoint(1.0, 0.0), projected)
        self.assertNotIn(FloatPoint(-1.0, 0.0), projected)
        self.assertNotIn(FloatPoint(1.0, 1.0), projected)

        boundary = projected.local_model_at(FloatPoint(0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 0.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0, 0.0), boundary.cone)
        self.assertNotIn(FloatPoint(0.0, 1.0), boundary.cone)

    def test_central_projection_onto_hyperplane(self):
        """Central projection should return a new geometric object."""
        space = EuclideanPlaneSpace()
        source_line = RiemannianGeometricObject.from_charted(
            space,
            Hyperplane((0.0, 1.0), offset=1.0),
            name="source-line",
        )
        source_half_line = source_line & space.half_plane((1.0, 0.0), offset=0.0)
        target_line = Hyperplane((0.0, 1.0), offset=0.0)

        projected = source_half_line.project_from_point_onto(
            Hyperplane((0.0, 1.0), offset=1.0),
            target_line,
            FloatPoint(0.0, 2.0),
            name="central-projected-half-line",
        )

        self.assertIn(FloatPoint(2.0, 0.0), projected)
        self.assertNotIn(FloatPoint(-1.0, 0.0), projected)
        self.assertNotIn(FloatPoint(0.0, 1.0), projected)

        boundary = projected.local_model_at(FloatPoint(0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 0.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0, 0.0), boundary.cone)
        self.assertNotIn(FloatPoint(0.0, 1.0), boundary.cone)


if __name__ == "__main__":
    unittest.main()
