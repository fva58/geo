"""Tests for metric spaces and geometric objects."""

import math
import unittest

from geo import (
    Ball,
    CircleSpace,
    EllipsoidSurface,
    EuclideanCone,
    EuclideanNeighborhood,
    EuclideanMetricSpace,
    EuclideanPlaneSpace,
    FloatCirclePoint,
    FloatPoint,
    FloatVector,
    Hyperplane,
    LazyMetricExpressionObject,
    LazyMetricObject,
    LazyMetricMappedObject,
    LocalConeModel,
    ManifoldChart,
    MetricGeometricObject,
    MetricSpace,
    RealLineSpace,
)


class TestMetricSpaces(unittest.TestCase):
    """Test standard metric spaces."""

    def test_real_line_space(self):
        """The real line should satisfy the metric-space protocol."""
        space = RealLineSpace()
        self.assertIsInstance(space, MetricSpace)
        self.assertEqual(space.distance(0.0, 4.0), 4.0)
        self.assertEqual(space.distance(4.0, 0.0), 4.0)

    def test_unit_circle_space(self):
        """The unit circle should use the shortest-arc metric."""
        space = CircleSpace()
        point = FloatCirclePoint(math.pi / 3.0)
        self.assertIsInstance(space, MetricSpace)
        self.assertAlmostEqual(space.distance(point, point), 0.0)
        self.assertAlmostEqual(
            space.distance(0.1, 2.0 * math.pi - 0.1),
            0.2,
        )

    def test_euclidean_plane_space(self):
        """The Euclidean plane should use Euclidean distance."""
        space = EuclideanPlaneSpace()
        self.assertIsInstance(space, MetricSpace)
        self.assertIsInstance(space, EuclideanMetricSpace)
        self.assertEqual(
            space.distance(FloatPoint(1.0, 2.0), FloatPoint(4.0, 6.0)),
            5.0,
        )


class TestMetricObjects(unittest.TestCase):
    """Test metric geometric objects in explicit ambient spaces."""

    def test_real_line_subset(self):
        """A real-line subset should keep its local cone models."""
        space = RealLineSpace()
        obj = space.subset((0.0, 2.0), 5.0, name="segment-and-point")
        self.assertIsInstance(obj, MetricGeometricObject)
        self.assertEqual(obj.space, space)
        self.assertIn(1.0, obj)
        self.assertIn(5.0, obj)
        self.assertNotIn(3.0, obj)

        boundary = obj.local_model_at(0.0)
        self.assertIn(FloatPoint(1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0), boundary.cone)

    def test_circle_arc(self):
        """An arc should be a geometric object in the unit circle space."""
        space = CircleSpace()
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

        disk = MetricGeometricObject.from_charted(
            space,
            Ball(FloatPoint(0.0, 0.0), 1.0),
            name="disk",
        )
        point_object = space.point(FloatPoint(0.0, 0.0))
        self.assertIn(FloatPoint(0.0, 0.0), disk)
        self.assertIn(FloatPoint(0.0, 0.0), point_object)

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

    def test_set_operations_build_lazy_expression_nodes(self):
        """Set operations should preserve an explicit lazy expression tree."""
        space = RealLineSpace()
        left = space.subset((0.0, 2.0), name="left")
        right = space.subset((1.0, 3.0), name="right")

        union = left | right
        difference = left - right

        self.assertIsInstance(union, LazyMetricExpressionObject)
        self.assertIsInstance(union, LazyMetricObject)
        self.assertEqual(union.operation, "union")
        self.assertEqual(union.node_kind, "binary")
        self.assertTrue(union.is_lazy)
        self.assertEqual(union.children, (left, right))
        self.assertIs(union.left, left)
        self.assertIs(union.right, right)

        self.assertIsInstance(difference, LazyMetricExpressionObject)
        self.assertEqual(difference.operation, "difference")
        self.assertIs(difference.left, left)
        self.assertIs(difference.right, right)

    def test_circle_set_operations(self):
        """Set-theoretic operations should work on circle objects."""
        space = CircleSpace()
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

    def test_set_operations_transport_local_models_across_charts(self):
        """Equivalent objects in different charts should keep boundary cones."""
        space = RealLineSpace()
        chart_identity = ManifoldChart(
            lambda point: FloatPoint(point),
            lambda coordinates: coordinates[0],
            dim=1,
            domain_contains=space.contains,
            image=EuclideanNeighborhood.whole(1),
            name="identity",
        )
        chart_reflected = ManifoldChart(
            lambda point: FloatPoint(-point),
            lambda coordinates: -coordinates[0],
            dim=1,
            domain_contains=space.contains,
            image=EuclideanNeighborhood.whole(1),
            name="reflected",
        )
        left = MetricGeometricObject(
            space,
            contains=lambda point: point >= 0.0,
            local_model=lambda point: LocalConeModel(
                chart_identity,
                EuclideanCone(
                    1,
                    contains=lambda coordinates: coordinates[0] >= 0.0,
                    neighborhood=EuclideanNeighborhood.whole(1),
                    name="positive-half-line",
                ),
            ),
            name="left",
        )
        right = MetricGeometricObject(
            space,
            contains=lambda point: point >= 0.0,
            local_model=lambda point: LocalConeModel(
                chart_reflected,
                EuclideanCone(
                    1,
                    contains=lambda coordinates: coordinates[0] <= 0.0,
                    neighborhood=EuclideanNeighborhood.whole(1),
                    name="reflected-positive-half-line",
                ),
            ),
            name="right",
        )

        intersection = left.intersection(right)
        boundary = intersection.local_model_at(0.0)

        self.assertIn(FloatPoint(1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0), boundary.cone)

    def test_parallel_projection_onto_hyperplane(self):
        """Parallel projection should return a new geometric object."""
        space = EuclideanPlaneSpace()
        source_line = MetricGeometricObject.from_charted(
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

        self.assertIsInstance(projected, MetricGeometricObject)
        self.assertIsInstance(projected, LazyMetricMappedObject)
        self.assertIsInstance(projected, LazyMetricObject)
        self.assertEqual(projected.operation, "project-along-direction")
        self.assertEqual(projected.node_kind, "unary")
        self.assertEqual(projected.children, (source_half_line,))
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
        source_line = MetricGeometricObject.from_charted(
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

        self.assertIsInstance(projected, LazyMetricMappedObject)
        self.assertEqual(projected.operation, "project-from-point")
        self.assertIn(FloatPoint(2.0, 0.0), projected)
        self.assertNotIn(FloatPoint(-1.0, 0.0), projected)
        self.assertNotIn(FloatPoint(0.0, 1.0), projected)

        boundary = projected.local_model_at(FloatPoint(0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 0.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0, 0.0), boundary.cone)
        self.assertNotIn(FloatPoint(0.0, 1.0), boundary.cone)

    def test_smooth_image_object(self):
        """A smooth image should define a new geometric object."""
        source_space = RealLineSpace()
        target_space = EuclideanPlaneSpace()
        source = source_space.subset((0.0, 2.0), name="segment")

        def target_chart(point: FloatPoint) -> ManifoldChart[FloatPoint]:
            center = FloatPoint(point)
            return ManifoldChart(
                lambda candidate: FloatPoint(candidate) - center,
                lambda coordinates: center + FloatVector(coordinates),
                dim=2,
                domain_contains=target_space.contains,
                image=EuclideanNeighborhood.whole(2),
                name="plane-chart",
            )

        image = source.image_under_smooth_map(
            lambda point: FloatPoint(point, point * point),
            lambda point: float(FloatPoint(point)[0]),
            target_space,
            target_chart,
            contains_image_point=lambda point: (
                0.0 <= FloatPoint(point)[0] <= 2.0 and
                math.isclose(
                    FloatPoint(point)[1],
                    FloatPoint(point)[0] * FloatPoint(point)[0],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ),
            name="parabola-segment",
        )

        self.assertIsInstance(image, MetricGeometricObject)
        self.assertIsInstance(image, LazyMetricMappedObject)
        self.assertEqual(image.operation, "image-under-smooth-map")
        self.assertIn(FloatPoint(1.0, 1.0), image)
        self.assertNotIn(FloatPoint(1.0, 0.0), image)
        self.assertNotIn(FloatPoint(3.0, 9.0), image)

        interior = image.local_model_at(FloatPoint(1.0, 1.0))
        self.assertIn(FloatPoint(1.0, 2.0), interior.cone)
        self.assertIn(FloatPoint(-1.0, -2.0), interior.cone)
        self.assertNotIn(FloatPoint(0.0, 1.0), interior.cone)

        boundary = image.local_model_at(FloatPoint(0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 0.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0, 0.0), boundary.cone)
        self.assertNotIn(FloatPoint(0.0, 1.0), boundary.cone)

    def test_visible_ball_from_direction(self):
        """A ball should expose the visible boundary cap from a direction."""
        space = EuclideanPlaneSpace()
        ball = MetricGeometricObject.from_charted(
            space,
            Ball(FloatPoint(0.0, 0.0), 1.0),
            name="disk",
        )

        visible = ball.visible_from_direction(FloatVector(0.0, 1.0))

        self.assertIsInstance(visible, MetricGeometricObject)
        self.assertIsInstance(visible, LazyMetricMappedObject)
        self.assertEqual(visible.operation, "visible-from-direction")
        self.assertIn(FloatPoint(0.0, 1.0), visible)
        self.assertIn(FloatPoint(1.0, 0.0), visible)
        self.assertNotIn(FloatPoint(0.0, -1.0), visible)
        self.assertNotIn(FloatPoint(0.0, 0.0), visible)

        silhouette = visible.local_model_at(FloatPoint(1.0, 0.0))
        self.assertIn(FloatPoint(0.0, 1.0), silhouette.cone)
        self.assertNotIn(FloatPoint(0.0, -1.0), silhouette.cone)

    def test_visible_ellipsoid_surface_from_point(self):
        """An ellipsoid surface should keep only the observer-facing part."""
        space = EuclideanPlaneSpace()
        surface = MetricGeometricObject.from_charted(
            space,
            EllipsoidSurface(
                FloatPoint(0.0, 0.0),
                ((2.0, 0.0), (0.0, 1.0)),
            ),
            name="ellipse",
        )

        visible = surface.visible_from_point(FloatPoint(0.0, 3.0))

        self.assertIsInstance(visible, LazyMetricMappedObject)
        self.assertEqual(visible.operation, "visible-from-point")
        self.assertIn(FloatPoint(0.0, 1.0), visible)
        self.assertIn(FloatPoint(1.2, 0.8), visible)
        self.assertNotIn(FloatPoint(0.0, -1.0), visible)

    def test_visible_half_plane_from_point(self):
        """A half-plane should expose its boundary only from the exterior."""
        space = EuclideanPlaneSpace()
        half_plane = space.half_plane((0.0, 1.0), offset=0.0, name="upper")

        visible = half_plane.visible_from_point(FloatPoint(0.0, -1.0))
        hidden = half_plane.visible_from_point(FloatPoint(0.0, 1.0))

        self.assertIn(FloatPoint(2.0, 0.0), visible)
        self.assertNotIn(FloatPoint(2.0, 1.0), visible)
        self.assertNotIn(FloatPoint(2.0, 0.0), hidden)


if __name__ == "__main__":
    unittest.main()
