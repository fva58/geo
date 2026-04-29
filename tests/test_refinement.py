"""Tests for neighborhood refinement and local object classification."""

import unittest

from geo import (
    Ball,
    AdaptiveDistanceResult,
    CircleSpace,
    ChartNeighborhood,
    EuclideanNeighborhood,
    EuclideanPlaneSpace,
    FloatPoint,
    LocalObjectModel,
    MetricGeometricObject,
    Neighborhood,
    NeighborhoodCover,
    RealLineSpace,
    SphereSpace,
    TorusSpace,
    adaptive_distance,
    classify_local_object,
    classify_cover,
    local_chart_cover_from_samples,
    refine_until,
)


class TestNeighborhoodRefinement(unittest.TestCase):
    """Test the refinement-oriented manifold layer."""

    def test_chart_neighborhood_on_real_line(self):
        """A chart neighborhood should expose center, diameter, and probes."""
        space = RealLineSpace()
        neighborhood = space.neighborhood_at(0.5, 0.5, name="unit-interval")

        self.assertIsInstance(neighborhood, Neighborhood)
        self.assertIn(0.5, neighborhood)
        self.assertNotIn(2.0, neighborhood)
        self.assertAlmostEqual(neighborhood.diameter(), 1.0)
        self.assertEqual(neighborhood.sample_point(), 0.5)
        self.assertEqual(neighborhood.probe_points(), (0.5, 0.0, 1.0))

    def test_neighborhood_cover_refines(self):
        """A cover should refine into smaller neighborhoods."""
        space = RealLineSpace()
        neighborhood = space.neighborhood_at(0.5, 0.5)
        cover = NeighborhoodCover((neighborhood,))
        refined = cover.refine()

        self.assertEqual(len(refined.neighborhoods), 2)
        self.assertLess(refined.max_diameter(), cover.max_diameter())
        self.assertAlmostEqual(refined.neighborhoods[0].diameter(), 0.5)

    def test_local_object_classification_on_real_line(self):
        """Local classification should distinguish empty and simple patches."""
        space = RealLineSpace()
        segment = space.subset((0.0, 1.0), name="segment")

        simple = classify_local_object(
            segment,
            space.neighborhood_at(0.5, 0.25),
        )
        empty = classify_local_object(
            segment,
            space.neighborhood_at(2.0, 0.4),
        )

        self.assertIsInstance(simple, LocalObjectModel)
        self.assertEqual(simple.status, "simple")
        self.assertIsNotNone(simple.local_model)
        self.assertEqual(empty.status, "empty")
        self.assertIsNone(empty.witness_point)

    def test_local_object_classification_can_request_refinement(self):
        """Mixed neighborhoods should be marked complex for refinement."""
        space = EuclideanPlaneSpace()
        upper = space.half_plane((0.0, 1.0), offset=0.0, name="upper")
        neighborhood = space.neighborhood_at(FloatPoint(0.0, -0.5), 1.0)

        local = classify_local_object(upper, neighborhood)

        self.assertEqual(local.status, "complex")
        self.assertIn(local.witness_point, upper)
        self.assertIsNone(local.local_model)

    def test_standard_spaces_expose_intrinsic_neighborhoods(self):
        """Standard spaces should provide centered intrinsic neighborhoods."""
        line = RealLineSpace()
        circle = CircleSpace()
        plane = EuclideanPlaneSpace()
        sphere = SphereSpace()
        torus = TorusSpace()

        self.assertIn(1.0, line.neighborhood_at(1.0, 0.5))
        self.assertIn(0.0, circle.neighborhood_at(0.0, 0.5))
        self.assertIn(FloatPoint(0.0, 0.0), plane.neighborhood_at((0.0, 0.0), 0.5))
        north = sphere.point_from_angles(0.0, 0.0)
        self.assertIn(north, sphere.neighborhood_at(north, 0.5))
        self.assertIn((0.0, 0.0), torus.neighborhood_at((0.0, 0.0), 0.5))

    def test_refine_until_reduces_active_diameter(self):
        """Refinement should reduce the size of active neighborhoods."""
        space = RealLineSpace()
        segment = space.subset((0.0, 1.0), name="segment")
        cover = local_chart_cover_from_samples(
            segment,
            radius=0.5,
            resolution=2,
        )

        initial = classify_cover(segment, cover)
        refined = refine_until(
            segment,
            cover,
            max_diameter=0.2,
            max_steps=4,
        )

        self.assertGreater(initial.max_diameter(), 0.2)
        self.assertLessEqual(refined.max_diameter(), 0.2)

    def test_adaptive_distance_on_real_segments(self):
        """Adaptive distance should bound the distance between segments."""
        space = RealLineSpace()
        left = space.subset((0.0, 1.0), name="left")
        right = space.subset((3.0, 4.0), name="right")

        result = adaptive_distance(
            left,
            right,
            neighborhood_radius=0.5,
            sample_resolution=4,
            target_diameter=0.2,
            max_refinement_steps=4,
        )

        self.assertIsInstance(result, AdaptiveDistanceResult)
        self.assertLessEqual(result.lower_bound, 2.0)
        self.assertGreaterEqual(result.upper_bound, 2.0)
        self.assertLessEqual(result.error, 0.5)

    def test_adaptive_distance_on_planar_disks(self):
        """Adaptive distance should work on Euclidean planar objects."""
        space = EuclideanPlaneSpace()
        left = MetricGeometricObject.from_charted(
            space,
            Ball(FloatPoint(0.0, 0.0), 1.0),
            name="left-disk",
        )
        right = MetricGeometricObject.from_charted(
            space,
            Ball(FloatPoint(3.0, 0.0), 1.0),
            name="right-disk",
        )

        result = adaptive_distance(
            left,
            right,
            neighborhood_radius=0.4,
            sample_resolution=12,
            target_diameter=0.3,
            max_refinement_steps=4,
        )

        self.assertLessEqual(result.lower_bound, 1.0)
        self.assertGreaterEqual(result.upper_bound, 1.0)
        self.assertLessEqual(result.error, 1.0)


if __name__ == "__main__":
    unittest.main()
