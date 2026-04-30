"""Tests for neighborhood refinement and local object classification."""

import itertools
import unittest

from geo import (
    CircleSpace,
    EuclideanPlaneSpace,
    FloatPoint,
    LocalObjectModel,
    Neighborhood,
    NeighborhoodCover,
    NeighborhoodMarking,
    RealLineSpace,
    SphereSpace,
    TorusSpace,
    classify_local_object,
    classify_cover,
    local_chart_cover_from_points,
    refine_until,
)


class TestNeighborhoodRefinement(unittest.TestCase):
    """Test the refinement-oriented manifold layer."""

    def test_chart_neighborhood_on_real_line(self):
        """A chart neighborhood should expose radial bounds and probes."""
        space = RealLineSpace()
        neighborhood = space.neighborhood_at(0.5, 0.5, name="unit-interval")

        self.assertIsInstance(neighborhood, Neighborhood)
        self.assertIn(0.5, neighborhood)
        self.assertNotIn(2.0, neighborhood)
        self.assertAlmostEqual(neighborhood.inner_radius(), 0.5)
        self.assertAlmostEqual(neighborhood.outer_radius(), 0.5)
        self.assertAlmostEqual(neighborhood.diameter(), 1.0)
        self.assertEqual(neighborhood.center_point(), 0.5)
        self.assertEqual(neighborhood.probe_points(), (0.5, 0.0, 1.0))

    def test_neighborhood_cover_refines(self):
        """A cover should refine into smaller neighborhoods."""
        space = RealLineSpace()
        neighborhood = space.neighborhood_at(0.5, 0.5)
        cover = NeighborhoodCover((neighborhood,))
        refined = cover.refine()

        self.assertEqual(len(refined.neighborhoods), 2)
        self.assertLess(refined.max_diameter(), cover.max_diameter())
        self.assertLess(refined.max_outer_radius(), cover.max_outer_radius())
        self.assertAlmostEqual(refined.neighborhoods[0].diameter(), 0.5)

    def test_local_object_classification_on_real_line(self):
        """Local classification should distinguish empty and cone patches."""
        space = RealLineSpace()
        segment = space.subset((0.0, 1.0), name="segment")

        conic = classify_local_object(
            segment,
            space.neighborhood_at(0.5, 0.25),
        )
        empty = classify_local_object(
            segment,
            space.neighborhood_at(2.0, 0.4),
        )

        self.assertIsInstance(conic, LocalObjectModel)
        self.assertEqual(conic.status, "cone")
        self.assertTrue(conic.is_cone)
        self.assertIsNotNone(conic.local_model)
        self.assertEqual(empty.status, "empty")
        self.assertIsNone(empty.witness_point)

    def test_object_marks_list_of_neighborhoods(self):
        """An object should classify a whole neighborhood list at once."""
        space = RealLineSpace()
        segment = space.subset((0.0, 1.0), name="segment")
        neighborhoods = (
            space.neighborhood_at(0.5, 0.25),
            space.neighborhood_at(2.0, 0.4),
            space.neighborhood_at(1.2, 0.4),
        )

        marking = segment.classify_neighborhoods(neighborhoods)

        self.assertIsInstance(marking, NeighborhoodMarking)
        self.assertEqual(len(marking), 3)
        self.assertEqual(tuple(model.status for model in marking), ("cone", "empty", "complex"))
        self.assertEqual(len(marking.cone), 1)
        self.assertEqual(len(marking.empty), 1)
        self.assertEqual(len(marking.complex), 1)

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

    def test_spaces_expose_full_cover_and_refinement_api(self):
        """Spaces should expose full covers and smaller-diameter refinements."""
        line = RealLineSpace()
        circle = CircleSpace()
        plane = EuclideanPlaneSpace()
        sphere = SphereSpace()
        torus = TorusSpace()

        line_cover = tuple(itertools.islice(line.full_cover(0.5), 5))
        plane_cover = tuple(itertools.islice(plane.full_cover(0.5), 9))
        circle_cover = circle.full_cover(0.8)
        sphere_cover = sphere.full_cover(0.8, resolution=8)
        torus_cover = torus.full_cover(1.0)

        self.assertTrue(any(0.0 in neighborhood for neighborhood in line_cover))
        self.assertTrue(any(FloatPoint(0.0, 0.0) in neighborhood for neighborhood in plane_cover))
        self.assertTrue(any(0.0 in neighborhood for neighborhood in circle_cover))
        north = sphere.point_from_angles(0.0, 0.0)
        self.assertTrue(any(north in neighborhood for neighborhood in sphere_cover))
        self.assertTrue(any((0.0, 0.0) in neighborhood for neighborhood in torus_cover))

        refined = circle.refine_cover(circle_cover[:1], factor=4)
        self.assertTrue(refined)
        self.assertLess(
            max(neighborhood.diameter() for neighborhood in refined),
            circle_cover[0].diameter(),
        )

    def test_refine_until_reduces_active_outer_radius(self):
        """Refinement should reduce the outer radius of active parts."""
        space = RealLineSpace()
        segment = space.subset((0.0, 1.0), name="segment")
        cover = local_chart_cover_from_points(
            space,
            (0.0, 1.0),
            radius=0.5,
        )

        initial = classify_cover(segment, cover)
        refined = refine_until(
            segment,
            cover,
            max_outer_radius=0.1,
            max_steps=4,
        )

        self.assertGreater(initial.max_outer_radius(), 0.1)
        self.assertLessEqual(refined.max_outer_radius(), 0.1)



if __name__ == "__main__":
    unittest.main()
