"""Tests for neighborhood refinement and local object classification."""

import itertools
import unittest

from geo.cone import LocalConeModel
from geo.euclidean import FloatPoint
from geo.space.base import refine_neighborhoods
from geo.operations import classify_cover, local_chart_cover_from_points, refine_until
from geo import space as space_pkg
from geo.space.base import Neighborhood


class TestNeighborhoodRefinement(unittest.TestCase):
    """Test the refinement-oriented manifold layer."""

    def test_chart_neighborhood_on_real_line(self):
        """A chart neighborhood should expose radial bounds."""
        space = space_pkg.line.Space()
        neighborhood = space.neighborhood_at(0.5, 0.5)

        self.assertIsInstance(neighborhood, Neighborhood)
        self.assertIn(0.5, neighborhood)
        self.assertNotIn(2.0, neighborhood)
        self.assertAlmostEqual(neighborhood.inner_radius(), 0.5)
        self.assertAlmostEqual(neighborhood.outer_radius(), 0.5)
        self.assertAlmostEqual(neighborhood.diameter(), 1.0)
        self.assertEqual(neighborhood.center_point(), 0.5)

    def test_neighborhood_cover_refines(self):
        """A cover should refine into smaller neighborhoods."""
        space = space_pkg.line.Space()
        neighborhood = space.neighborhood_at(0.5, 0.5)
        cover = (neighborhood,)
        refined = refine_neighborhoods(cover, factor=2)

        self.assertEqual(len(refined), 2)
        self.assertLess(
            max(n.diameter() for n in refined),
            max(n.diameter() for n in cover),
        )
        self.assertLess(
            max(n.outer_radius() for n in refined),
            max(n.outer_radius() for n in cover),
        )
        self.assertAlmostEqual(refined[0].diameter(), 0.5)

    def test_local_object_classification_on_real_line(self):
        """Local classification should distinguish empty and cone patches."""
        space = space_pkg.line.Space()
        segment = space.subset((0.0, 1.0))

        conic = segment.classify_neighborhood(
            space.neighborhood_at(0.5, 0.25),
        )
        empty = segment.classify_neighborhood(
            space.neighborhood_at(2.0, 0.4),
        )

        self.assertIsInstance(conic, LocalConeModel)
        self.assertIsNone(empty)

    def test_object_marks_list_of_neighborhoods(self):
        """An object should classify a whole neighborhood list at once."""
        space = space_pkg.line.Space()
        segment = space.subset((0.0, 1.0))
        neighborhoods = (
            space.neighborhood_at(0.5, 0.25),
            space.neighborhood_at(2.0, 0.4),
            space.neighborhood_at(1.2, 0.4),
        )

        marking = segment.classify_neighborhoods(neighborhoods)

        self.assertIsInstance(marking, tuple)
        self.assertEqual(len(marking), 3)
        self.assertIsInstance(marking[0], LocalConeModel)
        self.assertIsNone(marking[1])
        self.assertIs(marking[2], Ellipsis)

    def test_local_object_classification_can_request_refinement(self):
        """Mixed neighborhoods should be marked complex for refinement."""
        space = space_pkg.euclidean.Space(2)
        upper = space.half_plane((0.0, 1.0), offset=0.0)
        neighborhood = space.neighborhood_at(FloatPoint(0.0, -0.5), 1.0)

        local = upper.classify_neighborhood(neighborhood)

        self.assertIs(local, Ellipsis)

    def test_large_neighborhood_is_not_forced_to_be_cone(self):
        """A large neighborhood can contain more than one local cone patch."""
        space = space_pkg.euclidean.Space(2)
        cube = space.cube(FloatPoint(1.0, 1.0), 1.0)
        neighborhood = space.neighborhood_at(FloatPoint(0.0, 0.0), 10.0)

        local = cube.classify_neighborhood(neighborhood)

        self.assertIs(local, Ellipsis)

    def test_standard_spaces_expose_intrinsic_neighborhoods(self):
        """Standard spaces should provide centered intrinsic neighborhoods."""
        line = space_pkg.line.Space()
        circle = space_pkg.circle.Space()
        plane = space_pkg.euclidean.Space(2)
        sphere = space_pkg.sphere.Space()
        torus = space_pkg.torus.Space()

        self.assertIn(1.0, line.neighborhood_at(1.0, 0.5))
        self.assertIn(0.0, circle.neighborhood_at(0.0, 0.5))
        self.assertIn(FloatPoint(0.0, 0.0), plane.neighborhood_at((0.0, 0.0), 0.5))
        north = sphere.point_from_angles(0.0, 0.0)
        self.assertIn(north, sphere.neighborhood_at(north, 0.5))
        self.assertIn((0.0, 0.0), torus.neighborhood_at((0.0, 0.0), 0.5))

    def test_spaces_expose_full_and_refinement_api(self):
        """Spaces should expose full covers and smaller-diameter refinements."""
        line = space_pkg.line.Space()
        circle = space_pkg.circle.Space()
        plane = space_pkg.euclidean.Space(2)
        sphere = space_pkg.sphere.Space()
        torus = space_pkg.torus.Space()

        line_cover = tuple(itertools.islice(line.full(0.5), 5))
        plane_cover = plane.full(0.5)
        circle_cover = circle.full(0.8)
        sphere_cover = sphere.full(0.8, resolution=8)
        torus_cover = torus.full(1.0)

        self.assertTrue(any(0.0 in neighborhood for neighborhood in line_cover))
        self.assertTrue(any(FloatPoint(0.0, 0.0) in neighborhood for neighborhood in plane_cover))
        self.assertTrue(any(0.0 in neighborhood for neighborhood in circle_cover))
        north = sphere.point_from_angles(0.0, 0.0)
        self.assertTrue(any(north in neighborhood for neighborhood in sphere_cover))
        self.assertTrue(any((0.0, 0.0) in neighborhood for neighborhood in torus_cover))

        refined = circle.refine(circle_cover[:1], factor=4)
        self.assertTrue(refined)
        self.assertLess(
            max(neighborhood.diameter() for neighborhood in refined),
            circle_cover[0].diameter(),
        )

    def test_euclidean_full_is_finite_and_bounded_by_max_size(self):
        """Euclidean full should respect the configured finite size."""
        plane = space_pkg.euclidean.Space(2, max_size=1.0)

        cover = plane.full(0.5)

        self.assertEqual(len(cover), 9)
        self.assertTrue(any(FloatPoint(0.0, 0.0) in neighborhood for neighborhood in cover))
        centers = {neighborhood.center_point().to_tuple() for neighborhood in cover}
        self.assertEqual(
            centers,
            {
                (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
                (0.0, -1.0), (0.0, 0.0), (0.0, 1.0),
                (1.0, -1.0), (1.0, 0.0), (1.0, 1.0),
            },
        )

    def test_refine_until_reduces_active_outer_radius(self):
        """Refinement should reduce the outer radius of active parts."""
        space = space_pkg.line.Space()
        segment = space.subset((0.0, 1.0))
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
