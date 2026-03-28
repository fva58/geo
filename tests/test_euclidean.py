"""Tests for Euclidean points and vectors."""

import math
import unittest

from geo import (
    AffineDiffeomorphism,
    Chart,
    EuclideanChart,
    EuclideanNeighborhood,
    FloatPoint,
    FloatVector,
)


class TestFloatVector(unittest.TestCase):
    """Test cases for ``FloatVector``."""

    def test_construction_and_dimension(self):
        """Vectors should accept variadic and sequence construction."""
        self.assertEqual(FloatVector(1, 2, 3).to_tuple(), (1.0, 2.0, 3.0))
        self.assertEqual(FloatVector([1, 2]).dim, 2)
        self.assertEqual(FloatVector.zero(3).to_tuple(), (0.0, 0.0, 0.0))

    def test_arithmetic(self):
        """Basic vector arithmetic should work coordinate-wise."""
        left = FloatVector(1.0, 2.0)
        right = FloatVector(3.0, 4.0)
        self.assertEqual((left + right).to_tuple(), (4.0, 6.0))
        self.assertEqual((right - left).to_tuple(), (2.0, 2.0))
        self.assertEqual((-left).to_tuple(), (-1.0, -2.0))
        self.assertEqual((2.0 * left).to_tuple(), (2.0, 4.0))
        self.assertEqual((right / 2.0).to_tuple(), (1.5, 2.0))

    def test_dot_and_norm(self):
        """Dot product and norm should follow Euclidean formulas."""
        vector = FloatVector(3.0, 4.0)
        self.assertEqual(vector.dot(FloatVector(1.0, 2.0)), 11.0)
        self.assertEqual(vector.norm(), 5.0)

    def test_dimension_mismatch(self):
        """Coordinate-wise operations should reject mixed dimensions."""
        with self.assertRaises(ValueError):
            FloatVector(1.0, 2.0) + FloatVector(1.0)


class TestFloatPoint(unittest.TestCase):
    """Test cases for ``FloatPoint``."""

    def test_construction_and_origin(self):
        """Points should support sequence construction and origins."""
        self.assertEqual(FloatPoint(1, 2).to_tuple(), (1.0, 2.0))
        self.assertEqual(FloatPoint.origin(2).to_tuple(), (0.0, 0.0))

    def test_point_vector_arithmetic(self):
        """Points and vectors should interact with affine semantics."""
        point = FloatPoint(1.0, 2.0)
        vector = FloatVector(3.0, 4.0)
        other = FloatPoint(4.0, 6.0)

        self.assertEqual((point + vector).to_tuple(), (4.0, 6.0))
        self.assertEqual((other - point).to_tuple(), (3.0, 4.0))
        self.assertEqual((other - vector).to_tuple(), (1.0, 2.0))

    def test_distance(self):
        """Distances should be induced by the Euclidean norm."""
        left = FloatPoint(0.0, 0.0)
        right = FloatPoint(3.0, 4.0)
        self.assertEqual(left.distance_to(right), 5.0)
        self.assertTrue(math.isclose(right.distance_to(left), 5.0))

    def test_dimension_mismatch(self):
        """Affine operations should reject mixed dimensions."""
        with self.assertRaises(ValueError):
            FloatPoint(1.0, 2.0) + FloatVector(1.0)


class TestEuclideanChart(unittest.TestCase):
    """Test cases for ``EuclideanChart``."""

    def test_identity_chart(self):
        """The identity chart should preserve points and satisfy Chart."""
        chart = EuclideanChart.identity(2)
        point = FloatPoint(1.0, 2.0)
        self.assertIsInstance(chart, Chart)
        self.assertEqual(chart(point).to_tuple(), (1.0, 2.0))
        self.assertEqual(chart.inverse(point).to_tuple(), (1.0, 2.0))

    def test_translation_chart(self):
        """A chart given by translation should be invertible."""
        shift = FloatVector(3.0, -1.0)
        chart = EuclideanChart(
            lambda point: point + shift,
            lambda point: point - shift,
            source_dim=2,
            target_dim=2,
            name="translate",
        )
        point = FloatPoint(1.0, 2.0)
        image = chart(point)
        self.assertEqual(image.to_tuple(), (4.0, 1.0))
        self.assertEqual(chart.inverse(image).to_tuple(), point.to_tuple())

    def test_inverse_chart_and_composition(self):
        """Inverse charts and composition should preserve chart semantics."""
        shift = FloatVector(1.0, 1.0)
        chart = EuclideanChart(
            lambda point: point + shift,
            lambda point: point - shift,
            source_dim=2,
            target_dim=2,
            name="shift",
        )
        inverse_chart = chart.inverse_chart()
        point = FloatPoint(2.0, 3.0)
        self.assertEqual(inverse_chart(chart(point)).to_tuple(), point.to_tuple())

        identity = chart.compose(inverse_chart)
        self.assertEqual(identity(point).to_tuple(), point.to_tuple())

    def test_dimension_checks(self):
        """Charts should reject points with incompatible dimensions."""
        chart = EuclideanChart.identity(2)
        with self.assertRaises(ValueError):
            chart(FloatPoint(1.0))

    def test_domain_and_image_checks(self):
        """Charts should enforce explicit domain and image neighborhoods."""
        domain = EuclideanNeighborhood.box((0.0, 1.0), (0.0, 1.0))
        image = EuclideanNeighborhood.box((1.0, 2.0), (1.0, 2.0))
        chart = EuclideanChart(
            lambda point: point + FloatVector(1.0, 1.0),
            lambda point: point - FloatVector(1.0, 1.0),
            source_dim=2,
            target_dim=2,
            domain=domain,
            image=image,
            name="unit-shift",
        )
        self.assertEqual(chart(FloatPoint(0.5, 0.25)).to_tuple(), (1.5, 1.25))
        with self.assertRaises(ValueError):
            chart(FloatPoint(2.0, 0.0))
        with self.assertRaises(ValueError):
            chart.inverse(FloatPoint(3.0, 3.0))


class TestEuclideanNeighborhood(unittest.TestCase):
    """Test cases for ``EuclideanNeighborhood``."""

    def test_box_membership(self):
        """Neighborhoods should check point-wise membership."""
        neighborhood = EuclideanNeighborhood.box((0.0, 1.0), (-1.0, 1.0))
        self.assertIn(FloatPoint(0.5, 0.0), neighborhood)
        self.assertNotIn(FloatPoint(1.5, 0.0), neighborhood)

    def test_whole_space(self):
        """The whole-space neighborhood should contain every point."""
        neighborhood = EuclideanNeighborhood.whole(2)
        self.assertIn(FloatPoint(-100.0, 1.5), neighborhood)


class TestAffineDiffeomorphism(unittest.TestCase):
    """Test cases for ``AffineDiffeomorphism``."""

    def test_translation(self):
        """Pure translation should be handled as an affine diffeomorphism."""
        mapping = AffineDiffeomorphism(((1.0, 0.0), (0.0, 1.0)), (3.0, -1.0))
        point = FloatPoint(1.0, 2.0)
        image = mapping(point)
        self.assertIsInstance(mapping, Chart)
        self.assertEqual(image.to_tuple(), (4.0, 1.0))
        self.assertEqual(mapping.inverse(image).to_tuple(), point.to_tuple())

    def test_linear_change_of_coordinates(self):
        """An invertible linear map should be reversible."""
        mapping = AffineDiffeomorphism(((2.0, 0.0), (0.0, 3.0)), (0.0, 0.0))
        point = FloatPoint(1.5, -2.0)
        image = mapping(point)
        self.assertEqual(image.to_tuple(), (3.0, -6.0))
        restored = mapping.inverse(image)
        self.assertTrue(math.isclose(restored[0], point[0]))
        self.assertTrue(math.isclose(restored[1], point[1]))

    def test_inverse_chart(self):
        """The inverse affine map should undo the forward map."""
        mapping = AffineDiffeomorphism(((1.0, 1.0), (0.0, 1.0)), (2.0, -1.0))
        inverse_mapping = mapping.inverse_chart()
        point = FloatPoint(2.0, 3.0)
        self.assertEqual(
            inverse_mapping(mapping(point)).to_tuple(),
            point.to_tuple(),
        )

    def test_singular_matrix_is_rejected(self):
        """A diffeomorphism must have an invertible linear part."""
        with self.assertRaises(ValueError):
            AffineDiffeomorphism(((1.0, 2.0), (2.0, 4.0)), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
