"""Tests for the visualization-aware space layer."""

import math
import unittest

from geo import (
    EuclideanPlaneSpace,
    FloatCirclePoint,
    FloatPoint,
    ObjectMesh,
    RealLineSpace,
    Space,
    SphereSpace,
    TorusPoint,
    TorusSpace,
    UnitCircleSpace,
)


class TestSpaceProtocol(unittest.TestCase):
    """Test the visualization-aware space protocol."""

    def test_standard_spaces_satisfy_space_protocol(self):
        """Standard spaces should satisfy the ``Space`` protocol."""
        self.assertIsInstance(RealLineSpace(), Space)
        self.assertIsInstance(UnitCircleSpace(), Space)
        self.assertIsInstance(EuclideanPlaneSpace(), Space)
        self.assertIsInstance(SphereSpace(), Space)
        self.assertIsInstance(TorusSpace(), Space)

    def test_real_line_visualization(self):
        """The real line should embed into 2D and 3D coordinates."""
        space = RealLineSpace()
        self.assertEqual(space.space_kind, "real-line")
        self.assertEqual(space.to_2d(3.5), (3.5, 0.0))
        self.assertEqual(space.to_3d(-2.0), (-2.0, 0.0, 0.0))

    def test_unit_circle_visualization(self):
        """The unit circle should use its standard Euclidean embedding."""
        space = UnitCircleSpace()
        point = FloatCirclePoint(math.pi / 2.0)
        self.assertEqual(space.space_kind, "unit-circle")
        self.assertAlmostEqual(space.to_2d(point)[0], 0.0)
        self.assertAlmostEqual(space.to_2d(point)[1], 1.0)
        self.assertEqual(space.to_3d(0.0), (1.0, 0.0, 0.0))

    def test_euclidean_plane_visualization(self):
        """Euclidean space should use orthographic coordinates."""
        space = EuclideanPlaneSpace()
        self.assertEqual(space.space_kind, "euclidean")
        self.assertEqual(space.to_2d(FloatPoint(1.0, 2.0)), (1.0, 2.0))
        self.assertEqual(
            space.to_3d(FloatPoint(1.0, 2.0)),
            (1.0, 2.0, 0.0),
        )

    def test_sphere_distance_and_visualization(self):
        """The sphere should use intrinsic distance and explicit embeddings."""
        space = SphereSpace(radius=2.0)
        equator_zero = space.point_from_angles(0.0, 0.0)
        equator_quarter = space.point_from_angles(math.pi / 2.0, 0.0)

        self.assertEqual(space.space_kind, "sphere")
        self.assertAlmostEqual(
            space.distance(equator_zero, equator_quarter),
            math.pi,
        )
        self.assertEqual(
            space.to_3d(equator_zero),
            (2.0, 0.0, 0.0),
        )
        longitude, latitude = space.to_2d(
            equator_quarter,
            method="equirectangular",
        )
        self.assertAlmostEqual(longitude, math.pi / 2.0)
        self.assertAlmostEqual(latitude, 0.0)

    def test_sphere_stereographic_projection_rejects_north_pole(self):
        """Stereographic projection should reject the north pole."""
        space = SphereSpace()
        north_pole = FloatPoint(0.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            space.to_2d(north_pole)

    def test_sphere_point_object_and_cap(self):
        """Sphere spaces should expose singleton and cap objects."""
        space = SphereSpace()
        north_pole = space.point_from_angles(0.0, math.pi / 2.0)
        equator = space.point_from_angles(0.0, 0.0)
        south_pole = space.point_from_angles(0.0, -math.pi / 2.0)

        point_object = space.point_object(north_pole)
        hemisphere = space.cap(north_pole, math.pi / 2.0)

        self.assertIn(north_pole, point_object)
        self.assertNotIn(equator, point_object)
        self.assertIn(north_pole, hemisphere)
        self.assertIn(equator, hemisphere)
        self.assertNotIn(south_pole, hemisphere)

        boundary = hemisphere.local_model_at(equator)
        self.assertIn(FloatPoint(0.0, 1.0), boundary.cone)
        self.assertNotIn(FloatPoint(0.0, -1.0), boundary.cone)

    def test_sphere_mesh_and_cap_mesh(self):
        """Sphere meshes should provide embedded vertices and cells."""
        space = SphereSpace(radius=2.0)
        north = space.point_from_angles(0.0, math.pi / 2.0)
        mesh = space.mesh(resolution=12)
        cap_mesh = space.cap_mesh(north, math.pi / 3.0, resolution=12)
        cap = space.cap(north, math.pi / 3.0)

        self.assertIsInstance(mesh, ObjectMesh)
        self.assertEqual(mesh.dim, 3)
        self.assertTrue(mesh.cells)
        self.assertTrue(all(space.contains(vertex) for vertex in mesh.vertices))

        self.assertIsInstance(cap_mesh, ObjectMesh)
        self.assertEqual(cap_mesh.dim, 3)
        self.assertTrue(cap_mesh.cells)
        self.assertTrue(all(vertex in cap for vertex in cap_mesh.vertices))

    def test_torus_distance_and_visualization(self):
        """The torus should use the flat product metric."""
        space = TorusSpace(major_radius=3.0, minor_radius=1.0)
        first = TorusPoint(0.0, 0.0)
        second = TorusPoint(math.pi, 0.0)

        self.assertEqual(space.space_kind, "torus")
        self.assertAlmostEqual(space.distance(first, second), math.pi)
        self.assertEqual(space.to_2d(first), (0.0, 0.0))
        self.assertEqual(space.to_3d(first), (4.0, 0.0, 0.0))

    def test_torus_point_requires_two_angles(self):
        """A torus point should reject ambiguous one-angle construction."""
        with self.assertRaises(TypeError):
            TorusPoint(1.0)

    def test_torus_patch_local_model(self):
        """Torus patches should expose product boundary cones."""
        space = TorusSpace()
        patch = space.patch((0.0, math.pi / 2.0), (0.0, math.pi / 2.0))

        self.assertIn(TorusPoint(math.pi / 4.0, math.pi / 4.0), patch)
        self.assertNotIn(TorusPoint(math.pi, math.pi / 4.0), patch)

        boundary = patch.local_model_at(TorusPoint(0.0, 0.0))
        self.assertIn(FloatPoint(1.0, 1.0), boundary.cone)
        self.assertNotIn(FloatPoint(-1.0, 1.0), boundary.cone)
        self.assertNotIn(FloatPoint(1.0, -1.0), boundary.cone)

    def test_torus_sampling_and_mesh(self):
        """Torus spaces should expose sample points and meshes."""
        space = TorusSpace()
        patch = space.patch((0.0, math.pi / 2.0), (0.0, math.pi / 2.0))
        samples = space.sample_points(resolution=10)
        mesh = space.mesh(resolution=12)
        patch_mesh = space.patch_mesh(
            (0.0, math.pi / 2.0),
            (0.0, math.pi / 2.0),
            resolution=12,
        )

        self.assertTrue(samples)
        self.assertTrue(all(sample in space for sample in samples))

        self.assertIsInstance(mesh, ObjectMesh)
        self.assertEqual(mesh.dim, 3)
        self.assertTrue(mesh.cells)

        self.assertIsInstance(patch_mesh, ObjectMesh)
        self.assertEqual(patch_mesh.dim, 3)
        self.assertTrue(patch_mesh.cells)
        self.assertTrue(
            all(
                TorusPoint(
                    math.atan2(vertex[1], vertex[0]),
                    math.atan2(
                        vertex[2],
                        math.hypot(vertex[0], vertex[1]) - space.major_radius,
                    ),
                ) in patch
                for vertex in patch_mesh.vertices
            )
        )
