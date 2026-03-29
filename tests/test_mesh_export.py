"""Tests for mesh export and plotting adapters."""

import unittest

from geo import (
    FloatPoint,
    ObjectMesh,
    mesh_to_matplotlib_data,
    mesh_to_plotly_data,
    mesh_to_threejs_data,
    mesh_to_wireframe_data,
)


class TestMeshExport(unittest.TestCase):
    """Test mesh export adapters."""

    def test_polyline_mesh_adapters(self):
        """Polyline meshes should export wireframe and 2D plotting data."""
        mesh = ObjectMesh(
            (
                FloatPoint(0.0, 0.0),
                FloatPoint(1.0, 0.0),
                FloatPoint(1.0, 1.0),
            ),
            (
                (0, 1),
                (1, 2),
            ),
        )

        wireframe = mesh_to_wireframe_data(mesh)
        matplotlib_data = mesh_to_matplotlib_data(mesh)
        plotly_data = mesh_to_plotly_data(mesh, name="polyline")

        self.assertEqual(wireframe["dim"], 2)
        self.assertEqual(wireframe["edges"], [(0, 1), (1, 2)])
        self.assertEqual(matplotlib_data["points"]["x"], [0.0, 1.0, 1.0])
        self.assertEqual(len(matplotlib_data["line_segments"]), 2)
        self.assertEqual(plotly_data[0]["type"], "scatter")
        self.assertEqual(plotly_data[1]["type"], "scatter")

    def test_triangle_mesh_adapters(self):
        """Triangle meshes should export 3D plotting and indexed data."""
        mesh = ObjectMesh(
            (
                FloatPoint(0.0, 0.0, 0.0),
                FloatPoint(1.0, 0.0, 0.0),
                FloatPoint(0.0, 1.0, 0.0),
            ),
            ((0, 1, 2),),
        )

        matplotlib_data = mesh.matplotlib_data()
        plotly_data = mesh.plotly_data(name="triangle")
        threejs_data = mesh.threejs_data()

        self.assertEqual(matplotlib_data["dim"], 3)
        self.assertEqual(len(matplotlib_data["triangles"]), 1)
        self.assertEqual(plotly_data[0]["type"], "mesh3d")
        self.assertEqual(plotly_data[0]["i"], [0])
        self.assertEqual(plotly_data[0]["j"], [1])
        self.assertEqual(plotly_data[0]["k"], [2])
        self.assertEqual(threejs_data["vertex_count"], 3)
        self.assertEqual(threejs_data["triangle_indices"], [0, 1, 2])

    def test_projection_aware_convenience_methods(self):
        """Convenience methods should support projected export views."""
        mesh = ObjectMesh(
            (
                FloatPoint(0.0, 0.0, 0.0),
                FloatPoint(1.0, 0.0, 0.0),
                FloatPoint(1.0, 1.0, 0.0),
            ),
            ((0, 1), (1, 2)),
        )

        wireframe = mesh.wireframe_data(axes=(0, 1))
        matplotlib_data = mesh.matplotlib_data(axes=(0, 1))
        plotly_data = mesh.plotly_data(axes=(0, 1), name="projected")

        self.assertEqual(wireframe["dim"], 2)
        self.assertEqual(matplotlib_data["dim"], 2)
        self.assertEqual(plotly_data[0]["type"], "scatter")
