"""Tests for mesh plotting helpers and file exporters."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from geo import (
    FloatPoint,
    ObjectMesh,
    mesh_to_gltf_json_data,
    mesh_to_obj_text,
    mesh_to_ply_text,
    plot_mesh_matplotlib,
    plot_mesh_plotly,
    write_gltf_json,
    write_obj,
    write_ply,
)


class _DummyAxes:
    def __init__(self, figure):
        self.figure = figure
        self.plots = []
        self.scatters = []

    def plot(self, *args, **kwargs):
        self.plots.append((args, kwargs))

    def scatter(self, *args, **kwargs):
        self.scatters.append((args, kwargs))


class _DummyFigure:
    def __init__(self):
        self.axes = []

    def add_subplot(self, *args, **kwargs):
        axis = _DummyAxes(self)
        self.axes.append(axis)
        return axis


class _DummyPyplot(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.created = []

    def subplots(self):
        figure = _DummyFigure()
        axis = _DummyAxes(figure)
        figure.axes.append(axis)
        self.created.append((figure, axis))
        return figure, axis

    def figure(self):
        figure = _DummyFigure()
        self.created.append((figure, None))
        return figure


class _DummyPlotlyFigure:
    def __init__(self):
        self.traces = []

    def add_trace(self, trace):
        self.traces.append(trace)


class TestMeshIoAndPlot(unittest.TestCase):
    """Test plotting helpers and file exporters."""

    def test_text_exporters(self):
        """OBJ, PLY, and glTF-friendly data should be produced."""
        mesh = ObjectMesh(
            (
                FloatPoint(0.0, 0.0, 0.0),
                FloatPoint(1.0, 0.0, 0.0),
                FloatPoint(0.0, 1.0, 0.0),
            ),
            ((0, 1, 2),),
        )

        obj_text = mesh_to_obj_text(mesh)
        ply_text = mesh_to_ply_text(mesh)
        gltf_data = mesh_to_gltf_json_data(mesh)

        self.assertIn("v 0.0 0.0 0.0", obj_text)
        self.assertIn("f 1 2 3", obj_text)
        self.assertIn("element vertex 3", ply_text)
        self.assertIn("element face 1", ply_text)
        self.assertEqual(gltf_data["asset"]["version"], "2.0")
        self.assertEqual(gltf_data["extras"]["vertex_count"], 3)

    def test_file_writers(self):
        """Mesh exporters should write files to disk."""
        mesh = ObjectMesh(
            (
                FloatPoint(0.0, 0.0),
                FloatPoint(1.0, 0.0),
            ),
            ((0, 1),),
        )

        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            obj_path = write_obj(mesh, temp_path / "line.obj")
            ply_path = write_ply(mesh, temp_path / "line.ply")
            gltf_path = write_gltf_json(mesh, temp_path / "line.gltf.json")

            self.assertTrue(obj_path.exists())
            self.assertTrue(ply_path.exists())
            self.assertTrue(gltf_path.exists())
            self.assertIn("v 0.0 0.0 0.0", obj_path.read_text(encoding="utf-8"))
            gltf_payload = json.loads(gltf_path.read_text(encoding="utf-8"))
            self.assertEqual(gltf_payload["asset"]["generator"], "geo")

    def test_matplotlib_plot_helper_with_fake_backend(self):
        """Matplotlib helper should use plotting calls on the backend."""
        mesh = ObjectMesh(
            (
                FloatPoint(0.0, 0.0),
                FloatPoint(1.0, 0.0),
            ),
            ((0, 1),),
        )
        pyplot = _DummyPyplot()

        with mock.patch.dict(
            "sys.modules",
            {"matplotlib.pyplot": pyplot},
        ):
            figure, axis = plot_mesh_matplotlib(mesh)

        self.assertIsNotNone(figure)
        self.assertTrue(axis.plots)
        self.assertTrue(axis.scatters)

    def test_plotly_plot_helper_with_fake_backend(self):
        """Plotly helper should add converted traces to a figure."""
        mesh = ObjectMesh(
            (
                FloatPoint(0.0, 0.0, 0.0),
                FloatPoint(1.0, 0.0, 0.0),
                FloatPoint(0.0, 1.0, 0.0),
            ),
            ((0, 1, 2),),
        )
        graph_objects = types.SimpleNamespace(
            Figure=_DummyPlotlyFigure,
            Scatter=lambda **kwargs: {"kind": "scatter", **kwargs},
            Scatter3d=lambda **kwargs: {"kind": "scatter3d", **kwargs},
            Mesh3d=lambda **kwargs: {"kind": "mesh3d", **kwargs},
        )
        plotly_module = types.SimpleNamespace(graph_objects=graph_objects)

        with mock.patch.dict(
            "sys.modules",
            {
                "plotly": plotly_module,
                "plotly.graph_objects": graph_objects,
            },
        ):
            figure = plot_mesh_plotly(mesh, name="triangle")

        self.assertTrue(figure.traces)
        self.assertEqual(figure.traces[0]["kind"], "mesh3d")
