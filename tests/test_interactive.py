"""Tests for optional interactive helpers."""

import math
import types
import unittest
from unittest import mock

from geo import (
    Ball,
    CircleSpace,
    EuclideanPlaneSpace,
    FloatCirclePoint,
    FloatPoint,
    ObjectMesh,
    RealLineSpace,
    SphereSpace,
    TorusPoint,
    TorusSpace,
)
from geo.interactive import (
    angle,
    arc,
    ball,
    cap,
    circle,
    current_space,
    disk,
    half_plane,
    mesh,
    patch,
    plot,
    point,
    reset_default_space,
    set_default_space,
    subset,
    using_space,
    wrap,
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


class TestInteractiveHelpers(unittest.TestCase):
    """Test the notebook-oriented helper layer."""

    def setUp(self):
        """Reset the shared interactive state before each test."""
        reset_default_space()

    def tearDown(self):
        """Reset the shared interactive state after each test."""
        reset_default_space()

    def test_default_space_is_the_euclidean_plane(self):
        """The default interactive space should be a plane."""
        self.assertIsInstance(current_space(), EuclideanPlaneSpace)

        origin = point(0.0, 0.0)
        disk_object = disk((0.0, 0.0), 1.0)
        circle_object = circle((0.0, 0.0), 1.0)
        upper = half_plane((0.0, 1.0))
        wedge = angle(FloatPoint(0.0, 0.0), 0.0, math.pi / 2.0)

        self.assertIn(FloatPoint(0.0, 0.0), origin)
        self.assertIn(FloatPoint(0.5, 0.0), disk_object)
        self.assertIn(FloatPoint(1.0, 0.0), circle_object)
        self.assertIn(FloatPoint(0.0, 1.0), upper)
        self.assertIn(FloatPoint(1.0, 1.0), wedge)

    def test_wrap_and_ball_helpers_use_the_selected_space(self):
        """Convenience helpers should wrap Euclidean zoo objects."""
        space = EuclideanPlaneSpace(name="plane")
        wrapped_ball = wrap(Ball(FloatPoint(0.0, 0.0), 1.0), space=space)
        direct_ball = ball((0.0, 0.0), 1.0, space=space)

        self.assertIs(wrapped_ball.space, space)
        self.assertIs(direct_ball.space, space)
        self.assertIn(FloatPoint(0.0, 0.0), wrapped_ball)
        self.assertIn(FloatPoint(0.5, 0.5), direct_ball)

    def test_using_space_temporarily_switches_the_default_space(self):
        """A context manager should isolate default-space changes."""
        original = current_space()

        with using_space(RealLineSpace()) as space:
            self.assertIs(current_space(), space)
            obj = subset((0.0, 2.0), 5.0)
            singleton = point(5.0)
            self.assertIn(1.0, obj)
            self.assertIn(5.0, singleton)

        self.assertIs(current_space(), original)

    def test_circle_and_arc_helpers(self):
        """Circle-specific helpers should route through the unit circle."""
        set_default_space(CircleSpace())
        quarter = arc(0.0, math.pi / 2.0)
        singleton = point(math.pi / 4.0)

        self.assertIn(FloatCirclePoint(math.pi / 4.0), quarter)
        self.assertIn(FloatCirclePoint(math.pi / 4.0), singleton)

    def test_sphere_and_torus_helpers(self):
        """Native-space helpers should use point objects and native families."""
        sphere = SphereSpace()
        north = sphere.point_from_angles(0.0, math.pi / 2.0)
        set_default_space(sphere)
        north_object = point(north)
        hemisphere = cap(north, math.pi / 2.0)

        self.assertIn(north, north_object)
        self.assertIn(sphere.point_from_angles(0.0, 0.0), hemisphere)

        torus = TorusSpace()
        set_default_space(torus)
        patch_object = patch((0.0, math.pi / 2.0), (0.0, math.pi / 2.0))
        torus_point = point(0.0, 0.0)

        self.assertIn(TorusPoint(math.pi / 4.0, math.pi / 4.0), patch_object)
        self.assertIn(TorusPoint(0.0, 0.0), torus_point)

    def test_mesh_and_plot_helpers_support_objects_and_meshes(self):
        """Notebook helpers should mesh and plot with little boilerplate."""
        pyplot = _DummyPyplot()
        disk_object = disk((0.0, 0.0), 1.0)
        disk_mesh = mesh(disk_object, resolution=8)

        self.assertIsInstance(disk_mesh, ObjectMesh)

        with mock.patch.dict(
            "sys.modules",
            {"matplotlib.pyplot": pyplot},
        ):
            figure, axis = plot(disk_object, resolution=8)
            mesh_figure, mesh_axis = disk_mesh.plot_matplotlib()
            object_figure, object_axis = disk_object.plot_matplotlib(
                resolution=8,
            )

        self.assertIsNotNone(figure)
        if hasattr(axis, "plots"):
            self.assertTrue(axis.plots)
            self.assertTrue(mesh_axis.plots)
            self.assertTrue(object_axis.plots)
        else:
            self.assertTrue(axis.lines)
            self.assertTrue(mesh_axis.lines)
            self.assertTrue(object_axis.lines)
        self.assertIs(mesh_figure, mesh_axis.figure)
        self.assertIs(object_figure, object_axis.figure)
