"""Tests for optional interactive helpers."""

import math
import unittest

from geo.euclidean import FloatPoint
from geo.gobject import ChartedGeometricObject
from geo.space import Circle, Euclidean, RealLine, Sphere, Torus
from geo.space.circle import Point as CirclePoint
from geo.space.torus import TorusPoint
from geo.interactive import (
    angle,
    arc,
    ball,
    cap,
    circle,
    current_space,
    disk,
    half_plane,
    patch,
    point,
    reset_default_space,
    set_default_space,
    subset,
    using_space,
    wrap,
)


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
        self.assertIsInstance(current_space(), Euclidean)

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
        space = Euclidean(2, name="plane")
        line = RealLine()
        wrapped_subset = wrap(
            ChartedGeometricObject(
                line,
                contains=lambda point: 0.0 <= float(point) <= 1.0,
                local_model=lambda point: line.subset((0.0, 1.0)).local_model_at(point),
                name="segment",
            ),
            space=line,
        )
        direct_ball = ball((0.0, 0.0), 1.0, space=space)

        self.assertIsInstance(wrapped_subset.space, RealLine)
        self.assertIs(direct_ball.space, space)
        self.assertIn(0.5, wrapped_subset)
        self.assertIn(FloatPoint(0.5, 0.5), direct_ball)

    def test_using_space_temporarily_switches_the_default_space(self):
        """A context manager should isolate default-space changes."""
        original = current_space()

        with using_space(RealLine()) as space:
            self.assertIs(current_space(), space)
            obj = subset((0.0, 2.0), 5.0)
            singleton = point(5.0)
            self.assertIn(1.0, obj)
            self.assertIn(5.0, singleton)

        self.assertIs(current_space(), original)

    def test_circle_and_arc_helpers(self):
        """Circle-specific helpers should route through the unit circle."""
        set_default_space(Circle())
        quarter = arc(0.0, math.pi / 2.0)
        singleton = point(math.pi / 4.0)

        self.assertIn(CirclePoint(math.pi / 4.0), quarter)
        self.assertIn(CirclePoint(math.pi / 4.0), singleton)

    def test_sphere_and_torus_helpers(self):
        """Native-space helpers should use point objects and native families."""
        sphere = Sphere()
        north = sphere.point_from_angles(0.0, math.pi / 2.0)
        set_default_space(sphere)
        north_object = point(north)
        hemisphere = cap(north, math.pi / 2.0)

        self.assertIn(north, north_object)
        self.assertIn(sphere.point_from_angles(0.0, 0.0), hemisphere)

        torus = Torus()
        set_default_space(torus)
        patch_object = patch((0.0, math.pi / 2.0), (0.0, math.pi / 2.0))
        torus_point = point(0.0, 0.0)

        self.assertIn(TorusPoint(math.pi / 4.0, math.pi / 4.0), patch_object)
        self.assertIn(TorusPoint(0.0, 0.0), torus_point)
