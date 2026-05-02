"""Minimal public API for geometric spaces and objects."""

from .gobject import GeometricObject
from .operations import (
    RefinedObjectCover,
    classify_cover,
    local_chart_cover_from_points,
    refine_until,
)
from .space import Circle, Euclidean, Point, RealLine, Sphere, Torus
from .space import make_euclidean, make_sphere, make_torus

__all__ = [
    "GeometricObject",
    "Point",
    "RealLine",
    "Circle",
    "Euclidean",
    "Sphere",
    "Torus",
    "make_euclidean",
    "make_sphere",
    "make_torus",
    "RefinedObjectCover",
    "classify_cover",
    "local_chart_cover_from_points",
    "refine_until",
]
