"""Minimal public API for geometric spaces and objects."""

from .gobject import (
    GeometricObject,
    classify_cover,
    local_chart_cover_from_points,
    refine_until,
)


def make_euclidean(dim: int):
    """Return the Euclidean family space of the requested dimension."""
    from .space import make_euclidean as _make

    return _make(dim)


def make_sphere(dim: int = 2, **kwargs):
    """Return the sphere of the requested intrinsic dimension."""
    from .space import make_sphere as _make

    return _make(dim, **kwargs)


def make_torus(dim: int, **kwargs):
    """Return the torus family space of the requested dimension."""
    from .space import make_torus as _make

    return _make(dim, **kwargs)


__all__ = [
    "GeometricObject",
    "make_euclidean",
    "make_sphere",
    "make_torus",
    "classify_cover",
    "local_chart_cover_from_points",
    "refine_until",
]
