"""Space namespace package."""

from . import base, circle, euclidean, line, point, sphere, torus


def make_euclidean(dim: int):
    """Return the Euclidean family space of the requested dimension."""
    dim = int(dim)
    if dim < 0:
        raise ValueError("Dimension must be non-negative")
    if dim == 0:
        return point.Space()
    if dim == 1:
        return line.Space()
    return euclidean.Space(dim)


def make_torus(dim: int, **kwargs):
    """Return the torus family space of the requested dimension."""
    dim = int(dim)
    if dim < 0:
        raise ValueError("Dimension must be non-negative")
    if dim == 0:
        return point.Space()
    if dim == 1:
        return circle.Space()
    return torus.Space(dim=dim, **kwargs)


def make_sphere(dim: int = 2, **kwargs):
    """Return the sphere of the requested intrinsic dimension."""
    return sphere.Space(dim=int(dim), **kwargs)


__all__ = [
    "base",
    "point",
    "line",
    "circle",
    "euclidean",
    "sphere",
    "torus",
    "make_euclidean",
    "make_torus",
    "make_sphere",
]
