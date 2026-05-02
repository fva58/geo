"""Space package for intrinsic spaces."""

from .circle import Circle
from .euclidean import Euclidean
from .line import RealLine
from .point import Point
from .sphere import Sphere
from .torus import Torus


def make_euclidean(dim: int, name: str = ""):
    """Return the Euclidean family space of the requested dimension."""
    dim = int(dim)
    if dim < 0:
        raise ValueError("Dimension must be non-negative")
    if dim == 0:
        return Point(name=name or "Point")
    if dim == 1:
        return RealLine(name=name or "R")
    return Euclidean(dim, name=name)


def make_torus(dim: int, **kwargs):
    """Return the torus family space of the requested dimension."""
    dim = int(dim)
    if dim < 0:
        raise ValueError("Dimension must be non-negative")
    if dim == 0:
        return Point(name=kwargs.pop("name", "") or "Point")
    if dim == 1:
        return Circle(name=kwargs.pop("name", "") or "S1")
    return Torus(dim=dim, **kwargs)


def make_sphere(dim: int = 2, **kwargs):
    """Return the sphere of the requested intrinsic dimension."""
    return Sphere(dim=int(dim), **kwargs)

__all__ = [
    "Point",
    "RealLine",
    "Circle",
    "Euclidean",
    "Sphere",
    "Torus",
    "make_euclidean",
    "make_torus",
    "make_sphere",
]
