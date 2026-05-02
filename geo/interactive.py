"""Notebook-friendly helpers for interactive geometry sessions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .space import Euclidean


_DEFAULT_SPACE = None


def _make_default_space() -> Euclidean:
    """Return the package-wide default space for interactive work."""
    return Euclidean(2, name="default-plane")


def current_space():
    """Return the current default space for interactive helpers."""
    global _DEFAULT_SPACE
    if _DEFAULT_SPACE is None:
        _DEFAULT_SPACE = _make_default_space()
    return _DEFAULT_SPACE


def set_default_space(space):
    """Set and return the current default space."""
    global _DEFAULT_SPACE
    _DEFAULT_SPACE = space
    return space


def use_space(space):
    """Alias for ``set_default_space()`` for notebook-style workflows."""
    return set_default_space(space)


def reset_default_space():
    """Reset the interactive default space to the Euclidean plane."""
    return set_default_space(_make_default_space())


@contextmanager
def using_space(space) -> Iterator[object]:
    """Temporarily use a space as the default interactive ambient space."""
    previous = current_space()
    set_default_space(space)
    try:
        yield space
    finally:
        set_default_space(previous)


def _resolve_space(space):
    """Return an explicit space or the current default space."""
    return current_space() if space is None else space


def _coerce_point_arguments(*coordinates: object) -> object:
    """Return one scalar or one tuple from notebook-style point arguments."""
    if not coordinates:
        raise TypeError("Need at least one coordinate or point value")
    if len(coordinates) == 1:
        return coordinates[0]
    return tuple(coordinates)


def wrap(obj, *, space=None, name: str = ""):
    """Wrap a charted object into an ambient metric space."""
    ambient = _resolve_space(space)
    wrap_method = getattr(ambient, "wrap", None)
    if wrap_method is None:
        raise TypeError(f"Space {ambient!r} does not support wrap()")
    return wrap_method(obj, name=name)


def point(*coordinates: object, space=None, name: str = ""):
    """Build a singleton object in the selected space."""
    ambient = _resolve_space(space)
    value = _coerce_point_arguments(*coordinates)
    if hasattr(ambient, "point_object"):
        return ambient.point_object(value, name=name)
    point_method = getattr(ambient, "point", None)
    if point_method is None:
        raise TypeError(f"Space {ambient!r} does not support point()")
    return point_method(value, name=name)


def subset(*point_set: object, space=None, name: str = ""):
    """Build a set object in the selected space when supported."""
    ambient = _resolve_space(space)
    subset_method = getattr(ambient, "subset", None)
    if subset_method is None:
        raise TypeError(f"Space {ambient!r} does not support subset()")
    return subset_method(*point_set, name=name)


def arc(start, end, *, space=None, name: str = ""):
    """Build an arc object in the selected space when supported."""
    ambient = _resolve_space(space)
    arc_method = getattr(ambient, "arc", None)
    if arc_method is None:
        raise TypeError(f"Space {ambient!r} does not support arc()")
    return arc_method(start, end, name=name)


def half_plane(normal, offset: float = 0.0, *, space=None, name: str = ""):
    """Build a half-plane in the selected space when supported."""
    ambient = _resolve_space(space)
    half_plane_method = getattr(ambient, "half_plane", None)
    if half_plane_method is None:
        raise TypeError(f"Space {ambient!r} does not support half_plane()")
    return half_plane_method(normal, offset=offset, name=name)


def angle(vertex, start_angle: float, end_angle: float, *,
          space=None, name: str = ""):
    """Build a planar angle in the selected space when supported."""
    ambient = _resolve_space(space)
    angle_method = getattr(ambient, "angle", None)
    if angle_method is None:
        raise TypeError(f"Space {ambient!r} does not support angle()")
    return angle_method(vertex, start_angle, end_angle, name=name)


def ball(center, radius: float, *, space=None, name: str = ""):
    """Build a Euclidean ball in the selected space when supported."""
    ambient = _resolve_space(space)
    ball_method = getattr(ambient, "ball", None)
    if ball_method is None:
        raise TypeError(f"Space {ambient!r} does not support ball()")
    return ball_method(center, radius, name=name)


def disk(center, radius: float, *, space=None, name: str = ""):
    """Alias for ``ball()`` in two-dimensional notebook workflows."""
    return ball(center, radius, space=space, name=name)


def circle(center, radius: float, *, space=None, name: str = ""):
    """Build a Euclidean sphere boundary in the selected space."""
    ambient = _resolve_space(space)
    circle_method = getattr(ambient, "circle", None)
    if circle_method is None:
        raise TypeError(f"Space {ambient!r} does not support circle()")
    return circle_method(center, radius, name=name)


def cap(center, radius: float, *, space=None, name: str = ""):
    """Build a spherical cap in the selected space when supported."""
    ambient = _resolve_space(space)
    cap_method = getattr(ambient, "cap", None)
    if cap_method is None:
        raise TypeError(f"Space {ambient!r} does not support cap()")
    return cap_method(center, radius, name=name)


def patch(major_set, minor_set, *, space=None, name: str = ""):
    """Build a torus patch in the selected space when supported."""
    ambient = _resolve_space(space)
    patch_method = getattr(ambient, "patch", None)
    if patch_method is None:
        raise TypeError(f"Space {ambient!r} does not support patch()")
    return patch_method(major_set, minor_set, name=name)


__all__ = [
    "arc",
    "angle",
    "ball",
    "cap",
    "circle",
    "current_space",
    "disk",
    "half_plane",
    "patch",
    "point",
    "reset_default_space",
    "set_default_space",
    "subset",
    "use_space",
    "using_space",
    "wrap",
]
