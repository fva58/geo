"""Computational refinement operations for geometric objects."""

from __future__ import annotations

from typing import NamedTuple

from .cone import LocalConeModel


class _CoverResult(NamedTuple):
    """Result of cover classification or refinement."""

    cone_parts: tuple
    complex_parts: tuple
    empty_parts: tuple

    @property
    def active_parts(self):
        """Return the non-empty parts."""
        return self.cone_parts + self.complex_parts

    def max_diameter(self) -> float:
        """Return the largest diameter among active parts."""
        if not self.active_parts:
            return 0.0
        return max(n.diameter() for n in self.active_parts)

    def max_outer_radius(self) -> float:
        """Return the largest outer radius among active parts."""
        if not self.active_parts:
            return 0.0
        return max(n.outer_radius() for n in self.active_parts)


def local_chart_cover_from_points(
    space,
    points,
    radius: float,
):
    """Build a neighborhood cover from explicit points in one space."""
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("Neighborhood radius must be positive")
    neighborhoods = tuple(
        space.neighborhood_at(point, radius)
        for point in points
    )
    if not neighborhoods:
        raise ValueError("Need at least one point to build a cover")
    return neighborhoods


def classify_cover(
    obj,
    cover,
):
    """Classify one object over all neighborhoods in a cover."""
    cone = []
    complex_ = []
    empty = []
    for neighborhood, result in zip(cover, obj.classify_neighborhoods(cover)):
        if isinstance(result, LocalConeModel):
            cone.append(neighborhood)
        elif result is Ellipsis:
            complex_.append(neighborhood)
        else:
            empty.append(neighborhood)
    return _CoverResult(tuple(cone), tuple(complex_), tuple(empty))


def refine_until(
    obj,
    cover,
    *,
    max_outer_radius: float,
    max_steps: int = 8,
):
    """Refine a cover until non-empty parts are small enough or steps end."""
    if max_outer_radius <= 0.0:
        raise ValueError("max_outer_radius must be positive")
    current_cover = cover
    current = classify_cover(obj, current_cover)
    for _ in range(max_steps):
        if (
            not current.complex_parts and
            current.max_outer_radius() <= max_outer_radius
        ):
            return current
        to_keep = list(current.cone_parts)
        to_refine = list(current.complex_parts)
        if current.max_outer_radius() > max_outer_radius:
            to_refine.extend(
                n for n in current.cone_parts
                if n.outer_radius() > max_outer_radius
            )
            to_keep = [
                n for n in current.cone_parts
                if n.outer_radius() <= max_outer_radius
            ]
        refined = tuple(
            child
            for neighborhood in to_refine
            for child in neighborhood.subdivide()
        )
        if not refined:
            return current
        current_cover = tuple(to_keep) + refined
        current = classify_cover(obj, current_cover)
    return current


__all__ = [
    "local_chart_cover_from_points",
    "classify_cover",
    "refine_until",
]
