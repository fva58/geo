"""Computational refinement operations for geometric objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .manifold import ChartNeighborhood, LocalObjectModel, NeighborhoodCover


PointT = TypeVar("PointT")


def local_chart_cover_from_points(
    space,
    points,
    radius: float,
    name: str = "",
) -> NeighborhoodCover[PointT]:
    """Build a neighborhood cover from explicit points in one space."""
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("Neighborhood radius must be positive")
    neighborhoods = tuple(
        space.neighborhood_at(point, radius, name=name)
        for point in points
    )
    if not neighborhoods:
        raise ValueError("Need at least one point to build a cover")
    return NeighborhoodCover(
        neighborhoods,
        name=name or getattr(space, "name", ""),
    )


@dataclass(frozen=True)
class RefinedObjectCover(Generic[PointT]):
    """Refinement state for one object over a neighborhood cover."""

    obj: object
    cone_parts: tuple[LocalObjectModel[PointT], ...]
    complex_parts: tuple[LocalObjectModel[PointT], ...]
    empty_parts: tuple[LocalObjectModel[PointT], ...]

    @property
    def active_parts(self) -> tuple[LocalObjectModel[PointT], ...]:
        """Return the non-empty parts of the current refinement."""
        return self.cone_parts + self.complex_parts

    def max_diameter(self) -> float:
        """Return the largest diameter among active parts."""
        if not self.active_parts:
            return 0.0
        return max(part.neighborhood.diameter() for part in self.active_parts)

    def max_outer_radius(self) -> float:
        """Return the largest outer radius among active parts."""
        if not self.active_parts:
            return 0.0
        return max(part.neighborhood.outer_radius() for part in self.active_parts)


def classify_cover(
    obj,
    cover: NeighborhoodCover[PointT],
) -> RefinedObjectCover[PointT]:
    """Classify one object over all neighborhoods in a cover."""
    marking = obj.classify_neighborhoods(
        cover.neighborhoods,
        name=cover.name,
    )
    return RefinedObjectCover(
        obj,
        marking.cone,
        marking.complex,
        marking.empty,
    )


def refine_until(
    obj,
    cover: NeighborhoodCover[PointT],
    *,
    max_outer_radius: float,
    max_steps: int = 8,
) -> RefinedObjectCover[PointT]:
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
        to_keep = [part.neighborhood for part in current.cone_parts]
        to_refine = [part.neighborhood for part in current.complex_parts]
        if current.max_outer_radius() > max_outer_radius:
            to_refine.extend(
                part.neighborhood
                for part in current.cone_parts
                if part.neighborhood.outer_radius() > max_outer_radius
            )
            to_keep = [
                part.neighborhood
                for part in current.cone_parts
                if part.neighborhood.outer_radius() <= max_outer_radius
            ]
        refined = tuple(
            child
            for neighborhood in to_refine
            for child in neighborhood.subdivide()
        )
        if not refined:
            return current
        current_cover = NeighborhoodCover(tuple(to_keep) + refined, name=cover.name)
        current = classify_cover(obj, current_cover)
    return current


__all__ = [
    "local_chart_cover_from_points",
    "RefinedObjectCover",
    "classify_cover",
    "refine_until",
]
