"""Base abstractions for spaces."""

from __future__ import annotations

from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

from ..euclidean import FloatPoint
from ..manifold import Manifold, Neighborhood


PointT = TypeVar("PointT")


@runtime_checkable
class Space(Manifold[PointT], Protocol[PointT]):
    """Protocol for spaces with distance and neighborhood covers."""

    def distance(self, left: PointT, right: PointT) -> float:
        """Return the distance between two points."""

    def full_cover(self, radius: float):
        """Return a full cover of the space by neighborhoods."""

    def refine_cover(self, neighborhoods, factor: int = 2):
        """Return a covering refinement with smaller diameters."""


class ChartedSpace(Generic[PointT]):
    """Concrete space given by a manifold and a distance function."""

    def __init__(
        self,
        manifold: Manifold[PointT],
        distance: Callable[[PointT, PointT], float],
        name: str = "",
    ) -> None:
        """Initialize the space."""
        self.manifold = manifold
        self._distance = distance
        self.name = name

    @property
    def dim(self) -> int:
        """Return the space dimension."""
        return self.manifold.dim

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"ChartedSpace(dim={self.dim}{label})"

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the underlying manifold."""
        return point in self.manifold

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the underlying manifold."""
        return self.contains(point)

    def distance(self, left: PointT, right: PointT) -> float:
        """Return the distance between two points."""
        if left not in self or right not in self:
            raise ValueError("Points must belong to the space")
        value = float(self._distance(left, right))
        if value < 0.0:
            raise ValueError("Distance must be non-negative")
        return value

    def wrap(self, obj, name: str = ""):
        """Wrap a charted object into this ambient space."""
        from ..gobject import GeometricObject

        return GeometricObject.from_charted(self, obj, name=name)


def refine_neighborhoods(
    neighborhoods: tuple[Neighborhood[PointT], ...],
    factor: int = 2,
) -> tuple[Neighborhood[PointT], ...]:
    """Refine neighborhoods until diameters drop by the requested factor."""
    if factor < 2:
        raise ValueError("Refinement factor must be at least 2")
    refined = tuple(neighborhoods)
    steps = 0
    current = factor
    while current > 1:
        if current % 2 != 0 and current != 1:
            raise ValueError("Refinement factor must be a power of 2")
        current //= 2
        steps += 1
    for _ in range(steps):
        refined = tuple(
            child
            for neighborhood in refined
            for child in neighborhood.subdivide()
        )
    return refined


def centered_real_chart(center: float, domain_contains) -> "ManifoldChart[float]":
    """Return the canonical centered chart on the real line."""
    from ..euclidean import EuclideanNeighborhood
    from ..manifold import ManifoldChart

    return ManifoldChart(
        lambda point: FloatPoint(float(point) - center),
        lambda coordinates: center + coordinates[0],
        dim=1,
        domain_contains=domain_contains,
        image=EuclideanNeighborhood.whole(1),
        name="real-centered",
    )


__all__ = [
    "Space",
    "ChartedSpace",
    "refine_neighborhoods",
    "centered_real_chart",
]
