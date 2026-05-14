"""Manifold protocols, local charts, atlases, and local refinement data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, Protocol, TypeVar, runtime_checkable

from .euclidean import EuclideanNeighborhood, Point


PointT = TypeVar("PointT")
LocalPointT = TypeVar("LocalPointT")

if TYPE_CHECKING:
    from .space.base import Neighborhood


@runtime_checkable
class Manifold(Protocol[PointT]):
    """Protocol for a manifold-like space with point membership."""

    @property
    def dim(self) -> int:
        """Return the manifold dimension."""

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the manifold."""

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the manifold."""


class ManifoldChart(Generic[PointT]):
    """Local chart from manifold points to Euclidean coordinates."""

    def __init__(
        self,
        forward: Callable[[PointT], Point],
        inverse: Callable[[Point], PointT],
        dim: int,
        domain_contains: Callable[[PointT], bool] | None = None,
        image: EuclideanNeighborhood | None = None,
    ) -> None:
        """Initialize a local manifold chart."""
        self._forward = forward
        self._inverse = inverse
        self.dim = dim
        self.domain_contains = domain_contains
        self.image = image

    def __repr__(self) -> str:
        """Return a debug representation."""
        image = f", image_dim={self.image.dim}" if self.image else ""
        return f"ManifoldChart(dim={self.dim}{image})"

    def __call__(self, point: PointT) -> Point:
        """Apply the chart map to a manifold point."""
        if self.domain_contains is not None and not self.domain_contains(point):
            raise ValueError("Point is outside the chart domain")
        coordinates = Point(self._forward(point))
        if coordinates.dim != self.dim:
            raise ValueError(
                f"Coordinate dimension mismatch: {coordinates.dim} != {self.dim}"
            )
        if self.image is not None and coordinates not in self.image:
            raise ValueError("Coordinates are outside the chart image")
        return coordinates

    def inverse(self, coordinates: Point) -> PointT:
        """Apply the inverse chart map."""
        coordinates = Point(coordinates)
        if coordinates.dim != self.dim:
            raise ValueError(
                f"Coordinate dimension mismatch: {coordinates.dim} != {self.dim}"
            )
        if self.image is not None and coordinates not in self.image:
            raise ValueError("Coordinates are outside the chart image")
        point = self._inverse(coordinates)
        if self.domain_contains is not None and not self.domain_contains(point):
            raise ValueError("Inverse image point is outside the chart domain")
        return point


__all__ = [
    "Manifold",
    "ManifoldChart",
]
