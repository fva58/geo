"""Manifold protocols, local charts, atlases, and local refinement data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, Protocol, TypeVar, runtime_checkable

from .euclidean import EuclideanNeighborhood, FloatPoint


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
        forward: Callable[[PointT], FloatPoint],
        inverse: Callable[[FloatPoint], PointT],
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

    def __call__(self, point: PointT) -> FloatPoint:
        """Apply the chart map to a manifold point."""
        if self.domain_contains is not None and not self.domain_contains(point):
            raise ValueError("Point is outside the chart domain")
        coordinates = FloatPoint(self._forward(point))
        if coordinates.dim != self.dim:
            raise ValueError(
                f"Coordinate dimension mismatch: {coordinates.dim} != {self.dim}"
            )
        if self.image is not None and coordinates not in self.image:
            raise ValueError("Coordinates are outside the chart image")
        return coordinates

    def inverse(self, coordinates: FloatPoint) -> PointT:
        """Apply the inverse chart map."""
        coordinates = FloatPoint(coordinates)
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


class ChartTransition(Generic[PointT]):
    """Coordinate transition between two charts on the same manifold patch."""

    def __init__(
        self,
        source_chart: ManifoldChart[PointT],
        target_chart: ManifoldChart[PointT],
    ) -> None:
        """Initialize a transition from source coordinates to target coordinates."""
        if source_chart.dim != target_chart.dim:
            raise ValueError(
                "Chart dimensions do not match: "
                f"{source_chart.dim} != {target_chart.dim}"
            )
        self.source_chart = source_chart
        self.target_chart = target_chart
        self.dim = source_chart.dim

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"ChartTransition(dim={self.dim})"

    def __call__(self, coordinates: FloatPoint) -> FloatPoint:
        """Apply the transition map in coordinates."""
        return self.target_chart(self.source_chart.inverse(coordinates))

    def inverse(self, coordinates: FloatPoint) -> FloatPoint:
        """Apply the inverse transition map."""
        return self.source_chart(self.target_chart.inverse(coordinates))


__all__ = [
    "Manifold",
    "ManifoldChart",
    "ChartTransition",
]
