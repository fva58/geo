"""Base abstractions for spaces."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

from ..euclidean import EuclideanNeighborhood, FloatPoint
from ..manifold import Manifold, ManifoldChart


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


@runtime_checkable
class Neighborhood(Protocol[PointT]):
    """Protocol for a local neighborhood in a space."""

    @property
    def manifold(self) -> Manifold[PointT]:
        """Return the ambient manifold."""

    @property
    def chart(self) -> ManifoldChart[PointT]:
        """Return chart coordinates for the neighborhood."""

    @property
    def center(self) -> PointT:
        """Return a distinguished point in the neighborhood."""

    @property
    def image(self) -> EuclideanNeighborhood:
        """Return the coordinate image of the neighborhood."""

    def inner_radius(self) -> float:
        """Return a guaranteed included ball radius around the center."""

    def outer_radius(self) -> float:
        """Return a guaranteed containing ball radius around the center."""

    def diameter(self) -> float:
        """Return an upper bound on the neighborhood diameter."""

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the neighborhood."""

    def center_point(self) -> PointT:
        """Return the distinguished center point."""

    def subdivide(self) -> tuple["Neighborhood[PointT]", ...]:
        """Return a finite refinement cover by smaller neighborhoods."""


def _single_interval_bounds(
    image: EuclideanNeighborhood,
) -> tuple[tuple[float, float], ...]:
    """Return one finite interval per coordinate of a box image."""
    bounds = []
    for coordinate_set in image:
        if len(coordinate_set) != 1:
            raise ValueError("Neighborhood subdivision requires box intervals")
        interval = coordinate_set[0]
        left = float(interval[0])
        right = float(interval[1])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Neighborhood subdivision requires finite bounds")
        bounds.append((left, right))
    return tuple(bounds)


@dataclass(frozen=True)
class BoxNeighborhood(Generic[PointT]):
    """Neighborhood represented by one chart patch and a box image."""

    manifold: Manifold[PointT]
    chart: ManifoldChart[PointT]
    center: PointT
    image: EuclideanNeighborhood

    def __post_init__(self) -> None:
        """Validate center and image dimensions."""
        if self.chart.dim != self.manifold.dim:
            raise ValueError("Chart dimension must match manifold dimension")
        if self.image.dim != self.chart.dim:
            raise ValueError("Image dimension must match chart dimension")
        center_coordinates = self.chart(self.center)
        if center_coordinates not in self.image:
            raise ValueError("Neighborhood center must belong to the image")

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the neighborhood patch."""
        if point not in self.manifold:
            return False
        try:
            return self.chart(point) in self.image
        except ValueError:
            return False

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the neighborhood patch."""
        return self.contains(point)

    def center_point(self) -> PointT:
        """Return the distinguished center point of the neighborhood."""
        return self.center

    def inner_radius(self) -> float:
        """Return the largest centered Euclidean ball inside the box image."""
        center_coordinates = self.chart(self.center)
        bounds = _single_interval_bounds(self.image)
        return min(
            min(
                center_coordinates[index] - left,
                right - center_coordinates[index],
            )
            for index, (left, right) in enumerate(bounds)
        )

    def outer_radius(self) -> float:
        """Return the farthest box-corner distance from the center."""
        center_coordinates = self.chart(self.center)
        bounds = _single_interval_bounds(self.image)
        squared = 0.0
        for index, (left, right) in enumerate(bounds):
            squared += max(
                abs(center_coordinates[index] - left),
                abs(right - center_coordinates[index]),
            ) ** 2
        return math.sqrt(squared)

    def diameter(self) -> float:
        """Return an upper bound on neighborhood diameter."""
        return 2.0 * self.outer_radius()

    def subdivide(self) -> tuple["BoxNeighborhood[PointT]", ...]:
        """Split the box image into ``2^dim`` smaller box neighborhoods."""
        bounds = _single_interval_bounds(self.image)
        split_bounds = []
        for left, right in bounds:
            midpoint = (left + right) / 2.0
            split_bounds.append(((left, midpoint), (midpoint, right)))

        neighborhoods = []
        for subbox in itertools.product(*split_bounds):
            image = EuclideanNeighborhood.box(*subbox)
            center_coordinates = FloatPoint(
                [(left + right) / 2.0 for left, right in subbox]
            )
            neighborhoods.append(
                type(self)(
                    self.manifold,
                    self.chart,
                    self.chart.inverse(center_coordinates),
                    image,
                )
            )
        return tuple(neighborhoods)


class ChartedSpace(Generic[PointT]):
    """Concrete space given by a manifold and a distance function."""

    def __init__(
        self,
        manifold: Manifold[PointT],
        distance: Callable[[PointT, PointT], float],
    ) -> None:
        """Initialize the space."""
        self.manifold = manifold
        self._distance = distance

    @property
    def dim(self) -> int:
        """Return the space dimension."""
        return self.manifold.dim

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"ChartedSpace(dim={self.dim})"

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

    def wrap(self, obj):
        """Wrap a charted object into this ambient space."""
        from ..gobject import GeometricObject

        return GeometricObject.from_charted(self, obj)


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
    )


__all__ = [
    "Space",
    "Neighborhood",
    "BoxNeighborhood",
    "ChartedSpace",
    "refine_neighborhoods",
    "centered_real_chart",
]
