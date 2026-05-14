"""Base abstractions for spaces."""

from __future__ import annotations

import abc
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
from collections.abc import Sequence

from ..euclidean import EuclideanNeighborhood, Point


PointT = TypeVar("PointT")
LocalPointT = TypeVar("LocalPointT")


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


class Space(abc.ABC, Generic[PointT]):
    """Abstract base class for spaces with distance and neighborhood covers."""

    def __init__(
        self,
        space: "Space[PointT] | None" = None,
        distance: Callable[[PointT, PointT], float] | None = None,
    ) -> None:
        """Initialize the space with an optional underlying space and distance."""
        if space is not None:
            self.space = space
        if distance is not None:
            self._distance = distance

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Return the space dimension."""

    @property
    @abc.abstractmethod
    def point_type(self) -> type:
        """Return the type of points in this space."""

    def __repr__(self) -> str:
        return f"Space(dim={self.dim})"

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the underlying manifold."""
        return point in self.space

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

    @abc.abstractmethod
    def full(self, radius: float):
        """Return a full cover of the space by neighborhoods."""

    @abc.abstractmethod
    def refine(self, neighborhoods, factor: int = 2):
        """Return a covering refinement with smaller diameters."""

    def wrap(self, obj):
        """Wrap a charted object into this ambient space."""
        from ..gobject import GeometricObject

        return GeometricObject.from_charted(self, obj)


class Neighborhood(abc.ABC, Generic[PointT]):
    """Abstract base class for local neighborhoods in a space."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        """Structural subtyping: accept any class with the expected methods."""
        if cls is Neighborhood:
            for attr in (
                "space", "chart", "center", "image",
                "inner_radius", "outer_radius", "diameter",
                "contains", "center_point", "subdivide",
            ):
                if not any(attr in vars(klass) for klass in subclass.__mro__):
                    return NotImplemented
            return True
        return NotImplemented


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
class BoxNeighborhood(Neighborhood[PointT]):
    """Neighborhood represented by one chart patch and a box image."""

    space: "Space[PointT]"
    chart: ManifoldChart[PointT]
    center: PointT
    image: EuclideanNeighborhood

    def __post_init__(self) -> None:
        """Validate center and image dimensions."""
        if self.chart.dim != self.space.dim:
            raise ValueError("Chart dimension must match manifold dimension")
        if self.image.dim != self.chart.dim:
            raise ValueError("Image dimension must match chart dimension")
        center_coordinates = self.chart(self.center)
        if center_coordinates not in self.image:
            raise ValueError("Neighborhood center must belong to the image")

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the neighborhood patch."""
        if point not in self.space:
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
            center_coordinates = Point(
                [(left + right) / 2.0 for left, right in subbox]
            )
            neighborhoods.append(
                type(self)(
                    self.space,
                    self.chart,
                    self.chart.inverse(center_coordinates),
                    image,
                )
            )
        return tuple(neighborhoods)


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

    return ManifoldChart(
        lambda point: Point(float(point) - center),
        lambda coordinates: center + coordinates[0],
        dim=1,
        domain_contains=domain_contains,
        image=EuclideanNeighborhood.whole(1),
    )


__all__ = [
    "ManifoldChart",
    "Space",
    "Neighborhood",
    "BoxNeighborhood",
    "refine_neighborhoods",
    "centered_real_chart",
]
