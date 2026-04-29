"""Manifold protocols, local charts, atlases, and refinement neighborhoods."""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

from .euclidean import EuclideanNeighborhood, FloatPoint


PointT = TypeVar("PointT")
LocalPointT = TypeVar("LocalPointT")


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
        name: str = "",
    ) -> None:
        """Initialize a local manifold chart."""
        self._forward = forward
        self._inverse = inverse
        self.dim = dim
        self.domain_contains = domain_contains
        self.image = image
        self.name = name

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        image = f", image_dim={self.image.dim}" if self.image else ""
        return f"ManifoldChart(dim={self.dim}{image}{label})"

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
        name: str = "",
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
        self.name = name

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"ChartTransition(dim={self.dim}{label})"

    def __call__(self, coordinates: FloatPoint) -> FloatPoint:
        """Apply the transition map in coordinates."""
        return self.target_chart(self.source_chart.inverse(coordinates))

    def inverse(self, coordinates: FloatPoint) -> FloatPoint:
        """Apply the inverse transition map."""
        return self.source_chart(self.target_chart.inverse(coordinates))


class Atlas(Generic[PointT]):
    """Finite atlas on a manifold."""

    def __init__(
        self,
        manifold: Manifold[PointT],
        *charts: ManifoldChart[PointT],
        name: str = "",
    ) -> None:
        """Initialize an atlas from a manifold and compatible charts."""
        if not charts:
            raise ValueError("Atlas must contain at least one chart")
        for chart in charts:
            if chart.dim != manifold.dim:
                raise ValueError(
                    "Chart dimension does not match manifold dimension: "
                    f"{chart.dim} != {manifold.dim}"
                )
        self.manifold = manifold
        self.charts = tuple(charts)
        self.name = name

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"Atlas(num_charts={len(self.charts)}, dim={self.manifold.dim}{label})"

    def __len__(self) -> int:
        """Return the number of charts."""
        return len(self.charts)

    def __iter__(self):
        """Iterate over the charts."""
        return iter(self.charts)

    def transition(
        self,
        source: int | ManifoldChart[PointT],
        target: int | ManifoldChart[PointT],
        name: str = "",
    ) -> ChartTransition[PointT]:
        """Build the transition map between two atlas charts."""
        source_chart = self.charts[source] if isinstance(source, int) else source
        target_chart = self.charts[target] if isinstance(target, int) else target
        if source_chart not in self.charts or target_chart not in self.charts:
            raise ValueError("Both charts must belong to the atlas")
        return ChartTransition(source_chart, target_chart, name=name)

@runtime_checkable
class Neighborhood(Protocol[PointT]):
    """Protocol for a local manifold neighborhood used in refinement."""

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

    def diameter(self) -> float:
        """Return an upper bound on the neighborhood diameter."""

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the neighborhood."""

    def sample_point(self) -> PointT:
        """Return one point in the neighborhood."""

    def probe_points(self) -> tuple[PointT, ...]:
        """Return finitely many points used for local classification."""

    def subdivide(self) -> tuple["Neighborhood[PointT]", ...]:
        """Return a finite refinement cover by smaller neighborhoods."""


def _single_interval_bounds(image: EuclideanNeighborhood) -> tuple[tuple[float, float], ...]:
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
class ChartNeighborhood(Generic[PointT]):
    """Neighborhood represented by one chart patch and a box image."""

    manifold: Manifold[PointT]
    chart: ManifoldChart[PointT]
    center: PointT
    image: EuclideanNeighborhood
    name: str = ""

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

    def sample_point(self) -> PointT:
        """Return a distinguished point in the neighborhood."""
        return self.center

    def diameter(self) -> float:
        """Return the Euclidean diameter of the chart image box."""
        bounds = _single_interval_bounds(self.image)
        return math.sqrt(sum((right - left) ** 2 for left, right in bounds))

    def probe_points(self) -> tuple[PointT, ...]:
        """Return the center and box corners mapped back to the manifold."""
        bounds = _single_interval_bounds(self.image)
        coordinate_points = [FloatPoint(
            [(left + right) / 2.0 for left, right in bounds]
        )]
        coordinate_points.extend(
            FloatPoint(vertex)
            for vertex in itertools.product(
                *((left, right) for left, right in bounds)
            )
        )
        probes = []
        for coordinates in coordinate_points:
            point = self.chart.inverse(coordinates)
            if point not in probes:
                probes.append(point)
        return tuple(probes)

    def subdivide(self) -> tuple["ChartNeighborhood[PointT]", ...]:
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
                ChartNeighborhood(
                    self.manifold,
                    self.chart,
                    self.chart.inverse(center_coordinates),
                    image,
                    name=self.name,
                )
            )
        return tuple(neighborhoods)


@dataclass(frozen=True)
class NeighborhoodCover(Generic[PointT]):
    """Finite cover by neighborhoods."""

    neighborhoods: tuple[Neighborhood[PointT], ...]
    name: str = ""

    def __post_init__(self) -> None:
        """Require a non-empty homogeneous cover."""
        if not self.neighborhoods:
            raise ValueError("NeighborhoodCover must not be empty")
        manifold = self.neighborhoods[0].manifold
        if any(neighborhood.manifold is not manifold for neighborhood in self.neighborhoods):
            raise ValueError("All neighborhoods in a cover must share a manifold")

    @property
    def manifold(self) -> Manifold[PointT]:
        """Return the common ambient manifold."""
        return self.neighborhoods[0].manifold

    def refine(self) -> "NeighborhoodCover[PointT]":
        """Refine every neighborhood in the cover."""
        refined = tuple(
            child
            for neighborhood in self.neighborhoods
            for child in neighborhood.subdivide()
        )
        return NeighborhoodCover(refined, name=self.name)

    def max_diameter(self) -> float:
        """Return the largest neighborhood diameter in the cover."""
        return max(neighborhood.diameter() for neighborhood in self.neighborhoods)


@dataclass(frozen=True)
class LocalObjectModel(Generic[PointT]):
    """Classification of one object inside one neighborhood."""

    status: str
    neighborhood: Neighborhood[PointT]
    witness_point: PointT | None = None
    local_model: object | None = None

    def __post_init__(self) -> None:
        """Require one of the supported statuses."""
        if self.status not in {"empty", "simple", "complex"}:
            raise ValueError(f"Unsupported local object status: {self.status!r}")

    @property
    def is_empty(self) -> bool:
        """Return whether the object is empty in the neighborhood."""
        return self.status == "empty"

    @property
    def is_simple(self) -> bool:
        """Return whether the object has a simple local model."""
        return self.status == "simple"

    @property
    def is_complex(self) -> bool:
        """Return whether the neighborhood should be refined."""
        return self.status == "complex"


def classify_local_object(
    obj,
    neighborhood: Neighborhood[PointT],
) -> LocalObjectModel[PointT]:
    """Classify an object inside one neighborhood by finite probing."""
    center = neighborhood.sample_point()
    if center in obj:
        return LocalObjectModel(
            "simple",
            neighborhood,
            witness_point=center,
            local_model=obj.local_model_at(center),
        )

    probes = neighborhood.probe_points()
    inside = [point for point in probes if point in obj]
    if not inside:
        return LocalObjectModel("empty", neighborhood)
    return LocalObjectModel(
        "complex",
        neighborhood,
        witness_point=inside[0],
    )


__all__ = [
    "Manifold",
    "ManifoldChart",
    "ChartTransition",
    "Atlas",
    "Neighborhood",
    "ChartNeighborhood",
    "NeighborhoodCover",
    "LocalObjectModel",
    "classify_local_object",
]
