"""Manifold protocols, local charts, atlases, and local refinement data."""

from __future__ import annotations

from dataclasses import dataclass
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

    def max_outer_radius(self) -> float:
        """Return the largest outer radius in the cover."""
        return max(neighborhood.outer_radius() for neighborhood in self.neighborhoods)


def refine_neighborhoods(
    neighborhoods: tuple[Neighborhood[PointT], ...],
    factor: int = 2,
) -> tuple[Neighborhood[PointT], ...]:
    """Compatibility wrapper for neighborhood refinement."""
    from .space.base import refine_neighborhoods as refine

    return refine(neighborhoods, factor=factor)


@dataclass(frozen=True)
class LocalObjectModel(Generic[PointT]):
    """Classification of one object inside one neighborhood."""

    status: str
    neighborhood: Neighborhood[PointT]
    witness_point: PointT | None = None
    local_model: object | None = None

    def __post_init__(self) -> None:
        """Require one of the supported statuses."""
        if self.status not in {"empty", "cone", "complex"}:
            raise ValueError(f"Unsupported local object status: {self.status!r}")

    @property
    def is_empty(self) -> bool:
        """Return whether the object is empty in the neighborhood."""
        return self.status == "empty"

    @property
    def is_cone(self) -> bool:
        """Return whether the object is conic in the neighborhood."""
        return self.status == "cone"

    @property
    def is_complex(self) -> bool:
        """Return whether the neighborhood should be refined."""
        return self.status == "complex"


@dataclass(frozen=True)
class NeighborhoodMarking(Generic[PointT]):
    """Object marking on a finite neighborhood family."""

    entries: tuple[LocalObjectModel[PointT], ...]
    name: str = ""

    def __iter__(self):
        """Iterate over marked neighborhoods."""
        return iter(self.entries)

    def __len__(self) -> int:
        """Return the number of marked neighborhoods."""
        return len(self.entries)

    @property
    def empty(self) -> tuple[LocalObjectModel[PointT], ...]:
        """Return neighborhoods marked empty."""
        return tuple(entry for entry in self.entries if entry.is_empty)

    @property
    def cone(self) -> tuple[LocalObjectModel[PointT], ...]:
        """Return neighborhoods marked cone."""
        return tuple(entry for entry in self.entries if entry.is_cone)

    @property
    def complex(self) -> tuple[LocalObjectModel[PointT], ...]:
        """Return neighborhoods marked complex."""
        return tuple(entry for entry in self.entries if entry.is_complex)


def classify_local_object(
    obj,
    neighborhood: Neighborhood[PointT],
) -> LocalObjectModel[PointT]:
    """Classify an object inside one neighborhood by finite probing."""
    center = neighborhood.center_point()
    if center in obj:
        return LocalObjectModel(
            "cone",
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


def classify_neighborhoods(
    obj,
    neighborhoods,
    name: str = "",
) -> NeighborhoodMarking[PointT]:
    """Return the object marking on a finite neighborhood family."""
    return NeighborhoodMarking(
        tuple(
            classify_local_object(obj, neighborhood)
            for neighborhood in neighborhoods
        ),
        name=name,
    )


__all__ = [
    "Manifold",
    "ManifoldChart",
    "ChartTransition",
    "Atlas",
    "NeighborhoodCover",
    "refine_neighborhoods",
    "LocalObjectModel",
    "NeighborhoodMarking",
    "classify_local_object",
    "classify_neighborhoods",
]
