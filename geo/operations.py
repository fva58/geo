"""Computational operations and invariants for geometric objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .euclidean import EuclideanNeighborhood, FloatPoint
from .manifold import (
    ChartNeighborhood,
    LocalObjectModel,
    Neighborhood,
    NeighborhoodCover,
    classify_local_object,
)


PointT = TypeVar("PointT")


@runtime_checkable
class SampledMetricObjectProtocol(Protocol[PointT]):
    """Protocol for metric objects with sampling support."""

    @property
    def space(self):
        """Return the ambient metric space."""

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the object."""

    def sample_points(self, resolution: int = 24):
        """Return sample points on the object."""

    def local_model_at(self, point: PointT):
        """Return a local model of the object at one point."""


def sample_points(
    obj: SampledMetricObjectProtocol[PointT],
    resolution: int = 64,
) -> tuple[PointT, ...]:
    """Return validated sample points for an object."""
    samples = tuple(obj.sample_points(resolution=resolution))
    if not samples:
        raise ValueError("Object sampling returned no points")
    return samples


def closest_sample_to_point(
    obj: SampledMetricObjectProtocol[PointT],
    point: PointT,
    resolution: int = 64,
) -> tuple[PointT, float]:
    """Return the closest sampled object point to one ambient point."""
    if point not in obj.space:
        raise ValueError("Point must belong to the ambient space")
    samples = sample_points(obj, resolution=resolution)
    closest = min(
        samples,
        key=lambda sample: obj.space.distance(sample, point),
    )
    return closest, obj.space.distance(closest, point)


def sampled_distance_to_point(
    obj: SampledMetricObjectProtocol[PointT],
    point: PointT,
    resolution: int = 64,
) -> float:
    """Return a sample-based approximation to distance from a point."""
    if point in obj:
        return 0.0
    _, distance = closest_sample_to_point(obj, point, resolution=resolution)
    return distance


def closest_sample_pair(
    left: SampledMetricObjectProtocol[PointT],
    right: SampledMetricObjectProtocol[PointT],
    resolution: int = 64,
) -> tuple[PointT, PointT, float]:
    """Return the closest sampled pair between two objects."""
    if left.space is not right.space:
        raise ValueError("Objects must share the same ambient space")

    left_samples = sample_points(left, resolution=resolution)
    right_samples = sample_points(right, resolution=resolution)

    best_left = left_samples[0]
    best_right = right_samples[0]
    best_distance = left.space.distance(best_left, best_right)

    for left_point in left_samples:
        if left_point in right:
            return left_point, left_point, 0.0
        for right_point in right_samples:
            current = left.space.distance(left_point, right_point)
            if current < best_distance:
                best_left = left_point
                best_right = right_point
                best_distance = current
    for right_point in right_samples:
        if right_point in left:
            return right_point, right_point, 0.0
    return best_left, best_right, best_distance


def sampled_distance(
    left: SampledMetricObjectProtocol[PointT],
    right: SampledMetricObjectProtocol[PointT],
    resolution: int = 64,
) -> float:
    """Return a sample-based approximation to distance between objects."""
    _, _, distance = closest_sample_pair(left, right, resolution=resolution)
    return distance


def sampled_diameter(
    obj: SampledMetricObjectProtocol[PointT],
    resolution: int = 64,
) -> float:
    """Return a sample-based approximation to the object diameter."""
    samples = sample_points(obj, resolution=resolution)
    diameter = 0.0
    for left_index, left_point in enumerate(samples):
        for right_point in samples[left_index + 1:]:
            diameter = max(diameter, obj.space.distance(left_point, right_point))
    return diameter


def sampled_hausdorff_distance(
    left: SampledMetricObjectProtocol[PointT],
    right: SampledMetricObjectProtocol[PointT],
    resolution: int = 64,
) -> float:
    """Return a sample-based approximation to the Hausdorff distance."""
    if left.space is not right.space:
        raise ValueError("Objects must share the same ambient space")

    left_samples = sample_points(left, resolution=resolution)
    right_samples = sample_points(right, resolution=resolution)

    def directed_distance(source_samples, target) -> float:
        return max(
            sampled_distance_to_point(target, source_point, resolution=resolution)
            for source_point in source_samples
        )

    return max(
        directed_distance(left_samples, right),
        directed_distance(right_samples, left),
    )


def local_chart_cover_from_samples(
    obj: SampledMetricObjectProtocol[PointT],
    radius: float,
    resolution: int = 16,
    name: str = "",
) -> NeighborhoodCover[PointT]:
    """Build an initial cover from sampled object points and local charts."""
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("Neighborhood radius must be positive")
    samples = sample_points(obj, resolution=resolution)
    neighborhoods = []
    for point in samples:
        if hasattr(obj.space, "neighborhood_at"):
            neighborhoods.append(
                obj.space.neighborhood_at(point, radius, name=name)
            )
            continue
        chart = obj.local_model_at(point).chart
        image = EuclideanNeighborhood.box(*(((-radius, radius),) * obj.space.dim))
        neighborhoods.append(
            ChartNeighborhood(
                obj.space,
                chart,
                point,
                image,
                name=name,
            )
        )
    return NeighborhoodCover(
        tuple(neighborhoods),
        name=name or getattr(obj, "name", ""),
    )


@dataclass(frozen=True)
class RefinedObjectCover(Generic[PointT]):
    """Refinement state for one object over a neighborhood cover."""

    obj: SampledMetricObjectProtocol[PointT]
    simple_parts: tuple[LocalObjectModel[PointT], ...]
    complex_parts: tuple[LocalObjectModel[PointT], ...]
    empty_parts: tuple[LocalObjectModel[PointT], ...]

    @property
    def active_parts(self) -> tuple[LocalObjectModel[PointT], ...]:
        """Return the non-empty parts of the current refinement."""
        return self.simple_parts + self.complex_parts

    def max_diameter(self) -> float:
        """Return the largest diameter among active parts."""
        if not self.active_parts:
            return 0.0
        return max(part.neighborhood.diameter() for part in self.active_parts)


def classify_cover(
    obj: SampledMetricObjectProtocol[PointT],
    cover: NeighborhoodCover[PointT],
) -> RefinedObjectCover[PointT]:
    """Classify one object over all neighborhoods in a cover."""
    simple = []
    complex_parts = []
    empty = []
    for neighborhood in cover.neighborhoods:
        local = classify_local_object(obj, neighborhood)
        if local.is_simple:
            simple.append(local)
        elif local.is_complex:
            complex_parts.append(local)
        else:
            empty.append(local)
    return RefinedObjectCover(
        obj,
        tuple(simple),
        tuple(complex_parts),
        tuple(empty),
    )


def refine_until(
    obj: SampledMetricObjectProtocol[PointT],
    cover: NeighborhoodCover[PointT],
    *,
    max_diameter: float,
    max_steps: int = 8,
) -> RefinedObjectCover[PointT]:
    """Refine a cover until non-empty parts are small enough or steps end."""
    if max_diameter <= 0.0:
        raise ValueError("max_diameter must be positive")
    current_cover = cover
    current = classify_cover(obj, current_cover)
    for _ in range(max_steps):
        if not current.complex_parts and current.max_diameter() <= max_diameter:
            return current
        to_keep = [part.neighborhood for part in current.simple_parts]
        to_refine = [part.neighborhood for part in current.complex_parts]
        if current.max_diameter() > max_diameter:
            to_refine.extend(
                part.neighborhood
                for part in current.simple_parts
                if part.neighborhood.diameter() > max_diameter
            )
            to_keep = [
                part.neighborhood
                for part in current.simple_parts
                if part.neighborhood.diameter() <= max_diameter
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


@dataclass(frozen=True)
class AdaptiveDistanceResult(Generic[PointT]):
    """Adaptive distance estimate between two objects."""

    lower_bound: float
    upper_bound: float
    left_point: PointT | None
    right_point: PointT | None
    left_cover: RefinedObjectCover[PointT]
    right_cover: RefinedObjectCover[PointT]

    @property
    def estimate(self) -> float:
        """Return the midpoint estimate between the current bounds."""
        return (self.lower_bound + self.upper_bound) / 2.0

    @property
    def error(self) -> float:
        """Return the current uncertainty width."""
        return self.upper_bound - self.lower_bound


def _distance_lower_bound(left_part, right_part) -> float:
    """Return a conservative distance lower bound between neighborhoods."""
    space = left_part.neighborhood.manifold
    center_distance = space.distance(
        left_part.neighborhood.center,
        right_part.neighborhood.center,
    )
    return max(
        0.0,
        center_distance -
        left_part.neighborhood.diameter() -
        right_part.neighborhood.diameter(),
    )


def adaptive_distance(
    left: SampledMetricObjectProtocol[PointT],
    right: SampledMetricObjectProtocol[PointT],
    *,
    neighborhood_radius: float,
    sample_resolution: int = 16,
    target_diameter: float = 0.25,
    max_refinement_steps: int = 8,
) -> AdaptiveDistanceResult[PointT]:
    """Approximate distance via adaptive refinement of local neighborhoods."""
    if left.space is not right.space:
        raise ValueError("Objects must share the same ambient space")

    left_cover = refine_until(
        left,
        local_chart_cover_from_samples(
            left,
            neighborhood_radius,
            resolution=sample_resolution,
            name="left-cover",
        ),
        max_diameter=target_diameter,
        max_steps=max_refinement_steps,
    )
    right_cover = refine_until(
        right,
        local_chart_cover_from_samples(
            right,
            neighborhood_radius,
            resolution=sample_resolution,
            name="right-cover",
        ),
        max_diameter=target_diameter,
        max_steps=max_refinement_steps,
    )

    if not left_cover.active_parts or not right_cover.active_parts:
        raise ValueError("Adaptive distance requires non-empty object covers")

    upper_left, upper_right, upper_bound = closest_sample_pair(
        left,
        right,
        resolution=max(8, sample_resolution),
    )
    lower_bound = math.inf
    for left_part in left_cover.active_parts:
        for right_part in right_cover.active_parts:
            lower_bound = min(
                lower_bound,
                _distance_lower_bound(left_part, right_part),
            )

    return AdaptiveDistanceResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        left_point=upper_left,
        right_point=upper_right,
        left_cover=left_cover,
        right_cover=right_cover,
    )


__all__ = [
    "SampledMetricObjectProtocol",
    "sample_points",
    "closest_sample_to_point",
    "sampled_distance_to_point",
    "closest_sample_pair",
    "sampled_distance",
    "sampled_diameter",
    "sampled_hausdorff_distance",
    "local_chart_cover_from_samples",
    "RefinedObjectCover",
    "classify_cover",
    "refine_until",
    "AdaptiveDistanceResult",
    "adaptive_distance",
]
