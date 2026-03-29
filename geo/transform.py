"""Transforms between spaces and visualization adapters."""

from __future__ import annotations

from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

from .diffeomorphism import Map
from .euclidean import FloatPoint
from .riemannian import EuclideanMetricSpace, MetricSpace
from .space import Space


SourceT = TypeVar("SourceT")
TargetT = TypeVar("TargetT")
NextT = TypeVar("NextT")


@runtime_checkable
class Transform(Map[SourceT, TargetT], Protocol[SourceT, TargetT]):
    """Protocol for a point transform between ambient spaces."""

    @property
    def source_space(self) -> MetricSpace[SourceT]:
        """Return the source ambient space."""

    @property
    def target_space(self) -> MetricSpace[TargetT]:
        """Return the target ambient space."""


class PointTransform(Generic[SourceT, TargetT]):
    """Concrete validated point transform between ambient spaces."""

    def __init__(
        self,
        source_space: MetricSpace[SourceT],
        target_space: MetricSpace[TargetT],
        forward: Callable[[SourceT], TargetT],
        name: str = "",
    ) -> None:
        """Initialize the transform."""
        self._source_space = source_space
        self._target_space = target_space
        self._forward = forward
        self.name = name

    @property
    def source_space(self) -> MetricSpace[SourceT]:
        """Return the source ambient space."""
        return self._source_space

    @property
    def target_space(self) -> MetricSpace[TargetT]:
        """Return the target ambient space."""
        return self._target_space

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return (
            "PointTransform("
            f"source_dim={self.source_space.dim}, "
            f"target_dim={self.target_space.dim}{label})"
        )

    def __call__(self, point: SourceT) -> TargetT:
        """Apply the transform to one point with membership checks."""
        if point not in self.source_space:
            raise ValueError("Point is outside the transform source space")
        image_point = self._forward(point)
        if image_point not in self.target_space:
            raise ValueError("Image point is outside the transform target space")
        return image_point

    def then(
        self,
        other: Transform[TargetT, NextT],
        name: str = "",
    ) -> "PointTransform[SourceT, NextT]":
        """Return the composition ``other(self(point))``."""
        if self.target_space is not other.source_space:
            raise ValueError(
                "Transform composition requires the same intermediate space"
            )
        chosen_name = name or self.name or getattr(other, "name", "")
        return PointTransform(
            self.source_space,
            other.target_space,
            forward=lambda point: other(self(point)),
            name=chosen_name,
        )


def identity_transform(
    space: MetricSpace[SourceT],
    name: str = "",
) -> PointTransform[SourceT, SourceT]:
    """Return the identity transform on one space."""
    return PointTransform(
        space,
        space,
        forward=lambda point: point,
        name=name or "identity",
    )


def visualization_transform_2d(
    space: Space[SourceT],
    method: str = "default",
    name: str = "",
) -> PointTransform[SourceT, FloatPoint]:
    """Return a transform from intrinsic points into ``R^2``."""
    target_space = EuclideanMetricSpace(2, name=f"{space.space_kind}-vis2d")
    return PointTransform(
        space,
        target_space,
        forward=lambda point: FloatPoint(space.to_2d(point, method=method)),
        name=name or f"{space.space_kind}-to-2d",
    )


def visualization_transform_3d(
    space: Space[SourceT],
    method: str = "default",
    name: str = "",
) -> PointTransform[SourceT, FloatPoint]:
    """Return a transform from intrinsic points into ``R^3``."""
    target_space = EuclideanMetricSpace(3, name=f"{space.space_kind}-vis3d")
    return PointTransform(
        space,
        target_space,
        forward=lambda point: FloatPoint(space.to_3d(point, method=method)),
        name=name or f"{space.space_kind}-to-3d",
    )


__all__ = [
    "Transform",
    "PointTransform",
    "identity_transform",
    "visualization_transform_2d",
    "visualization_transform_3d",
]
