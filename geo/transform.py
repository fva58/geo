"""Transforms between spaces."""

from __future__ import annotations

from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

from .diffeomorphism import Map
from .space.base import Space


SourceT = TypeVar("SourceT")
TargetT = TypeVar("TargetT")
NextT = TypeVar("NextT")


@runtime_checkable
class Transform(Map[SourceT, TargetT], Protocol[SourceT, TargetT]):
    """Protocol for a point transform between ambient spaces."""

    @property
    def source_space(self) -> Space[SourceT]:
        """Return the source ambient space."""

    @property
    def target_space(self) -> Space[TargetT]:
        """Return the target ambient space."""


class PointTransform(Generic[SourceT, TargetT]):
    """Concrete validated point transform between ambient spaces."""

    def __init__(
        self,
        source_space: Space[SourceT],
        target_space: Space[TargetT],
        forward: Callable[[SourceT], TargetT],
    ) -> None:
        """Initialize the transform."""
        self._source_space = source_space
        self._target_space = target_space
        self._forward = forward

    @property
    def source_space(self) -> Space[SourceT]:
        """Return the source ambient space."""
        return self._source_space

    @property
    def target_space(self) -> Space[TargetT]:
        """Return the target ambient space."""
        return self._target_space

    def __repr__(self) -> str:
        """Return a debug representation."""
        return (
            "PointTransform("
            f"source_dim={self.source_space.dim}, "
            f"target_dim={self.target_space.dim})"
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
    ) -> "PointTransform[SourceT, NextT]":
        """Return the composition ``other(self(point))``."""
        if self.target_space is not other.source_space:
            raise ValueError(
                "Transform composition requires the same intermediate space"
            )
        return PointTransform(
            self.source_space,
            other.target_space,
            forward=lambda point: other(self(point)),
        )


def identity_transform(
    space: Space[SourceT],
) -> PointTransform[SourceT, SourceT]:
    """Return the identity transform on one space."""
    return PointTransform(
        space,
        space,
        forward=lambda point: point,
    )


__all__ = [
    "Transform",
    "PointTransform",
    "identity_transform",
]
