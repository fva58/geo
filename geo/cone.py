"""Cones and local cone models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from .circle import Point as CirclePoint, Set as CircleSet
from .euclidean import EuclideanNeighborhood, FloatPoint, FloatVector
from .manifold import ManifoldChart


PointT = TypeVar("PointT")


def _signed_circle_offset(center: CirclePoint, point: CirclePoint) -> float:
    """Return the signed angular offset from ``center`` to ``point``."""
    offset = float(point) - float(center)
    if offset > math.pi:
        offset -= 2.0 * math.pi
    elif offset <= -math.pi:
        offset += 2.0 * math.pi
    return offset


def _cross_2d(left: FloatVector, right: FloatVector) -> float:
    """Return the scalar two-dimensional cross product."""
    return left[0] * right[1] - left[1] * right[0]


def _isclose(value: float, target: float = 0.0) -> bool:
    """Return a small-tolerance comparison for Euclidean predicates."""
    return math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12)


def _point_array(point: FloatPoint) -> np.ndarray:
    """Return a point as a NumPy float array."""
    return np.asarray(FloatPoint(point), dtype=float)


def _vector_array(vector: FloatVector) -> np.ndarray:
    """Return a vector as a NumPy float array."""
    return np.asarray(FloatVector(vector), dtype=float)


def _coerce_nonzero_normal(normal: FloatVector) -> FloatVector:
    """Normalize a hyperplane normal input."""
    normal = FloatVector(normal)
    if normal.norm() == 0.0:
        raise ValueError("Normal must be non-zero")
    return normal


def _coerce_affine_matrix(
    vectors: Sequence[Sequence[float]],
    dim: int,
) -> np.ndarray:
    """Return an invertible matrix whose columns are the given vectors."""
    columns = [_vector_array(FloatVector(vector)) for vector in vectors]
    if len(columns) != dim:
        raise ValueError(f"Need {dim} spanning vectors, got {len(columns)}")
    matrix = np.column_stack(columns)
    if matrix.shape != (dim, dim):
        raise ValueError(
            f"Matrix dimension mismatch: {matrix.shape} != ({dim}, {dim})"
        )
    if _isclose(float(np.linalg.det(matrix))):
        raise ValueError("Spanning vectors must be linearly independent")
    return matrix


def _active_box_constraints(local_point: np.ndarray) -> list[tuple[int, float]]:
    """Return active coordinate constraints for a unit box point."""
    active = []
    for index, value in enumerate(local_point):
        if _isclose(abs(value), 1.0):
            sign = 1.0 if value >= 0.0 else -1.0
            active.append((index, sign))
    return active


@runtime_checkable
class Cone(Protocol):
    """Protocol for a cone in Euclidean coordinates."""

    @property
    def dim(self) -> int:
        """Return the cone dimension."""

    @property
    def apex(self) -> FloatPoint:
        """Return the cone apex."""

    def contains(self, point: FloatPoint) -> bool:
        """Check whether a coordinate point belongs to the cone."""

    def __contains__(self, point: FloatPoint) -> bool:
        """Check whether a coordinate point belongs to the cone."""


@runtime_checkable
class SphereObject(Protocol):
    """Protocol for an object on the unit sphere."""

    @property
    def dim(self) -> int:
        """Return the ambient dimension."""

    def contains(self, direction: FloatVector) -> bool:
        """Check whether a unit direction belongs to the sphere object."""

    def __contains__(self, direction: FloatVector) -> bool:
        """Check whether a unit direction belongs to the sphere object."""


def point_cone(dim: int) -> "EuclideanCone":
    """Return the zero-dimensional cone supported at the apex only."""
    origin = FloatPoint.origin(dim)
    return EuclideanCone(
        dim,
        contains=lambda point: FloatPoint(point) == origin,
        apex=origin,
        neighborhood=EuclideanNeighborhood.whole(dim),
    )


def positive_half_line_cone() -> "EuclideanCone":
    """Return the standard half-line ``[0, +inf)``."""
    return EuclideanCone(
        1,
        contains=lambda point: point[0] >= 0.0,
        apex=FloatPoint.origin(1),
        neighborhood=EuclideanNeighborhood.whole(1),
    )


def negative_half_line_cone() -> "EuclideanCone":
    """Return the standard half-line ``(-inf, 0]``."""
    return EuclideanCone(
        1,
        contains=lambda point: point[0] <= 0.0,
        apex=FloatPoint.origin(1),
        neighborhood=EuclideanNeighborhood.whole(1),
    )


def half_space_cone(
    normal: FloatVector,
    *,
    reverse: bool = False,
) -> "EuclideanCone":
    """Return a cone defined by one linear inequality."""
    orientation = -1.0 if reverse else 1.0
    dim = normal.dim
    return EuclideanCone(
        dim,
        contains=lambda coordinates: (
            orientation * FloatVector(coordinates).dot(normal) >= 0.0
        ),
        apex=FloatPoint.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
    )


def hyperplane_cone(normal: FloatVector) -> "EuclideanCone":
    """Return a cone given by one linear equality."""
    dim = normal.dim
    return EuclideanCone(
        dim,
        contains=lambda coordinates: _isclose(
            FloatVector(coordinates).dot(normal)
        ),
        apex=FloatPoint.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
    )


class EuclideanCone:
    """Concrete cone in Euclidean coordinates."""

    def __init__(
        self,
        dim: int,
        contains: Callable[[FloatPoint], bool],
        apex: FloatPoint | None = None,
        neighborhood: EuclideanNeighborhood | None = None,
    ) -> None:
        self._dim = dim
        self._contains = contains
        self._apex = FloatPoint.origin(dim) if apex is None else FloatPoint(apex)
        if self._apex.dim != dim:
            raise ValueError(f"Apex dimension mismatch: {self._apex.dim} != {dim}")
        if neighborhood is not None and neighborhood.dim != dim:
            raise ValueError(
                f"Neighborhood dimension mismatch: {neighborhood.dim} != {dim}"
            )
        self.neighborhood = neighborhood

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def apex(self) -> FloatPoint:
        return self._apex

    def __repr__(self) -> str:
        return f"EuclideanCone(dim={self.dim}, apex={self.apex.to_tuple()})"

    def contains(self, point: FloatPoint) -> bool:
        point = FloatPoint(point)
        if point.dim != self.dim:
            return False
        if self.neighborhood is not None and point not in self.neighborhood:
            return False
        return self._contains(point)

    def __contains__(self, point: FloatPoint) -> bool:
        return self.contains(point)

    @classmethod
    def whole(cls, dim: int) -> "EuclideanCone":
        return cls(
            dim,
            contains=lambda point: True,
            apex=FloatPoint.origin(dim),
            neighborhood=EuclideanNeighborhood.whole(dim),
        )


class RadialCone(EuclideanCone):
    """Cone defined by an admissible set of directions."""

    def __init__(
        self,
        dim: int,
        contains_direction: Callable[[FloatVector], bool],
        apex: FloatPoint | None = None,
        neighborhood: EuclideanNeighborhood | None = None,
    ) -> None:
        self._contains_direction = contains_direction
        chosen_apex = FloatPoint.origin(dim) if apex is None else FloatPoint(apex)

        def contains(point: FloatPoint) -> bool:
            point = FloatPoint(point)
            displacement = point - chosen_apex
            if displacement.norm() == 0.0:
                return True
            direction = displacement / displacement.norm()
            return self._contains_direction(direction)

        super().__init__(
            dim,
            contains=contains,
            apex=chosen_apex,
            neighborhood=neighborhood,
        )

    @classmethod
    def whole(cls, dim: int) -> "RadialCone":
        return cls(
            dim,
            contains_direction=lambda direction: True,
            apex=FloatPoint.origin(dim),
            neighborhood=EuclideanNeighborhood.whole(dim),
        )


class DirectionSetSphereObject:
    """Concrete object on the unit sphere defined by a direction predicate."""

    def __init__(
        self,
        dim: int,
        contains: Callable[[FloatVector], bool],
    ) -> None:
        self._dim = dim
        self._contains = contains

    @property
    def dim(self) -> int:
        return self._dim

    def __repr__(self) -> str:
        return f"DirectionSetSphereObject(dim={self.dim})"

    def contains(self, direction: FloatVector) -> bool:
        direction = FloatVector(direction)
        if direction.dim != self.dim or direction.norm() == 0.0:
            return False
        normalized = direction / direction.norm()
        return self._contains(normalized)

    def __contains__(self, direction: FloatVector) -> bool:
        return self.contains(direction)


class CircleSphereObject:
    """Sphere object in dimension 2 defined by a circle subset."""

    def __init__(self, circle_set: CircleSet) -> None:
        self.circle_set = CircleSet(circle_set)

    @property
    def dim(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"CircleSphereObject({self.circle_set!r})"

    def contains(self, direction: FloatVector) -> bool:
        direction = FloatVector(direction)
        if direction.dim != 2 or direction.norm() == 0.0:
            return False
        point = CirclePoint.from_cartesian(direction[0], direction[1])
        return point in self.circle_set

    def __contains__(self, direction: FloatVector) -> bool:
        return self.contains(direction)


class SphericalCone(RadialCone):
    """Cone induced by an explicit object on the unit sphere."""

    def __init__(
        self,
        sphere_object: SphereObject,
        apex: FloatPoint | None = None,
        neighborhood: EuclideanNeighborhood | None = None,
    ) -> None:
        self.sphere_object = sphere_object
        super().__init__(
            sphere_object.dim,
            contains_direction=sphere_object.contains,
            apex=apex,
            neighborhood=neighborhood,
        )


@dataclass(frozen=True)
class LocalConeModel(Generic[PointT]):
    """Local model of a geometric object by a cone in chart coordinates."""

    chart: ManifoldChart[PointT]
    cone: Cone

    def __post_init__(self) -> None:
        if self.chart.dim != self.cone.dim:
            raise ValueError(
                "Chart/cone dimension mismatch: "
                f"{self.chart.dim} != {self.cone.dim}"
            )


__all__ = [
    "Cone",
    "SphereObject",
    "DirectionSetSphereObject",
    "CircleSphereObject",
    "EuclideanCone",
    "RadialCone",
    "SphericalCone",
    "LocalConeModel",
    "point_cone",
    "positive_half_line_cone",
    "negative_half_line_cone",
    "half_space_cone",
    "hyperplane_cone",
    "_signed_circle_offset",
    "_cross_2d",
    "_isclose",
    "_point_array",
    "_vector_array",
    "_coerce_nonzero_normal",
    "_coerce_affine_matrix",
    "_active_box_constraints",
]
