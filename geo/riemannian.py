"""Riemannian spaces and geometric objects inside them."""

from __future__ import annotations

import math
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from .euclidean import FloatPoint, FloatVector
from .floatcircle import FloatCirclePoint
from .geometric import (
    Circle,
    CirclePointObject,
    CircleSetObject,
    ChartedGeometricObject,
    EuclideanPointObject,
    EuclideanSpace,
    HalfPlane,
    PlanarAngle,
    RealLine,
    RealPointObject,
    RealSetObject,
    WholePlane,
)
from .manifold import Manifold


PointT = TypeVar("PointT")
MetricTensor = tuple[tuple[float, ...], ...]


def _whole_neighborhood(dim: int):
    """Return the default unconstrained neighborhood for a cone."""
    from .euclidean import EuclideanNeighborhood

    return EuclideanNeighborhood.whole(dim)


def _coerce_metric_tensor(
    matrix: MetricTensor,
    dim: int,
) -> MetricTensor:
    """Normalize a metric tensor into a square float matrix."""
    array = np.asarray(matrix, dtype=float)
    if array.shape != (dim, dim):
        raise ValueError(
            "Metric tensor dimension mismatch: "
            f"{array.shape} != ({dim}, {dim})"
        )
    if not np.allclose(array, array.T):
        raise ValueError("Metric tensor must be symmetric")
    return tuple(
        tuple(float(value) for value in row)
        for row in array
    )


@runtime_checkable
class RiemannianSpace(Manifold[PointT], Protocol[PointT]):
    """Protocol for a manifold equipped with a Riemannian metric."""

    def metric_tensor(self, point: PointT) -> MetricTensor:
        """Return the metric tensor in local coordinates at a point."""

    def inner_product(
        self,
        point: PointT,
        left: FloatVector,
        right: FloatVector,
    ) -> float:
        """Return the metric inner product of two tangent vectors."""

    def norm(
        self,
        point: PointT,
        vector: FloatVector,
    ) -> float:
        """Return the metric norm of a tangent vector."""


class ChartedRiemannianSpace(Generic[PointT]):
    """Concrete Riemannian space given by a manifold and a metric tensor."""

    def __init__(
        self,
        manifold: Manifold[PointT],
        metric_tensor: Callable[[PointT], MetricTensor],
        name: str = "",
    ) -> None:
        """Initialize the Riemannian space."""
        self.manifold = manifold
        self._metric_tensor = metric_tensor
        self.name = name

    @property
    def dim(self) -> int:
        """Return the space dimension."""
        return self.manifold.dim

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"ChartedRiemannianSpace(dim={self.dim}{label})"

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the underlying manifold."""
        return point in self.manifold

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the underlying manifold."""
        return self.contains(point)

    def metric_tensor(self, point: PointT) -> MetricTensor:
        """Return the metric tensor at the given point."""
        if point not in self:
            raise ValueError("Point is outside the Riemannian space")
        return _coerce_metric_tensor(self._metric_tensor(point), self.dim)

    def inner_product(
        self,
        point: PointT,
        left: FloatVector,
        right: FloatVector,
    ) -> float:
        """Return the metric inner product at a point."""
        left = FloatVector(left)
        right = FloatVector(right)
        if left.dim != self.dim or right.dim != self.dim:
            raise ValueError(
                "Tangent vector dimension mismatch: "
                f"{left.dim}, {right.dim} != {self.dim}"
            )
        metric = np.asarray(self.metric_tensor(point), dtype=float)
        result = np.asarray(left, dtype=float) @ metric @ np.asarray(
            right,
            dtype=float,
        )
        return float(result)

    def norm(
        self,
        point: PointT,
        vector: FloatVector,
    ) -> float:
        """Return the metric norm at a point."""
        squared_norm = self.inner_product(point, vector, vector)
        if squared_norm < -1e-12:
            raise ValueError("Metric tensor is not positive semidefinite")
        return math.sqrt(max(0.0, squared_norm))


class RiemannianGeometricObject(ChartedGeometricObject[PointT]):
    """Geometric object with an explicitly Riemannian ambient space."""

    def __init__(
        self,
        space: RiemannianSpace[PointT],
        contains: Callable[[PointT], bool],
        local_model,
        name: str = "",
    ) -> None:
        """Initialize the Riemannian geometric object."""
        self.space = space
        super().__init__(space, contains, local_model, name=name)

    @classmethod
    def from_charted(
        cls,
        space: RiemannianSpace[PointT],
        obj: ChartedGeometricObject[PointT],
        name: str = "",
    ) -> "RiemannianGeometricObject[PointT]":
        """Wrap an existing charted object in a Riemannian ambient space."""
        chosen_name = name or getattr(obj, "name", "")
        return cls(
            space,
            contains=obj.contains,
            local_model=obj.local_model_at,
            name=chosen_name,
        )

    def _require_same_space(
        self,
        other: "RiemannianGeometricObject[PointT]",
    ) -> None:
        """Require two objects to live in the same ambient space instance."""
        if self.space is not other.space:
            raise ValueError(
                "Set-theoretic operations require the same Riemannian space"
            )

    @staticmethod
    def _combine_local_models(
        left_model,
        right_model,
        operation: Callable[[bool, bool], bool],
        name: str,
    ):
        """Combine two local cone models in a shared coordinate chart."""
        from .geometric import EuclideanCone, LocalConeModel

        if left_model.chart.dim != right_model.chart.dim:
            raise ValueError("Local model dimensions do not match")
        dim = left_model.chart.dim
        cone = EuclideanCone(
            dim,
            contains=lambda point: operation(
                left_model.cone.contains(point),
                right_model.cone.contains(point),
            ),
            neighborhood=_whole_neighborhood(dim),
            name=name,
        )
        return LocalConeModel(left_model.chart, cone)

    def union(
        self,
        other: "RiemannianGeometricObject[PointT]",
        name: str = "",
    ) -> "RiemannianGeometricObject[PointT]":
        """Return the set-theoretic union of two objects."""
        self._require_same_space(other)

        def local_model(point: PointT):
            if point in self and point in other:
                return self._combine_local_models(
                    self.local_model_at(point),
                    other.local_model_at(point),
                    lambda left, right: left or right,
                    "union",
                )
            if point in self:
                return self.local_model_at(point)
            return other.local_model_at(point)

        chosen_name = name or f"({self.name})|({other.name})"
        return RiemannianGeometricObject(
            self.space,
            contains=lambda point: point in self or point in other,
            local_model=local_model,
            name=chosen_name,
        )

    def intersection(
        self,
        other: "RiemannianGeometricObject[PointT]",
        name: str = "",
    ) -> "RiemannianGeometricObject[PointT]":
        """Return the set-theoretic intersection of two objects."""
        self._require_same_space(other)

        def local_model(point: PointT):
            return self._combine_local_models(
                self.local_model_at(point),
                other.local_model_at(point),
                lambda left, right: left and right,
                "intersection",
            )

        chosen_name = name or f"({self.name})&({other.name})"
        return RiemannianGeometricObject(
            self.space,
            contains=lambda point: point in self and point in other,
            local_model=local_model,
            name=chosen_name,
        )

    def difference(
        self,
        other: "RiemannianGeometricObject[PointT]",
        name: str = "",
    ) -> "RiemannianGeometricObject[PointT]":
        """Return the set-theoretic difference ``self \\ other``."""
        self._require_same_space(other)

        def local_model(point: PointT):
            if point in other:
                return self._combine_local_models(
                    self.local_model_at(point),
                    other.local_model_at(point),
                    lambda left, right: left and not right,
                    "difference",
                )
            return self.local_model_at(point)

        chosen_name = name or f"({self.name})-({other.name})"
        return RiemannianGeometricObject(
            self.space,
            contains=lambda point: point in self and point not in other,
            local_model=local_model,
            name=chosen_name,
        )

    def symmetric_difference(
        self,
        other: "RiemannianGeometricObject[PointT]",
        name: str = "",
    ) -> "RiemannianGeometricObject[PointT]":
        """Return the symmetric difference of two objects."""
        self._require_same_space(other)

        def local_model(point: PointT):
            if point in self and point not in other:
                return self.local_model_at(point)
            if point in other and point not in self:
                return other.local_model_at(point)
            return self._combine_local_models(
                self.local_model_at(point),
                other.local_model_at(point),
                lambda left, right: left ^ right,
                "symmetric-difference",
            )

        chosen_name = name or f"({self.name})^({other.name})"
        return RiemannianGeometricObject(
            self.space,
            contains=lambda point: (point in self) ^ (point in other),
            local_model=local_model,
            name=chosen_name,
        )

    def __or__(
        self,
        other: "RiemannianGeometricObject[PointT]",
    ) -> "RiemannianGeometricObject[PointT]":
        """Return the union operator result."""
        return self.union(other)

    def __and__(
        self,
        other: "RiemannianGeometricObject[PointT]",
    ) -> "RiemannianGeometricObject[PointT]":
        """Return the intersection operator result."""
        return self.intersection(other)

    def __sub__(
        self,
        other: "RiemannianGeometricObject[PointT]",
    ) -> "RiemannianGeometricObject[PointT]":
        """Return the difference operator result."""
        return self.difference(other)

    def __xor__(
        self,
        other: "RiemannianGeometricObject[PointT]",
    ) -> "RiemannianGeometricObject[PointT]":
        """Return the symmetric-difference operator result."""
        return self.symmetric_difference(other)

    def project_along_direction_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        direction: FloatVector,
        name: str = "",
    ) -> "RiemannianGeometricObject[PointT]":
        """Project an Euclidean object along a direction onto a hyperplane."""
        projected = super().project_along_direction_onto(
            source_hyperplane,
            target_hyperplane,
            direction,
            name=name,
        )
        chosen_name = name or getattr(projected, "name", "")
        return RiemannianGeometricObject.from_charted(
            self.space,
            projected,
            name=chosen_name,
        )

    def project_from_point_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        center: FloatPoint,
        name: str = "",
    ) -> "RiemannianGeometricObject[PointT]":
        """Project an Euclidean object from a point onto a hyperplane."""
        projected = super().project_from_point_onto(
            source_hyperplane,
            target_hyperplane,
            center,
            name=name,
        )
        chosen_name = name or getattr(projected, "name", "")
        return RiemannianGeometricObject.from_charted(
            self.space,
            projected,
            name=chosen_name,
        )


class RealLineSpace(ChartedRiemannianSpace[float]):
    """The real line with its standard Euclidean metric."""

    def __init__(self, name: str = "") -> None:
        """Initialize the standard Riemannian real line."""
        super().__init__(
            RealLine(),
            metric_tensor=lambda point: ((1.0,),),
            name=name or "R",
        )

    def point(
        self,
        point: float,
        name: str = "",
    ) -> RiemannianGeometricObject[float]:
        """Return a singleton object in the real line."""
        return RiemannianGeometricObject.from_charted(
            self,
            RealPointObject(point, name=name),
            name=name,
        )

    def subset(
        self,
        *point_set: object,
        name: str = "",
    ) -> RiemannianGeometricObject[float]:
        """Return a geometric object built from a ``FloatSet``."""
        return RiemannianGeometricObject.from_charted(
            self,
            RealSetObject(*point_set, name=name),
            name=name,
        )


class UnitCircleSpace(ChartedRiemannianSpace[FloatCirclePoint]):
    """The unit circle with its standard angular metric."""

    def __init__(self, name: str = "") -> None:
        """Initialize the standard Riemannian unit circle."""
        super().__init__(
            Circle(),
            metric_tensor=lambda point: ((1.0,),),
            name=name or "S1",
        )

    def point(
        self,
        point: FloatCirclePoint,
        name: str = "",
    ) -> RiemannianGeometricObject[FloatCirclePoint]:
        """Return a singleton object on the unit circle."""
        return RiemannianGeometricObject.from_charted(
            self,
            CirclePointObject(point, name=name),
            name=name,
        )

    def subset(
        self,
        *point_set: object,
        name: str = "",
    ) -> RiemannianGeometricObject[FloatCirclePoint]:
        """Return a geometric object built from a ``FloatCircleSet``."""
        return RiemannianGeometricObject.from_charted(
            self,
            CircleSetObject(*point_set, name=name),
            name=name,
        )

    def arc(
        self,
        start: FloatCirclePoint,
        end: FloatCirclePoint,
        name: str = "",
    ) -> RiemannianGeometricObject[FloatCirclePoint]:
        """Return a connected arc on the unit circle."""
        return self.subset((start, end), name=name)


class EuclideanRiemannianSpace(ChartedRiemannianSpace[FloatPoint]):
    """Euclidean space with its standard Riemannian metric."""

    def __init__(self, dim: int, name: str = "") -> None:
        """Initialize Euclidean space with the identity metric."""
        self._dim = dim
        identity = tuple(
            tuple(1.0 if i == j else 0.0 for j in range(dim))
            for i in range(dim)
        )
        super().__init__(
            EuclideanSpace(dim),
            metric_tensor=lambda point: identity,
            name=name or f"R^{dim}",
        )

    def point(
        self,
        point: FloatPoint,
        name: str = "",
    ) -> RiemannianGeometricObject[FloatPoint]:
        """Return a singleton object in Euclidean space."""
        return RiemannianGeometricObject.from_charted(
            self,
            EuclideanPointObject(point, name=name),
            name=name,
        )


class EuclideanPlaneSpace(EuclideanRiemannianSpace):
    """The Euclidean plane with several standard geometric objects."""

    def __init__(self, name: str = "") -> None:
        """Initialize the standard Riemannian plane."""
        super().__init__(2, name=name or "R^2")

    def whole_plane(
        self,
        name: str = "",
    ) -> RiemannianGeometricObject[FloatPoint]:
        """Return the whole plane as a geometric object."""
        return RiemannianGeometricObject.from_charted(
            self,
            WholePlane(name=name),
            name=name,
        )

    def half_plane(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> RiemannianGeometricObject[FloatPoint]:
        """Return a closed half-plane."""
        return RiemannianGeometricObject.from_charted(
            self,
            HalfPlane(normal, offset=offset, name=name),
            name=name,
        )

    def angle(
        self,
        apex: FloatPoint,
        start: FloatCirclePoint,
        end: FloatCirclePoint,
        name: str = "",
    ) -> RiemannianGeometricObject[FloatPoint]:
        """Return a closed planar angle with interior."""
        return RiemannianGeometricObject.from_charted(
            self,
            PlanarAngle(apex, start, end, name=name),
            name=name,
        )


__all__ = [
    "MetricTensor",
    "RiemannianSpace",
    "ChartedRiemannianSpace",
    "RiemannianGeometricObject",
    "RealLineSpace",
    "UnitCircleSpace",
    "EuclideanRiemannianSpace",
    "EuclideanPlaneSpace",
]
