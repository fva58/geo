"""Metric spaces and geometric objects inside them."""

from __future__ import annotations

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
TargetT = TypeVar("TargetT")
MetricTensor = tuple[tuple[float, ...], ...]


def _whole_neighborhood(dim: int):
    """Return the default unconstrained neighborhood for a cone."""
    from .euclidean import EuclideanNeighborhood

    return EuclideanNeighborhood.whole(dim)


def _point_array(point: FloatPoint) -> np.ndarray:
    """Return local coordinates as a NumPy float array."""
    return np.asarray(FloatPoint(point), dtype=float)


def _numeric_local_jacobian(
    mapping: Callable[[FloatPoint], FloatPoint],
    point: FloatPoint,
    step: float = 1e-6,
) -> np.ndarray:
    """Approximate the Jacobian of a local coordinate transition."""
    point = FloatPoint(point)
    base = _point_array(point)
    image_dim = FloatPoint(mapping(point)).dim
    jacobian = np.zeros((image_dim, point.dim), dtype=float)
    for index in range(point.dim):
        delta = np.zeros(point.dim, dtype=float)
        delta[index] = step
        image_plus = _point_array(mapping(FloatPoint(base + delta)))
        image_minus = _point_array(mapping(FloatPoint(base - delta)))
        jacobian[:, index] = (image_plus - image_minus) / (2.0 * step)
    return jacobian


@runtime_checkable
class MetricSpace(Manifold[PointT], Protocol[PointT]):
    """Protocol for a manifold equipped with a distance function.

    The contract is intentionally weak enough to model pseudometric spaces:
    implementations are expected to provide a non-negative symmetric distance,
    and different points may still have distance zero.
    """

    def distance(self, left: PointT, right: PointT) -> float:
        """Return the distance between two points."""


@runtime_checkable
class RiemannianSpace(MetricSpace[PointT], Protocol[PointT]):
    """Compatibility alias for the older metric-space naming."""


class ChartedMetricSpace(Generic[PointT]):
    """Concrete metric space given by a manifold and a distance function."""

    def __init__(
        self,
        manifold: Manifold[PointT],
        distance: Callable[[PointT, PointT], float],
        name: str = "",
    ) -> None:
        """Initialize the metric space."""
        self.manifold = manifold
        self._distance = distance
        self.name = name

    @property
    def dim(self) -> int:
        """Return the space dimension."""
        return self.manifold.dim

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"ChartedMetricSpace(dim={self.dim}{label})"

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the underlying manifold."""
        return point in self.manifold

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the underlying manifold."""
        return self.contains(point)

    def distance(
        self,
        left: PointT,
        right: PointT,
    ) -> float:
        """Return the distance between two points."""
        if left not in self or right not in self:
            raise ValueError("Points must belong to the metric space")
        distance = float(self._distance(left, right))
        if distance < 0.0:
            raise ValueError("Distance must be non-negative")
        return distance


class ChartedRiemannianSpace(ChartedMetricSpace[PointT]):
    """Compatibility alias for the older charted metric-space naming."""


class MetricGeometricObject(ChartedGeometricObject[PointT]):
    """Geometric object with an explicitly metric ambient space."""

    def __init__(
        self,
        space: MetricSpace[PointT],
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
        space: MetricSpace[PointT],
        obj: ChartedGeometricObject[PointT],
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Wrap an existing charted object in a metric ambient space."""
        chosen_name = name or getattr(obj, "name", "")
        wrapped = cls(
            space,
            contains=obj.contains,
            local_model=obj.local_model_at,
            name=chosen_name,
        )
        wrapped._charted_source_object = obj
        return wrapped

    def _require_same_space(
        self,
        other: "MetricGeometricObject[PointT]",
    ) -> None:
        """Require two objects to live in the same ambient space instance."""
        if self.space is not other.space:
            raise ValueError(
                "Set-theoretic operations require the same ambient space"
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
        origin = FloatPoint.origin(dim)

        def transition(point: FloatPoint) -> FloatPoint:
            manifold_point = left_model.chart.inverse(point)
            return FloatPoint(right_model.chart(manifold_point))

        transition_origin = FloatPoint(transition(origin))
        if not np.allclose(
            _point_array(transition_origin),
            np.zeros(dim, dtype=float),
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError(
                "Local charts do not share the same base point coordinates"
            )

        transition_jacobian = _numeric_local_jacobian(transition, origin)
        if np.linalg.matrix_rank(transition_jacobian) != dim:
            raise ValueError("Local chart transition is singular at the base point")

        cone = EuclideanCone(
            dim,
            contains=lambda point: operation(
                left_model.cone.contains(point),
                right_model.cone.contains(
                    FloatPoint(transition_jacobian @ _point_array(point))
                ),
            ),
            neighborhood=_whole_neighborhood(dim),
            name=name,
        )
        return LocalConeModel(left_model.chart, cone)

    def union(
        self,
        other: "MetricGeometricObject[PointT]",
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
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
        return MetricGeometricObject(
            self.space,
            contains=lambda point: point in self or point in other,
            local_model=local_model,
            name=chosen_name,
        )

    def intersection(
        self,
        other: "MetricGeometricObject[PointT]",
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
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
        return MetricGeometricObject(
            self.space,
            contains=lambda point: point in self and point in other,
            local_model=local_model,
            name=chosen_name,
        )

    def difference(
        self,
        other: "MetricGeometricObject[PointT]",
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
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
        return MetricGeometricObject(
            self.space,
            contains=lambda point: point in self and point not in other,
            local_model=local_model,
            name=chosen_name,
        )

    def symmetric_difference(
        self,
        other: "MetricGeometricObject[PointT]",
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
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
        return MetricGeometricObject(
            self.space,
            contains=lambda point: (point in self) ^ (point in other),
            local_model=local_model,
            name=chosen_name,
        )

    def __or__(
        self,
        other: "MetricGeometricObject[PointT]",
    ) -> "MetricGeometricObject[PointT]":
        """Return the union operator result."""
        return self.union(other)

    def __and__(
        self,
        other: "MetricGeometricObject[PointT]",
    ) -> "MetricGeometricObject[PointT]":
        """Return the intersection operator result."""
        return self.intersection(other)

    def __sub__(
        self,
        other: "MetricGeometricObject[PointT]",
    ) -> "MetricGeometricObject[PointT]":
        """Return the difference operator result."""
        return self.difference(other)

    def __xor__(
        self,
        other: "MetricGeometricObject[PointT]",
    ) -> "MetricGeometricObject[PointT]":
        """Return the symmetric-difference operator result."""
        return self.symmetric_difference(other)

    def project_along_direction_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        direction: FloatVector,
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Project an Euclidean object along a direction onto a hyperplane."""
        projected = super().project_along_direction_onto(
            source_hyperplane,
            target_hyperplane,
            direction,
            name=name,
        )
        chosen_name = name or getattr(projected, "name", "")
        return MetricGeometricObject.from_charted(
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
    ) -> "MetricGeometricObject[PointT]":
        """Project an Euclidean object from a point onto a hyperplane."""
        projected = super().project_from_point_onto(
            source_hyperplane,
            target_hyperplane,
            center,
            name=name,
        )
        chosen_name = name or getattr(projected, "name", "")
        return MetricGeometricObject.from_charted(
            self.space,
            projected,
            name=chosen_name,
        )

    def visible_from_direction(
        self,
        direction: FloatVector,
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Return the part of an Euclidean object visible from a direction."""
        source_object = getattr(self, "_charted_source_object", self)
        visible = source_object.visible_from_direction(direction, name=name)
        chosen_name = name or getattr(visible, "name", "")
        return MetricGeometricObject.from_charted(
            self.space,
            visible,
            name=chosen_name,
        )

    def visible_from_point(
        self,
        point: FloatPoint,
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Return the part of an Euclidean object visible from a point."""
        source_object = getattr(self, "_charted_source_object", self)
        visible = source_object.visible_from_point(point, name=name)
        chosen_name = name or getattr(visible, "name", "")
        return MetricGeometricObject.from_charted(
            self.space,
            visible,
            name=chosen_name,
        )

    def image_under_smooth_map(
        self,
        forward: Callable[[PointT], TargetT],
        preimage_on_image: Callable[[TargetT], PointT],
        target_space: MetricSpace[TargetT],
        target_chart,
        contains_image_point: Callable[[TargetT], bool] | None = None,
        name: str = "",
    ) -> "MetricGeometricObject[TargetT]":
        """Return the image object under a smooth map into a metric space."""
        image_object = super().image_under_smooth_map(
            forward,
            preimage_on_image,
            target_space,
            target_chart,
            contains_image_point=contains_image_point,
            name=name,
        )
        chosen_name = name or getattr(image_object, "name", "")
        return MetricGeometricObject.from_charted(
            target_space,
            image_object,
            name=chosen_name,
        )


RiemannianGeometricObject = MetricGeometricObject


class RealLineSpace(ChartedMetricSpace[float]):
    """The real line with its standard metric."""

    def __init__(self, name: str = "") -> None:
        """Initialize the standard metric real line."""
        super().__init__(
            RealLine(),
            distance=lambda left, right: abs(float(left) - float(right)),
            name=name or "R",
        )

    def point(
        self,
        point: float,
        name: str = "",
    ) -> MetricGeometricObject[float]:
        """Return a singleton object in the real line."""
        return MetricGeometricObject.from_charted(
            self,
            RealPointObject(point, name=name),
            name=name,
        )

    def subset(
        self,
        *point_set: object,
        name: str = "",
    ) -> MetricGeometricObject[float]:
        """Return a geometric object built from a ``FloatSet``."""
        return MetricGeometricObject.from_charted(
            self,
            RealSetObject(*point_set, name=name),
            name=name,
        )


class UnitCircleSpace(ChartedMetricSpace[FloatCirclePoint]):
    """The unit circle with its standard arc-length metric."""

    def __init__(self, name: str = "") -> None:
        """Initialize the standard metric unit circle."""
        super().__init__(
            Circle(),
            distance=lambda left, right: float(
                FloatCirclePoint(left).distance_to(FloatCirclePoint(right))
            ),
            name=name or "S1",
        )

    def point(
        self,
        point: FloatCirclePoint,
        name: str = "",
    ) -> MetricGeometricObject[FloatCirclePoint]:
        """Return a singleton object on the unit circle."""
        return MetricGeometricObject.from_charted(
            self,
            CirclePointObject(point, name=name),
            name=name,
        )

    def subset(
        self,
        *point_set: object,
        name: str = "",
    ) -> MetricGeometricObject[FloatCirclePoint]:
        """Return a geometric object built from a ``FloatCircleSet``."""
        return MetricGeometricObject.from_charted(
            self,
            CircleSetObject(*point_set, name=name),
            name=name,
        )

    def arc(
        self,
        start: FloatCirclePoint,
        end: FloatCirclePoint,
        name: str = "",
    ) -> MetricGeometricObject[FloatCirclePoint]:
        """Return a connected arc on the unit circle."""
        return self.subset((start, end), name=name)


class EuclideanMetricSpace(ChartedMetricSpace[FloatPoint]):
    """Euclidean space with its standard metric."""

    def __init__(self, dim: int, name: str = "") -> None:
        """Initialize Euclidean space with the Euclidean distance."""
        self._dim = dim
        super().__init__(
            EuclideanSpace(dim),
            distance=lambda left, right: FloatPoint(left).distance_to(
                FloatPoint(right)
            ),
            name=name or f"R^{dim}",
        )

    def point(
        self,
        point: FloatPoint,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return a singleton object in Euclidean space."""
        return MetricGeometricObject.from_charted(
            self,
            EuclideanPointObject(point, name=name),
            name=name,
        )


class EuclideanRiemannianSpace(EuclideanMetricSpace):
    """Compatibility alias for the older Euclidean metric-space naming."""


class EuclideanPlaneSpace(EuclideanMetricSpace):
    """The Euclidean plane with several standard geometric objects."""

    def __init__(self, name: str = "") -> None:
        """Initialize the standard metric plane."""
        super().__init__(2, name=name or "R^2")

    def whole_plane(
        self,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return the whole plane as a geometric object."""
        return MetricGeometricObject.from_charted(
            self,
            WholePlane(name=name),
            name=name,
        )

    def half_plane(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return a closed half-plane."""
        return MetricGeometricObject.from_charted(
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
    ) -> MetricGeometricObject[FloatPoint]:
        """Return a closed planar angle with interior."""
        return MetricGeometricObject.from_charted(
            self,
            PlanarAngle(apex, start, end, name=name),
            name=name,
        )


__all__ = [
    "MetricTensor",
    "MetricSpace",
    "ChartedMetricSpace",
    "MetricGeometricObject",
    "RiemannianSpace",
    "ChartedRiemannianSpace",
    "RiemannianGeometricObject",
    "RealLineSpace",
    "UnitCircleSpace",
    "EuclideanMetricSpace",
    "EuclideanRiemannianSpace",
    "EuclideanPlaneSpace",
]
