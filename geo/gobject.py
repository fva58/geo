"""Geometric objects in spaces and operations over them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from .cone import EuclideanCone, LocalConeModel
from .euclidean import FloatPoint, FloatVector
from .manifold import (
    LocalObjectModel,
    Manifold,
    ManifoldChart,
    NeighborhoodMarking,
    classify_local_object,
    classify_neighborhoods,
)
from .space.base import Neighborhood

if TYPE_CHECKING:
    from .space.base import Space


PointT = TypeVar("PointT")
TargetT = TypeVar("TargetT")


def _whole_neighborhood(dim: int):
    """Return the default unconstrained neighborhood for a cone."""
    from .euclidean import EuclideanNeighborhood

    return EuclideanNeighborhood.whole(dim)


@runtime_checkable
class GeometricObjectProtocol(Protocol[PointT]):
    """Protocol for a geometric object with local cone models."""

    @property
    def manifold(self) -> Manifold[PointT]:
        """Return the ambient manifold."""

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the object."""

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the object."""

    def local_model_at(self, point: PointT) -> LocalConeModel[PointT]:
        """Return a local cone model at a point of the object."""


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


def _centered_chart_at(
    chart: ManifoldChart[PointT],
    point: PointT,
) -> ManifoldChart[PointT]:
    """Return a chart recentered so that ``point`` maps to the origin."""
    origin = FloatPoint(chart(point))
    return ManifoldChart(
        lambda candidate: FloatPoint(chart(candidate)) - FloatVector(origin),
        lambda coordinates: chart.inverse(
            FloatPoint(coordinates) + FloatVector(origin)
        ),
        dim=chart.dim,
        domain_contains=chart.domain_contains,
        image=_whole_neighborhood(chart.dim),
        name=f"{chart.name}-centered" if chart.name else "",
    )


def _contains_linear_image(
    matrix: np.ndarray,
    source_cone,
    target_coordinates: FloatPoint,
) -> bool:
    """Check whether a vector belongs to the image of a source cone."""
    target_array = _point_array(target_coordinates)
    source_array, _, _, _ = np.linalg.lstsq(matrix, target_array, rcond=None)
    if not np.allclose(matrix @ source_array, target_array, atol=1e-7, rtol=1e-7):
        return False
    return source_cone.contains(FloatPoint(source_array))


class ChartedGeometricObject(Generic[PointT]):
    """Geometric object with local cone models on an ambient manifold."""

    def __init__(
        self,
        manifold: Manifold[PointT],
        contains: Callable[[PointT], bool],
        local_model: Callable[[PointT], LocalConeModel[PointT]],
        name: str = "",
    ) -> None:
        self.manifold = manifold
        self._contains = contains
        self._local_model = local_model
        self.name = name

    def __repr__(self) -> str:
        label = f", name={self.name!r}" if self.name else ""
        return f"ChartedGeometricObject(dim={self.manifold.dim}{label})"

    def contains(self, point: PointT) -> bool:
        return point in self.manifold and self._contains(point)

    def __contains__(self, point: PointT) -> bool:
        return self.contains(point)

    def local_model_at(self, point: PointT) -> LocalConeModel[PointT]:
        if point not in self:
            raise ValueError("Point is outside the geometric object")
        model = self._local_model(point)
        if model.chart.dim != self.manifold.dim:
            raise ValueError(
                "Local model chart dimension does not match manifold dimension"
            )
        return model

    def classify_neighborhood(
        self,
        neighborhood: Neighborhood[PointT],
    ) -> LocalObjectModel[PointT]:
        return classify_local_object(self, neighborhood)

    def classify_neighborhoods(
        self,
        neighborhoods: Sequence[Neighborhood[PointT]],
        name: str = "",
    ) -> NeighborhoodMarking[PointT]:
        return classify_neighborhoods(self, neighborhoods, name=name or self.name)

    def visible_from_direction(
        self,
        direction: FloatVector,
        name: str = "",
    ) -> "ChartedGeometricObject[FloatPoint]":
        from .space._euclidean_impl import visible_part_from_direction

        direction = FloatVector(direction)
        if direction.norm() == 0.0:
            raise ValueError("Visibility direction must be non-zero")
        return visible_part_from_direction(self, direction, name=name)

    def visible_from_point(
        self,
        point: FloatPoint,
        name: str = "",
    ) -> "ChartedGeometricObject[FloatPoint]":
        from .space._euclidean_impl import visible_part_from_point

        return visible_part_from_point(self, FloatPoint(point), name=name)

    def image_under_smooth_map(
        self,
        forward: Callable[[PointT], TargetT],
        preimage_on_image: Callable[[TargetT], PointT],
        target_manifold: Manifold[TargetT],
        target_chart: Callable[[TargetT], ManifoldChart[TargetT]],
        contains_image_point: Callable[[TargetT], bool] | None = None,
        name: str = "",
    ) -> "SmoothImageObject[PointT, TargetT]":
        return SmoothImageObject(
            self,
            forward,
            preimage_on_image,
            target_manifold,
            target_chart,
            contains_image_point=contains_image_point,
            name=name,
        )

    def project_along_direction_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        direction: FloatVector,
        name: str = "",
    ) -> "ChartedGeometricObject[FloatPoint]":
        from .space._euclidean_impl import (
            parallel_projection_inverse,
            projected_local_model,
        )

        inverse_map = parallel_projection_inverse(
            source_hyperplane,
            target_hyperplane,
            direction,
        )
        chosen_name = name or "parallel-projection"

        def contains(point: FloatPoint) -> bool:
            point = FloatPoint(point)
            if point not in target_hyperplane:
                return False
            try:
                source_point = inverse_map(point)
            except ValueError:
                return False
            return source_point in self

        def local_model(point: FloatPoint) -> LocalConeModel[FloatPoint]:
            target_point = FloatPoint(point)
            source_point = inverse_map(target_point)
            return projected_local_model(
                self,
                source_point,
                target_hyperplane,
                target_point,
                inverse_map,
                chosen_name,
            )

        return ChartedGeometricObject(
            self.manifold,
            contains=contains,
            local_model=local_model,
            name=chosen_name,
        )

    def project_from_point_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        center: FloatPoint,
        name: str = "",
    ) -> "ChartedGeometricObject[FloatPoint]":
        from .space._euclidean_impl import (
            central_projection_inverse,
            projected_local_model,
        )

        inverse_map = central_projection_inverse(
            source_hyperplane,
            target_hyperplane,
            center,
        )
        chosen_name = name or "central-projection"

        def contains(point: FloatPoint) -> bool:
            point = FloatPoint(point)
            if point not in target_hyperplane:
                return False
            try:
                source_point = inverse_map(point)
            except ValueError:
                return False
            return source_point in self

        def local_model(point: FloatPoint) -> LocalConeModel[FloatPoint]:
            target_point = FloatPoint(point)
            source_point = inverse_map(target_point)
            return projected_local_model(
                self,
                source_point,
                target_hyperplane,
                target_point,
                inverse_map,
                chosen_name,
            )

        return ChartedGeometricObject(
            self.manifold,
            contains=contains,
            local_model=local_model,
            name=chosen_name,
        )


class SmoothImageObject(ChartedGeometricObject[TargetT], Generic[PointT, TargetT]):
    """Image of a geometric object under a smooth map with local inverse."""

    def __init__(
        self,
        source_object: ChartedGeometricObject[PointT],
        forward: Callable[[PointT], TargetT],
        preimage_on_image: Callable[[TargetT], PointT],
        target_manifold: Manifold[TargetT],
        target_chart: Callable[[TargetT], ManifoldChart[TargetT]],
        contains_image_point: Callable[[TargetT], bool] | None = None,
        name: str = "",
    ) -> None:
        self.source_object = source_object
        self.forward = forward
        self.preimage_on_image = preimage_on_image
        self.target_chart = target_chart

        if contains_image_point is None:
            def image_contains(point: TargetT) -> bool:
                try:
                    return self.forward(self.preimage_on_image(point)) == point
                except (TypeError, ValueError):
                    return False
        else:
            image_contains = contains_image_point

        super().__init__(
            target_manifold,
            contains=lambda point: (
                point in target_manifold
                and image_contains(point)
                and self.preimage_on_image(point) in self.source_object
            ),
            local_model=self._local_model,
            name=name,
        )

    def _local_model(self, point: TargetT) -> LocalConeModel[TargetT]:
        source_point = self.preimage_on_image(point)
        source_model = self.source_object.local_model_at(source_point)
        centered_source_chart = _centered_chart_at(source_model.chart, source_point)
        base_target_chart = self.target_chart(point)
        centered_target_chart = _centered_chart_at(base_target_chart, point)

        def local_forward(source_coordinates: FloatPoint) -> FloatPoint:
            source_local_point = centered_source_chart.inverse(source_coordinates)
            return centered_target_chart(self.forward(source_local_point))

        jacobian = _numeric_local_jacobian(
            local_forward,
            FloatPoint.origin(centered_source_chart.dim),
        )
        if np.linalg.matrix_rank(jacobian) < centered_source_chart.dim:
            raise ValueError(
                "Smooth image local model requires an immersive map at the point"
            )
        cone = EuclideanCone(
            centered_target_chart.dim,
            contains=lambda coordinates: _contains_linear_image(
                jacobian,
                source_model.cone,
                FloatPoint(coordinates),
            ),
            apex=FloatPoint.origin(centered_target_chart.dim),
            neighborhood=_whole_neighborhood(centered_target_chart.dim),
            name="smooth-image",
        )
        return LocalConeModel(centered_target_chart, cone)


class GeometricObject(ChartedGeometricObject[PointT]):
    """Geometric object with an explicit ambient space."""

    def __init__(
        self,
        space: Space[PointT],
        contains: Callable[[PointT], bool],
        local_model,
        name: str = "",
    ) -> None:
        """Initialize the ambient-space geometric object."""
        self.space = space
        super().__init__(space, contains, local_model, name=name)

    @classmethod
    def from_charted(
        cls,
        space: Space[PointT],
        obj: ChartedGeometricObject[PointT],
        name: str = "",
    ) -> "GeometricObject[PointT]":
        """Wrap an existing charted object in an ambient space."""
        chosen_name = name or getattr(obj, "name", "")
        wrapped = cls(
            space,
            contains=obj.contains,
            local_model=obj.local_model_at,
            name=chosen_name,
        )
        wrapped._charted_source_object = obj
        return wrapped

    @staticmethod
    def _combine_local_models(
        left_model,
        right_model,
        operation: Callable[[bool, bool], bool],
        name: str,
    ):
        """Combine two local cone models in a shared coordinate chart."""
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
        other: "GeometricObject[PointT]",
        name: str = "",
    ) -> "GeometricObject[PointT]":
        return LazyExpressionObject("union", self, other, name=name)

    def intersection(
        self,
        other: "GeometricObject[PointT]",
        name: str = "",
    ) -> "GeometricObject[PointT]":
        return LazyExpressionObject("intersection", self, other, name=name)

    def difference(
        self,
        other: "GeometricObject[PointT]",
        name: str = "",
    ) -> "GeometricObject[PointT]":
        return LazyExpressionObject("difference", self, other, name=name)

    def symmetric_difference(
        self,
        other: "GeometricObject[PointT]",
        name: str = "",
    ) -> "GeometricObject[PointT]":
        return LazyExpressionObject("symmetric-difference", self, other, name=name)

    def __or__(self, other: "GeometricObject[PointT]") -> "GeometricObject[PointT]":
        return self.union(other)

    def __and__(self, other: "GeometricObject[PointT]") -> "GeometricObject[PointT]":
        return self.intersection(other)

    def __sub__(self, other: "GeometricObject[PointT]") -> "GeometricObject[PointT]":
        return self.difference(other)

    def __xor__(self, other: "GeometricObject[PointT]") -> "GeometricObject[PointT]":
        return self.symmetric_difference(other)

    def project_along_direction_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        direction: FloatVector,
        name: str = "",
    ) -> "GeometricObject[PointT]":
        return LazyMappedObject(
            self.space,
            "project-along-direction",
            self,
            name=name,
            source_hyperplane=source_hyperplane,
            target_hyperplane=target_hyperplane,
            direction=FloatVector(direction),
        )

    def project_from_point_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        center: FloatPoint,
        name: str = "",
    ) -> "GeometricObject[PointT]":
        return LazyMappedObject(
            self.space,
            "project-from-point",
            self,
            name=name,
            source_hyperplane=source_hyperplane,
            target_hyperplane=target_hyperplane,
            center=FloatPoint(center),
        )

    def visible_from_direction(
        self,
        direction: FloatVector,
        name: str = "",
    ) -> "GeometricObject[PointT]":
        return LazyMappedObject(
            self.space,
            "visible-from-direction",
            self,
            name=name,
            direction=FloatVector(direction),
        )

    def visible_from_point(
        self,
        point: FloatPoint,
        name: str = "",
    ) -> "GeometricObject[PointT]":
        return LazyMappedObject(
            self.space,
            "visible-from-point",
            self,
            name=name,
            point=FloatPoint(point),
        )

    def image_under_smooth_map(
        self,
        forward: Callable[[PointT], TargetT],
        preimage_on_image: Callable[[TargetT], PointT],
        target_space: Space[TargetT],
        target_chart,
        contains_image_point: Callable[[TargetT], bool] | None = None,
        name: str = "",
    ) -> "GeometricObject[TargetT]":
        return LazyMappedObject(
            target_space,
            "image-under-smooth-map",
            self,
            name=name,
            forward=forward,
            preimage_on_image=preimage_on_image,
            target_chart=target_chart,
            contains_image_point=contains_image_point,
        )


class LazyObject(GeometricObject[PointT]):
    """Base class for lazy object expression-tree nodes."""

    def __init__(self, space: Space[PointT], operation: str, name: str = "") -> None:
        self.operation = operation
        super().__init__(
            space,
            contains=self._contains_lazy,
            local_model=self._local_model_lazy,
            name=name or self._default_name(),
        )

    @property
    def children(self) -> tuple[GeometricObject, ...]:
        return tuple()

    @property
    def is_lazy(self) -> bool:
        return True

    @property
    def node_kind(self) -> str:
        raise NotImplementedError

    def _default_name(self) -> str:
        raise NotImplementedError

    def _contains_lazy(self, point: PointT) -> bool:
        raise NotImplementedError

    def _local_model_lazy(self, point: PointT):
        raise NotImplementedError


class LazyExpressionObject(LazyObject[PointT]):
    """Lazy set-theoretic expression over geometric objects."""

    def __init__(
        self,
        operation: str,
        left: GeometricObject[PointT],
        right: GeometricObject[PointT],
        name: str = "",
    ) -> None:
        if left.space is not right.space:
            raise ValueError("Set-theoretic operations require the same ambient space")
        self.left = left
        self.right = right
        super().__init__(left.space, operation, name=name)

    def __repr__(self) -> str:
        return f"LazyExpressionObject(operation={self.operation!r}, name={self.name!r})"

    @property
    def children(self) -> tuple[GeometricObject[PointT], ...]:
        return (self.left, self.right)

    @property
    def node_kind(self) -> str:
        return "binary"

    def _default_name(self) -> str:
        symbols = {
            "union": "|",
            "intersection": "&",
            "difference": "-",
            "symmetric-difference": "^",
        }
        symbol = symbols.get(self.operation, "?")
        return f"({self.left.name}){symbol}({self.right.name})"

    def _contains_lazy(self, point: PointT) -> bool:
        left_contains = point in self.left
        right_contains = point in self.right
        if self.operation == "union":
            return left_contains or right_contains
        if self.operation == "intersection":
            return left_contains and right_contains
        if self.operation == "difference":
            return left_contains and not right_contains
        if self.operation == "symmetric-difference":
            return left_contains ^ right_contains
        raise ValueError(f"Unsupported lazy operation: {self.operation!r}")

    def _local_model_lazy(self, point: PointT):
        if self.operation == "union":
            if point in self.left and point in self.right:
                return GeometricObject._combine_local_models(
                    self.left.local_model_at(point),
                    self.right.local_model_at(point),
                    lambda left, right: left or right,
                    self.operation,
                )
            if point in self.left:
                return self.left.local_model_at(point)
            return self.right.local_model_at(point)
        if self.operation == "intersection":
            return GeometricObject._combine_local_models(
                self.left.local_model_at(point),
                self.right.local_model_at(point),
                lambda left, right: left and right,
                self.operation,
            )
        if self.operation == "difference":
            if point in self.right:
                return GeometricObject._combine_local_models(
                    self.left.local_model_at(point),
                    self.right.local_model_at(point),
                    lambda left, right: left and not right,
                    self.operation,
                )
            return self.left.local_model_at(point)
        if self.operation == "symmetric-difference":
            if point in self.left and point not in self.right:
                return self.left.local_model_at(point)
            if point in self.right and point not in self.left:
                return self.right.local_model_at(point)
            return GeometricObject._combine_local_models(
                self.left.local_model_at(point),
                self.right.local_model_at(point),
                lambda left, right: left ^ right,
                self.operation,
            )
        raise ValueError(f"Unsupported lazy operation: {self.operation!r}")


class LazyMappedObject(LazyObject):
    """Lazy unary operation over a geometric object."""

    def __init__(
        self,
        space: Space,
        operation: str,
        source: GeometricObject,
        name: str = "",
        **parameters,
    ) -> None:
        self.source = source
        self.parameters = parameters
        super().__init__(space, operation, name=name)

    def __repr__(self) -> str:
        return f"LazyMappedObject(operation={self.operation!r}, name={self.name!r})"

    @property
    def children(self) -> tuple[GeometricObject, ...]:
        return (self.source,)

    @property
    def node_kind(self) -> str:
        return "unary"

    def _default_name(self) -> str:
        return f"{self.operation}({self.source.name})"

    def _materialize_charted_object(self):
        source_object = getattr(self.source, "_charted_source_object", self.source)
        if self.operation == "project-along-direction":
            return ChartedGeometricObject.project_along_direction_onto(
                source_object,
                self.parameters["source_hyperplane"],
                self.parameters["target_hyperplane"],
                self.parameters["direction"],
                name=self.name,
            )
        if self.operation == "project-from-point":
            return ChartedGeometricObject.project_from_point_onto(
                source_object,
                self.parameters["source_hyperplane"],
                self.parameters["target_hyperplane"],
                self.parameters["center"],
                name=self.name,
            )
        if self.operation == "visible-from-direction":
            return ChartedGeometricObject.visible_from_direction(
                source_object,
                self.parameters["direction"],
                name=self.name,
            )
        if self.operation == "visible-from-point":
            return ChartedGeometricObject.visible_from_point(
                source_object,
                self.parameters["point"],
                name=self.name,
            )
        if self.operation == "image-under-smooth-map":
            return ChartedGeometricObject.image_under_smooth_map(
                source_object,
                self.parameters["forward"],
                self.parameters["preimage_on_image"],
                self.space,
                self.parameters["target_chart"],
                contains_image_point=self.parameters["contains_image_point"],
                name=self.name,
            )
        raise ValueError(f"Unsupported lazy mapped operation: {self.operation!r}")

    def _materialize_object(self):
        return GeometricObject.from_charted(
            self.space,
            self._materialize_charted_object(),
            name=self.name,
        )

    def _contains_lazy(self, point):
        return point in self._materialize_charted_object()

    def _local_model_lazy(self, point):
        return self._materialize_charted_object().local_model_at(point)


__all__ = [
    "GeometricObjectProtocol",
    "ChartedGeometricObject",
    "SmoothImageObject",
    "GeometricObject",
    "LazyObject",
    "LazyExpressionObject",
    "LazyMappedObject",
]
