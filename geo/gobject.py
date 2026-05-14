"""Geometric objects in spaces and operations over them."""

from __future__ import annotations

from collections.abc import Sequence
import itertools
from typing import TYPE_CHECKING, Callable, Generic, NamedTuple, Protocol, TypeVar, runtime_checkable

import numpy as np

from .cone import EuclideanCone, LocalConeModel
from .euclidean import Point, Vector
from .manifold import (
    Manifold,
    ManifoldChart,
)

if TYPE_CHECKING:
    from .space.base import Neighborhood, Space


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


def _point_array(point: Point) -> np.ndarray:
    """Return local coordinates as a NumPy float array."""
    return np.asarray(Point(point), dtype=float)


def _numeric_local_jacobian(
    mapping: Callable[[Point], Point],
    point: Point,
    step: float = 1e-6,
) -> np.ndarray:
    """Approximate the Jacobian of a local coordinate transition."""
    point = Point(point)
    base = _point_array(point)
    image_dim = Point(mapping(point)).dim
    jacobian = np.zeros((image_dim, point.dim), dtype=float)
    for index in range(point.dim):
        delta = np.zeros(point.dim, dtype=float)
        delta[index] = step
        image_plus = _point_array(mapping(Point(base + delta)))
        image_minus = _point_array(mapping(Point(base - delta)))
        jacobian[:, index] = (image_plus - image_minus) / (2.0 * step)
    return jacobian


def _centered_chart_at(
    chart: ManifoldChart[PointT],
    point: PointT,
) -> ManifoldChart[PointT]:
    """Return a chart recentered so that ``point`` maps to the origin."""
    origin = Point(chart(point))
    return ManifoldChart(
        lambda candidate: Point(chart(candidate)) - Vector(origin),
        lambda coordinates: chart.inverse(
            Point(coordinates) + Vector(origin)
        ),
        dim=chart.dim,
        domain_contains=chart.domain_contains,
        image=_whole_neighborhood(chart.dim),
    )


def _contains_linear_image(
    matrix: np.ndarray,
    source_cone,
    target_coordinates: Point,
) -> bool:
    """Check whether a vector belongs to the image of a source cone."""
    target_array = _point_array(target_coordinates)
    source_array, _, _, _ = np.linalg.lstsq(matrix, target_array, rcond=None)
    if not np.allclose(matrix @ source_array, target_array, atol=1e-7, rtol=1e-7):
        return False
    return source_cone.contains(Point(source_array))


def _classification_points(neighborhood: Neighborhood[PointT]) -> tuple[PointT, ...]:
    """Return private test points used by object-side classification."""
    bounds = []
    for coordinate_set in neighborhood.image:
        if len(coordinate_set) != 1:
            raise ValueError("Classification requires box neighborhoods")
        interval = coordinate_set[0]
        bounds.append((float(interval[0]), float(interval[1])))
    coordinate_points = [Point(
        [(left + right) / 2.0 for left, right in bounds]
    )]
    coordinate_points.extend(
        Point(vertex)
        for vertex in itertools.product(
            *((left, right) for left, right in bounds)
        )
    )
    points = []
    for coordinates in coordinate_points:
        point = neighborhood.chart.inverse(coordinates)
        if point not in points:
            points.append(point)
    return tuple(points)


class ChartedGeometricObject(Generic[PointT]):
    """Geometric object with local cone models on an ambient manifold."""

    def __init__(
        self,
        manifold: Manifold[PointT],
        contains: Callable[[PointT], bool],
        local_model: Callable[[PointT], LocalConeModel[PointT]],
    ) -> None:
        self.manifold = manifold
        self._contains = contains
        self._local_model = local_model

    def __repr__(self) -> str:
        return f"ChartedGeometricObject(dim={self.manifold.dim})"

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
    ):
        center = neighborhood.center_point()
        if center in self:
            local_model = self.local_model_at(center)
            for point in _classification_points(neighborhood):
                actual = point in self
                try:
                    predicted = local_model.chart(point) in local_model.cone
                except ValueError:
                    return Ellipsis
                if actual != predicted:
                    return Ellipsis
            return local_model

        for point in _classification_points(neighborhood):
            if point in self:
                return Ellipsis
        return None

    def classify_neighborhoods(
        self,
        neighborhoods: Sequence[Neighborhood[PointT]],
    ):
        return tuple(
            self.classify_neighborhood(neighborhood)
            for neighborhood in neighborhoods
        )

    def visible_from_direction(
        self,
        direction: Vector,
    ) -> "ChartedGeometricObject[Point]":
        from .space._euclidean_impl import visible_part_from_direction

        direction = Vector(direction)
        if direction.norm() == 0.0:
            raise ValueError("Visibility direction must be non-zero")
        return visible_part_from_direction(self, direction)

    def visible_from_point(
        self,
        point: Point,
    ) -> "ChartedGeometricObject[Point]":
        from .space._euclidean_impl import visible_part_from_point

        return visible_part_from_point(self, Point(point))

    def image_under_smooth_map(
        self,
        forward: Callable[[PointT], TargetT],
        preimage_on_image: Callable[[TargetT], PointT],
        target_manifold: Manifold[TargetT],
        target_chart: Callable[[TargetT], ManifoldChart[TargetT]],
        contains_image_point: Callable[[TargetT], bool] | None = None,
    ) -> "SmoothImageObject[PointT, TargetT]":
        return SmoothImageObject(
            self,
            forward,
            preimage_on_image,
            target_manifold,
            target_chart,
            contains_image_point=contains_image_point,
        )

    def project_along_direction_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        direction: Vector,
    ) -> "ChartedGeometricObject[Point]":
        from .space._euclidean_impl import (
            parallel_projection_inverse,
            projected_local_model,
        )

        inverse_map = parallel_projection_inverse(
            source_hyperplane,
            target_hyperplane,
            direction,
        )

        def contains(point: Point) -> bool:
            point = Point(point)
            if point not in target_hyperplane:
                return False
            try:
                source_point = inverse_map(point)
            except ValueError:
                return False
            return source_point in self

        def local_model(point: Point) -> LocalConeModel[Point]:
            target_point = Point(point)
            source_point = inverse_map(target_point)
            return projected_local_model(
                self,
                source_point,
                target_hyperplane,
                target_point,
                inverse_map,
            )

        return ChartedGeometricObject(
            self.manifold,
            contains=contains,
            local_model=local_model,
        )

    def project_from_point_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        center: Point,
    ) -> "ChartedGeometricObject[Point]":
        from .space._euclidean_impl import (
            central_projection_inverse,
            projected_local_model,
        )

        inverse_map = central_projection_inverse(
            source_hyperplane,
            target_hyperplane,
            center,
        )

        def contains(point: Point) -> bool:
            point = Point(point)
            if point not in target_hyperplane:
                return False
            try:
                source_point = inverse_map(point)
            except ValueError:
                return False
            return source_point in self

        def local_model(point: Point) -> LocalConeModel[Point]:
            target_point = Point(point)
            source_point = inverse_map(target_point)
            return projected_local_model(
                self,
                source_point,
                target_hyperplane,
                target_point,
                inverse_map,
            )

        return ChartedGeometricObject(
            self.manifold,
            contains=contains,
            local_model=local_model,
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
        )

    def _local_model(self, point: TargetT) -> LocalConeModel[TargetT]:
        source_point = self.preimage_on_image(point)
        source_model = self.source_object.local_model_at(source_point)
        centered_source_chart = _centered_chart_at(source_model.chart, source_point)
        base_target_chart = self.target_chart(point)
        centered_target_chart = _centered_chart_at(base_target_chart, point)

        def local_forward(source_coordinates: Point) -> Point:
            source_local_point = centered_source_chart.inverse(source_coordinates)
            return centered_target_chart(self.forward(source_local_point))

        jacobian = _numeric_local_jacobian(
            local_forward,
            Point.origin(centered_source_chart.dim),
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
                Point(coordinates),
            ),
            apex=Point.origin(centered_target_chart.dim),
            neighborhood=_whole_neighborhood(centered_target_chart.dim),
        )
        return LocalConeModel(centered_target_chart, cone)


class GeometricObject(ChartedGeometricObject[PointT]):
    """Geometric object with an explicit ambient space."""

    def __init__(
        self,
        space: Space[PointT],
        contains: Callable[[PointT], bool],
        local_model,
    ) -> None:
        """Initialize the ambient-space geometric object."""
        self.space = space
        super().__init__(space, contains, local_model)

    @classmethod
    def from_charted(
        cls,
        space: Space[PointT],
        obj: ChartedGeometricObject[PointT],
    ) -> "GeometricObject[PointT]":
        """Wrap an existing charted object in an ambient space."""
        wrapped = cls(
            space,
            contains=obj.contains,
            local_model=obj.local_model_at,
        )
        wrapped._charted_source_object = obj
        return wrapped

    @staticmethod
    def _combine_local_models(
        left_model,
        right_model,
        operation: Callable[[bool, bool], bool],
    ):
        """Combine two local cone models in a shared coordinate chart."""
        if left_model.chart.dim != right_model.chart.dim:
            raise ValueError("Local model dimensions do not match")
        dim = left_model.chart.dim
        origin = Point.origin(dim)

        def transition(point: Point) -> Point:
            manifold_point = left_model.chart.inverse(point)
            return Point(right_model.chart(manifold_point))

        transition_origin = Point(transition(origin))
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
                    Point(transition_jacobian @ _point_array(point))
                ),
            ),
            neighborhood=_whole_neighborhood(dim),
        )
        return LocalConeModel(left_model.chart, cone)

    def union(
        self,
        other: "GeometricObject[PointT]",
    ) -> "GeometricObject[PointT]":
        return LazyExpressionObject("union", self, other)

    def intersection(
        self,
        other: "GeometricObject[PointT]",
    ) -> "GeometricObject[PointT]":
        return LazyExpressionObject("intersection", self, other)

    def difference(
        self,
        other: "GeometricObject[PointT]",
    ) -> "GeometricObject[PointT]":
        return LazyExpressionObject("difference", self, other)

    def symmetric_difference(
        self,
        other: "GeometricObject[PointT]",
    ) -> "GeometricObject[PointT]":
        return LazyExpressionObject("symmetric-difference", self, other)

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
        direction: Vector,
    ) -> "GeometricObject[PointT]":
        return LazyMappedObject(
            self.space,
            "project-along-direction",
            self,
            source_hyperplane=source_hyperplane,
            target_hyperplane=target_hyperplane,
            direction=Vector(direction),
        )

    def project_from_point_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        center: Point,
    ) -> "GeometricObject[PointT]":
        return LazyMappedObject(
            self.space,
            "project-from-point",
            self,
            source_hyperplane=source_hyperplane,
            target_hyperplane=target_hyperplane,
            center=Point(center),
        )

    def visible_from_direction(
        self,
        direction: Vector,
    ) -> "GeometricObject[PointT]":
        return LazyMappedObject(
            self.space,
            "visible-from-direction",
            self,
            direction=Vector(direction),
        )

    def visible_from_point(
        self,
        point: Point,
    ) -> "GeometricObject[PointT]":
        return LazyMappedObject(
            self.space,
            "visible-from-point",
            self,
            point=Point(point),
        )

    def image_under_smooth_map(
        self,
        forward: Callable[[PointT], TargetT],
        preimage_on_image: Callable[[TargetT], PointT],
        target_space: Space[TargetT],
        target_chart,
        contains_image_point: Callable[[TargetT], bool] | None = None,
    ) -> "GeometricObject[TargetT]":
        return LazyMappedObject(
            target_space,
            "image-under-smooth-map",
            self,
            forward=forward,
            preimage_on_image=preimage_on_image,
            target_chart=target_chart,
            contains_image_point=contains_image_point,
        )


class LazyObject(GeometricObject[PointT]):
    """Base class for lazy object expression-tree nodes."""

    def __init__(self, space: Space[PointT], operation: str) -> None:
        self.operation = operation
        super().__init__(
            space,
            contains=self._contains_lazy,
            local_model=self._local_model_lazy,
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
    ) -> None:
        if left.space is not right.space:
            raise ValueError("Set-theoretic operations require the same ambient space")
        self.left = left
        self.right = right
        super().__init__(left.space, operation)

    def __repr__(self) -> str:
        return f"LazyExpressionObject(operation={self.operation!r})"

    @property
    def children(self) -> tuple[GeometricObject[PointT], ...]:
        return (self.left, self.right)

    @property
    def node_kind(self) -> str:
        return "binary"

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
                )
            if point in self.left:
                return self.left.local_model_at(point)
            return self.right.local_model_at(point)
        if self.operation == "intersection":
            return GeometricObject._combine_local_models(
                self.left.local_model_at(point),
                self.right.local_model_at(point),
                lambda left, right: left and right,
            )
        if self.operation == "difference":
            if point in self.right:
                return GeometricObject._combine_local_models(
                    self.left.local_model_at(point),
                    self.right.local_model_at(point),
                    lambda left, right: left and not right,
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
            )
        raise ValueError(f"Unsupported lazy operation: {self.operation!r}")


class LazyMappedObject(LazyObject):
    """Lazy unary operation over a geometric object."""

    def __init__(
        self,
        space: Space,
        operation: str,
        source: GeometricObject,
        **parameters,
    ) -> None:
        self.source = source
        self.parameters = parameters
        super().__init__(space, operation)

    def __repr__(self) -> str:
        return f"LazyMappedObject(operation={self.operation!r})"

    @property
    def children(self) -> tuple[GeometricObject, ...]:
        return (self.source,)

    @property
    def node_kind(self) -> str:
        return "unary"

    def _materialize_charted_object(self):
        source_object = getattr(self.source, "_charted_source_object", self.source)
        if self.operation == "project-along-direction":
            return ChartedGeometricObject.project_along_direction_onto(
                source_object,
                self.parameters["source_hyperplane"],
                self.parameters["target_hyperplane"],
                self.parameters["direction"],
            )
        if self.operation == "project-from-point":
            return ChartedGeometricObject.project_from_point_onto(
                source_object,
                self.parameters["source_hyperplane"],
                self.parameters["target_hyperplane"],
                self.parameters["center"],
            )
        if self.operation == "visible-from-direction":
            return ChartedGeometricObject.visible_from_direction(
                source_object,
                self.parameters["direction"],
            )
        if self.operation == "visible-from-point":
            return ChartedGeometricObject.visible_from_point(
                source_object,
                self.parameters["point"],
            )
        if self.operation == "image-under-smooth-map":
            return ChartedGeometricObject.image_under_smooth_map(
                source_object,
                self.parameters["forward"],
                self.parameters["preimage_on_image"],
                self.space,
                self.parameters["target_chart"],
                contains_image_point=self.parameters["contains_image_point"],
            )
        raise ValueError(f"Unsupported lazy mapped operation: {self.operation!r}")

    def _materialize_object(self):
        return GeometricObject.from_charted(
            self.space,
            self._materialize_charted_object(),
        )

    def _contains_lazy(self, point):
        return point in self._materialize_charted_object()

    def _local_model_lazy(self, point):
        return self._materialize_charted_object().local_model_at(point)


class _CoverResult(NamedTuple):
    """Result of cover classification or refinement."""

    cone_parts: tuple
    complex_parts: tuple
    empty_parts: tuple

    @property
    def active_parts(self):
        """Return the non-empty parts."""
        return self.cone_parts + self.complex_parts

    def max_diameter(self) -> float:
        """Return the largest diameter among active parts."""
        if not self.active_parts:
            return 0.0
        return max(n.diameter() for n in self.active_parts)

    def max_outer_radius(self) -> float:
        """Return the largest outer radius among active parts."""
        if not self.active_parts:
            return 0.0
        return max(n.outer_radius() for n in self.active_parts)


def local_chart_cover_from_points(
    space,
    points,
    radius: float,
):
    """Build a neighborhood cover from explicit points in one space."""
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("Neighborhood radius must be positive")
    neighborhoods = tuple(
        space.neighborhood_at(point, radius)
        for point in points
    )
    if not neighborhoods:
        raise ValueError("Need at least one point to build a cover")
    return neighborhoods


def classify_cover(
    obj,
    cover,
):
    """Classify one object over all neighborhoods in a cover."""
    cone = []
    complex_ = []
    empty = []
    for neighborhood, result in zip(cover, obj.classify_neighborhoods(cover)):
        if isinstance(result, LocalConeModel):
            cone.append(neighborhood)
        elif result is Ellipsis:
            complex_.append(neighborhood)
        else:
            empty.append(neighborhood)
    return _CoverResult(tuple(cone), tuple(complex_), tuple(empty))


def refine_until(
    obj,
    cover,
    *,
    max_outer_radius: float,
    max_steps: int = 8,
):
    """Refine a cover until non-empty parts are small enough or steps end."""
    if max_outer_radius <= 0.0:
        raise ValueError("max_outer_radius must be positive")
    current_cover = cover
    current = classify_cover(obj, current_cover)
    for _ in range(max_steps):
        if (
            not current.complex_parts and
            current.max_outer_radius() <= max_outer_radius
        ):
            return current
        to_keep = list(current.cone_parts)
        to_refine = list(current.complex_parts)
        if current.max_outer_radius() > max_outer_radius:
            to_refine.extend(
                n for n in current.cone_parts
                if n.outer_radius() > max_outer_radius
            )
            to_keep = [
                n for n in current.cone_parts
                if n.outer_radius() <= max_outer_radius
            ]
        refined = tuple(
            child
            for neighborhood in to_refine
            for child in neighborhood.subdivide()
        )
        if not refined:
            return current
        current_cover = tuple(to_keep) + refined
        current = classify_cover(obj, current_cover)
    return current


__all__ = [
    "GeometricObjectProtocol",
    "ChartedGeometricObject",
    "SmoothImageObject",
    "GeometricObject",
    "LazyObject",
    "LazyExpressionObject",
    "LazyMappedObject",
    "classify_cover",
    "local_chart_cover_from_points",
    "refine_until",
]
