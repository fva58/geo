"""Geometric objects and local cone models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from .euclidean import EuclideanNeighborhood, FloatPoint, FloatVector
from .floatcircle import (
    FloatAngle,
    FloatCircleInterval,
    FloatCirclePoint,
    FloatCircleSet,
)
from .floatset import FloatInterval, FloatSet
from .manifold import LocalPointT, Manifold, ManifoldChart


PointT = TypeVar("PointT")
TargetT = TypeVar("TargetT")


def _point_cone(dim: int) -> EuclideanCone:
    """Return the zero-dimensional cone supported at the apex only."""
    origin = FloatPoint.origin(dim)
    return EuclideanCone(
        dim,
        contains=lambda point: FloatPoint(point) == origin,
        apex=origin,
        neighborhood=EuclideanNeighborhood.whole(dim),
        name="point",
    )


def _positive_half_line_cone() -> EuclideanCone:
    """Return the standard half-line ``[0, +inf)``."""
    return EuclideanCone(
        1,
        contains=lambda point: point[0] >= 0.0,
        apex=FloatPoint.origin(1),
        neighborhood=EuclideanNeighborhood.whole(1),
        name="positive-half-line",
    )


def _negative_half_line_cone() -> EuclideanCone:
    """Return the standard half-line ``(-inf, 0]``."""
    return EuclideanCone(
        1,
        contains=lambda point: point[0] <= 0.0,
        apex=FloatPoint.origin(1),
        neighborhood=EuclideanNeighborhood.whole(1),
        name="negative-half-line",
    )


def _signed_circle_offset(
    center: FloatCirclePoint,
    point: FloatCirclePoint,
) -> float:
    """Return the signed angular offset from ``center`` to ``point``."""
    offset = float(FloatAngle(point) - FloatAngle(center))
    if offset > math.pi:
        offset -= FloatAngle.TWO_PI
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


def _half_space_cone(
    normal: FloatVector,
    *,
    reverse: bool = False,
    name: str = "",
) -> EuclideanCone:
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
        name=name,
    )


def _hyperplane_cone(normal: FloatVector, name: str = "") -> EuclideanCone:
    """Return a cone given by one linear equality."""
    dim = normal.dim
    return EuclideanCone(
        dim,
        contains=lambda coordinates: _isclose(
            FloatVector(coordinates).dot(normal)
        ),
        apex=FloatPoint.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
        name=name,
    )


def _coerce_affine_matrix(
    vectors: Sequence[Sequence[float]],
    dim: int,
) -> np.ndarray:
    """Return an invertible matrix whose columns are the given vectors."""
    columns = [_vector_array(FloatVector(vector)) for vector in vectors]
    if len(columns) != dim:
        raise ValueError(
            f"Need {dim} spanning vectors, got {len(columns)}"
        )
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


def _coerce_real_set_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    """Translate constructor-like input into ``FloatSet`` arguments."""
    intervals: list[object] = []
    for arg in args:
        if isinstance(arg, FloatSet):
            intervals.extend(
                FloatInterval.from_tuple(interval)
                for interval in arg
            )
            continue
        if isinstance(arg, FloatInterval):
            intervals.append(arg)
            continue
        if isinstance(arg, (float, int)):
            intervals.append(float(arg))
            continue
        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
            if len(arg) == 2 and all(
                isinstance(item, (float, int)) for item in arg
            ):
                intervals.append(FloatInterval(float(arg[0]), float(arg[1])))
                continue
            intervals.extend(_coerce_real_set_arguments(tuple(arg)))
            continue
        raise TypeError(f"Unsupported real-set argument: {arg!r}")
    return tuple(intervals)


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


@runtime_checkable
class GeometricObject(Protocol[PointT]):
    """Protocol for a geometric object with local cone models."""

    @property
    def manifold(self) -> Manifold[PointT]:
        """Return the ambient manifold."""

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the object."""

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the object."""

    def local_model_at(self, point: PointT) -> "LocalConeModel[PointT]":
        """Return a local cone model at a point of the object."""


class RealLine:
    """Canonical one-dimensional manifold modeled by finite floats."""

    dim = 1

    def contains(self, point: float) -> bool:
        """Check whether the point is a finite real coordinate."""
        return isinstance(point, (int, float)) and math.isfinite(float(point))

    def __contains__(self, point: float) -> bool:
        """Check whether the point is a finite real coordinate."""
        return self.contains(point)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return "RealLine()"


class Circle:
    """Canonical unit-circle manifold parameterized by ``FloatCirclePoint``."""

    dim = 1

    def contains(self, point: object) -> bool:
        """Check whether the point can be interpreted on the unit circle."""
        try:
            FloatCirclePoint(point)
        except (TypeError, ValueError):
            return False
        return True

    def __contains__(self, point: object) -> bool:
        """Check whether the point can be interpreted on the unit circle."""
        return self.contains(point)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return "Circle()"


class EuclideanSpace:
    """Canonical Euclidean manifold ``R^n``."""

    def __init__(self, dim: int) -> None:
        """Initialize the Euclidean space dimension."""
        if dim <= 0:
            raise ValueError("Euclidean space dimension must be positive")
        self.dim = dim

    def contains(self, point: object) -> bool:
        """Check whether the point is a coordinate tuple of matching size."""
        try:
            coordinates = FloatPoint(point)
        except (TypeError, ValueError):
            return False
        return coordinates.dim == self.dim

    def __contains__(self, point: object) -> bool:
        """Check whether the point is a coordinate tuple of matching size."""
        return self.contains(point)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"EuclideanSpace(dim={self.dim})"


def _real_chart(center: float) -> ManifoldChart[float]:
    """Return the canonical centered chart on the real line."""
    return ManifoldChart(
        lambda point: FloatPoint(float(point) - center),
        lambda coordinates: center + coordinates[0],
        dim=1,
        domain_contains=RealLine().contains,
        image=EuclideanNeighborhood.whole(1),
        name="real-centered",
    )


def _circle_chart(center: FloatCirclePoint) -> ManifoldChart[FloatCirclePoint]:
    """Return a local angular chart centered at a circle point."""

    def forward(point: FloatCirclePoint) -> FloatPoint:
        offset = _signed_circle_offset(center, FloatCirclePoint(point))
        return FloatPoint(offset)

    def inverse(coordinates: FloatPoint) -> FloatCirclePoint:
        return FloatCirclePoint(float(center) + coordinates[0])

    return ManifoldChart(
        forward,
        inverse,
        dim=1,
        domain_contains=lambda point: (
            Circle().contains(point) and
            abs(
                _signed_circle_offset(center, FloatCirclePoint(point))
            ) < math.pi
        ),
        image=EuclideanNeighborhood.box((-math.pi, math.pi)),
        name="circle-centered",
    )


def _euclidean_chart(
    center: FloatPoint,
) -> ManifoldChart[FloatPoint]:
    """Return the canonical translated chart in Euclidean space."""
    dim = center.dim
    space = EuclideanSpace(dim)
    return ManifoldChart(
        lambda point: FloatPoint(point) - center,
        lambda coordinates: center + FloatVector(coordinates),
        dim=dim,
        domain_contains=space.contains,
        image=EuclideanNeighborhood.whole(dim),
        name="euclidean-centered",
    )


def _numeric_jacobian(
    mapping: Callable[[FloatPoint], FloatPoint],
    point: FloatPoint,
    step: float = 1e-6,
) -> np.ndarray:
    """Approximate the Jacobian matrix of a local Euclidean map."""
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


def _validate_projection_hyperplanes(
    source_hyperplane,
    target_hyperplane,
) -> None:
    """Require compatible ambient dimensions for projection data."""
    if source_hyperplane.normal.dim != target_hyperplane.normal.dim:
        raise ValueError("Projection hyperplanes must have equal dimension")


def _parallel_projection_inverse(
    source_hyperplane,
    target_hyperplane,
    direction: FloatVector,
) -> Callable[[FloatPoint], FloatPoint]:
    """Return the inverse map of a parallel projection between hyperplanes."""
    direction = FloatVector(direction)
    _validate_projection_hyperplanes(source_hyperplane, target_hyperplane)
    if direction.dim != source_hyperplane.normal.dim:
        raise ValueError("Projection direction dimension mismatch")
    denominator = source_hyperplane.normal.dot(direction)
    if _isclose(denominator):
        raise ValueError(
            "Projection direction must not be parallel to the source hyperplane"
        )

    def inverse_map(point: FloatPoint) -> FloatPoint:
        point = FloatPoint(point)
        scalar = (
            source_hyperplane.offset -
            source_hyperplane.normal.dot(FloatVector(point))
        ) / denominator
        return point + scalar * direction

    return inverse_map


def _central_projection_inverse(
    source_hyperplane,
    target_hyperplane,
    center: FloatPoint,
) -> Callable[[FloatPoint], FloatPoint]:
    """Return the inverse map of a central projection between hyperplanes."""
    center = FloatPoint(center)
    _validate_projection_hyperplanes(source_hyperplane, target_hyperplane)
    if center.dim != source_hyperplane.normal.dim:
        raise ValueError("Projection center dimension mismatch")
    source_offset = source_hyperplane.offset
    source_center_value = source_hyperplane.normal.dot(FloatVector(center))
    if _isclose(source_offset, source_center_value):
        raise ValueError("Projection center must not lie in the source hyperplane")

    def inverse_map(point: FloatPoint) -> FloatPoint:
        point = FloatPoint(point)
        direction = point - center
        denominator = source_hyperplane.normal.dot(direction)
        if _isclose(denominator):
            raise ValueError("Projection ray is parallel to the source hyperplane")
        scalar = (source_offset - source_center_value) / denominator
        return center + scalar * direction

    return inverse_map


def _projected_local_model(
    source_object,
    source_point: FloatPoint,
    target_hyperplane,
    target_point: FloatPoint,
    inverse_map: Callable[[FloatPoint], FloatPoint],
    name: str,
) -> LocalConeModel[FloatPoint]:
    """Transport a local cone model through a Euclidean projection."""
    source_model = source_object.local_model_at(source_point)
    target_model = target_hyperplane.local_model_at(target_point)
    jacobian = _numeric_jacobian(inverse_map, target_point)
    dim = target_point.dim
    cone = EuclideanCone(
        dim,
        contains=lambda coordinates: (
            target_model.cone.contains(coordinates) and
            source_model.cone.contains(
                FloatPoint(jacobian @ _point_array(coordinates))
            )
        ),
        apex=FloatPoint.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
        name=name,
    )
    return LocalConeModel(target_model.chart, cone)


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
        image=EuclideanNeighborhood.whole(chart.dim),
        name=f"{chart.name}-centered" if chart.name else "",
    )


def _contains_linear_image(
    matrix: np.ndarray,
    source_cone: Cone,
    target_coordinates: FloatPoint,
) -> bool:
    """Check whether a vector belongs to the image of a source cone."""
    target_array = _point_array(target_coordinates)
    source_array, residuals, _, _ = np.linalg.lstsq(
        matrix,
        target_array,
        rcond=None,
    )
    if not np.allclose(matrix @ source_array, target_array, atol=1e-7, rtol=1e-7):
        return False
    return source_cone.contains(FloatPoint(source_array))


class EuclideanCone:
    """Concrete cone in Euclidean coordinates.

    The implementation stores an explicit apex, an optional neighborhood, and a
    membership predicate. The cone property itself remains a semantic contract
    of the supplied predicate.
    """

    def __init__(
        self,
        dim: int,
        contains: Callable[[FloatPoint], bool],
        apex: FloatPoint | None = None,
        neighborhood: EuclideanNeighborhood | None = None,
        name: str = "",
    ) -> None:
        """Initialize a Euclidean cone."""
        self._dim = dim
        self._contains = contains
        if apex is None:
            self._apex = FloatPoint.origin(dim)
        else:
            self._apex = FloatPoint(apex)
        if self._apex.dim != dim:
            raise ValueError(
                f"Apex dimension mismatch: {self._apex.dim} != {dim}"
            )
        if neighborhood is not None and neighborhood.dim != dim:
            raise ValueError(
                f"Neighborhood dimension mismatch: {neighborhood.dim} != {dim}"
            )
        self.neighborhood = neighborhood
        self.name = name

    @property
    def dim(self) -> int:
        """Return the cone dimension."""
        return self._dim

    @property
    def apex(self) -> FloatPoint:
        """Return the cone apex."""
        return self._apex

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return (
            "EuclideanCone("
            f"dim={self.dim}, apex={self.apex.to_tuple()}{label})"
        )

    def contains(self, point: FloatPoint) -> bool:
        """Check whether a point belongs to the cone."""
        point = FloatPoint(point)
        if point.dim != self.dim:
            return False
        if self.neighborhood is not None and point not in self.neighborhood:
            return False
        return self._contains(point)

    def __contains__(self, point: FloatPoint) -> bool:
        """Check whether a point belongs to the cone."""
        return self.contains(point)

    @classmethod
    def whole(cls, dim: int) -> "EuclideanCone":
        """Return the whole Euclidean space as a cone."""
        return cls(
            dim,
            contains=lambda point: True,
            apex=FloatPoint.origin(dim),
            neighborhood=EuclideanNeighborhood.whole(dim),
            name="whole",
        )


class RadialCone(EuclideanCone):
    """Cone defined by an admissible set of directions.

    A point belongs to the cone if it is the apex or if the normalized
    direction from the apex to that point satisfies ``contains_direction``.
    This is a concrete approximation of the mathematical idea that a cone is
    determined by an object on the unit sphere.
    """

    def __init__(
        self,
        dim: int,
        contains_direction: Callable[[FloatVector], bool],
        apex: FloatPoint | None = None,
        neighborhood: EuclideanNeighborhood | None = None,
        name: str = "",
    ) -> None:
        """Initialize a radial cone from a direction predicate."""
        self._contains_direction = contains_direction
        if apex is None:
            chosen_apex = FloatPoint.origin(dim)
        else:
            chosen_apex = FloatPoint(apex)

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
            name=name,
        )

    @classmethod
    def whole(cls, dim: int) -> "RadialCone":
        """Return the whole Euclidean space as a radial cone."""
        return cls(
            dim,
            contains_direction=lambda direction: True,
            apex=FloatPoint.origin(dim),
            neighborhood=EuclideanNeighborhood.whole(dim),
            name="whole-radial",
        )


class DirectionSetSphereObject:
    """Concrete object on the unit sphere defined by a direction predicate."""

    def __init__(
        self,
        dim: int,
        contains: Callable[[FloatVector], bool],
        name: str = "",
    ) -> None:
        """Initialize a sphere object from a direction predicate."""
        self._dim = dim
        self._contains = contains
        self.name = name

    @property
    def dim(self) -> int:
        """Return the ambient dimension."""
        return self._dim

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"DirectionSetSphereObject(dim={self.dim}{label})"

    def contains(self, direction: FloatVector) -> bool:
        """Check whether a unit direction belongs to the sphere object."""
        direction = FloatVector(direction)
        if direction.dim != self.dim:
            return False
        if direction.norm() == 0.0:
            return False
        normalized = direction / direction.norm()
        return self._contains(normalized)

    def __contains__(self, direction: FloatVector) -> bool:
        """Check whether a unit direction belongs to the sphere object."""
        return self.contains(direction)


class CircleSphereObject:
    """Sphere object in dimension 2 defined by a ``FloatCircleSet``.

    This identifies the unit sphere in ``R^2`` with the unit circle already
    implemented by the package.
    """

    def __init__(self, circle_set: FloatCircleSet, name: str = "") -> None:
        """Initialize the sphere object from a circle subset."""
        self.circle_set = FloatCircleSet(circle_set)
        self.name = name

    @property
    def dim(self) -> int:
        """Return the ambient Euclidean dimension."""
        return 2

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"CircleSphereObject({self.circle_set!r}{label})"

    def contains(self, direction: FloatVector) -> bool:
        """Check whether a direction belongs to the spherical object."""
        direction = FloatVector(direction)
        if direction.dim != 2:
            return False
        if direction.norm() == 0.0:
            return False
        point = FloatCirclePoint.from_cartesian(direction[0], direction[1])
        return point in self.circle_set

    def __contains__(self, direction: FloatVector) -> bool:
        """Check whether a direction belongs to the spherical object."""
        return self.contains(direction)


class SphericalCone(RadialCone):
    """Cone induced by an explicit object on the unit sphere."""

    def __init__(
        self,
        sphere_object: SphereObject,
        apex: FloatPoint | None = None,
        neighborhood: EuclideanNeighborhood | None = None,
        name: str = "",
    ) -> None:
        """Initialize a cone from a spherical base object."""
        self.sphere_object = sphere_object
        super().__init__(
            sphere_object.dim,
            contains_direction=sphere_object.contains,
            apex=apex,
            neighborhood=neighborhood,
            name=name,
        )


@dataclass(frozen=True)
class LocalConeModel(Generic[PointT]):
    """Local model of a geometric object by a cone in chart coordinates."""

    chart: ManifoldChart[PointT]
    cone: Cone

    def __post_init__(self) -> None:
        """Require matching dimensions."""
        if self.chart.dim != self.cone.dim:
            raise ValueError(
                "Chart/cone dimension mismatch: "
                f"{self.chart.dim} != {self.cone.dim}"
            )


class ChartedGeometricObject(Generic[PointT]):
    """Geometric object with local cone models on an ambient manifold."""

    def __init__(
        self,
        manifold: Manifold[PointT],
        contains: Callable[[PointT], bool],
        local_model: Callable[[PointT], LocalConeModel[PointT]],
        name: str = "",
    ) -> None:
        """Initialize a geometric object."""
        self.manifold = manifold
        self._contains = contains
        self._local_model = local_model
        self.name = name

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"ChartedGeometricObject(dim={self.manifold.dim}{label})"

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the object."""
        return point in self.manifold and self._contains(point)

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the object."""
        return self.contains(point)

    def local_model_at(self, point: PointT) -> LocalConeModel[PointT]:
        """Return a local cone model at a point of the object."""
        if point not in self:
            raise ValueError("Point is outside the geometric object")
        model = self._local_model(point)
        if model.chart.dim != self.manifold.dim:
            raise ValueError(
                "Local model chart dimension does not match manifold dimension"
            )
        return model

    def image_under_smooth_map(
        self,
        forward: Callable[[PointT], TargetT],
        preimage_on_image: Callable[[TargetT], PointT],
        target_manifold: Manifold[TargetT],
        target_chart: Callable[[TargetT], ManifoldChart[TargetT]],
        contains_image_point: Callable[[TargetT], bool] | None = None,
        name: str = "",
    ) -> "SmoothImageObject[PointT, TargetT]":
        """Return the image object under a smooth map with local inverse."""
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
        """Project an Euclidean object along a direction onto a hyperplane."""
        ambient_manifold = getattr(self.manifold, "manifold", self.manifold)
        if not isinstance(ambient_manifold, EuclideanSpace):
            raise ValueError("Projection is only implemented in Euclidean spaces")
        inverse_map = _parallel_projection_inverse(
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
            return _projected_local_model(
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
        """Project an Euclidean object from a center point onto a hyperplane."""
        ambient_manifold = getattr(self.manifold, "manifold", self.manifold)
        if not isinstance(ambient_manifold, EuclideanSpace):
            raise ValueError("Projection is only implemented in Euclidean spaces")
        inverse_map = _central_projection_inverse(
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
            return _projected_local_model(
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
        """Initialize an image object under a smooth map."""
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
                point in target_manifold and
                image_contains(point) and
                self.preimage_on_image(point) in self.source_object
            ),
            local_model=self._local_model,
            name=name,
        )

    def _local_model(self, point: TargetT) -> LocalConeModel[TargetT]:
        """Transport the source local model through the smooth map."""
        source_point = self.preimage_on_image(point)
        source_model = self.source_object.local_model_at(source_point)
        centered_source_chart = _centered_chart_at(source_model.chart, source_point)
        base_target_chart = self.target_chart(point)
        centered_target_chart = _centered_chart_at(base_target_chart, point)

        def local_forward(source_coordinates: FloatPoint) -> FloatPoint:
            source_local_point = centered_source_chart.inverse(source_coordinates)
            return centered_target_chart(self.forward(source_local_point))

        jacobian = _numeric_jacobian(
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
            neighborhood=EuclideanNeighborhood.whole(centered_target_chart.dim),
            name="smooth-image",
        )
        return LocalConeModel(centered_target_chart, cone)


class RealPointObject(ChartedGeometricObject[float]):
    """Zero-dimensional object supported at one point of the real line."""

    def __init__(self, point: float, name: str = "") -> None:
        """Initialize the singleton real point."""
        self.point = float(point)
        manifold = RealLine()
        super().__init__(
            manifold,
            contains=lambda candidate: float(candidate) == self.point,
            local_model=lambda candidate: LocalConeModel(
                _real_chart(self.point),
                _point_cone(1),
            ),
            name=name,
        )


class CirclePointObject(ChartedGeometricObject[FloatCirclePoint]):
    """Zero-dimensional object supported at one point of the circle."""

    def __init__(self, point: FloatCirclePoint, name: str = "") -> None:
        """Initialize the singleton circle point."""
        self.point = FloatCirclePoint(point)
        manifold = Circle()
        super().__init__(
            manifold,
            contains=lambda candidate: (
                FloatCirclePoint(candidate) == self.point
            ),
            local_model=lambda candidate: LocalConeModel(
                _circle_chart(self.point),
                _point_cone(1),
            ),
            name=name,
        )


class EuclideanPointObject(ChartedGeometricObject[FloatPoint]):
    """Zero-dimensional object supported at one Euclidean point."""

    def __init__(self, point: FloatPoint, name: str = "") -> None:
        """Initialize the singleton Euclidean point."""
        self.point = FloatPoint(point)
        manifold = EuclideanSpace(self.point.dim)
        super().__init__(
            manifold,
            contains=lambda candidate: FloatPoint(candidate) == self.point,
            local_model=lambda candidate: LocalConeModel(
                _euclidean_chart(self.point),
                _point_cone(self.point.dim),
            ),
            name=name,
        )


class RealSetObject(ChartedGeometricObject[float]):
    """One-dimensional object on the real line built from ``FloatSet``."""

    def __init__(self, *point_set: object, name: str = "") -> None:
        """Initialize the real-line object from a float set."""
        self.point_set = FloatSet(*_coerce_real_set_arguments(point_set))
        manifold = RealLine()
        super().__init__(
            manifold,
            contains=lambda point: float(point) in self.point_set,
            local_model=self._local_model,
            name=name,
        )

    def _local_model(self, point: float) -> LocalConeModel[float]:
        """Build the local cone model at a real-line point."""
        point = float(point)
        previous = math.nextafter(point, -math.inf)
        following = math.nextafter(point, math.inf)
        left_in = previous in self.point_set
        right_in = following in self.point_set

        if left_in and right_in:
            cone = EuclideanCone.whole(1)
        elif right_in:
            cone = _positive_half_line_cone()
        elif left_in:
            cone = _negative_half_line_cone()
        else:
            cone = _point_cone(1)

        return LocalConeModel(_real_chart(point), cone)


class CircleSetObject(ChartedGeometricObject[FloatCirclePoint]):
    """One-dimensional object on the unit circle from ``FloatCircleSet``."""

    def __init__(self, *point_set: object, name: str = "") -> None:
        """Initialize the circle object from a circle set."""
        self.point_set = FloatCircleSet(*point_set)
        manifold = Circle()
        super().__init__(
            manifold,
            contains=lambda point: FloatCirclePoint(point) in self.point_set,
            local_model=self._local_model,
            name=name,
        )

    def _local_model(
        self,
        point: FloatCirclePoint,
    ) -> LocalConeModel[FloatCirclePoint]:
        """Build the local cone model at a circle point."""
        point = FloatCirclePoint(point)
        if float(point) == 0.0:
            previous = FloatCirclePoint(FloatAngle.MAX_ANGLE)
        else:
            previous = FloatCirclePoint(
                math.nextafter(float(point), -math.inf)
            )
        following = FloatCirclePoint(math.nextafter(float(point), math.inf))
        left_in = previous in self.point_set
        right_in = following in self.point_set

        if left_in and right_in:
            cone = EuclideanCone.whole(1)
        elif right_in:
            cone = _positive_half_line_cone()
        elif left_in:
            cone = _negative_half_line_cone()
        else:
            cone = _point_cone(1)

        return LocalConeModel(_circle_chart(point), cone)


class WholeSpace(ChartedGeometricObject[FloatPoint]):
    """The whole Euclidean space ``R^n``."""

    def __init__(self, dim: int, name: str = "") -> None:
        """Initialize the full Euclidean space."""
        self.dim = dim
        manifold = EuclideanSpace(dim)
        super().__init__(
            manifold,
            contains=lambda point: True,
            local_model=lambda point: LocalConeModel(
                _euclidean_chart(FloatPoint(point)),
                EuclideanCone.whole(dim),
            ),
            name=name,
        )


class Hyperplane(ChartedGeometricObject[FloatPoint]):
    """Affine hyperplane ``{x : <normal, x> = offset}``."""

    def __init__(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> None:
        """Initialize the hyperplane from a normal and offset."""
        self.normal = _coerce_nonzero_normal(normal)
        self.offset = float(offset)
        manifold = EuclideanSpace(self.normal.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a point belongs to the hyperplane."""
        point = FloatPoint(point)
        value = FloatVector(point).dot(self.normal)
        return _isclose(value, self.offset)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the local cone model at a hyperplane point."""
        point = FloatPoint(point)
        cone = _hyperplane_cone(self.normal, name="hyperplane")
        return LocalConeModel(_euclidean_chart(point), cone)


class HalfSpace(ChartedGeometricObject[FloatPoint]):
    """Closed half-space ``{x : <normal, x> >= offset}``."""

    def __init__(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> None:
        """Initialize the half-space from a normal and offset."""
        self.normal = _coerce_nonzero_normal(normal)
        self.offset = float(offset)
        manifold = EuclideanSpace(self.normal.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a point belongs to the half-space."""
        point = FloatPoint(point)
        value = FloatVector(point).dot(self.normal)
        return value >= self.offset or _isclose(value, self.offset)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the local cone model at a half-space point."""
        point = FloatPoint(point)
        value = FloatVector(point).dot(self.normal) - self.offset
        if value > 0.0 and not _isclose(value):
            cone = EuclideanCone.whole(self.normal.dim)
        else:
            cone = _half_space_cone(
                self.normal,
                name="half-space-boundary",
            )
        return LocalConeModel(_euclidean_chart(point), cone)


class Sphere(ChartedGeometricObject[FloatPoint]):
    """Euclidean sphere surface ``{x : ||x - c|| = r}``."""

    def __init__(
        self,
        center: FloatPoint,
        radius: float,
        name: str = "",
    ) -> None:
        """Initialize the sphere from its center and radius."""
        self.center = FloatPoint(center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Sphere radius must be positive")
        manifold = EuclideanSpace(self.center.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _radial_vector(self, point: FloatPoint) -> FloatVector:
        """Return the vector from the center to the point."""
        return FloatPoint(point) - self.center

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a point lies on the sphere."""
        return _isclose(self._radial_vector(point).norm(), self.radius)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the tangent hyperplane cone on the sphere."""
        normal = self._radial_vector(point)
        cone = _hyperplane_cone(normal, name="sphere-tangent")
        return LocalConeModel(_euclidean_chart(FloatPoint(point)), cone)


class Ball(ChartedGeometricObject[FloatPoint]):
    """Closed Euclidean ball ``{x : ||x - c|| <= r}``."""

    def __init__(
        self,
        center: FloatPoint,
        radius: float,
        name: str = "",
    ) -> None:
        """Initialize the ball from its center and radius."""
        self.center = FloatPoint(center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Ball radius must be positive")
        manifold = EuclideanSpace(self.center.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _radial_vector(self, point: FloatPoint) -> FloatVector:
        """Return the vector from the center to the point."""
        return FloatPoint(point) - self.center

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a point belongs to the ball."""
        norm = self._radial_vector(point).norm()
        return norm <= self.radius or _isclose(norm, self.radius)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the local cone model at a ball point."""
        point = FloatPoint(point)
        normal = self._radial_vector(point)
        if normal.norm() < self.radius and not _isclose(normal.norm(), self.radius):
            cone = EuclideanCone.whole(self.center.dim)
        else:
            cone = _half_space_cone(
                -normal,
                name="ball-boundary",
            )
        return LocalConeModel(_euclidean_chart(point), cone)


class EllipsoidSurface(ChartedGeometricObject[FloatPoint]):
    """Affine image of the unit sphere."""

    def __init__(
        self,
        center: FloatPoint,
        semiaxes: Sequence[Sequence[float]],
        name: str = "",
    ) -> None:
        """Initialize the ellipsoid surface from center and axes."""
        self.center = FloatPoint(center)
        self.dim = self.center.dim
        self.matrix = _coerce_affine_matrix(semiaxes, self.dim)
        self.inverse_matrix = np.linalg.inv(self.matrix)
        manifold = EuclideanSpace(self.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _local_coordinates(self, point: FloatPoint) -> np.ndarray:
        """Return affine unit-ball coordinates of a point."""
        displacement = _point_array(FloatPoint(point) - self.center)
        return self.inverse_matrix @ displacement

    def _normal(self, point: FloatPoint) -> FloatVector:
        """Return the outward normal covector at a boundary point."""
        local = self._local_coordinates(point)
        normal = self.inverse_matrix.T @ local
        return FloatVector(normal)

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a point lies on the ellipsoid surface."""
        local = self._local_coordinates(point)
        value = float(local @ local)
        return _isclose(value, 1.0)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the tangent hyperplane cone on the ellipsoid."""
        normal = self._normal(point)
        cone = _hyperplane_cone(normal, name="ellipsoid-tangent")
        return LocalConeModel(_euclidean_chart(FloatPoint(point)), cone)


class Ellipsoid(ChartedGeometricObject[FloatPoint]):
    """Affine image of the closed unit ball."""

    def __init__(
        self,
        center: FloatPoint,
        semiaxes: Sequence[Sequence[float]],
        name: str = "",
    ) -> None:
        """Initialize the ellipsoid from center and axes."""
        self.surface = EllipsoidSurface(center, semiaxes, name=name)
        manifold = EuclideanSpace(self.surface.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a point belongs to the ellipsoid."""
        local = self.surface._local_coordinates(point)
        value = float(local @ local)
        return value <= 1.0 or _isclose(value, 1.0)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the local cone model at an ellipsoid point."""
        point = FloatPoint(point)
        local = self.surface._local_coordinates(point)
        value = float(local @ local)
        if value < 1.0 and not _isclose(value, 1.0):
            cone = EuclideanCone.whole(self.surface.dim)
        else:
            cone = _half_space_cone(
                -self.surface._normal(point),
                name="ellipsoid-boundary",
            )
        return LocalConeModel(_euclidean_chart(point), cone)


class ParallelepipedSurface(ChartedGeometricObject[FloatPoint]):
    """Affine image of the boundary of the unit cube."""

    def __init__(
        self,
        center: FloatPoint,
        spanning_vectors: Sequence[Sequence[float]],
        name: str = "",
    ) -> None:
        """Initialize the parallelepiped surface."""
        self.center = FloatPoint(center)
        self.dim = self.center.dim
        self.matrix = _coerce_affine_matrix(spanning_vectors, self.dim)
        self.inverse_matrix = np.linalg.inv(self.matrix)
        manifold = EuclideanSpace(self.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _local_coordinates(self, point: FloatPoint) -> np.ndarray:
        """Return local cube coordinates of a point."""
        displacement = _point_array(FloatPoint(point) - self.center)
        return self.inverse_matrix @ displacement

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a point lies on the parallelepiped surface."""
        local = self._local_coordinates(point)
        max_abs = float(np.max(np.abs(local)))
        return (
            max_abs <= 1.0 or _isclose(max_abs, 1.0)
        ) and _isclose(max_abs, 1.0)

    def _surface_cone(
        self,
        local_point: np.ndarray,
    ) -> EuclideanCone:
        """Return the tangent cone to the boundary of the unit cube."""
        active = _active_box_constraints(local_point)
        if not active:
            return EuclideanCone.whole(self.dim)

        def contains(coordinates: FloatPoint) -> bool:
            local_disp = self.inverse_matrix @ _point_array(coordinates)
            values = [
                sign * float(local_disp[index])
                for index, sign in active
            ]
            return all(value <= 0.0 or _isclose(value) for value in values) and any(
                _isclose(value) for value in values
            )

        return EuclideanCone(
            self.dim,
            contains=contains,
            apex=FloatPoint.origin(self.dim),
            neighborhood=EuclideanNeighborhood.whole(self.dim),
            name="parallelepiped-surface",
        )

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the local cone model on the parallelepiped surface."""
        point = FloatPoint(point)
        local = self._local_coordinates(point)
        cone = self._surface_cone(local)
        return LocalConeModel(_euclidean_chart(point), cone)


class Parallelepiped(ChartedGeometricObject[FloatPoint]):
    """Affine image of the closed unit cube."""

    def __init__(
        self,
        center: FloatPoint,
        spanning_vectors: Sequence[Sequence[float]],
        name: str = "",
    ) -> None:
        """Initialize the parallelepiped."""
        self.surface = ParallelepipedSurface(
            center,
            spanning_vectors,
            name=name,
        )
        manifold = EuclideanSpace(self.surface.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a point belongs to the parallelepiped."""
        local = self.surface._local_coordinates(point)
        max_abs = float(np.max(np.abs(local)))
        return max_abs <= 1.0 or _isclose(max_abs, 1.0)

    def _solid_cone(self, local_point: np.ndarray) -> EuclideanCone:
        """Return the tangent cone to the solid unit cube."""
        active = _active_box_constraints(local_point)
        if not active:
            return EuclideanCone.whole(self.surface.dim)

        def contains(coordinates: FloatPoint) -> bool:
            local_disp = self.surface.inverse_matrix @ _point_array(coordinates)
            values = [
                sign * float(local_disp[index])
                for index, sign in active
            ]
            return all(value <= 0.0 or _isclose(value) for value in values)

        return EuclideanCone(
            self.surface.dim,
            contains=contains,
            apex=FloatPoint.origin(self.surface.dim),
            neighborhood=EuclideanNeighborhood.whole(self.surface.dim),
            name="parallelepiped-boundary",
        )

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the local cone model at a parallelepiped point."""
        point = FloatPoint(point)
        local = self.surface._local_coordinates(point)
        max_abs = float(np.max(np.abs(local)))
        if max_abs < 1.0 and not _isclose(max_abs, 1.0):
            cone = EuclideanCone.whole(self.surface.dim)
        else:
            cone = self._solid_cone(local)
        return LocalConeModel(_euclidean_chart(point), cone)


class CubeSurface(ParallelepipedSurface):
    """Boundary of an axis-aligned cube centered at a point."""

    def __init__(
        self,
        center: FloatPoint,
        half_extent: float,
        name: str = "",
    ) -> None:
        """Initialize the cube surface."""
        center = FloatPoint(center)
        half_extent = float(half_extent)
        if half_extent <= 0.0:
            raise ValueError("Cube half-extent must be positive")
        dim = center.dim
        spanning_vectors = tuple(
            tuple(
                half_extent if i == j else 0.0
                for j in range(dim)
            )
            for i in range(dim)
        )
        super().__init__(center, spanning_vectors, name=name)


class Cube(Parallelepiped):
    """Closed axis-aligned cube centered at a point."""

    def __init__(
        self,
        center: FloatPoint,
        half_extent: float,
        name: str = "",
    ) -> None:
        """Initialize the cube."""
        center = FloatPoint(center)
        half_extent = float(half_extent)
        if half_extent <= 0.0:
            raise ValueError("Cube half-extent must be positive")
        dim = center.dim
        spanning_vectors = tuple(
            tuple(
                half_extent if i == j else 0.0
                for j in range(dim)
            )
            for i in range(dim)
        )
        super().__init__(center, spanning_vectors, name=name)


class WholePlane(WholeSpace):
    """The whole Euclidean plane."""

    def __init__(self, name: str = "") -> None:
        """Initialize the full plane."""
        super().__init__(2, name=name)


class HalfPlane(HalfSpace):
    """Closed half-plane ``{x : <normal, x> >= offset}``."""

    def __init__(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> None:
        """Initialize the half-plane from a normal and offset."""
        normal = FloatVector(normal)
        if normal.dim != 2:
            raise ValueError("HalfPlane is only defined in dimension 2")
        super().__init__(normal, offset=offset, name=name)


class PlanarAngle(ChartedGeometricObject[FloatPoint]):
    """Closed planar angle with interior and apex in ``R^2``."""

    def __init__(
        self,
        apex: FloatPoint,
        start: FloatCirclePoint,
        end: FloatCirclePoint,
        name: str = "",
    ) -> None:
        """Initialize the planar angle from its apex and boundary rays."""
        self.apex = FloatPoint(apex)
        if self.apex.dim != 2:
            raise ValueError("PlanarAngle apex must be two-dimensional")
        self.interval = FloatCircleInterval(start, end)
        if self.interval.is_point():
            raise ValueError("PlanarAngle must have a non-zero opening")
        self.direction_set = FloatCircleSet(self.interval)
        manifold = EuclideanSpace(2)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a plane point belongs to the angle."""
        point = FloatPoint(point)
        displacement = point - self.apex
        if displacement.norm() == 0.0:
            return True
        direction = FloatCirclePoint.from_cartesian(
            displacement[0],
            displacement[1],
        )
        return direction in self.direction_set

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the local cone model at an angle point."""
        point = FloatPoint(point)
        chart = _euclidean_chart(point)

        if point == self.apex:
            if self.direction_set.is_full():
                cone = EuclideanCone.whole(2)
            else:
                cone = SphericalCone(
                    CircleSphereObject(self.direction_set),
                    neighborhood=EuclideanNeighborhood.whole(2),
                    name="angle-apex",
                )
            return LocalConeModel(chart, cone)

        if self.direction_set.is_full():
            return LocalConeModel(chart, EuclideanCone.whole(2))

        displacement = point - self.apex
        direction = FloatCirclePoint.from_cartesian(
            displacement[0],
            displacement[1],
        )

        if direction == FloatCirclePoint(self.interval.start):
            boundary = FloatVector(
                math.cos(self.interval.start),
                math.sin(self.interval.start),
            )
            cone = EuclideanCone(
                2,
                contains=lambda coordinates: (
                    _cross_2d(boundary, FloatVector(coordinates)) >= 0.0
                ),
                apex=FloatPoint.origin(2),
                neighborhood=EuclideanNeighborhood.whole(2),
                name="angle-start-boundary",
            )
            return LocalConeModel(chart, cone)

        if direction == FloatCirclePoint(self.interval.end):
            boundary = FloatVector(
                math.cos(self.interval.end),
                math.sin(self.interval.end),
            )
            cone = EuclideanCone(
                2,
                contains=lambda coordinates: (
                    _cross_2d(boundary, FloatVector(coordinates)) <= 0.0
                ),
                apex=FloatPoint.origin(2),
                neighborhood=EuclideanNeighborhood.whole(2),
                name="angle-end-boundary",
            )
            return LocalConeModel(chart, cone)

        return LocalConeModel(chart, EuclideanCone.whole(2))


__all__ = [
    "RealLine",
    "Circle",
    "EuclideanSpace",
    "WholeSpace",
    "Hyperplane",
    "HalfSpace",
    "Sphere",
    "Ball",
    "EllipsoidSurface",
    "Ellipsoid",
    "ParallelepipedSurface",
    "Parallelepiped",
    "CubeSurface",
    "Cube",
    "Cone",
    "SphereObject",
    "GeometricObject",
    "DirectionSetSphereObject",
    "CircleSphereObject",
    "EuclideanCone",
    "RadialCone",
    "SphericalCone",
    "LocalConeModel",
    "ChartedGeometricObject",
    "SmoothImageObject",
    "RealPointObject",
    "CirclePointObject",
    "EuclideanPointObject",
    "RealSetObject",
    "CircleSetObject",
    "WholePlane",
    "HalfPlane",
    "PlanarAngle",
]
