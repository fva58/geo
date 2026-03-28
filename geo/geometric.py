"""Geometric objects and local cone models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

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


class WholePlane(ChartedGeometricObject[FloatPoint]):
    """The whole Euclidean plane."""

    def __init__(self, name: str = "") -> None:
        """Initialize the full plane."""
        manifold = EuclideanSpace(2)
        super().__init__(
            manifold,
            contains=lambda point: True,
            local_model=lambda point: LocalConeModel(
                _euclidean_chart(FloatPoint(point)),
                EuclideanCone.whole(2),
            ),
            name=name,
        )


class HalfPlane(ChartedGeometricObject[FloatPoint]):
    """Closed half-plane ``{x : <normal, x> >= offset}``."""

    def __init__(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> None:
        """Initialize the half-plane from a normal and offset."""
        self.normal = FloatVector(normal)
        if self.normal.dim != 2:
            raise ValueError("HalfPlane is only defined in dimension 2")
        if self.normal.norm() == 0.0:
            raise ValueError("HalfPlane normal must be non-zero")
        self.offset = float(offset)
        manifold = EuclideanSpace(2)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        """Check whether a plane point belongs to the half-plane."""
        point = FloatPoint(point)
        return FloatVector(point).dot(self.normal) >= self.offset

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        """Build the local cone model at a half-plane point."""
        point = FloatPoint(point)
        value = FloatVector(point).dot(self.normal) - self.offset
        if value > 0.0:
            cone = EuclideanCone.whole(2)
        else:
            cone = EuclideanCone(
                2,
                contains=lambda coordinates: (
                    FloatVector(coordinates).dot(self.normal) >= 0.0
                ),
                apex=FloatPoint.origin(2),
                neighborhood=EuclideanNeighborhood.whole(2),
                name="half-plane-boundary",
            )
        return LocalConeModel(_euclidean_chart(point), cone)


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
    "RealPointObject",
    "CirclePointObject",
    "EuclideanPointObject",
    "RealSetObject",
    "CircleSetObject",
    "WholePlane",
    "HalfPlane",
    "PlanarAngle",
]
