"""Geometric objects and local cone models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

from .euclidean import EuclideanNeighborhood, FloatPoint, FloatVector
from .floatcircle import FloatCirclePoint, FloatCircleSet
from .manifold import LocalPointT, Manifold, ManifoldChart


PointT = TypeVar("PointT")


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
        self._apex = FloatPoint.origin(dim) if apex is None else FloatPoint(apex)
        if self._apex.dim != dim:
            raise ValueError(f"Apex dimension mismatch: {self._apex.dim} != {dim}")
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
        return f"EuclideanCone(dim={self.dim}, apex={self.apex.to_tuple()}{label})"

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
                f"Chart/cone dimension mismatch: {self.chart.dim} != {self.cone.dim}"
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


__all__ = [
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
]
