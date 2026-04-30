"""General space protocols and standard visualizable spaces."""

from __future__ import annotations

import math
import itertools
from collections.abc import Sequence
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

from .euclidean import EuclideanNeighborhood, FloatPoint, FloatVector
from .floatcircle import (
    FloatAngle,
    FloatCirclePoint,
    FloatCircleSet,
)
from .geometric import EuclideanCone, LocalConeModel
from .manifold import ManifoldChart
from .manifold import ChartNeighborhood
from .manifold import refine_neighborhoods as _refine_neighborhoods
from .riemannian import MetricGeometricObject, MetricSpace


PointT = TypeVar("PointT")
Embedding2D = tuple[float, float]
Embedding3D = tuple[float, float, float]
_MISSING = object()


def _coerce_point3(value: object) -> FloatPoint:
    """Return a three-dimensional embedded point."""
    point = FloatPoint(value)
    if point.dim != 3:
        raise ValueError("Expected a three-dimensional point")
    return point


def _coerce_point_dim(value: object, dim: int) -> FloatPoint:
    """Return an embedded point of the requested dimension."""
    point = FloatPoint(value)
    if point.dim != dim:
        raise ValueError(f"Expected a {dim}-dimensional point")
    return point


def _coerce_nonzero_point_dim(value: object, dim: int) -> FloatPoint:
    """Return a nonzero embedded point of the requested dimension."""
    point = _coerce_point_dim(value, dim)
    if math.isclose(FloatVector(point).norm(), 0.0, abs_tol=1e-15):
        raise ValueError("Expected a nonzero vector")
    return point


def _clamp_unit(value: float) -> float:
    """Clamp a value into ``[-1, 1]`` for inverse trigonometric formulas."""
    return max(-1.0, min(1.0, value))


def _point_cone(dim: int) -> EuclideanCone:
    """Return the zero cone at the origin."""
    return EuclideanCone(
        dim,
        contains=lambda point: FloatPoint(point) == FloatPoint.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
        name="point",
    )


def _previous_circle_point(point: FloatCirclePoint) -> FloatCirclePoint:
    """Return the previous representable point on the circle."""
    if float(point) == 0.0:
        return FloatCirclePoint(FloatAngle.MAX_ANGLE)
    return FloatCirclePoint(math.nextafter(float(point), -math.inf))


def _following_circle_point(point: FloatCirclePoint) -> FloatCirclePoint:
    """Return the next representable point on the circle."""
    following = math.nextafter(float(point), math.inf)
    if following >= FloatAngle.TWO_PI:
        following = 0.0
    return FloatCirclePoint(following)


def _signed_circle_difference(
    target: FloatCirclePoint,
    base: FloatCirclePoint,
) -> float:
    """Return the signed shortest angular difference."""
    difference = float(FloatCirclePoint(target)) - float(FloatCirclePoint(base))
    if difference > math.pi:
        difference -= FloatAngle.TWO_PI
    elif difference <= -math.pi:
        difference += FloatAngle.TWO_PI
    return difference


def _normalize_vector(vector: FloatPoint | FloatVector) -> FloatVector:
    """Return the normalized vector."""
    array = FloatVector(vector)
    norm = array.norm()
    if math.isclose(norm, 0.0, abs_tol=1e-15):
        raise ValueError("Zero vector cannot be normalized")
    return array / norm


def _cross(left: FloatPoint | FloatVector,
           right: FloatPoint | FloatVector) -> FloatVector:
    """Return the 3D cross product."""
    left = FloatVector(left)
    right = FloatVector(right)
    return FloatVector(
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _orthonormal_basis(vectors: Sequence[FloatVector]) -> tuple[FloatVector, ...]:
    """Return an orthonormal basis from spanning vectors."""
    basis: list[FloatVector] = []
    for vector in vectors:
        candidate = FloatVector(vector)
        for basis_vector in basis:
            candidate = candidate - candidate.dot(basis_vector) * basis_vector
        if candidate.norm() > 1e-12:
            basis.append(_normalize_vector(candidate))
    return tuple(basis)


def _sphere_tangent_basis(point: FloatPoint) -> tuple[FloatVector, ...]:
    """Return an orthonormal tangent basis at a sphere point."""
    point = FloatPoint(point)
    ambient_dim = point.dim
    normal = _normalize_vector(point)
    candidates = []
    for axis in range(ambient_dim):
        basis_vector = FloatVector([
            1.0 if index == axis else 0.0
            for index in range(ambient_dim)
        ])
        tangent_candidate = basis_vector - basis_vector.dot(normal) * normal
        if tangent_candidate.norm() > 1e-12:
            candidates.append(tangent_candidate)
    basis = _orthonormal_basis(candidates)
    if len(basis) != ambient_dim - 1:
        raise ValueError("Could not build a full tangent basis")
    return basis


def _sphere_chart(base_point: FloatPoint) -> ManifoldChart[FloatPoint]:
    """Return a gnomonic chart centered at a sphere point."""
    base_point = FloatPoint(base_point)
    if base_point.dim < 2:
        raise ValueError("Sphere ambient dimension must be at least two")
    radius = math.sqrt(sum(coordinate * coordinate for coordinate in base_point))
    normal = _normalize_vector(base_point)
    basis = _sphere_tangent_basis(base_point)
    dim = base_point.dim - 1

    def forward(point: FloatPoint) -> FloatPoint:
        point = _coerce_point_dim(point, base_point.dim)
        denominator = normal.dot(FloatVector(point))
        if math.isclose(denominator, 0.0, abs_tol=1e-12):
            raise ValueError("Point is outside the local sphere chart")
        return FloatPoint([
            radius * FloatVector(point).dot(basis_vector) / denominator
            for basis_vector in basis
        ])

    def inverse(coordinates: FloatPoint) -> FloatPoint:
        coordinates = FloatPoint(coordinates)
        if coordinates.dim != dim:
            raise ValueError(
                f"Sphere chart coordinates must be {dim}-dimensional"
            )
        candidate = FloatVector(base_point)
        for coordinate, basis_vector in zip(coordinates, basis):
            candidate = candidate + coordinate * basis_vector
        normalized = _normalize_vector(candidate)
        return FloatPoint([radius * coordinate for coordinate in normalized])

    return ManifoldChart(
        forward,
        inverse,
        dim=dim,
        name="sphere-gnomonic",
    )


def _torus_chart(base_point: "TorusPoint") -> ManifoldChart["TorusPoint"]:
    """Return a local angular chart centered at a torus point."""
    base_point = TorusPoint(base_point)
    dim = base_point.dim

    def forward(point: TorusPoint) -> FloatPoint:
        point = TorusPoint(point)
        return FloatPoint([
            _signed_circle_difference(point[index], base_point[index])
            for index in range(dim)
        ])

    def inverse(coordinates: FloatPoint) -> TorusPoint:
        coordinates = FloatPoint(coordinates)
        if coordinates.dim != dim:
            raise ValueError(f"Torus chart coordinates must be {dim}-dimensional")
        return TorusPoint(
            tuple(
                float(base_point[index]) + coordinates[index]
                for index in range(dim)
            )
        )

    return ManifoldChart(
        forward,
        inverse,
        dim=dim,
        name="torus-angle",
    )


def _product_cone(axis_flags: Sequence[tuple[bool, bool]], name: str) -> EuclideanCone:
    """Return the product cone from axis-wise membership flags."""

    def axis_ok(value: float, left_in: bool, right_in: bool) -> bool:
        if left_in and right_in:
            return True
        if right_in:
            return value >= 0.0
        if left_in:
            return value <= 0.0
        return math.isclose(value, 0.0, abs_tol=1e-12)

    dim = len(axis_flags)
    return EuclideanCone(
        dim,
        contains=lambda point: all(
            axis_ok(FloatPoint(point)[index], left_in, right_in)
            for index, (left_in, right_in) in enumerate(axis_flags)
        ),
        neighborhood=EuclideanNeighborhood.whole(dim),
        name=name,
    )


def _torus_embedding(point: "TorusPoint", radii: Sequence[float]) -> FloatPoint:
    """Return the standard nested torus embedding into ``R^(n+1)``."""
    point = TorusPoint(point)
    if len(radii) != point.dim:
        raise ValueError("Torus radii dimension does not match point dimension")
    radius = float(radii[-1])
    coordinates = [0.0] * (point.dim + 1)
    coordinates[-1] = radius * math.sin(float(point[-1]))
    radius = float(radii[-2]) + radius * math.cos(float(point[-1])) if point.dim > 1 else radius
    for axis in range(point.dim - 2, 0, -1):
        coordinates[axis + 1] = radius * math.sin(float(point[axis]))
        radius = float(radii[axis - 1]) + radius * math.cos(float(point[axis]))
    if point.dim == 1:
        coordinates[0] = radius * math.cos(float(point[0]))
        coordinates[1] = radius * math.sin(float(point[0]))
    else:
        coordinates[0] = radius * math.cos(float(point[0]))
        coordinates[1] = radius * math.sin(float(point[0]))
    return FloatPoint(coordinates)


@runtime_checkable
class Space(MetricSpace[PointT], Protocol[PointT]):
    """Protocol for spaces with metric and visualization embeddings.

    Visualization methods are not required to preserve the metric. Their role
    is only to provide deterministic coordinate images in 2D or 3D.
    """

    @property
    def space_kind(self) -> str:
        """Return a short identifier for the kind of space."""

    def to_2d(
        self,
        point: PointT,
        method: str = "default",
    ) -> Embedding2D:
        """Return a 2D visualization embedding of a point."""

    def to_3d(
        self,
        point: PointT,
        method: str = "default",
    ) -> Embedding3D:
        """Return a 3D visualization embedding of a point."""

    def full_cover(self, radius: float):
        """Return a full cover of the space by neighborhoods."""

    def refine_cover(
        self,
        neighborhoods,
        factor: int = 2,
    ):
        """Return a covering refinement with smaller diameters."""


class SpherePoint(FloatPoint):
    """Point on a sphere represented by a nonzero ambient vector.

    The constructor accepts any nonzero vector in ``R^(n+1)`` and normalizes
    it to the sphere of the requested radius.
    """

    __slots__ = ()

    def __new__(
        cls,
        *coordinates: object,
        dim: int | None = None,
        radius: float = 1.0,
    ) -> "SpherePoint":
        """Create a sphere point from a nonzero ambient vector."""
        if len(coordinates) == 1:
            vector = _coerce_nonzero_point_dim(
                coordinates[0],
                dim + 1 if dim is not None else FloatPoint(coordinates[0]).dim,
            )
        else:
            if dim is None:
                dim = len(coordinates) - 1
            vector = _coerce_nonzero_point_dim(coordinates, dim + 1)
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Sphere radius must be positive")
        normalized = (radius / FloatVector(vector).norm()) * FloatVector(vector)
        return super().__new__(cls, normalized)

    @property
    def ambient_dim(self) -> int:
        """Return the ambient Euclidean dimension."""
        return super().dim

    @property
    def sphere_dim(self) -> int:
        """Return the intrinsic sphere dimension."""
        return self.ambient_dim - 1

    @property
    def radius(self) -> float:
        """Return the sphere radius."""
        return FloatVector(self).norm()

    def as_float_point(self) -> FloatPoint:
        """Return the embedded Euclidean representative."""
        return FloatPoint(self)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return (
            "SpherePoint("
            f"{tuple(float(value) for value in self)}, "
            f"dim={self.sphere_dim}, radius={self.radius})"
        )


class SphereSpace:
    """Metric sphere represented by embedded points in ``R^(n+1)``.

    Distances use the intrinsic great-circle metric. Visualization uses either
    low-dimensional projections of the embedding or, for ``S^2``, the explicit
    longitude/latitude formulas.
    """

    def __init__(
        self,
        dim: int = 2,
        radius: float = 1.0,
        name: str = "",
    ) -> None:
        """Initialize the sphere radius."""
        self.dim = int(dim)
        if self.dim < 1:
            raise ValueError("Sphere dimension must be positive")
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Sphere radius must be positive")
        self.name = name or f"S^{self.dim}({self.radius})"

    @property
    def space_kind(self) -> str:
        """Return the space kind identifier."""
        return "sphere"

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"SphereSpace(dim={self.dim}, radius={self.radius}{label})"

    def contains(self, point: object) -> bool:
        """Check whether a point lies on the sphere."""
        try:
            embedded = SpherePoint(point, dim=self.dim, radius=self.radius)
        except (TypeError, ValueError):
            return False
        return math.isclose(
            math.sqrt(sum(coordinate * coordinate for coordinate in embedded)),
            self.radius,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    def __contains__(self, point: object) -> bool:
        """Check whether a point lies on the sphere."""
        return self.contains(point)

    def point(self, point: object) -> "SpherePoint":
        """Return a validated point on the sphere."""
        return SpherePoint(point, dim=self.dim, radius=self.radius)

    def point_from_angles(
        self,
        *angles: float,
    ) -> "SpherePoint":
        """Build a sphere point from hyperspherical angles."""
        if self.dim == 2 and len(angles) == 2:
            longitude = float(angles[0])
            latitude = float(angles[1])
            cos_lat = math.cos(latitude)
            return SpherePoint(
                self.radius * cos_lat * math.cos(longitude),
                self.radius * cos_lat * math.sin(longitude),
                self.radius * math.sin(latitude),
                dim=self.dim,
                radius=self.radius,
            )
        if len(angles) != self.dim:
            raise ValueError(f"Expected {self.dim} angles, got {len(angles)}")
        angles = tuple(float(angle) for angle in angles)
        coordinates = []
        prefix = self.radius
        for index in range(self.dim):
            if index < self.dim - 1:
                coordinates.append(prefix * math.cos(angles[index]))
                prefix *= math.sin(angles[index])
            else:
                coordinates.append(prefix * math.cos(angles[index]))
                coordinates.append(prefix * math.sin(angles[index]))
        return SpherePoint(
            coordinates,
            dim=self.dim,
            radius=self.radius,
        )

    def distance(self, left: object, right: object) -> float:
        """Return the intrinsic great-circle distance."""
        left_point = self.point(left)
        right_point = self.point(right)
        cosine = _clamp_unit(
            sum(a * b for a, b in zip(left_point, right_point)) /
            (self.radius * self.radius)
        )
        return self.radius * math.acos(cosine)

    def to_2d(self, point: object, method: str = "default") -> Embedding2D:
        """Return a 2D visualization embedding of a sphere point."""
        embedded = self.point(point)
        method = (
            "stereographic" if method == "default" and self.dim == 2
            else "orthographic" if method == "default"
            else method
        )

        if method == "stereographic":
            if self.dim != 2:
                raise ValueError(
                    "Stereographic projection is implemented only for S^2"
                )
            x, y, z = embedded
            denominator = self.radius - z
            if math.isclose(denominator, 0.0, abs_tol=1e-12):
                raise ValueError(
                    "Stereographic projection is undefined at the north pole"
                )
            return (
                self.radius * x / denominator,
                self.radius * y / denominator,
            )

        if method == "equirectangular":
            if self.dim != 2:
                raise ValueError(
                    "Equirectangular projection is implemented only for S^2"
                )
            x, y, z = embedded
            longitude = math.atan2(y, x)
            latitude = math.asin(_clamp_unit(z / self.radius))
            return (longitude, latitude)

        if method == "orthographic":
            values = tuple(embedded)
            return (
                values[0],
                values[1] if len(values) > 1 else 0.0,
            )

        raise ValueError(f"Unknown 2D visualization method: {method!r}")

    def to_3d(self, point: object, method: str = "default") -> Embedding3D:
        """Return a 3D visualization embedding of a sphere point."""
        embedded = self.point(point)
        method = "embedding" if method == "default" else method
        if method != "embedding":
            raise ValueError(f"Unknown 3D visualization method: {method!r}")
        values = tuple(embedded)
        padded = values + (0.0, 0.0, 0.0)
        return (padded[0], padded[1], padded[2])

    def point_object(
        self,
        point: object,
        name: str = "",
    ) -> MetricGeometricObject["SpherePoint"]:
        """Return a singleton object on the sphere."""
        sphere_point = self.point(point)
        return MetricGeometricObject(
            self,
            contains=lambda candidate: self.point(candidate) == sphere_point,
            local_model=lambda candidate: LocalConeModel(
                _sphere_chart(sphere_point),
                _point_cone(self.dim),
            ),
            name=name or "sphere-point",
        )

    def neighborhood_at(
        self,
        point: object,
        radius: float,
        name: str = "",
    ) -> ChartNeighborhood["SpherePoint"]:
        """Return a centered intrinsic neighborhood on the sphere."""
        center = self.point(point)
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        chart = _sphere_chart(center)
        return ChartNeighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box(*(((-radius, radius),) * self.dim)),
            name=name or "sphere-neighborhood",
        )

    def full_cover(
        self,
        radius: float,
        resolution: int | None = None,
    ) -> tuple[ChartNeighborhood["SpherePoint"], ...]:
        """Return a finite full cover of the sphere by local neighborhoods."""
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Cover radius must be positive")
        if resolution is None:
            resolution = max(8, int(math.ceil((math.pi * self.radius) / radius)))
        if self.dim == 2:
            longitude_steps = max(4, int(resolution))
            latitude_steps = max(3, int(resolution // 2))
            return tuple(
                self.neighborhood_at(
                    self.point_from_angles(
                        2.0 * math.pi * longitude_index / longitude_steps,
                        -math.pi / 2.0 + math.pi * latitude_index / (latitude_steps - 1),
                    ),
                    radius,
                )
                for latitude_index in range(latitude_steps)
                for longitude_index in range(longitude_steps)
            )
        directions = _orthonormal_basis(
            tuple(
                FloatVector([
                    1.0 if index == axis else 0.0
                    for index in range(self.dim + 1)
                ])
                for axis in range(self.dim + 1)
            )
        )
        return tuple(
            self.neighborhood_at(
                SpherePoint(
                    [self.radius * coordinate for coordinate in direction],
                    dim=self.dim,
                    radius=self.radius,
                ),
                radius,
            )
            for direction in directions
        )

    def refine_cover(
        self,
        neighborhoods,
        factor: int = 2,
    ) -> tuple[ChartNeighborhood["SpherePoint"], ...]:
        """Return a refinement of a sphere neighborhood cover."""
        return _refine_neighborhoods(tuple(neighborhoods), factor=factor)

    def cap(
        self,
        center: object,
        radius: float,
        name: str = "",
    ) -> MetricGeometricObject["SpherePoint"]:
        """Return the closed geodesic cap around a center point."""
        center_point = self.point(center)
        cap_radius = float(radius)
        max_radius = math.pi * self.radius
        if cap_radius < 0.0:
            raise ValueError("Cap radius must be non-negative")
        if cap_radius > max_radius:
            raise ValueError("Cap radius must not exceed the sphere diameter")
        if math.isclose(cap_radius, 0.0, abs_tol=1e-12):
            return self.point_object(center_point, name=name or "sphere-cap")

        threshold = (self.radius * self.radius) * math.cos(
            cap_radius / self.radius
        )
        if math.isclose(cap_radius, max_radius, abs_tol=1e-12):
            return MetricGeometricObject(
                self,
                contains=lambda point: point in self,
                local_model=lambda point: LocalConeModel(
                    _sphere_chart(self.point(point)),
                    EuclideanCone.whole(self.dim),
                ),
                name=name or "sphere-cap",
            )

        def contains(point: FloatPoint) -> bool:
            sphere_point = self.point(point)
            return (
                sum(a * b for a, b in zip(center_point, sphere_point)) >=
                threshold - 1e-12
            )

        def local_model(point: FloatPoint) -> LocalConeModel[FloatPoint]:
            sphere_point = self.point(point)
            chart = _sphere_chart(sphere_point)
            score = sum(a * b for a, b in zip(center_point, sphere_point))
            if score > threshold + 1e-10:
                cone = EuclideanCone.whole(2)
            else:
                tangent_basis = _sphere_tangent_basis(sphere_point)
                gradient = FloatVector([
                    FloatVector(center_point).dot(basis_vector)
                    for basis_vector in tangent_basis
                ])
                if gradient.norm() < 1e-12:
                    cone = EuclideanCone.whole(self.dim)
                else:
                    cone = EuclideanCone(
                        self.dim,
                        contains=lambda coordinates: (
                            gradient.dot(FloatVector(coordinates)) >= -1e-12
                        ),
                        neighborhood=EuclideanNeighborhood.whole(self.dim),
                        name="sphere-cap-boundary",
                    )
            return LocalConeModel(chart, cone)

        return MetricGeometricObject(
            self,
            contains=contains,
            local_model=local_model,
            name=name or "sphere-cap",
        )


class TorusPoint(tuple):
    """Point on a torus represented by angular coordinates."""

    __slots__ = ()

    def __new__(
        cls,
        *angles: object,
    ) -> "TorusPoint":
        """Create a torus point from angles or one sequence of angles."""
        if len(angles) == 1 and isinstance(angles[0], cls):
            return angles[0]
        if len(angles) == 1:
            point = angles[0]
            if (
                isinstance(point, Sequence) and
                not isinstance(point, (str, bytes)) and
                len(point) >= 1
            ):
                return super().__new__(
                    cls,
                    tuple(FloatCirclePoint(angle) for angle in point),
                )
            if point == 0.0:
                angles = (0.0, 0.0)
            else:
                raise TypeError(
                    "TorusPoint requires at least two angles or one angle tuple"
                )
        return super().__new__(
            cls,
            tuple(FloatCirclePoint(angle) for angle in angles),
        )

    @property
    def dim(self) -> int:
        """Return the torus dimension."""
        return len(self)

    @property
    def major_angle(self) -> FloatCirclePoint:
        """Return the angle around the main circle."""
        if self.dim < 1:
            raise ValueError("TorusPoint has no angles")
        return self[0]

    @property
    def minor_angle(self) -> FloatCirclePoint:
        """Return the angle around the tube circle."""
        if self.dim < 2:
            raise ValueError("TorusPoint has no minor angle")
        return self[1]

    def to_tuple(self) -> tuple[float, ...]:
        """Return angular coordinates as plain floats."""
        return tuple(float(angle) for angle in self)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"TorusPoint{self.to_tuple()}"


class TorusSpace:
    """Flat torus modeled as a product of circles.

    The intrinsic metric is the product metric of circle distances. The 3D
    embedding is a visualization of the torus; it is not used for distance.
    """

    def __init__(
        self,
        dim: int = 2,
        major_radius: float = 2.0,
        minor_radius: float = 0.5,
        radii: Sequence[float] | None = None,
        name: str = "",
    ) -> None:
        """Initialize the torus visualization radii."""
        self.dim = int(dim)
        if self.dim < 1:
            raise ValueError("Torus dimension must be positive")
        if radii is None:
            if self.dim == 1:
                radii = (float(major_radius),)
            elif self.dim == 2:
                radii = (float(major_radius), float(minor_radius))
            else:
                base = float(major_radius)
                step = float(minor_radius)
                radii = tuple(base - step * index for index in range(self.dim))
        self.radii = tuple(float(radius) for radius in radii)
        if len(self.radii) != self.dim:
            raise ValueError("Torus radii length must match the torus dimension")
        if any(radius <= 0.0 for radius in self.radii):
            raise ValueError("Torus radii must be positive")
        self.major_radius = self.radii[0]
        self.minor_radius = self.radii[1] if self.dim > 1 else self.radii[0]
        self.name = name or f"T^{self.dim}"

    @property
    def space_kind(self) -> str:
        """Return the space kind identifier."""
        return "torus"

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return (
            "TorusSpace("
            f"dim={self.dim}, radii={self.radii}{label})"
        )

    def contains(self, point: object) -> bool:
        """Check whether a point can be interpreted on the torus."""
        try:
            return TorusPoint(point).dim == self.dim
        except (TypeError, ValueError):
            return False

    def __contains__(self, point: object) -> bool:
        """Check whether a point can be interpreted on the torus."""
        return self.contains(point)

    def point(self, point: object) -> TorusPoint:
        """Return a validated point on the torus."""
        torus_point = TorusPoint(point)
        if torus_point.dim != self.dim:
            raise ValueError(
                f"Expected a {self.dim}-dimensional torus point"
            )
        return torus_point

    def distance(self, left: object, right: object) -> float:
        """Return the flat-torus product distance."""
        left_point = self.point(left)
        right_point = self.point(right)
        squared = 0.0
        for index in range(self.dim):
            diff = float(left_point[index].distance_to(right_point[index]))
            squared += diff * diff
        return math.sqrt(squared)

    def to_2d(self, point: object, method: str = "default") -> Embedding2D:
        """Return a 2D visualization embedding of a torus point."""
        torus_point = self.point(point)
        method = "flat" if method == "default" else method
        if method != "flat":
            raise ValueError(f"Unknown 2D visualization method: {method!r}")
        values = torus_point.to_tuple()
        padded = values + (0.0, 0.0)
        return (padded[0], padded[1])

    def to_3d(self, point: object, method: str = "default") -> Embedding3D:
        """Return a 3D donut embedding of a torus point."""
        torus_point = self.point(point)
        method = "embedding" if method == "default" else method
        if method != "embedding":
            raise ValueError(f"Unknown 3D visualization method: {method!r}")

        embedded = tuple(_torus_embedding(torus_point, self.radii))
        padded = embedded + (0.0, 0.0, 0.0)
        return (padded[0], padded[1], padded[2])

    def point_object(
        self,
        point: object,
        name: str = "",
    ) -> MetricGeometricObject[TorusPoint]:
        """Return a singleton object on the torus."""
        torus_point = self.point(point)
        return MetricGeometricObject(
            self,
            contains=lambda candidate: self.point(candidate) == torus_point,
            local_model=lambda candidate: LocalConeModel(
                _torus_chart(torus_point),
                _point_cone(self.dim),
            ),
            name=name or "torus-point",
        )

    def neighborhood_at(
        self,
        point: object,
        radius: float,
        name: str = "",
    ) -> ChartNeighborhood[TorusPoint]:
        """Return a centered intrinsic neighborhood on the torus."""
        center = self.point(point)
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        if radius >= math.pi:
            raise ValueError("Torus angular neighborhoods must have radius < pi")
        chart = _torus_chart(center)
        return ChartNeighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box(*(((-radius, radius),) * self.dim)),
            name=name or "torus-neighborhood",
        )

    def full_cover(
        self,
        radius: float,
    ) -> tuple[ChartNeighborhood[TorusPoint], ...]:
        """Return a finite full cover of the torus by local neighborhoods."""
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Cover radius must be positive")
        if radius >= math.pi:
            raise ValueError("Torus cover radius must be < pi")
        steps = max(2, int(math.ceil((2.0 * math.pi) / (2.0 * radius))))
        axis_values = tuple(
            tuple(2.0 * math.pi * index / steps for index in range(steps))
            for _ in range(self.dim)
        )
        return tuple(
            self.neighborhood_at(TorusPoint(point), radius)
            for point in itertools.product(*axis_values)
        )

    def refine_cover(
        self,
        neighborhoods,
        factor: int = 2,
    ) -> tuple[ChartNeighborhood[TorusPoint], ...]:
        """Return a refinement of a torus neighborhood cover."""
        return _refine_neighborhoods(tuple(neighborhoods), factor=factor)

    def patch(
        self,
        *angle_sets: object,
        name: str = "",
    ) -> MetricGeometricObject[TorusPoint]:
        """Return an axis-aligned angular patch on the torus."""
        if len(angle_sets) != self.dim:
            raise ValueError(
                f"Expected {self.dim} angular sets, got {len(angle_sets)}"
            )
        circle_sets = tuple(FloatCircleSet(angle_set) for angle_set in angle_sets)

        def contains(point: TorusPoint) -> bool:
            torus_point = self.point(point)
            return all(
                torus_point[index] in circle_sets[index]
                for index in range(self.dim)
            )

        def local_model(point: TorusPoint) -> LocalConeModel[TorusPoint]:
            torus_point = self.point(point)
            chart = _torus_chart(torus_point)
            axis_flags = [
                (
                    _previous_circle_point(torus_point[index]) in circle_sets[index],
                    _following_circle_point(torus_point[index]) in circle_sets[index],
                )
                for index in range(self.dim)
            ]
            cone = _product_cone(
                axis_flags,
                "torus-patch",
            )
            return LocalConeModel(chart, cone)

        return MetricGeometricObject(
            self,
            contains=contains,
            local_model=local_model,
            name=name or "torus-patch",
        )


__all__ = [
    "Embedding2D",
    "Embedding3D",
    "Space",
    "SpherePoint",
    "SphereSpace",
    "TorusPoint",
    "TorusSpace",
]
