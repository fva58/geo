"""Spherical spaces."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..cone import EuclideanCone, LocalConeModel
from ..euclidean import EuclideanNeighborhood, FloatPoint, FloatVector
from ..gobject import GeometricObject
from ..manifold import ManifoldChart
from .base import BoxNeighborhood, Space as SpaceBase, refine_neighborhoods as _refine_neighborhoods


class Neighborhood(BoxNeighborhood["SpherePoint"]):
    """Neighborhood on a sphere."""


def _coerce_point_dim(value: object, dim: int) -> FloatPoint:
    point = FloatPoint(value)
    if point.dim != dim:
        raise ValueError(f"Expected a {dim}-dimensional point")
    return point


def _coerce_nonzero_point_dim(value: object, dim: int) -> FloatPoint:
    point = _coerce_point_dim(value, dim)
    if math.isclose(FloatVector(point).norm(), 0.0, abs_tol=1e-15):
        raise ValueError("Expected a nonzero vector")
    return point


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _point_cone(dim: int) -> EuclideanCone:
    return EuclideanCone(
        dim,
        contains=lambda point: FloatPoint(point) == FloatPoint.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
    )


def _normalize_vector(vector: FloatPoint | FloatVector) -> FloatVector:
    array = FloatVector(vector)
    norm = array.norm()
    if math.isclose(norm, 0.0, abs_tol=1e-15):
        raise ValueError("Zero vector cannot be normalized")
    return array / norm


def _orthonormal_basis(vectors: Sequence[FloatVector]) -> tuple[FloatVector, ...]:
    basis: list[FloatVector] = []
    for vector in vectors:
        candidate = FloatVector(vector)
        for basis_vector in basis:
            candidate = candidate - candidate.dot(basis_vector) * basis_vector
        if candidate.norm() > 1e-12:
            basis.append(_normalize_vector(candidate))
    return tuple(basis)


def _sphere_tangent_basis(point: FloatPoint) -> tuple[FloatVector, ...]:
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
    )


class SpherePoint(FloatPoint):
    """Point on a sphere represented by a nonzero ambient vector."""

    __slots__ = ()

    def __new__(
        cls,
        *coordinates: object,
        dim: int | None = None,
        radius: float = 1.0,
    ) -> "SpherePoint":
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
        return super().dim

    @property
    def sphere_dim(self) -> int:
        return self.ambient_dim - 1

    @property
    def radius(self) -> float:
        return FloatVector(self).norm()

    def as_float_point(self) -> FloatPoint:
        return FloatPoint(self)

    def __repr__(self) -> str:
        return (
            "SpherePoint("
            f"{tuple(float(value) for value in self)}, "
            f"dim={self.sphere_dim}, radius={self.radius})"
        )


class Space(SpaceBase):
    """Sphere represented by embedded points in ``R^(n+1)``."""

    def __init__(
        self,
        dim: int = 2,
        radius: float = 1.0,
    ) -> None:
        self._dim = int(dim)
        if self._dim < 1:
            raise ValueError("Sphere dimension must be positive")
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Sphere radius must be positive")

    @property
    def dim(self) -> int:
        """Return the sphere dimension."""
        return self._dim

    def __repr__(self) -> str:
        return f"Space(dim={self.dim}, radius={self.radius})"

    def contains(self, point: object) -> bool:
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
        return self.contains(point)

    def point(self, point: object) -> SpherePoint:
        return SpherePoint(point, dim=self.dim, radius=self.radius)

    def point_from_angles(self, *angles: float) -> SpherePoint:
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
        return SpherePoint(coordinates, dim=self.dim, radius=self.radius)

    def distance(self, left: object, right: object) -> float:
        left_point = self.point(left)
        right_point = self.point(right)
        cosine = _clamp_unit(
            sum(a * b for a, b in zip(left_point, right_point)) /
            (self.radius * self.radius)
        )
        return self.radius * math.acos(cosine)

    def whole(self) -> GeometricObject[SpherePoint]:
        return GeometricObject(
            self,
            contains=lambda point: point in self,
            local_model=lambda point: LocalConeModel(
                _sphere_chart(self.point(point)),
                EuclideanCone.whole(self.dim),
            ),
        )

    def point_object(
        self,
        point: object,
    ) -> GeometricObject[SpherePoint]:
        sphere_point = self.point(point)
        return GeometricObject(
            self,
            contains=lambda candidate: self.point(candidate) == sphere_point,
            local_model=lambda candidate: LocalConeModel(
                _sphere_chart(sphere_point),
                _point_cone(self.dim),
            ),
        )

    def neighborhood_at(
        self,
        point: object,
        radius: float,
    ) -> Neighborhood:
        center = self.point(point)
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        chart = _sphere_chart(center)
        return Neighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box(*(((-radius, radius),) * self.dim)),
        )

    def full(
        self,
        radius: float,
        resolution: int | None = None,
    ) -> tuple[Neighborhood, ...]:
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
                        -math.pi / 2.0
                        + math.pi * latitude_index / (latitude_steps - 1),
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

    def refine(
        self,
        neighborhoods,
        factor: int = 2,
    ) -> tuple[Neighborhood, ...]:
        return _refine_neighborhoods(tuple(neighborhoods), factor=factor)

    def cap(
        self,
        center: object,
        radius: float,
    ) -> GeometricObject[SpherePoint]:
        center_point = self.point(center)
        cap_radius = float(radius)
        max_radius = math.pi * self.radius
        if cap_radius < 0.0:
            raise ValueError("Cap radius must be non-negative")
        if cap_radius > max_radius:
            raise ValueError("Cap radius must not exceed the sphere diameter")
        if math.isclose(cap_radius, 0.0, abs_tol=1e-12):
            return self.point_object(center_point)
        threshold = (self.radius * self.radius) * math.cos(
            cap_radius / self.radius
        )
        if math.isclose(cap_radius, max_radius, abs_tol=1e-12):
            return GeometricObject(
                self,
                contains=lambda point: point in self,
                local_model=lambda point: LocalConeModel(
                    _sphere_chart(self.point(point)),
                    EuclideanCone.whole(self.dim),
                ),
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
                    )
            return LocalConeModel(chart, cone)

        return GeometricObject(
            self,
            contains=contains,
            local_model=local_model,
        )


__all__ = ["Neighborhood", "Space", "SpherePoint"]
