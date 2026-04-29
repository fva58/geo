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
    FloatCircleInterval,
    FloatCircleSet,
)
from .geometric import EuclideanCone, LocalConeModel, ObjectMesh, Sphere
from .manifold import ManifoldChart
from .manifold import ChartNeighborhood
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


def _merge_meshes(meshes: Sequence[ObjectMesh]) -> ObjectMesh:
    """Return one mesh from several disjoint mesh parts."""
    vertices: list[FloatPoint] = []
    cells: list[tuple[int, ...]] = []
    offset = 0
    for mesh in meshes:
        vertices.extend(mesh.vertices)
        cells.extend(
            tuple(index + offset for index in cell)
            for cell in mesh.cells
        )
        offset += len(mesh.vertices)
    return ObjectMesh(tuple(vertices), tuple(cells))


def _sphere_point_from_local(
    center: FloatPoint,
    distance: float,
    direction: FloatVector,
) -> FloatPoint:
    """Return a sphere point from local polar data around a center."""
    radius = math.sqrt(sum(coordinate * coordinate for coordinate in center))
    normal = _normalize_vector(center)
    direction = _normalize_vector(direction)
    angle = distance / radius
    embedded = radius * (
        math.cos(angle) * normal +
        math.sin(angle) * direction
    )
    return FloatPoint(embedded)


def _sample_sphere_directions(dim: int, steps: int) -> tuple[FloatVector, ...]:
    """Return deterministic tangent directions on the unit sphere."""
    dim = max(1, int(dim))
    steps = max(dim + 1, int(steps))
    directions = []
    for axis in range(dim):
        vector = [0.0] * dim
        vector[axis] = 1.0
        directions.append(FloatVector(vector))
        vector = [0.0] * dim
        vector[axis] = -1.0
        directions.append(FloatVector(vector))
    for index in range(steps - len(directions)):
        directions.append(
            _normalize_vector(
                FloatVector([
                    math.cos((index + 1) * (axis + 1))
                    for axis in range(dim)
                ])
            )
        )
    return tuple(directions)


def _sample_torus_axis_sets(
    angle_sets: Sequence[FloatCircleSet],
    resolution: int,
) -> tuple[tuple[float, ...], ...]:
    """Return sampled angles for each torus axis."""
    base_resolution = max(2, int(resolution))
    return tuple(
        _sample_circle_set(
            circle_set,
            max(2, int(base_resolution // (2 if index else 1))),
        )
        for index, circle_set in enumerate(angle_sets)
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


def _sample_linear_interval(
    left: float,
    right: float,
    steps: int,
) -> tuple[float, ...]:
    """Return evenly sampled values on one closed interval."""
    steps = max(1, int(steps))
    if math.isclose(left, right, abs_tol=1e-15):
        return (left,)
    if steps == 1:
        return ((left + right) / 2.0,)
    return tuple(
        left + (right - left) * index / (steps - 1)
        for index in range(steps)
    )


def _sample_circle_set(
    circle_set: FloatCircleSet,
    steps: int,
) -> tuple[float, ...]:
    """Return evenly sampled angles on a circle set."""
    intervals = [FloatCircleInterval(interval) for interval in circle_set]
    if not intervals:
        return ()

    total_length = sum(float(interval.length()) for interval in intervals)
    samples: list[float] = []
    for interval in intervals:
        interval_length = float(interval.length())
        if math.isclose(interval_length, 0.0, abs_tol=1e-15):
            part_steps = 1
        elif total_length <= 0.0:
            part_steps = max(2, steps)
        else:
            part_steps = max(
                2,
                int(round(steps * interval_length / total_length)),
            )
        samples.extend(
            _sample_linear_interval(
                float(interval.start),
                float(interval.end),
                part_steps,
            )
        )

    deduplicated: list[float] = []
    for value in samples:
        if deduplicated and math.isclose(
            deduplicated[-1],
            value,
            abs_tol=1e-12,
        ):
            continue
        deduplicated.append(value)
    return tuple(deduplicated)


def _grid_torus_mesh(
    major_values: Sequence[float],
    minor_values: Sequence[float],
    major_radius: float,
    minor_radius: float,
    wrap_major: bool,
    wrap_minor: bool,
) -> ObjectMesh:
    """Return a torus mesh from angular grids."""
    if len(major_values) < 2 or len(minor_values) < 2:
        raise ValueError("Torus mesh grids need at least two samples per axis")

    major_count = len(major_values)
    minor_count = len(minor_values)
    vertices = tuple(
        FloatPoint(
            (major_radius + minor_radius * math.cos(minor)) * math.cos(major),
            (major_radius + minor_radius * math.cos(minor)) * math.sin(major),
            minor_radius * math.sin(minor),
        )
        for major in major_values
        for minor in minor_values
    )

    def vertex_index(major_index: int, minor_index: int) -> int:
        return major_index * minor_count + minor_index

    major_limit = major_count if wrap_major else major_count - 1
    minor_limit = minor_count if wrap_minor else minor_count - 1
    cells = []
    for major_index in range(major_limit):
        next_major = (major_index + 1) % major_count
        if not wrap_major and next_major == 0:
            continue
        for minor_index in range(minor_limit):
            next_minor = (minor_index + 1) % minor_count
            if not wrap_minor and next_minor == 0:
                continue
            lower_left = vertex_index(major_index, minor_index)
            lower_right = vertex_index(next_major, minor_index)
            upper_left = vertex_index(major_index, next_minor)
            upper_right = vertex_index(next_major, next_minor)
            cells.append((lower_left, lower_right, upper_left))
            cells.append((lower_right, upper_right, upper_left))
    return ObjectMesh(vertices, tuple(cells))


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


class SampledMetricObject(MetricGeometricObject[PointT], Generic[PointT]):
    """Metric object with explicit sampling and mesh hooks."""

    def __init__(
        self,
        space: MetricSpace[PointT],
        contains: Callable[[PointT], bool],
        local_model: Callable[[PointT], LocalConeModel[PointT]],
        sample_points: Callable[[int], tuple[PointT, ...]],
        mesh: Callable[[int], ObjectMesh],
        name: str = "",
    ) -> None:
        """Initialize the sampled metric object."""
        super().__init__(space, contains, local_model, name=name)
        self._sample_points = sample_points
        self._mesh = mesh

    def sample_points(self, resolution: int = 24) -> tuple[PointT, ...]:
        """Return sample points on the object."""
        return tuple(self._sample_points(int(resolution)))

    def mesh(self, resolution: int = 24) -> ObjectMesh:
        """Return a visualization mesh of the object."""
        return self._mesh(int(resolution))


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
    ) -> SampledMetricObject["SpherePoint"]:
        """Return a singleton object on the sphere."""
        sphere_point = self.point(point)
        return SampledMetricObject(
            self,
            contains=lambda candidate: self.point(candidate) == sphere_point,
            local_model=lambda candidate: LocalConeModel(
                _sphere_chart(sphere_point),
                _point_cone(self.dim),
            ),
            sample_points=lambda resolution: (sphere_point,),
            mesh=lambda resolution: ObjectMesh((sphere_point,), ((0,),)),
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

    def cap(
        self,
        center: object,
        radius: float,
        name: str = "",
    ) -> SampledMetricObject["SpherePoint"]:
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
            return SampledMetricObject(
                self,
                contains=lambda point: point in self,
                local_model=lambda point: LocalConeModel(
                    _sphere_chart(self.point(point)),
                    EuclideanCone.whole(self.dim),
                ),
                sample_points=lambda resolution: self.sample_points(resolution),
                mesh=lambda resolution: self.mesh(resolution),
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

        return SampledMetricObject(
            self,
            contains=contains,
            local_model=local_model,
            sample_points=lambda resolution: tuple(
                self.cap_mesh(center_point, cap_radius, resolution).vertices
            ),
            mesh=lambda resolution: self.cap_mesh(
                center_point,
                cap_radius,
                resolution,
            ),
            name=name or "sphere-cap",
        )

    def sample_points(self, resolution: int = 24) -> tuple[FloatPoint, ...]:
        """Return sample points on the whole sphere."""
        if self.dim != 2:
            directions = _sample_sphere_directions(self.dim + 1, resolution * 2)
            return tuple(
                FloatPoint([
                    self.radius * coordinate for coordinate in direction
                ])
                for direction in directions
            )
        return self.mesh(resolution=resolution).vertices

    def mesh(self, resolution: int = 24) -> ObjectMesh:
        """Return a visualization mesh for the whole sphere."""
        if self.dim != 2:
            vertices = self.sample_points(resolution=resolution)
            return ObjectMesh(
                tuple(FloatPoint(self.to_3d(vertex)) for vertex in vertices),
                tuple((index,) for index in range(len(vertices))),
            )
        return Sphere(FloatPoint(0.0, 0.0, 0.0), self.radius).mesh(
            resolution=resolution,
        )

    def cap_mesh(
        self,
        center: object,
        radius: float,
        resolution: int = 24,
    ) -> ObjectMesh:
        """Return a visualization mesh for a spherical cap."""
        center_point = self.point(center)
        cap_radius = float(radius)
        if cap_radius < 0.0:
            raise ValueError("Cap radius must be non-negative")
        if math.isclose(cap_radius, 0.0, abs_tol=1e-12):
            return ObjectMesh((center_point,), ((0,),))
        if math.isclose(cap_radius, math.pi * self.radius, abs_tol=1e-12):
            return self.mesh(resolution=resolution)
        if self.dim != 2:
            local_radius = max(2, int(resolution))
            directions = _sample_sphere_directions(self.dim, local_radius * 2)
            tangent_basis = _sphere_tangent_basis(center_point)
            vertices = [center_point]
            for radial_index in range(1, local_radius + 1):
                geodesic_distance = cap_radius * radial_index / local_radius
                for direction in directions:
                    tangent_direction = FloatVector(
                        [
                            sum(
                            direction[basis_index] * basis_vector[axis]
                            for basis_index, basis_vector in enumerate(tangent_basis)
                            )
                            for axis in range(self.dim + 1)
                        ]
                    )
                    vertices.append(
                        _sphere_point_from_local(
                            center_point,
                            geodesic_distance,
                            tangent_direction,
                        )
                    )
            return ObjectMesh(
                tuple(FloatPoint(self.to_3d(vertex)) for vertex in vertices),
                tuple((index,) for index in range(len(vertices))),
            )

        radial_steps = max(2, int(resolution // 2))
        angular_steps = max(8, int(resolution))
        first_basis, second_basis = _sphere_tangent_basis(center_point)
        vertices = [center_point]
        cells: list[tuple[int, ...]] = []

        for radial_index in range(1, radial_steps + 1):
            geodesic_distance = cap_radius * radial_index / radial_steps
            for angular_index in range(angular_steps):
                azimuth = 2.0 * math.pi * angular_index / angular_steps
                vertices.append(
                    _sphere_point_from_local(
                        center_point,
                        geodesic_distance,
                        math.cos(azimuth) * first_basis +
                        math.sin(azimuth) * second_basis,
                    )
                )

        def ring_index(radial_index: int, angular_index: int) -> int:
            return 1 + (radial_index - 1) * angular_steps + angular_index

        for angular_index in range(angular_steps):
            next_index = (angular_index + 1) % angular_steps
            cells.append((0, ring_index(1, next_index), ring_index(1, angular_index)))

        for radial_index in range(1, radial_steps):
            for angular_index in range(angular_steps):
                next_index = (angular_index + 1) % angular_steps
                lower_left = ring_index(radial_index, angular_index)
                lower_right = ring_index(radial_index, next_index)
                upper_left = ring_index(radial_index + 1, angular_index)
                upper_right = ring_index(radial_index + 1, next_index)
                cells.append((lower_left, lower_right, upper_left))
                cells.append((lower_right, upper_right, upper_left))

        return ObjectMesh(tuple(vertices), tuple(cells))


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
    ) -> SampledMetricObject[TorusPoint]:
        """Return a singleton object on the torus."""
        torus_point = self.point(point)
        return SampledMetricObject(
            self,
            contains=lambda candidate: self.point(candidate) == torus_point,
            local_model=lambda candidate: LocalConeModel(
                _torus_chart(torus_point),
                _point_cone(self.dim),
            ),
            sample_points=lambda resolution: (torus_point,),
            mesh=lambda resolution: ObjectMesh(
                (FloatPoint(self.to_3d(torus_point)),),
                ((0,),),
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

    def patch(
        self,
        *angle_sets: object,
        name: str = "",
    ) -> SampledMetricObject[TorusPoint]:
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

        return SampledMetricObject(
            self,
            contains=contains,
            local_model=local_model,
            sample_points=lambda resolution: tuple(
                TorusPoint(point)
                for point in itertools.product(
                    *_sample_torus_axis_sets(circle_sets, resolution)
                )
            ),
            mesh=lambda resolution: self.patch_mesh(
                *circle_sets,
                resolution=resolution,
            ),
            name=name or "torus-patch",
        )

    def sample_points(self, resolution: int = 24) -> tuple[TorusPoint, ...]:
        """Return sample points on the whole torus."""
        axis_steps = tuple(
            max(4, int(resolution // (2 if index else 1)))
            for index in range(self.dim)
        )
        import itertools
        return tuple(
            TorusPoint(
                tuple(
                    2.0 * math.pi * axis_index / steps
                    for axis_index, steps in zip(indices, axis_steps)
                )
            )
            for indices in itertools.product(*(range(steps) for steps in axis_steps))
        )

    def mesh(self, resolution: int = 24) -> ObjectMesh:
        """Return a visualization mesh for the whole torus."""
        if self.dim != 2:
            vertices = tuple(
                FloatPoint(self.to_3d(point))
                for point in self.sample_points(resolution=resolution)
            )
            return ObjectMesh(vertices, tuple((index,) for index in range(len(vertices))))
        major_steps = max(8, int(resolution))
        minor_steps = max(8, int(resolution // 2))
        major_values = tuple(
            2.0 * math.pi * index / major_steps
            for index in range(major_steps)
        )
        minor_values = tuple(
            2.0 * math.pi * index / minor_steps
            for index in range(minor_steps)
        )
        return _grid_torus_mesh(
            major_values,
            minor_values,
            self.major_radius,
            self.minor_radius,
            wrap_major=True,
            wrap_minor=True,
        )

    def patch_mesh(
        self,
        *angle_sets: object,
        resolution: int = 24,
    ) -> ObjectMesh:
        """Return a visualization mesh for an angular patch."""
        if len(angle_sets) != self.dim:
            raise ValueError(
                f"Expected {self.dim} angular sets, got {len(angle_sets)}"
            )
        circle_sets = tuple(FloatCircleSet(angle_set) for angle_set in angle_sets)
        if self.dim != 2:
            patch = self.patch(*circle_sets)
            vertices = tuple(
                FloatPoint(self.to_3d(point))
                for point in patch.sample_points(resolution=resolution)
            )
            return ObjectMesh(
                vertices,
                tuple((index,) for index in range(len(vertices))),
            )
        major_circle_set, minor_circle_set = circle_sets
        if major_circle_set.is_full() and minor_circle_set.is_full():
            return self.mesh(resolution=resolution)

        major_intervals = [
            FloatCircleInterval(interval)
            for interval in major_circle_set
        ]
        minor_intervals = [
            FloatCircleInterval(interval)
            for interval in minor_circle_set
        ]
        meshes = []
        interval_steps = max(4, int(resolution // max(
            1,
            len(major_intervals) * len(minor_intervals),
        )))
        for major_interval in major_intervals:
            major_values = _sample_linear_interval(
                float(major_interval.start),
                float(major_interval.end),
                interval_steps,
            )
            for minor_interval in minor_intervals:
                minor_values = _sample_linear_interval(
                    float(minor_interval.start),
                    float(minor_interval.end),
                    interval_steps,
                )
                if len(major_values) < 2 or len(minor_values) < 2:
                    continue
                meshes.append(
                    _grid_torus_mesh(
                        major_values,
                        minor_values,
                        self.major_radius,
                        self.minor_radius,
                        wrap_major=False,
                        wrap_minor=False,
                    )
                )
        if not meshes:
            samples = [
                self.to_3d(TorusPoint(major, minor))
                for major in _sample_circle_set(major_circle_set, 1)
                for minor in _sample_circle_set(minor_circle_set, 1)
            ]
            vertices = tuple(FloatPoint(sample) for sample in samples)
            cells = tuple((index,) for index in range(len(vertices)))
            return ObjectMesh(vertices, cells)
        return _merge_meshes(meshes)


__all__ = [
    "Embedding2D",
    "Embedding3D",
    "Space",
    "SpherePoint",
    "SphereSpace",
    "TorusPoint",
    "TorusSpace",
]
