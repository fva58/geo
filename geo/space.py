"""General space protocols and standard visualizable spaces."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from .euclidean import EuclideanNeighborhood, FloatPoint, FloatVector
from .floatcircle import (
    FloatAngle,
    FloatCirclePoint,
    FloatCircleSet,
)
from .geometric import EuclideanCone, LocalConeModel
from .manifold import ManifoldChart
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


def _sphere_tangent_basis(
    point: FloatPoint,
) -> tuple[FloatVector, FloatVector]:
    """Return an orthonormal tangent basis at a sphere point."""
    normal = _normalize_vector(point)
    reference = FloatVector(0.0, 0.0, 1.0)
    if abs(normal.dot(reference)) > 0.9:
        reference = FloatVector(1.0, 0.0, 0.0)
    first = _normalize_vector(_cross(reference, normal))
    second = _normalize_vector(_cross(normal, first))
    return (first, second)


def _sphere_chart(base_point: FloatPoint) -> ManifoldChart[FloatPoint]:
    """Return a gnomonic chart centered at a sphere point."""
    base_point = _coerce_point3(base_point)
    radius = math.sqrt(sum(coordinate * coordinate for coordinate in base_point))
    normal = _normalize_vector(base_point)
    first, second = _sphere_tangent_basis(base_point)

    def forward(point: FloatPoint) -> FloatPoint:
        point = _coerce_point3(point)
        denominator = normal.dot(FloatVector(point))
        if math.isclose(denominator, 0.0, abs_tol=1e-12):
            raise ValueError("Point is outside the local sphere chart")
        return FloatPoint(
            radius * FloatVector(point).dot(first) / denominator,
            radius * FloatVector(point).dot(second) / denominator,
        )

    def inverse(coordinates: FloatPoint) -> FloatPoint:
        coordinates = FloatPoint(coordinates)
        if coordinates.dim != 2:
            raise ValueError("Sphere chart coordinates must be two-dimensional")
        candidate = (
            FloatVector(base_point) +
            coordinates[0] * first +
            coordinates[1] * second
        )
        normalized = _normalize_vector(candidate)
        return FloatPoint(radius * coordinate for coordinate in normalized)

    return ManifoldChart(
        forward,
        inverse,
        dim=2,
        name="sphere-gnomonic",
    )


def _torus_chart(base_point: "TorusPoint") -> ManifoldChart["TorusPoint"]:
    """Return a local angular chart centered at a torus point."""
    base_point = TorusPoint(base_point)

    def forward(point: TorusPoint) -> FloatPoint:
        point = TorusPoint(point)
        return FloatPoint(
            _signed_circle_difference(point.major_angle, base_point.major_angle),
            _signed_circle_difference(point.minor_angle, base_point.minor_angle),
        )

    def inverse(coordinates: FloatPoint) -> TorusPoint:
        coordinates = FloatPoint(coordinates)
        if coordinates.dim != 2:
            raise ValueError("Torus chart coordinates must be two-dimensional")
        return TorusPoint(
            float(base_point.major_angle) + coordinates[0],
            float(base_point.minor_angle) + coordinates[1],
        )

    return ManifoldChart(
        forward,
        inverse,
        dim=2,
        name="torus-angle",
    )


def _product_cone(
    x_negative: bool,
    x_positive: bool,
    y_negative: bool,
    y_positive: bool,
    name: str,
) -> EuclideanCone:
    """Return the product cone from axis-wise membership flags."""

    def axis_ok(value: float, left_in: bool, right_in: bool) -> bool:
        if left_in and right_in:
            return True
        if right_in:
            return value >= 0.0
        if left_in:
            return value <= 0.0
        return math.isclose(value, 0.0, abs_tol=1e-12)

    return EuclideanCone(
        2,
        contains=lambda point: (
            axis_ok(FloatPoint(point)[0], x_negative, x_positive) and
            axis_ok(FloatPoint(point)[1], y_negative, y_positive)
        ),
        neighborhood=EuclideanNeighborhood.whole(2),
        name=name,
    )


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


class SphereSpace:
    """Metric sphere represented by embedded points in ``R^3``.

    Distances use the intrinsic great-circle metric. Visualization can use the
    embedded 3D coordinates directly or simple 2D projections derived from
    longitude and latitude.
    """

    dim = 2

    def __init__(self, radius: float = 1.0, name: str = "") -> None:
        """Initialize the sphere radius."""
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Sphere radius must be positive")
        self.name = name or f"S^2({self.radius})"

    @property
    def space_kind(self) -> str:
        """Return the space kind identifier."""
        return "sphere"

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"SphereSpace(radius={self.radius}{label})"

    def contains(self, point: object) -> bool:
        """Check whether a point lies on the sphere."""
        try:
            embedded = _coerce_point3(point)
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

    def point(self, point: object) -> FloatPoint:
        """Return a validated point on the sphere."""
        embedded = _coerce_point3(point)
        if embedded not in self:
            raise ValueError("Point is outside the sphere")
        return embedded

    def point_from_angles(
        self,
        longitude: float,
        latitude: float,
    ) -> FloatPoint:
        """Build a sphere point from longitude and latitude."""
        lon = float(longitude)
        lat = float(latitude)
        cos_lat = math.cos(lat)
        return FloatPoint(
            self.radius * cos_lat * math.cos(lon),
            self.radius * cos_lat * math.sin(lon),
            self.radius * math.sin(lat),
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
        x, y, z = embedded
        method = "stereographic" if method == "default" else method

        if method == "stereographic":
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
            longitude = math.atan2(y, x)
            latitude = math.asin(_clamp_unit(z / self.radius))
            return (longitude, latitude)

        raise ValueError(f"Unknown 2D visualization method: {method!r}")

    def to_3d(self, point: object, method: str = "default") -> Embedding3D:
        """Return a 3D visualization embedding of a sphere point."""
        embedded = self.point(point)
        method = "embedding" if method == "default" else method
        if method != "embedding":
            raise ValueError(f"Unknown 3D visualization method: {method!r}")
        return (embedded[0], embedded[1], embedded[2])

    def point_object(
        self,
        point: object,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return a singleton object on the sphere."""
        sphere_point = self.point(point)
        return MetricGeometricObject(
            self,
            contains=lambda candidate: self.point(candidate) == sphere_point,
            local_model=lambda candidate: LocalConeModel(
                _sphere_chart(sphere_point),
                _point_cone(2),
            ),
            name=name or "sphere-point",
        )

    def point(self, point: object) -> FloatPoint:
        """Return a validated point on the sphere."""
        embedded = _coerce_point3(point)
        if embedded not in self:
            raise ValueError("Point is outside the sphere")
        return embedded

    def cap(
        self,
        center: object,
        radius: float,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
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
                    EuclideanCone.whole(2),
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
                first, second = _sphere_tangent_basis(sphere_point)
                gradient = FloatVector(
                    FloatVector(center_point).dot(first),
                    FloatVector(center_point).dot(second),
                )
                if gradient.norm() < 1e-12:
                    cone = EuclideanCone.whole(2)
                else:
                    cone = EuclideanCone(
                        2,
                        contains=lambda coordinates: (
                            gradient.dot(FloatVector(coordinates)) >= -1e-12
                        ),
                        neighborhood=EuclideanNeighborhood.whole(2),
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
    """Point on a torus represented by two angular coordinates."""

    __slots__ = ()

    def __new__(
        cls,
        major_angle: object = 0.0,
        minor_angle: object = _MISSING,
    ) -> "TorusPoint":
        """Create a torus point from two angles or one pair."""
        if isinstance(major_angle, cls) and minor_angle is _MISSING:
            return major_angle
        if (
            minor_angle is _MISSING and
            isinstance(major_angle, Sequence) and
            not isinstance(major_angle, (str, bytes)) and
            len(major_angle) == 2
        ):
            major_angle, minor_angle = major_angle
        elif minor_angle is _MISSING:
            if major_angle == 0.0:
                minor_angle = 0.0
            else:
                raise TypeError(
                    "TorusPoint requires two angles or one pair of angles"
                )
        return super().__new__(
            cls,
            (
                FloatCirclePoint(major_angle),
                FloatCirclePoint(minor_angle),
            ),
        )

    @property
    def major_angle(self) -> FloatCirclePoint:
        """Return the angle around the main circle."""
        return self[0]

    @property
    def minor_angle(self) -> FloatCirclePoint:
        """Return the angle around the tube circle."""
        return self[1]

    def to_tuple(self) -> tuple[float, float]:
        """Return angular coordinates as plain floats."""
        return (float(self.major_angle), float(self.minor_angle))

    def __repr__(self) -> str:
        """Return a debug representation."""
        return (
            "TorusPoint("
            f"{float(self.major_angle)}, {float(self.minor_angle)})"
        )


class TorusSpace:
    """Flat torus modeled as a product of two circles.

    The intrinsic metric is the product metric of two circle distances. The
    3D embedding is a visualization of the torus with chosen major and minor
    radii; it is not used for the distance.
    """

    dim = 2

    def __init__(
        self,
        major_radius: float = 2.0,
        minor_radius: float = 0.5,
        name: str = "",
    ) -> None:
        """Initialize the torus visualization radii."""
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        if self.major_radius <= 0.0 or self.minor_radius <= 0.0:
            raise ValueError("Torus radii must be positive")
        self.name = name or "T^2"

    @property
    def space_kind(self) -> str:
        """Return the space kind identifier."""
        return "torus"

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return (
            "TorusSpace("
            f"major_radius={self.major_radius}, "
            f"minor_radius={self.minor_radius}{label})"
        )

    def contains(self, point: object) -> bool:
        """Check whether a point can be interpreted on the torus."""
        try:
            TorusPoint(point)
        except (TypeError, ValueError):
            return False
        return True

    def __contains__(self, point: object) -> bool:
        """Check whether a point can be interpreted on the torus."""
        return self.contains(point)

    def point(self, point: object) -> TorusPoint:
        """Return a validated point on the torus."""
        return TorusPoint(point)

    def distance(self, left: object, right: object) -> float:
        """Return the flat-torus product distance."""
        left_point = self.point(left)
        right_point = self.point(right)
        first = float(left_point.major_angle.distance_to(right_point.major_angle))
        second = float(left_point.minor_angle.distance_to(right_point.minor_angle))
        return math.sqrt(first * first + second * second)

    def to_2d(self, point: object, method: str = "default") -> Embedding2D:
        """Return a 2D visualization embedding of a torus point."""
        torus_point = self.point(point)
        method = "flat" if method == "default" else method
        if method != "flat":
            raise ValueError(f"Unknown 2D visualization method: {method!r}")
        return torus_point.to_tuple()

    def to_3d(self, point: object, method: str = "default") -> Embedding3D:
        """Return a 3D donut embedding of a torus point."""
        torus_point = self.point(point)
        method = "embedding" if method == "default" else method
        if method != "embedding":
            raise ValueError(f"Unknown 3D visualization method: {method!r}")

        major = float(torus_point.major_angle)
        minor = float(torus_point.minor_angle)
        ring_radius = self.major_radius + self.minor_radius * math.cos(minor)
        return (
            ring_radius * math.cos(major),
            ring_radius * math.sin(major),
            self.minor_radius * math.sin(minor),
        )

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
                _point_cone(2),
            ),
            name=name or "torus-point",
        )

    def patch(
        self,
        major_set: object,
        minor_set: object,
        name: str = "",
    ) -> MetricGeometricObject[TorusPoint]:
        """Return a rectangular angular patch on the torus."""
        major_circle_set = FloatCircleSet(major_set)
        minor_circle_set = FloatCircleSet(minor_set)

        def contains(point: TorusPoint) -> bool:
            torus_point = self.point(point)
            return (
                torus_point.major_angle in major_circle_set and
                torus_point.minor_angle in minor_circle_set
            )

        def local_model(point: TorusPoint) -> LocalConeModel[TorusPoint]:
            torus_point = self.point(point)
            chart = _torus_chart(torus_point)

            major_previous = _previous_circle_point(torus_point.major_angle)
            major_following = _following_circle_point(torus_point.major_angle)
            minor_previous = _previous_circle_point(torus_point.minor_angle)
            minor_following = _following_circle_point(torus_point.minor_angle)

            cone = _product_cone(
                major_previous in major_circle_set,
                major_following in major_circle_set,
                minor_previous in minor_circle_set,
                minor_following in minor_circle_set,
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
    "SphereSpace",
    "TorusPoint",
    "TorusSpace",
]
