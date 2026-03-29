"""General space protocols and standard visualizable spaces."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from .euclidean import FloatPoint
from .floatcircle import FloatCirclePoint
from .riemannian import MetricSpace


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


__all__ = [
    "Embedding2D",
    "Embedding3D",
    "Space",
    "SphereSpace",
    "TorusPoint",
    "TorusSpace",
]
