"""Toroidal spaces."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from ..cone import EuclideanCone, LocalConeModel
from ..euclidean import EuclideanNeighborhood, FloatPoint
from ..gobject import GeometricObject
from ..circle import Angle, Point, Set
from ..manifold import ManifoldChart
from .base import BoxNeighborhood, refine_neighborhoods as _refine_neighborhoods


class Neighborhood(BoxNeighborhood["TorusPoint"]):
    """Neighborhood on a torus."""


def _point_cone(dim: int) -> EuclideanCone:
    return EuclideanCone(
        dim,
        contains=lambda point: FloatPoint(point) == FloatPoint.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
        name="point",
    )


def _previous_circle_point(point: Point) -> Point:
    if float(point) == 0.0:
        return Point(Angle.MAX_ANGLE)
    return Point(math.nextafter(float(point), -math.inf))


def _following_circle_point(point: Point) -> Point:
    following = math.nextafter(float(point), math.inf)
    if following >= Angle.TWO_PI:
        following = 0.0
    return Point(following)


def _signed_circle_difference(
    target: Point,
    base: Point,
) -> float:
    difference = float(Point(target)) - float(Point(base))
    if difference > math.pi:
        difference -= Angle.TWO_PI
    elif difference <= -math.pi:
        difference += Angle.TWO_PI
    return difference


def _product_cone(axis_flags: Sequence[tuple[bool, bool]], name: str) -> EuclideanCone:
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


class TorusPoint(tuple):
    """Point on a torus represented by angular coordinates."""

    __slots__ = ()

    def __new__(cls, *angles: object) -> "TorusPoint":
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
                    tuple(Point(angle) for angle in point),
                )
            if point == 0.0:
                angles = (0.0, 0.0)
            else:
                raise TypeError(
                    "TorusPoint requires at least two angles or one angle tuple"
                )
        return super().__new__(
            cls,
            tuple(Point(angle) for angle in angles),
        )

    @property
    def dim(self) -> int:
        return len(self)

    @property
    def major_angle(self) -> Point:
        if self.dim < 1:
            raise ValueError("TorusPoint has no angles")
        return self[0]

    @property
    def minor_angle(self) -> Point:
        if self.dim < 2:
            raise ValueError("TorusPoint has no minor angle")
        return self[1]

    def to_tuple(self) -> tuple[float, ...]:
        return tuple(float(angle) for angle in self)

    def __repr__(self) -> str:
        return f"TorusPoint{self.to_tuple()}"


def _torus_chart(base_point: TorusPoint) -> ManifoldChart[TorusPoint]:
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


class Space:
    """Flat torus modeled as a product of circles."""

    def __init__(
        self,
        dim: int = 2,
        major_radius: float = 2.0,
        minor_radius: float = 0.5,
        radii: Sequence[float] | None = None,
        name: str = "",
    ) -> None:
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

    def __repr__(self) -> str:
        label = f", name={self.name!r}" if self.name else ""
        return f"Space(dim={self.dim}, radii={self.radii}{label})"

    def contains(self, point: object) -> bool:
        try:
            return TorusPoint(point).dim == self.dim
        except (TypeError, ValueError):
            return False

    def __contains__(self, point: object) -> bool:
        return self.contains(point)

    def point(self, point: object) -> TorusPoint:
        torus_point = TorusPoint(point)
        if torus_point.dim != self.dim:
            raise ValueError(
                f"Expected a {self.dim}-dimensional torus point"
            )
        return torus_point

    def distance(self, left: object, right: object) -> float:
        left_point = self.point(left)
        right_point = self.point(right)
        squared = 0.0
        for index in range(self.dim):
            diff = float(left_point[index].distance_to(right_point[index]))
            squared += diff * diff
        return math.sqrt(squared)

    def whole(self, name: str = "") -> GeometricObject[TorusPoint]:
        return GeometricObject(
            self,
            contains=lambda point: point in self,
            local_model=lambda point: LocalConeModel(
                _torus_chart(self.point(point)),
                EuclideanCone.whole(self.dim),
            ),
            name=name or "torus",
        )

    def point_object(
        self,
        point: object,
        name: str = "",
    ) -> GeometricObject[TorusPoint]:
        torus_point = self.point(point)
        return GeometricObject(
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
    ) -> Neighborhood:
        center = self.point(point)
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        if radius >= math.pi:
            raise ValueError("Torus angular neighborhoods must have radius < pi")
        chart = _torus_chart(center)
        return Neighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box(*(((-radius, radius),) * self.dim)),
            name=name or "torus-neighborhood",
        )

    def full_cover(
        self,
        radius: float,
    ) -> tuple[Neighborhood, ...]:
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
    ) -> tuple[Neighborhood, ...]:
        return _refine_neighborhoods(tuple(neighborhoods), factor=factor)

    def patch(
        self,
        *angle_sets: object,
        name: str = "",
    ) -> GeometricObject[TorusPoint]:
        if len(angle_sets) != self.dim:
            raise ValueError(
                f"Expected {self.dim} angular sets, got {len(angle_sets)}"
            )
        circle_sets = tuple(Set(angle_set) for angle_set in angle_sets)

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
            cone = _product_cone(axis_flags, "torus-patch")
            return LocalConeModel(chart, cone)

        return GeometricObject(
            self,
            contains=contains,
            local_model=local_model,
            name=name or "torus-patch",
        )


__all__ = ["Neighborhood", "Space", "TorusPoint"]
