"""Unit circle as a space."""

from __future__ import annotations

import math

from ..cone import EuclideanCone, LocalConeModel, negative_half_line_cone, point_cone, positive_half_line_cone, _signed_circle_offset
from ..circle import Angle, FULL_INTERVAL, FULL_SET, Interval, Point, Set
from ..euclidean import EuclideanNeighborhood, Point as EuclideanPoint
from ..gobject import GeometricObject
from .base import ManifoldChart
from .base import (
    BoxNeighborhood,
    Space as SpaceBase,
    refine_neighborhoods as _refine_neighborhoods,
)


class Neighborhood(BoxNeighborhood[Point]):
    """Neighborhood in the unit circle."""


def _circle_contains(point: object) -> bool:
    try:
        Point(point)
    except (TypeError, ValueError):
        return False
    return True


def _circle_chart(center: Point) -> ManifoldChart[Point]:
    def forward(point: Point):
        return EuclideanPoint(_signed_circle_offset(center, Point(point)))

    def inverse(coordinates):
        return Point(float(center) + coordinates[0])

    return ManifoldChart(
        forward,
        inverse,
        dim=1,
        domain_contains=lambda point: (
            _circle_contains(point)
            and abs(_signed_circle_offset(center, Point(point))) < math.pi
        ),
        image=EuclideanNeighborhood.box((-math.pi, math.pi)),
    )


class Space(SpaceBase[Point]):
    """The unit circle with its standard arc-length metric."""

    def __init__(self) -> None:
        self._distance = lambda left, right: float(Point(left).distance_to(Point(right)))

    @property
    def dim(self) -> int:
        return 1

    @property
    def point_type(self) -> type:
        return Point

    def contains(self, point: object) -> bool:
        return _circle_contains(point)

    def __contains__(self, point: object) -> bool:
        return self.contains(point)

    def point(
        self,
        point: object,
    ) -> GeometricObject[Point]:
        point = Point(point)
        return GeometricObject(
            self,
            contains=lambda candidate: Point(candidate) == point,
            local_model=lambda candidate: LocalConeModel(
                _circle_chart(point),
                point_cone(1),
            ),
        )

    def neighborhood_at(
        self,
        point: object,
        radius: float,
    ) -> Neighborhood:
        center = Point(point)
        if center not in self:
            raise ValueError("Point is outside the unit circle")
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        if radius >= math.pi:
            raise ValueError("Circle neighborhoods must have radius < pi")
        chart = self.point(center).local_model_at(center).chart
        return Neighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box((-radius, radius)),
        )

    def full(self, radius: float) -> tuple[Neighborhood, ...]:
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Cover radius must be positive")
        if radius >= math.pi:
            raise ValueError("Circle cover radius must be < pi")
        steps = max(3, int(math.ceil((2.0 * math.pi) / (2.0 * radius))))
        return tuple(
            self.neighborhood_at(2.0 * math.pi * index / steps, radius)
            for index in range(steps)
        )

    def refine(
        self,
        neighborhoods,
        factor: int = 2,
    ) -> tuple[Neighborhood, ...]:
        return _refine_neighborhoods(tuple(neighborhoods), factor=factor)

    def subset(
        self,
        *point_set: object,
    ) -> GeometricObject[Point]:
        point_set = Set(*point_set)

        def local_model(point: Point) -> LocalConeModel[Point]:
            point = Point(point)
            if float(point) == 0.0:
                previous = Point(Angle.MAX_ANGLE)
            else:
                previous = Point(math.nextafter(float(point), -math.inf))
            following = Point(math.nextafter(float(point), math.inf))
            left_in = previous in point_set
            right_in = following in point_set
            if left_in and right_in:
                cone = EuclideanCone.whole(1)
            elif right_in:
                cone = positive_half_line_cone()
            elif left_in:
                cone = negative_half_line_cone()
            else:
                cone = point_cone(1)
            return LocalConeModel(_circle_chart(point), cone)

        return GeometricObject(
            self,
            contains=lambda point: Point(point) in point_set,
            local_model=local_model,
        )

    def arc(
        self,
        start: object,
        end: object,
    ) -> GeometricObject[Point]:
        return self.subset((start, end))


__all__ = [
    "Angle",
    "Point",
    "Interval",
    "Set",
    "FULL_INTERVAL",
    "FULL_SET",
    "Neighborhood",
    "Space",
]
