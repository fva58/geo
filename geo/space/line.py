"""Real line as a space."""

from __future__ import annotations

import math

from ..cone import EuclideanCone, LocalConeModel, negative_half_line_cone, point_cone, positive_half_line_cone
from ..euclidean import EuclideanNeighborhood, Point
from ..line import (
    ALL_REALS_INTERVAL,
    EMPTY_INTERVAL,
    FULL_INTERVAL,
    Interval,
    Point,
    Set,
)
from ..gobject import GeometricObject
from .base import SpaceChart
from .base import (
    BoxNeighborhood,
    Space as SpaceBase,
    refine_neighborhoods as _refine_neighborhoods,
)


class Neighborhood(BoxNeighborhood[float]):
    """Neighborhood in the real line."""


def _real_line_contains(point: object) -> bool:
    return isinstance(point, (int, float)) and math.isfinite(float(point))


def _real_chart(center: float) -> SpaceChart[float]:
    return SpaceChart(
        lambda point: Point(float(point) - center),
        lambda coordinates: center + coordinates[0],
        dim=1,
        domain_contains=_real_line_contains,
        image=EuclideanNeighborhood.whole(1),
    )


class Space(SpaceBase[float]):
    """The real line with its standard metric."""

    def __init__(self) -> None:
        self._distance = lambda left, right: abs(float(left) - float(right))

    @property
    def dim(self) -> int:
        return 1

    @property
    def point_type(self) -> type:
        return float

    def contains(self, point: object) -> bool:
        return _real_line_contains(point)

    def __contains__(self, point: object) -> bool:
        return self.contains(point)

    def point(
        self,
        point: object,
    ) -> GeometricObject[float]:
        point = float(Point(point))
        return GeometricObject(
            self,
            contains=lambda candidate: float(candidate) == point,
            local_model=lambda candidate: LocalConeModel(
                _real_chart(point),
                point_cone(1),
            ),
        )

    def neighborhood_at(
        self,
        point: object,
        radius: float,
    ) -> Neighborhood:
        center = float(Point(point))
        if center not in self:
            raise ValueError("Point is outside the real line")
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        chart = self.point(center).local_model_at(center).chart
        return Neighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box((-radius, radius)),
        )

    def full(self, radius: float):
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Cover radius must be positive")

        def generator():
            yield self.neighborhood_at(0.0, radius)
            step = 2.0 * radius
            index = 1
            while True:
                yield self.neighborhood_at(index * step, radius)
                yield self.neighborhood_at(-index * step, radius)
                index += 1

        return generator()

    def refine(
        self,
        neighborhoods,
        factor: int = 2,
    ) -> tuple[Neighborhood, ...]:
        return _refine_neighborhoods(tuple(neighborhoods), factor=factor)

    def subset(
        self,
        *point_set: object,
    ) -> GeometricObject[float]:
        subset = Set(*point_set)

        def local_model(point: float) -> LocalConeModel[float]:
            point = float(point)
            previous = math.nextafter(point, -math.inf)
            following = math.nextafter(point, math.inf)
            left_in = previous in subset
            right_in = following in subset
            if left_in and right_in:
                cone = EuclideanCone.whole(1)
            elif right_in:
                cone = positive_half_line_cone()
            elif left_in:
                cone = negative_half_line_cone()
            else:
                cone = point_cone(1)
            return LocalConeModel(_real_chart(point), cone)

        return GeometricObject(
            self,
            contains=lambda point: float(point) in subset,
            local_model=local_model,
        )


__all__ = [
    "Point",
    "Interval",
    "Set",
    "EMPTY_INTERVAL",
    "FULL_INTERVAL",
    "ALL_REALS_INTERVAL",
    "Neighborhood",
    "Space",
]
