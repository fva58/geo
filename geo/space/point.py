"""Zero-dimensional one-point space."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..cone import EuclideanCone, LocalConeModel
from ..euclidean import EuclideanNeighborhood, Point
from ..gobject import GeometricObject
from .base import SpaceChart
from .base import Neighborhood as NeighborhoodBase, Space as SpaceBase


_POINT = Point.origin(0)


def _point_contains(point: object) -> bool:
    try:
        return Point(point) == _POINT
    except (TypeError, ValueError):
        return False


def _point_chart() -> SpaceChart[Point]:
    """Return the unique zero-dimensional chart."""
    return SpaceChart(
        lambda point: _POINT,
        lambda coordinates: _POINT,
        dim=0,
        domain_contains=_point_contains,
        image=EuclideanNeighborhood.box(),
    )


@dataclass(frozen=True)
class Neighborhood(NeighborhoodBase[Point]):
    """The unique neighborhood in the one-point space."""

    space: SpaceBase[Point]
    chart: SpaceChart[Point]
    center: Point = field(default_factory=lambda: _POINT)

    @property
    def image(self) -> EuclideanNeighborhood:
        """Return the zero-dimensional chart image."""
        return EuclideanNeighborhood.box()

    def contains(self, point: Point) -> bool:
        """Check whether the unique point belongs."""
        return Point(point) == _POINT

    def __contains__(self, point: Point) -> bool:
        """Check whether the unique point belongs."""
        return self.contains(point)

    def center_point(self) -> Point:
        """Return the unique point."""
        return _POINT

    def inner_radius(self) -> float:
        """Return the maximal included radius."""
        return math.inf

    def outer_radius(self) -> float:
        """Return the minimal containing radius."""
        return 0.0

    def diameter(self) -> float:
        """Return the neighborhood diameter."""
        return 0.0

    def subdivide(self) -> tuple["Neighborhood", ...]:
        """Return the unique refinement."""
        return (self,)


class Space(SpaceBase):
    """Zero-dimensional metric space with one point."""

    def __init__(self) -> None:
        self._chart = _point_chart()

    @property
    def dim(self) -> int:
        return 0

    @property
    def point_type(self) -> type:
        return Point

    def __repr__(self) -> str:
        return "Space()"

    def contains(self, point: object) -> bool:
        return _point_contains(point)

    def __contains__(self, point: object) -> bool:
        return self.contains(point)

    def distance(self, left: object, right: object) -> float:
        if left not in self or right not in self:
            raise ValueError("Points must belong to the point space")
        return 0.0

    def point(self, point: object = ()) -> Point:
        """Return the unique point."""
        if point not in self:
            raise ValueError("Point is outside the point space")
        return _POINT

    def point_object(
        self,
        point: object = (),
    ) -> GeometricObject[Point]:
        """Return the singleton object in the point space."""
        self.point(point)
        return GeometricObject(
            self,
            contains=lambda candidate: candidate in self,
            local_model=lambda candidate: LocalConeModel(
                self._chart,
                EuclideanCone.whole(0),
            ),
        )

    def neighborhood_at(
        self,
        point: object = (),
        radius: float = 1.0,
    ) -> Neighborhood:
        """Return the unique neighborhood."""
        self.point(point)
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        return Neighborhood(self, self._chart)

    def full(self, radius: float) -> tuple[Neighborhood, ...]:
        """Return the unique full cover."""
        return (self.neighborhood_at((), radius),)

    def refine(
        self,
        neighborhoods,
        factor: int = 2,
    ) -> tuple[Neighborhood, ...]:
        """Return the unchanged refinement."""
        del factor
        return tuple(neighborhoods)


__all__ = ["Neighborhood", "Space"]
