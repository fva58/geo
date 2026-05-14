"""Zero-dimensional one-point space."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..cone import EuclideanCone, LocalConeModel
from ..euclidean import EuclideanNeighborhood, Point
from ..gobject import GeometricObject
from ..manifold import ManifoldChart
from .base import Neighborhood as NeighborhoodBase, Space as SpaceBase


_POINT = Point.origin(0)


class _PointManifold:
    """Underlying one-point manifold."""

    dim = 0

    def contains(self, point: object) -> bool:
        try:
            return Point(point) == _POINT
        except (TypeError, ValueError):
            return False

    def __contains__(self, point: object) -> bool:
        return self.contains(point)


def _point_chart() -> ManifoldChart[Point]:
    """Return the unique zero-dimensional chart."""
    return ManifoldChart(
        lambda point: _POINT,
        lambda coordinates: _POINT,
        dim=0,
        domain_contains=lambda point: Point(point) == _POINT,
        image=EuclideanNeighborhood.box(),
    )


@dataclass(frozen=True)
class Neighborhood(NeighborhoodBase[Point]):
    """The unique neighborhood in the one-point space."""

    manifold: SpaceBase[Point]
    chart: ManifoldChart[Point]
    center: Point = _POINT

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
        self.manifold = _PointManifold()
        self._chart = _point_chart()

    @property
    def dim(self) -> int:
        """Return the space dimension."""
        return 0

    @property
    def point_type(self) -> type:
        """Return the type of points in this space."""
        return Point

    def __repr__(self) -> str:
        """Return a debug representation."""
        return "Space()"

    def contains(self, point: object) -> bool:
        """Check whether a point belongs to the space."""
        return point in self.manifold

    def __contains__(self, point: object) -> bool:
        """Check whether a point belongs to the space."""
        return self.contains(point)

    def distance(self, left: object, right: object) -> float:
        """Return the unique distance."""
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
