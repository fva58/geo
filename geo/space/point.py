"""Zero-dimensional one-point space."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..cone import EuclideanCone, LocalConeModel
from ..euclidean import EuclideanNeighborhood, FloatPoint
from ..gobject import GeometricObject
from ..manifold import ManifoldChart
from .base import Space


_POINT = FloatPoint.origin(0)


class _PointManifold:
    """Underlying one-point manifold."""

    dim = 0

    def contains(self, point: object) -> bool:
        try:
            return FloatPoint(point) == _POINT
        except (TypeError, ValueError):
            return False

    def __contains__(self, point: object) -> bool:
        return self.contains(point)


def _point_chart() -> ManifoldChart[FloatPoint]:
    """Return the unique zero-dimensional chart."""
    return ManifoldChart(
        lambda point: _POINT,
        lambda coordinates: _POINT,
        dim=0,
        domain_contains=lambda point: FloatPoint(point) == _POINT,
        image=EuclideanNeighborhood.box(),
        name="point-chart",
    )


@dataclass(frozen=True)
class PointNeighborhood:
    """The unique neighborhood in the one-point space."""

    manifold: Space[FloatPoint]
    chart: ManifoldChart[FloatPoint]
    center: FloatPoint = _POINT
    name: str = ""

    @property
    def image(self) -> EuclideanNeighborhood:
        """Return the zero-dimensional chart image."""
        return EuclideanNeighborhood.box()

    def contains(self, point: FloatPoint) -> bool:
        """Check whether the unique point belongs."""
        return FloatPoint(point) == _POINT

    def __contains__(self, point: FloatPoint) -> bool:
        """Check whether the unique point belongs."""
        return self.contains(point)

    def center_point(self) -> FloatPoint:
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

    def probe_points(self) -> tuple[FloatPoint, ...]:
        """Return points used for local probing."""
        return (_POINT,)

    def subdivide(self) -> tuple["PointNeighborhood", ...]:
        """Return the unique refinement."""
        return (self,)


class Point:
    """Zero-dimensional metric space with one point."""

    def __init__(self, name: str = "") -> None:
        self.manifold = _PointManifold()
        self.name = name or "Point"
        self._chart = _point_chart()

    @property
    def dim(self) -> int:
        """Return the space dimension."""
        return 0

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"Point({label[2:]})" if label else "Point()"

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

    def point(self, point: object = ()) -> FloatPoint:
        """Return the unique point."""
        if point not in self:
            raise ValueError("Point is outside the point space")
        return _POINT

    def point_object(
        self,
        point: object = (),
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        """Return the singleton object in the point space."""
        self.point(point)
        return GeometricObject(
            self,
            contains=lambda candidate: candidate in self,
            local_model=lambda candidate: LocalConeModel(
                self._chart,
                EuclideanCone.whole(0),
            ),
            name=name or "point-space-point",
        )

    def neighborhood_at(
        self,
        point: object = (),
        radius: float = 1.0,
        name: str = "",
    ) -> PointNeighborhood:
        """Return the unique neighborhood."""
        self.point(point)
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        return PointNeighborhood(self, self._chart, name=name or "point-neighborhood")

    def full_cover(self, radius: float) -> tuple[PointNeighborhood, ...]:
        """Return the unique full cover."""
        return (self.neighborhood_at((), radius),)

    def refine_cover(
        self,
        neighborhoods,
        factor: int = 2,
    ) -> tuple[PointNeighborhood, ...]:
        """Return the unchanged refinement."""
        del factor
        return tuple(neighborhoods)


__all__ = ["Point", "PointNeighborhood"]
