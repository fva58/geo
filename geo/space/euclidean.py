"""Euclidean spaces."""

from __future__ import annotations

import itertools

from ..euclidean import EuclideanNeighborhood, FloatPoint, FloatVector
from ..gobject import GeometricObject
from ..manifold import ChartNeighborhood
from ._euclidean_impl import (
    Ball,
    Cube,
    CubeSurface,
    Ellipsoid,
    EllipsoidSurface,
    EuclideanPointObject,
    EuclideanSpace,
    HalfSpace,
    HalfPlane,
    Hyperplane,
    Parallelepiped,
    ParallelepipedSurface,
    PlanarAngle,
    Sphere as EuclideanSphereObject,
    WholeSpace,
    WholePlane,
)
from .base import ChartedSpace, refine_neighborhoods as _refine_neighborhoods


class Euclidean(ChartedSpace[FloatPoint]):
    """Euclidean space with its standard metric."""

    def __init__(self, dim: int, name: str = "") -> None:
        self._dim = int(dim)
        super().__init__(
            EuclideanSpace(self._dim),
            distance=lambda left, right: FloatPoint(left).distance_to(
                FloatPoint(right)
            ),
            name=name or f"R^{self._dim}",
        )

    def _coerce_point(self, point: object) -> FloatPoint:
        point = FloatPoint(point)
        if point.dim != self.dim:
            raise ValueError(f"Expected a point in R^{self.dim}")
        return point

    def _coerce_vector(self, vector: object) -> FloatVector:
        vector = FloatVector(vector)
        if vector.dim != self.dim:
            raise ValueError(f"Expected a vector in R^{self.dim}")
        return vector

    def point(
        self,
        point: FloatPoint,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        point = self._coerce_point(point)
        return GeometricObject.from_charted(
            self,
            EuclideanPointObject(point, name=name),
            name=name,
        )

    def neighborhood_at(
        self,
        point: FloatPoint,
        radius: float,
        name: str = "",
    ) -> ChartNeighborhood[FloatPoint]:
        center = FloatPoint(point)
        if center not in self:
            raise ValueError("Point is outside the Euclidean space")
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        chart = self.point(center).local_model_at(center).chart
        return ChartNeighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box(*(((-radius, radius),) * self.dim)),
            name=name or "euclidean-neighborhood",
        )

    def full_cover(self, radius: float):
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Cover radius must be positive")

        def shell_indices(shell: int):
            if self.dim == 1:
                if shell == 0:
                    yield (0,)
                else:
                    yield (shell,)
                    yield (-shell,)
                return
            ranges = [range(-shell, shell + 1) for _ in range(self.dim)]
            for index in itertools.product(*ranges):
                if max(abs(value) for value in index) == shell:
                    yield index

        def generator():
            step = 2.0 * radius
            shell = 0
            while True:
                for index in shell_indices(shell):
                    yield self.neighborhood_at(
                        FloatPoint([step * value for value in index]),
                        radius,
                    )
                shell += 1

        return generator()

    def refine_cover(
        self,
        neighborhoods,
        factor: int = 2,
    ) -> tuple[ChartNeighborhood[FloatPoint], ...]:
        return _refine_neighborhoods(tuple(neighborhoods), factor=factor)

    def whole(self, name: str = "") -> GeometricObject[FloatPoint]:
        return self.wrap(WholeSpace(self.dim, name=name), name=name)

    def whole_space(self, name: str = "") -> GeometricObject[FloatPoint]:
        return self.whole(name=name)

    def hyperplane(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        normal = self._coerce_vector(normal)
        return self.wrap(Hyperplane(normal, offset=offset, name=name), name=name)

    def half_space(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        normal = self._coerce_vector(normal)
        return self.wrap(HalfSpace(normal, offset=offset, name=name), name=name)

    def whole_plane(
        self,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        if self.dim != 2:
            raise ValueError("whole_plane() is only defined for Euclidean(2)")
        return self.wrap(WholePlane(name=name), name=name)

    def sphere(
        self,
        center: FloatPoint,
        radius: float,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        center = self._coerce_point(center)
        return self.wrap(EuclideanSphereObject(center, radius, name=name), name=name)

    def ball(
        self,
        center: FloatPoint,
        radius: float,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        center = self._coerce_point(center)
        return self.wrap(Ball(center, radius, name=name), name=name)

    def disk(
        self,
        center: FloatPoint,
        radius: float,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        return self.ball(center, radius, name=name)

    def circle(
        self,
        center: FloatPoint,
        radius: float,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        if self.dim != 2:
            raise ValueError("circle() is only defined for Euclidean(2)")
        return self.sphere(center, radius, name=name)

    def ellipsoid_surface(
        self,
        center: FloatPoint,
        semiaxes,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        center = self._coerce_point(center)
        return self.wrap(EllipsoidSurface(center, semiaxes, name=name), name=name)

    def ellipsoid(
        self,
        center: FloatPoint,
        semiaxes,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        center = self._coerce_point(center)
        return self.wrap(Ellipsoid(center, semiaxes, name=name), name=name)

    def parallelepiped_surface(
        self,
        center: FloatPoint,
        spanning_vectors,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        center = self._coerce_point(center)
        return self.wrap(
            ParallelepipedSurface(center, spanning_vectors, name=name),
            name=name,
        )

    def parallelepiped(
        self,
        center: FloatPoint,
        spanning_vectors,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        center = self._coerce_point(center)
        return self.wrap(
            Parallelepiped(center, spanning_vectors, name=name),
            name=name,
        )

    def cube_surface(
        self,
        center: FloatPoint,
        half_extent: float,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        center = self._coerce_point(center)
        return self.wrap(CubeSurface(center, half_extent, name=name), name=name)

    def cube(
        self,
        center: FloatPoint,
        half_extent: float,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        center = self._coerce_point(center)
        return self.wrap(Cube(center, half_extent, name=name), name=name)

    def half_plane(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        if self.dim != 2:
            raise ValueError("half_plane() is only defined for Euclidean(2)")
        normal = self._coerce_vector(normal)
        return self.wrap(HalfPlane(normal, offset=offset, name=name), name=name)

    def angle(
        self,
        apex: FloatPoint,
        start,
        end,
        name: str = "",
    ) -> GeometricObject[FloatPoint]:
        if self.dim != 2:
            raise ValueError("angle() is only defined for Euclidean(2)")
        apex = self._coerce_point(apex)
        return self.wrap(PlanarAngle(apex, start, end, name=name), name=name)


__all__ = ["Euclidean"]
