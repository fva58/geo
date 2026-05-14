"""Euclidean spaces."""

from __future__ import annotations

import itertools
import math

from ..euclidean import EuclideanNeighborhood, Point, Vector
from ..gobject import GeometricObject
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
from .base import (
    BoxNeighborhood,
    ChartedSpace,
    refine_neighborhoods as _refine_neighborhoods,
)


class Neighborhood(BoxNeighborhood[Point]):
    """Neighborhood in Euclidean space."""


class Space(ChartedSpace[Point]):
    """Euclidean space with its standard metric."""

    def __init__(
        self,
        dim: int,
        max_size: float | None = None,
    ) -> None:
        self._dim = int(dim)
        self.max_size = None if max_size is None else float(max_size)
        if self.max_size is not None and (
            self.max_size <= 0.0 or not math.isfinite(self.max_size)
        ):
            raise ValueError("max_size must be a positive finite number")
        super().__init__(
            EuclideanSpace(self._dim),
            distance=lambda left, right: Point(left).distance_to(
                Point(right)
            ),
        )

    @property
    def point_type(self) -> type:
        """Return the type of points in this space."""
        return Point

    def _coerce_point(self, point: object) -> Point:
        point = Point(point)
        if point.dim != self.dim:
            raise ValueError(f"Expected a point in R^{self.dim}")
        return point

    def _coerce_vector(self, vector: object) -> Vector:
        vector = Vector(vector)
        if vector.dim != self.dim:
            raise ValueError(f"Expected a vector in R^{self.dim}")
        return vector

    def point(
        self,
        point: Point,
    ) -> GeometricObject[Point]:
        point = self._coerce_point(point)
        return GeometricObject.from_charted(
            self,
            EuclideanPointObject(point),
        )

    def neighborhood_at(
        self,
        point: Point,
        radius: float,
    ) -> Neighborhood:
        center = Point(point)
        if center not in self:
            raise ValueError("Point is outside the Euclidean space")
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        chart = self.point(center).local_model_at(center).chart
        return Neighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box(*(((-radius, radius),) * self.dim)),
        )

    def full(self, radius: float):
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Cover radius must be positive")
        step = 2.0 * radius

        if self.max_size is None:
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
                shell = 0
                while True:
                    for index in shell_indices(shell):
                        yield self.neighborhood_at(
                            Point([step * value for value in index]),
                            radius,
                        )
                    shell += 1

            return generator()

        shell = int(math.ceil(self.max_size / step))
        return tuple(
            self.neighborhood_at(
                Point([step * value for value in index]),
                radius,
            )
            for index in itertools.product(
                range(-shell, shell + 1),
                repeat=self.dim,
            )
        )

    def refine(
        self,
        neighborhoods,
        factor: int = 2,
    ) -> tuple[Neighborhood, ...]:
        return _refine_neighborhoods(tuple(neighborhoods), factor=factor)

    def whole(self) -> GeometricObject[Point]:
        return self.wrap(WholeSpace(self.dim))

    def whole_space(self) -> GeometricObject[Point]:
        return self.whole()

    def hyperplane(
        self,
        normal: Vector,
        offset: float = 0.0,
    ) -> GeometricObject[Point]:
        normal = self._coerce_vector(normal)
        return self.wrap(Hyperplane(normal, offset=offset))

    def half_space(
        self,
        normal: Vector,
        offset: float = 0.0,
    ) -> GeometricObject[Point]:
        normal = self._coerce_vector(normal)
        return self.wrap(HalfSpace(normal, offset=offset))

    def whole_plane(
        self,
    ) -> GeometricObject[Point]:
        if self.dim != 2:
            raise ValueError("whole_plane() is only defined for euclidean.Space(2)")
        return self.wrap(WholePlane())

    def sphere(
        self,
        center: Point,
        radius: float,
    ) -> GeometricObject[Point]:
        center = self._coerce_point(center)
        return self.wrap(EuclideanSphereObject(center, radius))

    def ball(
        self,
        center: Point,
        radius: float,
    ) -> GeometricObject[Point]:
        center = self._coerce_point(center)
        return self.wrap(Ball(center, radius))

    def disk(
        self,
        center: Point,
        radius: float,
    ) -> GeometricObject[Point]:
        return self.ball(center, radius)

    def circle(
        self,
        center: Point,
        radius: float,
    ) -> GeometricObject[Point]:
        if self.dim != 2:
            raise ValueError("circle() is only defined for euclidean.Space(2)")
        return self.sphere(center, radius)

    def ellipsoid_surface(
        self,
        center: Point,
        semiaxes,
    ) -> GeometricObject[Point]:
        center = self._coerce_point(center)
        return self.wrap(EllipsoidSurface(center, semiaxes))

    def ellipsoid(
        self,
        center: Point,
        semiaxes,
    ) -> GeometricObject[Point]:
        center = self._coerce_point(center)
        return self.wrap(Ellipsoid(center, semiaxes))

    def parallelepiped_surface(
        self,
        center: Point,
        spanning_vectors,
    ) -> GeometricObject[Point]:
        center = self._coerce_point(center)
        return self.wrap(
            ParallelepipedSurface(center, spanning_vectors),
        )

    def parallelepiped(
        self,
        center: Point,
        spanning_vectors,
    ) -> GeometricObject[Point]:
        center = self._coerce_point(center)
        return self.wrap(
            Parallelepiped(center, spanning_vectors),
        )

    def cube_surface(
        self,
        center: Point,
        half_extent: float,
    ) -> GeometricObject[Point]:
        center = self._coerce_point(center)
        return self.wrap(CubeSurface(center, half_extent))

    def cube(
        self,
        center: Point,
        half_extent: float,
    ) -> GeometricObject[Point]:
        center = self._coerce_point(center)
        return self.wrap(Cube(center, half_extent))

    def half_plane(
        self,
        normal: Vector,
        offset: float = 0.0,
    ) -> GeometricObject[Point]:
        if self.dim != 2:
            raise ValueError("half_plane() is only defined for euclidean.Space(2)")
        normal = self._coerce_vector(normal)
        return self.wrap(HalfPlane(normal, offset=offset))

    def angle(
        self,
        apex: Point,
        start,
        end,
    ) -> GeometricObject[Point]:
        if self.dim != 2:
            raise ValueError("angle() is only defined for euclidean.Space(2)")
        apex = self._coerce_point(apex)
        return self.wrap(PlanarAngle(apex, start, end))


__all__ = ["Neighborhood", "Space"]
