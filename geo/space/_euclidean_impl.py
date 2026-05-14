"""Private Euclidean object implementations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable

import numpy as np

from ..circle import Interval as CircleInterval, Point as CirclePoint, Set as CircleSet
from ..cone import (
    CircleSphereObject,
    EuclideanCone,
    LocalConeModel,
    SphericalCone,
    _active_box_constraints,
    _coerce_affine_matrix,
    _coerce_nonzero_normal,
    _cross_2d,
    _isclose,
    _point_array,
    half_space_cone,
    hyperplane_cone,
    point_cone,
)
from ..euclidean import EuclideanNeighborhood, Point, Vector
from ..gobject import ChartedGeometricObject
from .base import ManifoldChart, Space as SpaceBase


class EuclideanSpace(SpaceBase):
    """Canonical Euclidean space ``R^n``."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("Euclidean space dimension must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def point_type(self) -> type:
        return Point

    def contains(self, point: object) -> bool:
        try:
            coordinates = Point(point)
        except (TypeError, ValueError):
            return False
        return coordinates.dim == self.dim

    def __contains__(self, point: object) -> bool:
        return self.contains(point)

    def distance(self, left: Point, right: Point) -> float:
        return Point(left).distance_to(Point(right))

    def full(self, radius: float):
        raise NotImplementedError("EuclideanSpace does not support full covers")

    def refine(self, neighborhoods, factor: int = 2):
        raise NotImplementedError("EuclideanSpace does not support refinement")


def euclidean_chart(center: Point) -> ManifoldChart[Point]:
    """Return the canonical translated chart in Euclidean space."""
    dim = center.dim
    space = EuclideanSpace(dim)
    return ManifoldChart(
        lambda point: Point(point) - center,
        lambda coordinates: center + Vector(coordinates),
        dim=dim,
        domain_contains=space.contains,
        image=EuclideanNeighborhood.whole(dim),
    )


def _projection_hyperplane_data(hyperplane):
    hyperplane = getattr(hyperplane, "_charted_source_object", hyperplane)
    if not hasattr(hyperplane, "normal") or not hasattr(hyperplane, "offset"):
        raise TypeError("Projection requires hyperplane objects")
    return hyperplane


def _validate_projection_hyperplanes(source_hyperplane, target_hyperplane) -> None:
    source = _projection_hyperplane_data(source_hyperplane)
    target = _projection_hyperplane_data(target_hyperplane)
    if source.normal.dim != target.normal.dim:
        raise ValueError("Projection hyperplanes must have equal dimension")


def _numeric_jacobian(
    mapping: Callable[[Point], Point],
    point: Point,
    step: float = 1e-6,
) -> np.ndarray:
    point = Point(point)
    base = _point_array(point)
    image_dim = Point(mapping(point)).dim
    jacobian = np.zeros((image_dim, point.dim), dtype=float)
    for index in range(point.dim):
        delta = np.zeros(point.dim, dtype=float)
        delta[index] = step
        image_plus = _point_array(mapping(Point(base + delta)))
        image_minus = _point_array(mapping(Point(base - delta)))
        jacobian[:, index] = (image_plus - image_minus) / (2.0 * step)
    return jacobian


def parallel_projection_inverse(
    source_hyperplane,
    target_hyperplane,
    direction: Vector,
) -> Callable[[Point], Point]:
    source_hyperplane = _projection_hyperplane_data(source_hyperplane)
    target_hyperplane = _projection_hyperplane_data(target_hyperplane)
    direction = Vector(direction)
    _validate_projection_hyperplanes(source_hyperplane, target_hyperplane)
    if direction.dim != source_hyperplane.normal.dim:
        raise ValueError("Projection direction dimension mismatch")
    denominator = source_hyperplane.normal.dot(direction)
    if _isclose(denominator):
        raise ValueError(
            "Projection direction must not be parallel to the source hyperplane"
        )

    def inverse_map(point: Point) -> Point:
        point = Point(point)
        scalar = (
            source_hyperplane.offset
            - source_hyperplane.normal.dot(Vector(point))
        ) / denominator
        return point + scalar * direction

    return inverse_map


def central_projection_inverse(
    source_hyperplane,
    target_hyperplane,
    center: Point,
) -> Callable[[Point], Point]:
    source_hyperplane = _projection_hyperplane_data(source_hyperplane)
    target_hyperplane = _projection_hyperplane_data(target_hyperplane)
    center = Point(center)
    _validate_projection_hyperplanes(source_hyperplane, target_hyperplane)
    if center.dim != source_hyperplane.normal.dim:
        raise ValueError("Projection center dimension mismatch")
    source_offset = source_hyperplane.offset
    source_center_value = source_hyperplane.normal.dot(Vector(center))
    if _isclose(source_offset, source_center_value):
        raise ValueError("Projection center must not lie in the source hyperplane")

    def inverse_map(point: Point) -> Point:
        point = Point(point)
        direction = point - center
        denominator = source_hyperplane.normal.dot(direction)
        if _isclose(denominator):
            raise ValueError("Projection ray is parallel to the source hyperplane")
        scalar = (source_offset - source_center_value) / denominator
        return center + scalar * direction

    return inverse_map


def projected_local_model(
    source_object,
    source_point: Point,
    target_hyperplane,
    target_point: Point,
    inverse_map: Callable[[Point], Point],
) -> LocalConeModel[Point]:
    source_model = source_object.local_model_at(source_point)
    target_model = target_hyperplane.local_model_at(target_point)
    jacobian = _numeric_jacobian(inverse_map, target_point)
    dim = target_point.dim
    cone = EuclideanCone(
        dim,
        contains=lambda coordinates: (
            target_model.cone.contains(coordinates)
            and source_model.cone.contains(
                Point(jacobian @ _point_array(coordinates))
            )
        ),
        apex=Point.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
    )
    return LocalConeModel(target_model.chart, cone)


def _centered_cone_model(source_model, point: Point) -> LocalConeModel[Point]:
    from ..gobject import _centered_chart_at

    centered_chart = _centered_chart_at(source_model.chart, point)
    chart_origin = Point(source_model.chart(point))
    translation = Vector(chart_origin)
    dim = centered_chart.dim
    centered_cone = EuclideanCone(
        dim,
        contains=lambda coordinates: source_model.cone.contains(
            Point(coordinates) + translation
        ),
        apex=Point.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
    )
    return LocalConeModel(centered_chart, centered_cone)


def _scalar_subset_local_model(
    source_object,
    point: Point,
    scalar: Callable[[Point], float],
) -> LocalConeModel[Point]:
    point = Point(point)
    source_model = _centered_cone_model(source_object.local_model_at(point), point)
    value = float(scalar(point))
    if value > 1e-9:
        return source_model

    def local_scalar(coordinates: Point) -> Point:
        local_point = source_model.chart.inverse(coordinates)
        return Point(float(scalar(local_point)))

    gradient = _numeric_jacobian(
        local_scalar,
        Point.origin(source_model.chart.dim),
    )[0]
    if np.linalg.norm(gradient) <= 1e-9:
        return source_model

    cone = EuclideanCone(
        source_model.chart.dim,
        contains=lambda coordinates: (
            source_model.cone.contains(coordinates)
            and float(gradient @ _point_array(Point(coordinates))) >= -1e-9
        ),
        apex=Point.origin(source_model.chart.dim),
        neighborhood=EuclideanNeighborhood.whole(source_model.chart.dim),
    )
    return LocalConeModel(source_model.chart, cone)


def _scalar_threshold_subset(source_object, scalar):
    return ChartedGeometricObject(
        source_object.space,
        contains=lambda point: (
            point in source_object and float(scalar(Point(point))) >= -1e-9
        ),
        local_model=lambda point: _scalar_subset_local_model(
            source_object,
            Point(point),
            scalar,
        ),
    )


def _empty_euclidean_object(space: EuclideanSpace):
    return ChartedGeometricObject(
        space,
        contains=lambda point: False,
        local_model=lambda point: (_ for _ in ()).throw(
            ValueError("Empty object has no local model")
        ),
    )


class EuclideanPointObject(ChartedGeometricObject[Point]):
    """Zero-dimensional object supported at one Euclidean point."""

    def __init__(self, point: Point) -> None:
        self.point = Point(point)
        object_space = EuclideanSpace(self.point.dim)
        super().__init__(
            object_space,
            contains=lambda candidate: Point(candidate) == self.point,
            local_model=lambda candidate: LocalConeModel(
                euclidean_chart(self.point),
                point_cone(self.point.dim),
            ),
        )


class WholeSpace(ChartedGeometricObject[Point]):
    """The whole Euclidean space ``R^n``."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        object_space = EuclideanSpace(dim)
        super().__init__(
            object_space,
            contains=lambda point: True,
            local_model=lambda point: LocalConeModel(
                euclidean_chart(Point(point)),
                EuclideanCone.whole(dim),
            ),
        )


class Hyperplane(ChartedGeometricObject[Point]):
    """Affine hyperplane ``{x : <normal, x> = offset}``."""

    def __init__(self, normal: Vector, offset: float = 0.0) -> None:
        self.normal = _coerce_nonzero_normal(normal)
        self.offset = float(offset)
        object_space = EuclideanSpace(self.normal.dim)
        super().__init__(
            object_space,
            contains=self._contains,
            local_model=self._local_model,
        )

    def _contains(self, point: Point) -> bool:
        point = Point(point)
        value = Vector(point).dot(self.normal)
        return _isclose(value, self.offset)

    def _local_model(self, point: Point) -> LocalConeModel[Point]:
        point = Point(point)
        cone = hyperplane_cone(self.normal)
        return LocalConeModel(euclidean_chart(point), cone)


class HalfSpace(ChartedGeometricObject[Point]):
    """Closed half-space ``{x : <normal, x> >= offset}``."""

    def __init__(self, normal: Vector, offset: float = 0.0) -> None:
        self.normal = _coerce_nonzero_normal(normal)
        self.offset = float(offset)
        object_space = EuclideanSpace(self.normal.dim)
        super().__init__(
            object_space,
            contains=self._contains,
            local_model=self._local_model,
        )

    def _contains(self, point: Point) -> bool:
        point = Point(point)
        value = Vector(point).dot(self.normal)
        return value >= self.offset or _isclose(value, self.offset)

    def _local_model(self, point: Point) -> LocalConeModel[Point]:
        point = Point(point)
        value = Vector(point).dot(self.normal) - self.offset
        if value > 0.0 and not _isclose(value):
            cone = EuclideanCone.whole(self.normal.dim)
        else:
            cone = half_space_cone(self.normal)
        return LocalConeModel(euclidean_chart(point), cone)


class Sphere(ChartedGeometricObject[Point]):
    """Euclidean sphere surface ``{x : ||x - c|| = r}``."""

    def __init__(self, center: Point, radius: float) -> None:
        self.center = Point(center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Sphere radius must be positive")
        object_space = EuclideanSpace(self.center.dim)
        super().__init__(
            object_space,
            contains=self._contains,
            local_model=self._local_model,
        )

    def _radial_vector(self, point: Point) -> Vector:
        return Point(point) - self.center

    def _contains(self, point: Point) -> bool:
        return _isclose(self._radial_vector(point).norm(), self.radius)

    def _local_model(self, point: Point) -> LocalConeModel[Point]:
        normal = self._radial_vector(point)
        cone = hyperplane_cone(normal)
        return LocalConeModel(euclidean_chart(Point(point)), cone)


class Ball(ChartedGeometricObject[Point]):
    """Closed Euclidean ball ``{x : ||x - c|| <= r}``."""

    def __init__(self, center: Point, radius: float) -> None:
        self.center = Point(center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Ball radius must be positive")
        object_space = EuclideanSpace(self.center.dim)
        super().__init__(
            object_space,
            contains=self._contains,
            local_model=self._local_model,
        )

    def _radial_vector(self, point: Point) -> Vector:
        return Point(point) - self.center

    def _contains(self, point: Point) -> bool:
        norm = self._radial_vector(point).norm()
        return norm <= self.radius or _isclose(norm, self.radius)

    def _local_model(self, point: Point) -> LocalConeModel[Point]:
        point = Point(point)
        normal = self._radial_vector(point)
        if normal.norm() < self.radius and not _isclose(normal.norm(), self.radius):
            cone = EuclideanCone.whole(self.center.dim)
        else:
            cone = half_space_cone(-normal)
        return LocalConeModel(euclidean_chart(point), cone)


class EllipsoidSurface(ChartedGeometricObject[Point]):
    """Affine image of the unit sphere."""

    def __init__(self, center: Point, semiaxes: Sequence[Sequence[float]]) -> None:
        self.center = Point(center)
        self.dim = self.center.dim
        self.matrix = _coerce_affine_matrix(semiaxes, self.dim)
        self.inverse_matrix = np.linalg.inv(self.matrix)
        object_space = EuclideanSpace(self.dim)
        super().__init__(
            object_space,
            contains=self._contains,
            local_model=self._local_model,
        )

    def _local_coordinates(self, point: Point) -> np.ndarray:
        displacement = _point_array(Point(point) - self.center)
        return self.inverse_matrix @ displacement

    def _normal(self, point: Point) -> Vector:
        local = self._local_coordinates(point)
        normal = self.inverse_matrix.T @ local
        return Vector(normal)

    def _contains(self, point: Point) -> bool:
        local = self._local_coordinates(point)
        return _isclose(float(local @ local), 1.0)

    def _local_model(self, point: Point) -> LocalConeModel[Point]:
        normal = self._normal(point)
        cone = hyperplane_cone(normal)
        return LocalConeModel(euclidean_chart(Point(point)), cone)


class Ellipsoid(ChartedGeometricObject[Point]):
    """Affine image of the closed unit ball."""

    def __init__(self, center: Point, semiaxes: Sequence[Sequence[float]]) -> None:
        self.surface = EllipsoidSurface(center, semiaxes)
        object_space = EuclideanSpace(self.surface.dim)
        super().__init__(
            object_space,
            contains=self._contains,
            local_model=self._local_model,
        )

    def _contains(self, point: Point) -> bool:
        local = self.surface._local_coordinates(point)
        value = float(local @ local)
        return value <= 1.0 or _isclose(value, 1.0)

    def _local_model(self, point: Point) -> LocalConeModel[Point]:
        point = Point(point)
        local = self.surface._local_coordinates(point)
        value = float(local @ local)
        if value < 1.0 and not _isclose(value, 1.0):
            cone = EuclideanCone.whole(self.surface.dim)
        else:
            cone = half_space_cone(-self.surface._normal(point))
        return LocalConeModel(euclidean_chart(point), cone)


class ParallelepipedSurface(ChartedGeometricObject[Point]):
    """Affine image of the boundary of the unit cube."""

    def __init__(self, center: Point, spanning_vectors: Sequence[Sequence[float]]) -> None:
        self.center = Point(center)
        self.dim = self.center.dim
        self.matrix = _coerce_affine_matrix(spanning_vectors, self.dim)
        self.inverse_matrix = np.linalg.inv(self.matrix)
        object_space = EuclideanSpace(self.dim)
        super().__init__(
            object_space,
            contains=self._contains,
            local_model=self._local_model,
        )

    def _local_coordinates(self, point: Point) -> np.ndarray:
        displacement = _point_array(Point(point) - self.center)
        return self.inverse_matrix @ displacement

    def _contains(self, point: Point) -> bool:
        local = self._local_coordinates(point)
        max_abs = float(np.max(np.abs(local)))
        return (max_abs <= 1.0 or _isclose(max_abs, 1.0)) and _isclose(max_abs, 1.0)

    def _surface_cone(self, local_point: np.ndarray) -> EuclideanCone:
        active = _active_box_constraints(local_point)
        if not active:
            return EuclideanCone.whole(self.dim)

        def contains(coordinates: Point) -> bool:
            local_disp = self.inverse_matrix @ _point_array(coordinates)
            values = [sign * float(local_disp[index]) for index, sign in active]
            return all(value <= 0.0 or _isclose(value) for value in values) and any(
                _isclose(value) for value in values
            )

        return EuclideanCone(
            self.dim,
            contains=contains,
            apex=Point.origin(self.dim),
            neighborhood=EuclideanNeighborhood.whole(self.dim),
        )

    def _local_model(self, point: Point) -> LocalConeModel[Point]:
        point = Point(point)
        local = self._local_coordinates(point)
        return LocalConeModel(euclidean_chart(point), self._surface_cone(local))


class Parallelepiped(ChartedGeometricObject[Point]):
    """Affine image of the closed unit cube."""

    def __init__(self, center: Point, spanning_vectors: Sequence[Sequence[float]]) -> None:
        self.surface = ParallelepipedSurface(center, spanning_vectors)
        object_space = EuclideanSpace(self.surface.dim)
        super().__init__(
            object_space,
            contains=self._contains,
            local_model=self._local_model,
        )

    def _contains(self, point: Point) -> bool:
        local = self.surface._local_coordinates(point)
        return float(np.max(np.abs(local))) <= 1.0 or _isclose(float(np.max(np.abs(local))), 1.0)

    def _solid_cone(self, local_point: np.ndarray) -> EuclideanCone:
        active = _active_box_constraints(local_point)
        if not active:
            return EuclideanCone.whole(self.surface.dim)

        def contains(coordinates: Point) -> bool:
            local_disp = self.surface.inverse_matrix @ _point_array(coordinates)
            values = [sign * float(local_disp[index]) for index, sign in active]
            return all(value <= 0.0 or _isclose(value) for value in values)

        return EuclideanCone(
            self.surface.dim,
            contains=contains,
            apex=Point.origin(self.surface.dim),
            neighborhood=EuclideanNeighborhood.whole(self.surface.dim),
        )

    def _local_model(self, point: Point) -> LocalConeModel[Point]:
        point = Point(point)
        local = self.surface._local_coordinates(point)
        max_abs = float(np.max(np.abs(local)))
        if max_abs < 1.0 and not _isclose(max_abs, 1.0):
            cone = EuclideanCone.whole(self.surface.dim)
        else:
            cone = self._solid_cone(local)
        return LocalConeModel(euclidean_chart(point), cone)


class CubeSurface(ParallelepipedSurface):
    """Boundary of an axis-aligned cube centered at a point."""

    def __init__(self, center: Point, half_extent: float) -> None:
        center = Point(center)
        half_extent = float(half_extent)
        if half_extent <= 0.0:
            raise ValueError("Cube half-extent must be positive")
        dim = center.dim
        spanning_vectors = tuple(
            tuple(half_extent if i == j else 0.0 for j in range(dim))
            for i in range(dim)
        )
        super().__init__(center, spanning_vectors)


class Cube(Parallelepiped):
    """Closed axis-aligned cube centered at a point."""

    def __init__(self, center: Point, half_extent: float) -> None:
        center = Point(center)
        half_extent = float(half_extent)
        if half_extent <= 0.0:
            raise ValueError("Cube half-extent must be positive")
        dim = center.dim
        spanning_vectors = tuple(
            tuple(half_extent if i == j else 0.0 for j in range(dim))
            for i in range(dim)
        )
        super().__init__(center, spanning_vectors)


class WholePlane(WholeSpace):
    """The whole Euclidean plane."""

    def __init__(self) -> None:
        super().__init__(2)


class HalfPlane(HalfSpace):
    """Closed half-plane ``{x : <normal, x> >= offset}``."""

    def __init__(self, normal: Vector, offset: float = 0.0) -> None:
        normal = Vector(normal)
        if normal.dim != 2:
            raise ValueError("HalfPlane is only defined in dimension 2")
        super().__init__(normal, offset=offset)


class PlanarAngle(ChartedGeometricObject[Point]):
    """Closed planar angle with interior and apex in ``R^2``."""

    def __init__(self, apex: Point, start: CirclePoint, end: CirclePoint) -> None:
        self.apex = Point(apex)
        if self.apex.dim != 2:
            raise ValueError("PlanarAngle apex must be two-dimensional")
        self.interval = CircleInterval(start, end)
        if self.interval.is_point():
            raise ValueError("PlanarAngle must have a non-zero opening")
        self.direction_set = CircleSet(self.interval)
        object_space = EuclideanSpace(2)
        super().__init__(
            object_space,
            contains=self._contains,
            local_model=self._local_model,
        )

    def _contains(self, point: Point) -> bool:
        point = Point(point)
        displacement = point - self.apex
        if displacement.norm() == 0.0:
            return True
        direction = CirclePoint.from_cartesian(displacement[0], displacement[1])
        return direction in self.direction_set

    def _local_model(self, point: Point) -> LocalConeModel[Point]:
        point = Point(point)
        chart = euclidean_chart(point)
        if point == self.apex:
            if self.direction_set.is_full():
                cone = EuclideanCone.whole(2)
            else:
                cone = SphericalCone(
                    CircleSphereObject(self.direction_set),
                    neighborhood=EuclideanNeighborhood.whole(2),
                )
            return LocalConeModel(chart, cone)
        if self.direction_set.is_full():
            return LocalConeModel(chart, EuclideanCone.whole(2))
        displacement = point - self.apex
        direction = CirclePoint.from_cartesian(displacement[0], displacement[1])
        if direction == CirclePoint(self.interval.start):
            boundary = Vector(
                math.cos(self.interval.start),
                math.sin(self.interval.start),
            )
            cone = EuclideanCone(
                2,
                contains=lambda coordinates: (
                    _cross_2d(boundary, Vector(coordinates)) >= 0.0
                ),
                apex=Point.origin(2),
                neighborhood=EuclideanNeighborhood.whole(2),
            )
            return LocalConeModel(chart, cone)
        if direction == CirclePoint(self.interval.end):
            boundary = Vector(
                math.cos(self.interval.end),
                math.sin(self.interval.end),
            )
            cone = EuclideanCone(
                2,
                contains=lambda coordinates: (
                    _cross_2d(boundary, Vector(coordinates)) <= 0.0
                ),
                apex=Point.origin(2),
                neighborhood=EuclideanNeighborhood.whole(2),
            )
            return LocalConeModel(chart, cone)
        return LocalConeModel(chart, EuclideanCone.whole(2))


def visible_part_from_direction(source_object, direction: Vector):
    direction = Vector(direction)
    if isinstance(source_object, Sphere):
        return _scalar_threshold_subset(
            source_object,
            lambda point: source_object._radial_vector(point).dot(direction),
        )
    if isinstance(source_object, Ball):
        return Sphere(
            source_object.center,
            source_object.radius,
        ).visible_from_direction(direction)
    if isinstance(source_object, EllipsoidSurface):
        return _scalar_threshold_subset(
            source_object,
            lambda point: source_object._normal(point).dot(direction),
        )
    if isinstance(source_object, Ellipsoid):
        return source_object.surface.visible_from_direction(direction)
    if isinstance(source_object, Hyperplane):
        return source_object
    if isinstance(source_object, HalfSpace):
        if (-source_object.normal).dot(direction) <= 0.0 and not _isclose(
            (-source_object.normal).dot(direction)
        ):
            return _empty_euclidean_object(source_object.space)
        return Hyperplane(source_object.normal, offset=source_object.offset)
    raise NotImplementedError(
        "Visibility is currently implemented for hyperplanes, half-spaces, "
        "spheres, balls, ellipsoids, and ellipsoid surfaces"
    )


def visible_part_from_point(source_object, observer: Point):
    observer = Point(observer)
    if isinstance(source_object, Sphere):
        return _scalar_threshold_subset(
            source_object,
            lambda point: source_object._radial_vector(point).dot(
                observer - Point(point)
            ),
        )
    if isinstance(source_object, Ball):
        return Sphere(
            source_object.center,
            source_object.radius,
        ).visible_from_point(observer)
    if isinstance(source_object, EllipsoidSurface):
        return _scalar_threshold_subset(
            source_object,
            lambda point: source_object._normal(point).dot(
                observer - Point(point)
            ),
        )
    if isinstance(source_object, Ellipsoid):
        return source_object.surface.visible_from_point(observer)
    if isinstance(source_object, Hyperplane):
        return source_object
    if isinstance(source_object, HalfSpace):
        signed_distance = (
            source_object.normal.dot(Vector(observer)) - source_object.offset
        )
        if signed_distance >= 0.0 and not _isclose(signed_distance):
            return _empty_euclidean_object(source_object.space)
        return Hyperplane(source_object.normal, offset=source_object.offset)
    raise NotImplementedError(
        "Visibility is currently implemented for hyperplanes, half-spaces, "
        "spheres, balls, ellipsoids, and ellipsoid surfaces"
    )

