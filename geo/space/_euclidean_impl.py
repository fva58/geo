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
from ..euclidean import EuclideanNeighborhood, FloatPoint, FloatVector
from ..gobject import ChartedGeometricObject
from ..manifold import ManifoldChart


class EuclideanSpace:
    """Canonical Euclidean manifold ``R^n``."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("Euclidean space dimension must be positive")
        self.dim = dim

    def contains(self, point: object) -> bool:
        try:
            coordinates = FloatPoint(point)
        except (TypeError, ValueError):
            return False
        return coordinates.dim == self.dim

    def __contains__(self, point: object) -> bool:
        return self.contains(point)


def euclidean_chart(center: FloatPoint) -> ManifoldChart[FloatPoint]:
    """Return the canonical translated chart in Euclidean space."""
    dim = center.dim
    space = EuclideanSpace(dim)
    return ManifoldChart(
        lambda point: FloatPoint(point) - center,
        lambda coordinates: center + FloatVector(coordinates),
        dim=dim,
        domain_contains=space.contains,
        image=EuclideanNeighborhood.whole(dim),
        name="euclidean-centered",
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
    mapping: Callable[[FloatPoint], FloatPoint],
    point: FloatPoint,
    step: float = 1e-6,
) -> np.ndarray:
    point = FloatPoint(point)
    base = _point_array(point)
    image_dim = FloatPoint(mapping(point)).dim
    jacobian = np.zeros((image_dim, point.dim), dtype=float)
    for index in range(point.dim):
        delta = np.zeros(point.dim, dtype=float)
        delta[index] = step
        image_plus = _point_array(mapping(FloatPoint(base + delta)))
        image_minus = _point_array(mapping(FloatPoint(base - delta)))
        jacobian[:, index] = (image_plus - image_minus) / (2.0 * step)
    return jacobian


def parallel_projection_inverse(
    source_hyperplane,
    target_hyperplane,
    direction: FloatVector,
) -> Callable[[FloatPoint], FloatPoint]:
    source_hyperplane = _projection_hyperplane_data(source_hyperplane)
    target_hyperplane = _projection_hyperplane_data(target_hyperplane)
    direction = FloatVector(direction)
    _validate_projection_hyperplanes(source_hyperplane, target_hyperplane)
    if direction.dim != source_hyperplane.normal.dim:
        raise ValueError("Projection direction dimension mismatch")
    denominator = source_hyperplane.normal.dot(direction)
    if _isclose(denominator):
        raise ValueError(
            "Projection direction must not be parallel to the source hyperplane"
        )

    def inverse_map(point: FloatPoint) -> FloatPoint:
        point = FloatPoint(point)
        scalar = (
            source_hyperplane.offset
            - source_hyperplane.normal.dot(FloatVector(point))
        ) / denominator
        return point + scalar * direction

    return inverse_map


def central_projection_inverse(
    source_hyperplane,
    target_hyperplane,
    center: FloatPoint,
) -> Callable[[FloatPoint], FloatPoint]:
    source_hyperplane = _projection_hyperplane_data(source_hyperplane)
    target_hyperplane = _projection_hyperplane_data(target_hyperplane)
    center = FloatPoint(center)
    _validate_projection_hyperplanes(source_hyperplane, target_hyperplane)
    if center.dim != source_hyperplane.normal.dim:
        raise ValueError("Projection center dimension mismatch")
    source_offset = source_hyperplane.offset
    source_center_value = source_hyperplane.normal.dot(FloatVector(center))
    if _isclose(source_offset, source_center_value):
        raise ValueError("Projection center must not lie in the source hyperplane")

    def inverse_map(point: FloatPoint) -> FloatPoint:
        point = FloatPoint(point)
        direction = point - center
        denominator = source_hyperplane.normal.dot(direction)
        if _isclose(denominator):
            raise ValueError("Projection ray is parallel to the source hyperplane")
        scalar = (source_offset - source_center_value) / denominator
        return center + scalar * direction

    return inverse_map


def projected_local_model(
    source_object,
    source_point: FloatPoint,
    target_hyperplane,
    target_point: FloatPoint,
    inverse_map: Callable[[FloatPoint], FloatPoint],
    name: str,
) -> LocalConeModel[FloatPoint]:
    source_model = source_object.local_model_at(source_point)
    target_model = target_hyperplane.local_model_at(target_point)
    jacobian = _numeric_jacobian(inverse_map, target_point)
    dim = target_point.dim
    cone = EuclideanCone(
        dim,
        contains=lambda coordinates: (
            target_model.cone.contains(coordinates)
            and source_model.cone.contains(
                FloatPoint(jacobian @ _point_array(coordinates))
            )
        ),
        apex=FloatPoint.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
        name=name,
    )
    return LocalConeModel(target_model.chart, cone)


def _centered_cone_model(source_model, point: FloatPoint) -> LocalConeModel[FloatPoint]:
    from ..gobject import _centered_chart_at

    centered_chart = _centered_chart_at(source_model.chart, point)
    chart_origin = FloatPoint(source_model.chart(point))
    translation = FloatVector(chart_origin)
    dim = centered_chart.dim
    centered_cone = EuclideanCone(
        dim,
        contains=lambda coordinates: source_model.cone.contains(
            FloatPoint(coordinates) + translation
        ),
        apex=FloatPoint.origin(dim),
        neighborhood=EuclideanNeighborhood.whole(dim),
        name=getattr(source_model.cone, "name", ""),
    )
    return LocalConeModel(centered_chart, centered_cone)


def _scalar_subset_local_model(
    source_object,
    point: FloatPoint,
    scalar: Callable[[FloatPoint], float],
    name: str,
) -> LocalConeModel[FloatPoint]:
    point = FloatPoint(point)
    source_model = _centered_cone_model(source_object.local_model_at(point), point)
    value = float(scalar(point))
    if value > 1e-9:
        return source_model

    def local_scalar(coordinates: FloatPoint) -> FloatPoint:
        local_point = source_model.chart.inverse(coordinates)
        return FloatPoint(float(scalar(local_point)))

    gradient = _numeric_jacobian(
        local_scalar,
        FloatPoint.origin(source_model.chart.dim),
    )[0]
    if np.linalg.norm(gradient) <= 1e-9:
        return source_model

    cone = EuclideanCone(
        source_model.chart.dim,
        contains=lambda coordinates: (
            source_model.cone.contains(coordinates)
            and float(gradient @ _point_array(FloatPoint(coordinates))) >= -1e-9
        ),
        apex=FloatPoint.origin(source_model.chart.dim),
        neighborhood=EuclideanNeighborhood.whole(source_model.chart.dim),
        name=name,
    )
    return LocalConeModel(source_model.chart, cone)


def _scalar_threshold_subset(source_object, scalar, name: str):
    return ChartedGeometricObject(
        source_object.manifold,
        contains=lambda point: (
            point in source_object and float(scalar(FloatPoint(point))) >= -1e-9
        ),
        local_model=lambda point: _scalar_subset_local_model(
            source_object,
            FloatPoint(point),
            scalar,
            name,
        ),
        name=name,
    )


def _empty_euclidean_object(manifold: EuclideanSpace, name: str):
    return ChartedGeometricObject(
        manifold,
        contains=lambda point: False,
        local_model=lambda point: (_ for _ in ()).throw(
            ValueError("Empty object has no local model")
        ),
        name=name,
    )


class EuclideanPointObject(ChartedGeometricObject[FloatPoint]):
    """Zero-dimensional object supported at one Euclidean point."""

    def __init__(self, point: FloatPoint, name: str = "") -> None:
        self.point = FloatPoint(point)
        manifold = EuclideanSpace(self.point.dim)
        super().__init__(
            manifold,
            contains=lambda candidate: FloatPoint(candidate) == self.point,
            local_model=lambda candidate: LocalConeModel(
                euclidean_chart(self.point),
                point_cone(self.point.dim),
            ),
            name=name,
        )


class WholeSpace(ChartedGeometricObject[FloatPoint]):
    """The whole Euclidean space ``R^n``."""

    def __init__(self, dim: int, name: str = "") -> None:
        self.dim = dim
        manifold = EuclideanSpace(dim)
        super().__init__(
            manifold,
            contains=lambda point: True,
            local_model=lambda point: LocalConeModel(
                euclidean_chart(FloatPoint(point)),
                EuclideanCone.whole(dim),
            ),
            name=name,
        )


class Hyperplane(ChartedGeometricObject[FloatPoint]):
    """Affine hyperplane ``{x : <normal, x> = offset}``."""

    def __init__(self, normal: FloatVector, offset: float = 0.0, name: str = "") -> None:
        self.normal = _coerce_nonzero_normal(normal)
        self.offset = float(offset)
        manifold = EuclideanSpace(self.normal.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        point = FloatPoint(point)
        value = FloatVector(point).dot(self.normal)
        return _isclose(value, self.offset)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        point = FloatPoint(point)
        cone = hyperplane_cone(self.normal, name="hyperplane")
        return LocalConeModel(euclidean_chart(point), cone)


class HalfSpace(ChartedGeometricObject[FloatPoint]):
    """Closed half-space ``{x : <normal, x> >= offset}``."""

    def __init__(self, normal: FloatVector, offset: float = 0.0, name: str = "") -> None:
        self.normal = _coerce_nonzero_normal(normal)
        self.offset = float(offset)
        manifold = EuclideanSpace(self.normal.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        point = FloatPoint(point)
        value = FloatVector(point).dot(self.normal)
        return value >= self.offset or _isclose(value, self.offset)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        point = FloatPoint(point)
        value = FloatVector(point).dot(self.normal) - self.offset
        if value > 0.0 and not _isclose(value):
            cone = EuclideanCone.whole(self.normal.dim)
        else:
            cone = half_space_cone(self.normal, name="half-space-boundary")
        return LocalConeModel(euclidean_chart(point), cone)


class Sphere(ChartedGeometricObject[FloatPoint]):
    """Euclidean sphere surface ``{x : ||x - c|| = r}``."""

    def __init__(self, center: FloatPoint, radius: float, name: str = "") -> None:
        self.center = FloatPoint(center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Sphere radius must be positive")
        manifold = EuclideanSpace(self.center.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _radial_vector(self, point: FloatPoint) -> FloatVector:
        return FloatPoint(point) - self.center

    def _contains(self, point: FloatPoint) -> bool:
        return _isclose(self._radial_vector(point).norm(), self.radius)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        normal = self._radial_vector(point)
        cone = hyperplane_cone(normal, name="sphere-tangent")
        return LocalConeModel(euclidean_chart(FloatPoint(point)), cone)


class Ball(ChartedGeometricObject[FloatPoint]):
    """Closed Euclidean ball ``{x : ||x - c|| <= r}``."""

    def __init__(self, center: FloatPoint, radius: float, name: str = "") -> None:
        self.center = FloatPoint(center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Ball radius must be positive")
        manifold = EuclideanSpace(self.center.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _radial_vector(self, point: FloatPoint) -> FloatVector:
        return FloatPoint(point) - self.center

    def _contains(self, point: FloatPoint) -> bool:
        norm = self._radial_vector(point).norm()
        return norm <= self.radius or _isclose(norm, self.radius)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        point = FloatPoint(point)
        normal = self._radial_vector(point)
        if normal.norm() < self.radius and not _isclose(normal.norm(), self.radius):
            cone = EuclideanCone.whole(self.center.dim)
        else:
            cone = half_space_cone(-normal, name="ball-boundary")
        return LocalConeModel(euclidean_chart(point), cone)


class EllipsoidSurface(ChartedGeometricObject[FloatPoint]):
    """Affine image of the unit sphere."""

    def __init__(self, center: FloatPoint, semiaxes: Sequence[Sequence[float]], name: str = "") -> None:
        self.center = FloatPoint(center)
        self.dim = self.center.dim
        self.matrix = _coerce_affine_matrix(semiaxes, self.dim)
        self.inverse_matrix = np.linalg.inv(self.matrix)
        manifold = EuclideanSpace(self.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _local_coordinates(self, point: FloatPoint) -> np.ndarray:
        displacement = _point_array(FloatPoint(point) - self.center)
        return self.inverse_matrix @ displacement

    def _normal(self, point: FloatPoint) -> FloatVector:
        local = self._local_coordinates(point)
        normal = self.inverse_matrix.T @ local
        return FloatVector(normal)

    def _contains(self, point: FloatPoint) -> bool:
        local = self._local_coordinates(point)
        return _isclose(float(local @ local), 1.0)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        normal = self._normal(point)
        cone = hyperplane_cone(normal, name="ellipsoid-tangent")
        return LocalConeModel(euclidean_chart(FloatPoint(point)), cone)


class Ellipsoid(ChartedGeometricObject[FloatPoint]):
    """Affine image of the closed unit ball."""

    def __init__(self, center: FloatPoint, semiaxes: Sequence[Sequence[float]], name: str = "") -> None:
        self.surface = EllipsoidSurface(center, semiaxes, name=name)
        manifold = EuclideanSpace(self.surface.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        local = self.surface._local_coordinates(point)
        value = float(local @ local)
        return value <= 1.0 or _isclose(value, 1.0)

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        point = FloatPoint(point)
        local = self.surface._local_coordinates(point)
        value = float(local @ local)
        if value < 1.0 and not _isclose(value, 1.0):
            cone = EuclideanCone.whole(self.surface.dim)
        else:
            cone = half_space_cone(-self.surface._normal(point), name="ellipsoid-boundary")
        return LocalConeModel(euclidean_chart(point), cone)


class ParallelepipedSurface(ChartedGeometricObject[FloatPoint]):
    """Affine image of the boundary of the unit cube."""

    def __init__(self, center: FloatPoint, spanning_vectors: Sequence[Sequence[float]], name: str = "") -> None:
        self.center = FloatPoint(center)
        self.dim = self.center.dim
        self.matrix = _coerce_affine_matrix(spanning_vectors, self.dim)
        self.inverse_matrix = np.linalg.inv(self.matrix)
        manifold = EuclideanSpace(self.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _local_coordinates(self, point: FloatPoint) -> np.ndarray:
        displacement = _point_array(FloatPoint(point) - self.center)
        return self.inverse_matrix @ displacement

    def _contains(self, point: FloatPoint) -> bool:
        local = self._local_coordinates(point)
        max_abs = float(np.max(np.abs(local)))
        return (max_abs <= 1.0 or _isclose(max_abs, 1.0)) and _isclose(max_abs, 1.0)

    def _surface_cone(self, local_point: np.ndarray) -> EuclideanCone:
        active = _active_box_constraints(local_point)
        if not active:
            return EuclideanCone.whole(self.dim)

        def contains(coordinates: FloatPoint) -> bool:
            local_disp = self.inverse_matrix @ _point_array(coordinates)
            values = [sign * float(local_disp[index]) for index, sign in active]
            return all(value <= 0.0 or _isclose(value) for value in values) and any(
                _isclose(value) for value in values
            )

        return EuclideanCone(
            self.dim,
            contains=contains,
            apex=FloatPoint.origin(self.dim),
            neighborhood=EuclideanNeighborhood.whole(self.dim),
            name="parallelepiped-surface",
        )

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        point = FloatPoint(point)
        local = self._local_coordinates(point)
        return LocalConeModel(euclidean_chart(point), self._surface_cone(local))


class Parallelepiped(ChartedGeometricObject[FloatPoint]):
    """Affine image of the closed unit cube."""

    def __init__(self, center: FloatPoint, spanning_vectors: Sequence[Sequence[float]], name: str = "") -> None:
        self.surface = ParallelepipedSurface(center, spanning_vectors, name=name)
        manifold = EuclideanSpace(self.surface.dim)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        local = self.surface._local_coordinates(point)
        return float(np.max(np.abs(local))) <= 1.0 or _isclose(float(np.max(np.abs(local))), 1.0)

    def _solid_cone(self, local_point: np.ndarray) -> EuclideanCone:
        active = _active_box_constraints(local_point)
        if not active:
            return EuclideanCone.whole(self.surface.dim)

        def contains(coordinates: FloatPoint) -> bool:
            local_disp = self.surface.inverse_matrix @ _point_array(coordinates)
            values = [sign * float(local_disp[index]) for index, sign in active]
            return all(value <= 0.0 or _isclose(value) for value in values)

        return EuclideanCone(
            self.surface.dim,
            contains=contains,
            apex=FloatPoint.origin(self.surface.dim),
            neighborhood=EuclideanNeighborhood.whole(self.surface.dim),
            name="parallelepiped-boundary",
        )

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        point = FloatPoint(point)
        local = self.surface._local_coordinates(point)
        max_abs = float(np.max(np.abs(local)))
        if max_abs < 1.0 and not _isclose(max_abs, 1.0):
            cone = EuclideanCone.whole(self.surface.dim)
        else:
            cone = self._solid_cone(local)
        return LocalConeModel(euclidean_chart(point), cone)


class CubeSurface(ParallelepipedSurface):
    """Boundary of an axis-aligned cube centered at a point."""

    def __init__(self, center: FloatPoint, half_extent: float, name: str = "") -> None:
        center = FloatPoint(center)
        half_extent = float(half_extent)
        if half_extent <= 0.0:
            raise ValueError("Cube half-extent must be positive")
        dim = center.dim
        spanning_vectors = tuple(
            tuple(half_extent if i == j else 0.0 for j in range(dim))
            for i in range(dim)
        )
        super().__init__(center, spanning_vectors, name=name)


class Cube(Parallelepiped):
    """Closed axis-aligned cube centered at a point."""

    def __init__(self, center: FloatPoint, half_extent: float, name: str = "") -> None:
        center = FloatPoint(center)
        half_extent = float(half_extent)
        if half_extent <= 0.0:
            raise ValueError("Cube half-extent must be positive")
        dim = center.dim
        spanning_vectors = tuple(
            tuple(half_extent if i == j else 0.0 for j in range(dim))
            for i in range(dim)
        )
        super().__init__(center, spanning_vectors, name=name)


class WholePlane(WholeSpace):
    """The whole Euclidean plane."""

    def __init__(self, name: str = "") -> None:
        super().__init__(2, name=name)


class HalfPlane(HalfSpace):
    """Closed half-plane ``{x : <normal, x> >= offset}``."""

    def __init__(self, normal: FloatVector, offset: float = 0.0, name: str = "") -> None:
        normal = FloatVector(normal)
        if normal.dim != 2:
            raise ValueError("HalfPlane is only defined in dimension 2")
        super().__init__(normal, offset=offset, name=name)


class PlanarAngle(ChartedGeometricObject[FloatPoint]):
    """Closed planar angle with interior and apex in ``R^2``."""

    def __init__(self, apex: FloatPoint, start: CirclePoint, end: CirclePoint, name: str = "") -> None:
        self.apex = FloatPoint(apex)
        if self.apex.dim != 2:
            raise ValueError("PlanarAngle apex must be two-dimensional")
        self.interval = CircleInterval(start, end)
        if self.interval.is_point():
            raise ValueError("PlanarAngle must have a non-zero opening")
        self.direction_set = CircleSet(self.interval)
        manifold = EuclideanSpace(2)
        super().__init__(
            manifold,
            contains=self._contains,
            local_model=self._local_model,
            name=name,
        )

    def _contains(self, point: FloatPoint) -> bool:
        point = FloatPoint(point)
        displacement = point - self.apex
        if displacement.norm() == 0.0:
            return True
        direction = CirclePoint.from_cartesian(displacement[0], displacement[1])
        return direction in self.direction_set

    def _local_model(self, point: FloatPoint) -> LocalConeModel[FloatPoint]:
        point = FloatPoint(point)
        chart = euclidean_chart(point)
        if point == self.apex:
            if self.direction_set.is_full():
                cone = EuclideanCone.whole(2)
            else:
                cone = SphericalCone(
                    CircleSphereObject(self.direction_set),
                    neighborhood=EuclideanNeighborhood.whole(2),
                    name="angle-apex",
                )
            return LocalConeModel(chart, cone)
        if self.direction_set.is_full():
            return LocalConeModel(chart, EuclideanCone.whole(2))
        displacement = point - self.apex
        direction = CirclePoint.from_cartesian(displacement[0], displacement[1])
        if direction == CirclePoint(self.interval.start):
            boundary = FloatVector(
                math.cos(self.interval.start),
                math.sin(self.interval.start),
            )
            cone = EuclideanCone(
                2,
                contains=lambda coordinates: (
                    _cross_2d(boundary, FloatVector(coordinates)) >= 0.0
                ),
                apex=FloatPoint.origin(2),
                neighborhood=EuclideanNeighborhood.whole(2),
                name="angle-start-boundary",
            )
            return LocalConeModel(chart, cone)
        if direction == CirclePoint(self.interval.end):
            boundary = FloatVector(
                math.cos(self.interval.end),
                math.sin(self.interval.end),
            )
            cone = EuclideanCone(
                2,
                contains=lambda coordinates: (
                    _cross_2d(boundary, FloatVector(coordinates)) <= 0.0
                ),
                apex=FloatPoint.origin(2),
                neighborhood=EuclideanNeighborhood.whole(2),
                name="angle-end-boundary",
            )
            return LocalConeModel(chart, cone)
        return LocalConeModel(chart, EuclideanCone.whole(2))


def visible_part_from_direction(source_object, direction: FloatVector, name: str = ""):
    direction = FloatVector(direction)
    chosen_name = name or "visible-from-direction"
    if isinstance(source_object, Sphere):
        return _scalar_threshold_subset(
            source_object,
            lambda point: source_object._radial_vector(point).dot(direction),
            chosen_name,
        )
    if isinstance(source_object, Ball):
        return Sphere(
            source_object.center,
            source_object.radius,
            name=chosen_name,
        ).visible_from_direction(direction, name=chosen_name)
    if isinstance(source_object, EllipsoidSurface):
        return _scalar_threshold_subset(
            source_object,
            lambda point: source_object._normal(point).dot(direction),
            chosen_name,
        )
    if isinstance(source_object, Ellipsoid):
        return source_object.surface.visible_from_direction(direction, name=chosen_name)
    if isinstance(source_object, Hyperplane):
        return source_object
    if isinstance(source_object, HalfSpace):
        if (-source_object.normal).dot(direction) <= 0.0 and not _isclose(
            (-source_object.normal).dot(direction)
        ):
            return _empty_euclidean_object(source_object.manifold, chosen_name)
        return Hyperplane(source_object.normal, offset=source_object.offset, name=chosen_name)
    raise NotImplementedError(
        "Visibility is currently implemented for hyperplanes, half-spaces, "
        "spheres, balls, ellipsoids, and ellipsoid surfaces"
    )


def visible_part_from_point(source_object, observer: FloatPoint, name: str = ""):
    observer = FloatPoint(observer)
    chosen_name = name or "visible-from-point"
    if isinstance(source_object, Sphere):
        return _scalar_threshold_subset(
            source_object,
            lambda point: source_object._radial_vector(point).dot(
                observer - FloatPoint(point)
            ),
            chosen_name,
        )
    if isinstance(source_object, Ball):
        return Sphere(
            source_object.center,
            source_object.radius,
            name=chosen_name,
        ).visible_from_point(observer, name=chosen_name)
    if isinstance(source_object, EllipsoidSurface):
        return _scalar_threshold_subset(
            source_object,
            lambda point: source_object._normal(point).dot(
                observer - FloatPoint(point)
            ),
            chosen_name,
        )
    if isinstance(source_object, Ellipsoid):
        return source_object.surface.visible_from_point(observer, name=chosen_name)
    if isinstance(source_object, Hyperplane):
        return source_object
    if isinstance(source_object, HalfSpace):
        signed_distance = (
            source_object.normal.dot(FloatVector(observer)) - source_object.offset
        )
        if signed_distance >= 0.0 and not _isclose(signed_distance):
            return _empty_euclidean_object(source_object.manifold, chosen_name)
        return Hyperplane(source_object.normal, offset=source_object.offset, name=chosen_name)
    raise NotImplementedError(
        "Visibility is currently implemented for hyperplanes, half-spaces, "
        "spheres, balls, ellipsoids, and ellipsoid surfaces"
    )

