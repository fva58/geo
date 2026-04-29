"""Metric spaces and geometric objects inside them."""

from __future__ import annotations

import math
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from .euclidean import EuclideanNeighborhood, FloatPoint, FloatVector
from .floatcircle import FloatAngle, FloatCircleInterval, FloatCirclePoint
from .geometric import (
    Circle,
    CirclePointObject,
    CircleSetObject,
    ChartedGeometricObject,
    Ball,
    EuclideanPointObject,
    EuclideanSpace,
    HalfPlane,
    ObjectMesh,
    PlanarAngle,
    RealLine,
    RealPointObject,
    RealSetObject,
    Sphere,
    WholePlane,
)
from .manifold import Manifold
from .manifold import ChartNeighborhood
from .operations import (
    closest_sample_pair as _closest_sample_pair,
    closest_sample_to_point as _closest_sample_to_point,
    sample_points as _validated_sample_points,
    sampled_distance as _sampled_distance,
    sampled_distance_to_point as _sampled_distance_to_point,
)


PointT = TypeVar("PointT")
TargetT = TypeVar("TargetT")
MetricTensor = tuple[tuple[float, ...], ...]


def _whole_neighborhood(dim: int):
    """Return the default unconstrained neighborhood for a cone."""
    from .euclidean import EuclideanNeighborhood

    return EuclideanNeighborhood.whole(dim)


def _point_array(point: FloatPoint) -> np.ndarray:
    """Return local coordinates as a NumPy float array."""
    return np.asarray(FloatPoint(point), dtype=float)


def _numeric_local_jacobian(
    mapping: Callable[[FloatPoint], FloatPoint],
    point: FloatPoint,
    step: float = 1e-6,
) -> np.ndarray:
    """Approximate the Jacobian of a local coordinate transition."""
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


def _sample_linear_values(
    left: float,
    right: float,
    steps: int,
) -> tuple[float, ...]:
    """Return evenly sampled values on one closed interval."""
    steps = max(1, int(steps))
    if math.isclose(left, right, abs_tol=1e-15):
        return (left,)
    if steps == 1:
        return ((left + right) / 2.0,)
    return tuple(
        left + (right - left) * index / (steps - 1)
        for index in range(steps)
    )


def _polyline_mesh_from_embedded_parts(
    parts: tuple[tuple[FloatPoint, ...], ...],
    closed: tuple[bool, ...] | None = None,
) -> ObjectMesh:
    """Return a polyline mesh from embedded point parts."""
    vertices: list[FloatPoint] = []
    cells: list[tuple[int, ...]] = []
    offset = 0
    closed = closed or tuple(False for _ in parts)
    for part, close_ring in zip(parts, closed):
        if not part:
            continue
        vertices.extend(part)
        if len(part) == 1:
            cells.append((offset,))
        else:
            cells.extend(
                (offset + index, offset + index + 1)
                for index in range(len(part) - 1)
            )
            if close_ring:
                cells.append((offset + len(part) - 1, offset))
        offset += len(part)
    return ObjectMesh(tuple(vertices), tuple(cells))


@runtime_checkable
class MetricSpace(Manifold[PointT], Protocol[PointT]):
    """Protocol for a manifold equipped with a distance function.

    The contract is intentionally weak enough to model pseudometric spaces:
    implementations are expected to provide a non-negative symmetric distance,
    and different points may still have distance zero.
    """

    def distance(self, left: PointT, right: PointT) -> float:
        """Return the distance between two points."""


@runtime_checkable
class RiemannianSpace(MetricSpace[PointT], Protocol[PointT]):
    """Compatibility alias for the older metric-space naming."""


class ChartedMetricSpace(Generic[PointT]):
    """Concrete metric space given by a manifold and a distance function."""

    def __init__(
        self,
        manifold: Manifold[PointT],
        distance: Callable[[PointT, PointT], float],
        name: str = "",
    ) -> None:
        """Initialize the metric space."""
        self.manifold = manifold
        self._distance = distance
        self.name = name

    @property
    def dim(self) -> int:
        """Return the space dimension."""
        return self.manifold.dim

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return f"ChartedMetricSpace(dim={self.dim}{label})"

    def contains(self, point: PointT) -> bool:
        """Check whether a point belongs to the underlying manifold."""
        return point in self.manifold

    def __contains__(self, point: PointT) -> bool:
        """Check whether a point belongs to the underlying manifold."""
        return self.contains(point)

    def distance(
        self,
        left: PointT,
        right: PointT,
    ) -> float:
        """Return the distance between two points."""
        if left not in self or right not in self:
            raise ValueError("Points must belong to the metric space")
        distance = float(self._distance(left, right))
        if distance < 0.0:
            raise ValueError("Distance must be non-negative")
        return distance

    def wrap(
        self,
        obj: ChartedGeometricObject[PointT],
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Wrap a charted object into this ambient metric space."""
        return MetricGeometricObject.from_charted(self, obj, name=name)


class ChartedRiemannianSpace(ChartedMetricSpace[PointT]):
    """Compatibility alias for the older charted metric-space naming."""


class MetricGeometricObject(ChartedGeometricObject[PointT]):
    """Geometric object with an explicitly metric ambient space."""

    def __init__(
        self,
        space: MetricSpace[PointT],
        contains: Callable[[PointT], bool],
        local_model,
        name: str = "",
    ) -> None:
        """Initialize the Riemannian geometric object."""
        self.space = space
        super().__init__(space, contains, local_model, name=name)

    @classmethod
    def from_charted(
        cls,
        space: MetricSpace[PointT],
        obj: ChartedGeometricObject[PointT],
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Wrap an existing charted object in a metric ambient space."""
        chosen_name = name or getattr(obj, "name", "")
        wrapped = cls(
            space,
            contains=obj.contains,
            local_model=obj.local_model_at,
            name=chosen_name,
        )
        wrapped._charted_source_object = obj
        return wrapped

    def _mesh_from_real_source(self, resolution: int) -> ObjectMesh:
        """Return a visualization mesh for a real-line source object."""
        source_object = getattr(self, "_charted_source_object")
        if isinstance(source_object, RealPointObject):
            point = float(source_object.point)
            return ObjectMesh((FloatPoint(self.space.to_2d(point)),), ((0,),))

        parts = []
        for interval in source_object.point_set:
            left, right = interval
            values = _sample_linear_values(left, right, max(2, resolution))
            parts.append(tuple(FloatPoint(self.space.to_2d(value)) for value in values))
        return _polyline_mesh_from_embedded_parts(tuple(parts))

    def _sample_points_from_real_source(
        self,
        resolution: int,
    ) -> tuple[float, ...]:
        """Return sampled points for a real-line source object."""
        source_object = getattr(self, "_charted_source_object")
        if isinstance(source_object, RealPointObject):
            return (float(source_object.point),)

        samples: list[float] = []
        for interval in source_object.point_set:
            left, right = interval
            samples.extend(_sample_linear_values(left, right, max(2, resolution)))
        return tuple(samples)

    def _mesh_from_circle_source(self, resolution: int) -> ObjectMesh:
        """Return a visualization mesh for a circle source object."""
        source_object = getattr(self, "_charted_source_object")
        if isinstance(source_object, CirclePointObject):
            point = FloatCirclePoint(source_object.point)
            return ObjectMesh((FloatPoint(self.space.to_2d(point)),), ((0,),))

        parts = []
        closed = []
        wrapped_arc = (
            len(source_object.point_set) == 2 and
            float(source_object.point_set[0][0]) == 0.0 and
            float(source_object.point_set[-1][1]) == FloatAngle.MAX_ANGLE
        )
        for index, interval in enumerate(source_object.point_set):
            circle_interval = FloatCircleInterval(interval)
            values = _sample_linear_values(
                float(circle_interval.start),
                float(circle_interval.end),
                max(2, resolution),
            )
            parts.append(tuple(FloatPoint(self.space.to_2d(value)) for value in values))
            closed.append(bool(source_object.point_set.is_full()))
            if wrapped_arc and index == len(source_object.point_set) - 1:
                parts[-1] = parts[-1] + (FloatPoint(self.space.to_2d(0.0)),)
        return _polyline_mesh_from_embedded_parts(tuple(parts), tuple(closed))

    def _sample_points_from_circle_source(
        self,
        resolution: int,
    ) -> tuple[FloatCirclePoint, ...]:
        """Return sampled points for a circle source object."""
        source_object = getattr(self, "_charted_source_object")
        if isinstance(source_object, CirclePointObject):
            return (FloatCirclePoint(source_object.point),)

        samples: list[FloatCirclePoint] = []
        for interval in source_object.point_set:
            circle_interval = FloatCircleInterval(interval)
            values = _sample_linear_values(
                float(circle_interval.start),
                float(circle_interval.end),
                max(2, resolution),
            )
            samples.extend(FloatCirclePoint(value) for value in values)
        return tuple(samples)

    def _mesh_from_euclidean_point_source(self) -> ObjectMesh:
        """Return a singleton mesh for an Euclidean point source object."""
        source_object = getattr(self, "_charted_source_object")
        return ObjectMesh((FloatPoint(source_object.point),), ((0,),))

    def _validated_samples(self, resolution: int) -> tuple[PointT, ...]:
        """Return sample points and require at least one sample."""
        return _validated_sample_points(self, resolution=resolution)

    def closest_sample_to_point(
        self,
        point: PointT,
        resolution: int = 64,
    ) -> tuple[PointT, float]:
        """Return the closest sampled object point to one ambient point."""
        return _closest_sample_to_point(self, point, resolution=resolution)

    def sampled_distance_to_point(
        self,
        point: PointT,
        resolution: int = 64,
    ) -> float:
        """Return a sample-based approximation to distance from a point."""
        return _sampled_distance_to_point(self, point, resolution=resolution)

    def closest_sample_pair(
        self,
        other: "MetricGeometricObject[PointT]",
        resolution: int = 64,
    ) -> tuple[PointT, PointT, float]:
        """Return the closest sampled pair between two objects."""
        return _closest_sample_pair(self, other, resolution=resolution)

    def sampled_distance_to(
        self,
        other: "MetricGeometricObject[PointT]",
        resolution: int = 64,
    ) -> float:
        """Return a sample-based approximation to distance between objects."""
        if self is other:
            return 0.0
        return _sampled_distance(self, other, resolution=resolution)

    def sample_points(self, resolution: int = 24):
        """Return sample points on the geometric object when supported."""
        resolution = max(1, int(resolution))
        if hasattr(self, "_sample_points_fn"):
            return tuple(self._sample_points_fn(resolution))

        source_object = getattr(self, "_charted_source_object", None)
        if isinstance(source_object, (RealPointObject, RealSetObject)):
            return self._sample_points_from_real_source(resolution)
        if isinstance(source_object, (CirclePointObject, CircleSetObject)):
            return self._sample_points_from_circle_source(resolution)
        if isinstance(source_object, EuclideanPointObject):
            return (FloatPoint(source_object.point),)

        if source_object is not None and hasattr(source_object, "mesh"):
            return tuple(source_object.mesh(resolution=resolution).vertices)
        return tuple(super().mesh(resolution=resolution).vertices)

    def mesh(
        self,
        resolution: int = 64,
        bounds=None,
    ) -> ObjectMesh:
        """Return a mesh or polyline approximation when supported."""
        resolution = max(1, int(resolution))
        if hasattr(self, "_mesh_fn"):
            return self._mesh_fn(resolution)

        source_object = getattr(self, "_charted_source_object", None)
        if isinstance(source_object, (RealPointObject, RealSetObject)):
            return self._mesh_from_real_source(resolution)
        if isinstance(source_object, (CirclePointObject, CircleSetObject)):
            return self._mesh_from_circle_source(resolution)
        if isinstance(source_object, EuclideanPointObject):
            return self._mesh_from_euclidean_point_source()
        if source_object is not None and hasattr(source_object, "mesh"):
            return source_object.mesh(resolution=resolution, bounds=bounds)
        return super().mesh(resolution=resolution, bounds=bounds)

    def _plotting_mesh(
        self,
        resolution: int = 64,
        bounds=None,
    ) -> ObjectMesh:
        """Return a plotting mesh while tolerating mesh APIs without bounds."""
        if bounds is None:
            return self.mesh(resolution=resolution)
        try:
            return self.mesh(resolution=resolution, bounds=bounds)
        except TypeError as exc:
            if "bounds" not in str(exc):
                raise
            return self.mesh(resolution=resolution)

    def plot_matplotlib(
        self,
        resolution: int = 64,
        bounds=None,
        axes=None,
        ax=None,
        *,
        color: str = "#1f1f1f",
        linewidth: float = 1.5,
        marker_size: float = 20.0,
    ):
        """Plot the object mesh through the Matplotlib helper."""
        from .mesh_plot import plot_mesh_matplotlib

        return plot_mesh_matplotlib(
            self._plotting_mesh(resolution=resolution, bounds=bounds),
            axes=axes,
            ax=ax,
            color=color,
            linewidth=linewidth,
            marker_size=marker_size,
        )

    def plot_plotly(
        self,
        resolution: int = 64,
        bounds=None,
        axes=None,
        figure=None,
        *,
        name: str | None = None,
    ):
        """Plot the object mesh through the Plotly helper."""
        from .mesh_plot import plot_mesh_plotly

        return plot_mesh_plotly(
            self._plotting_mesh(resolution=resolution, bounds=bounds),
            axes=axes,
            figure=figure,
            name=name or self.name or "object",
        )

    def show(
        self,
        resolution: int = 64,
        bounds=None,
        axes=None,
        ax=None,
        *,
        color: str = "#1f1f1f",
        linewidth: float = 1.5,
        marker_size: float = 20.0,
    ):
        """Alias for ``plot_matplotlib()`` for interactive sessions."""
        return self.plot_matplotlib(
            resolution=resolution,
            bounds=bounds,
            axes=axes,
            ax=ax,
            color=color,
            linewidth=linewidth,
            marker_size=marker_size,
        )

    @staticmethod
    def _combine_local_models(
        left_model,
        right_model,
        operation: Callable[[bool, bool], bool],
        name: str,
    ):
        """Combine two local cone models in a shared coordinate chart."""
        from .geometric import EuclideanCone, LocalConeModel

        if left_model.chart.dim != right_model.chart.dim:
            raise ValueError("Local model dimensions do not match")
        dim = left_model.chart.dim
        origin = FloatPoint.origin(dim)

        def transition(point: FloatPoint) -> FloatPoint:
            manifold_point = left_model.chart.inverse(point)
            return FloatPoint(right_model.chart(manifold_point))

        transition_origin = FloatPoint(transition(origin))
        if not np.allclose(
            _point_array(transition_origin),
            np.zeros(dim, dtype=float),
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError(
                "Local charts do not share the same base point coordinates"
            )

        transition_jacobian = _numeric_local_jacobian(transition, origin)
        if np.linalg.matrix_rank(transition_jacobian) != dim:
            raise ValueError("Local chart transition is singular at the base point")

        cone = EuclideanCone(
            dim,
            contains=lambda point: operation(
                left_model.cone.contains(point),
                right_model.cone.contains(
                    FloatPoint(transition_jacobian @ _point_array(point))
                ),
            ),
            neighborhood=_whole_neighborhood(dim),
            name=name,
        )
        return LocalConeModel(left_model.chart, cone)

    def union(
        self,
        other: "MetricGeometricObject[PointT]",
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Return the set-theoretic union of two objects."""
        return LazyMetricExpressionObject("union", self, other, name=name)

    def intersection(
        self,
        other: "MetricGeometricObject[PointT]",
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Return the set-theoretic intersection of two objects."""
        return LazyMetricExpressionObject(
            "intersection",
            self,
            other,
            name=name,
        )

    def difference(
        self,
        other: "MetricGeometricObject[PointT]",
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Return the set-theoretic difference ``self \\ other``."""
        return LazyMetricExpressionObject(
            "difference",
            self,
            other,
            name=name,
        )

    def symmetric_difference(
        self,
        other: "MetricGeometricObject[PointT]",
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Return the symmetric difference of two objects."""
        return LazyMetricExpressionObject(
            "symmetric-difference",
            self,
            other,
            name=name,
        )

    def __or__(
        self,
        other: "MetricGeometricObject[PointT]",
    ) -> "MetricGeometricObject[PointT]":
        """Return the union operator result."""
        return self.union(other)

    def __and__(
        self,
        other: "MetricGeometricObject[PointT]",
    ) -> "MetricGeometricObject[PointT]":
        """Return the intersection operator result."""
        return self.intersection(other)

    def __sub__(
        self,
        other: "MetricGeometricObject[PointT]",
    ) -> "MetricGeometricObject[PointT]":
        """Return the difference operator result."""
        return self.difference(other)

    def __xor__(
        self,
        other: "MetricGeometricObject[PointT]",
    ) -> "MetricGeometricObject[PointT]":
        """Return the symmetric-difference operator result."""
        return self.symmetric_difference(other)

    def project_along_direction_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        direction: FloatVector,
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Project an Euclidean object along a direction onto a hyperplane."""
        return LazyMetricMappedObject(
            self.space,
            "project-along-direction",
            self,
            name=name,
            source_hyperplane=source_hyperplane,
            target_hyperplane=target_hyperplane,
            direction=FloatVector(direction),
        )

    def project_from_point_onto(
        self,
        source_hyperplane,
        target_hyperplane,
        center: FloatPoint,
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Project an Euclidean object from a point onto a hyperplane."""
        return LazyMetricMappedObject(
            self.space,
            "project-from-point",
            self,
            name=name,
            source_hyperplane=source_hyperplane,
            target_hyperplane=target_hyperplane,
            center=FloatPoint(center),
        )

    def visible_from_direction(
        self,
        direction: FloatVector,
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Return the part of an Euclidean object visible from a direction."""
        return LazyMetricMappedObject(
            self.space,
            "visible-from-direction",
            self,
            name=name,
            direction=FloatVector(direction),
        )

    def visible_from_point(
        self,
        point: FloatPoint,
        name: str = "",
    ) -> "MetricGeometricObject[PointT]":
        """Return the part of an Euclidean object visible from a point."""
        return LazyMetricMappedObject(
            self.space,
            "visible-from-point",
            self,
            name=name,
            point=FloatPoint(point),
        )

    def image_under_smooth_map(
        self,
        forward: Callable[[PointT], TargetT],
        preimage_on_image: Callable[[TargetT], PointT],
        target_space: MetricSpace[TargetT],
        target_chart,
        contains_image_point: Callable[[TargetT], bool] | None = None,
        name: str = "",
    ) -> "MetricGeometricObject[TargetT]":
        """Return the image object under a smooth map into a metric space."""
        return LazyMetricMappedObject(
            target_space,
            "image-under-smooth-map",
            self,
            name=name,
            forward=forward,
            preimage_on_image=preimage_on_image,
            target_chart=target_chart,
            contains_image_point=contains_image_point,
        )


class LazyMetricObject(MetricGeometricObject[PointT]):
    """Base class for lazy metric expression-tree nodes."""

    def __init__(
        self,
        space: MetricSpace[PointT],
        operation: str,
        name: str = "",
    ) -> None:
        """Initialize shared lazy-node metadata."""
        self.operation = operation
        super().__init__(
            space,
            contains=self._contains_lazy,
            local_model=self._local_model_lazy,
            name=name or self._default_name(),
        )

    @property
    def children(self) -> tuple[MetricGeometricObject, ...]:
        """Return child nodes of this expression-tree node."""
        return tuple()

    @property
    def is_lazy(self) -> bool:
        """Return whether this object is a lazy expression-tree node."""
        return True

    @property
    def node_kind(self) -> str:
        """Return the expression-tree node kind."""
        raise NotImplementedError

    def _default_name(self) -> str:
        """Return the default node name."""
        raise NotImplementedError

    def _contains_lazy(self, point: PointT) -> bool:
        """Evaluate point membership for the lazy node."""
        raise NotImplementedError

    def _local_model_lazy(self, point: PointT):
        """Evaluate a local model for the lazy node."""
        raise NotImplementedError


class LazyMetricExpressionObject(LazyMetricObject[PointT]):
    """Lazy set-theoretic expression over metric geometric objects."""

    def __init__(
        self,
        operation: str,
        left: MetricGeometricObject[PointT],
        right: MetricGeometricObject[PointT],
        name: str = "",
    ) -> None:
        """Initialize a lazy binary expression node."""
        if left.space is not right.space:
            raise ValueError(
                "Set-theoretic operations require the same ambient space"
            )
        self.left = left
        self.right = right
        super().__init__(
            left.space,
            operation,
            name=name,
        )

    def __repr__(self) -> str:
        """Return a debug representation."""
        return (
            "LazyMetricExpressionObject("
            f"operation={self.operation!r}, name={self.name!r})"
        )

    @property
    def children(self) -> tuple[MetricGeometricObject[PointT], ...]:
        """Return the two children of the binary expression node."""
        return (self.left, self.right)

    @property
    def node_kind(self) -> str:
        """Return the expression-tree node kind."""
        return "binary"

    def _default_name(self) -> str:
        """Return the default expression name."""
        symbols = {
            "union": "|",
            "intersection": "&",
            "difference": "-",
            "symmetric-difference": "^",
        }
        symbol = symbols.get(self.operation, "?")
        return f"({self.left.name}){symbol}({self.right.name})"

    def _contains_lazy(self, point: PointT) -> bool:
        """Evaluate point membership lazily from the expression tree."""
        left_contains = point in self.left
        right_contains = point in self.right
        if self.operation == "union":
            return left_contains or right_contains
        if self.operation == "intersection":
            return left_contains and right_contains
        if self.operation == "difference":
            return left_contains and not right_contains
        if self.operation == "symmetric-difference":
            return left_contains ^ right_contains
        raise ValueError(f"Unsupported lazy operation: {self.operation!r}")

    def _local_model_lazy(self, point: PointT):
        """Evaluate a local model lazily from the expression tree."""
        if self.operation == "union":
            if point in self.left and point in self.right:
                return MetricGeometricObject._combine_local_models(
                    self.left.local_model_at(point),
                    self.right.local_model_at(point),
                    lambda left, right: left or right,
                    self.operation,
                )
            if point in self.left:
                return self.left.local_model_at(point)
            return self.right.local_model_at(point)

        if self.operation == "intersection":
            return MetricGeometricObject._combine_local_models(
                self.left.local_model_at(point),
                self.right.local_model_at(point),
                lambda left, right: left and right,
                self.operation,
            )

        if self.operation == "difference":
            if point in self.right:
                return MetricGeometricObject._combine_local_models(
                    self.left.local_model_at(point),
                    self.right.local_model_at(point),
                    lambda left, right: left and not right,
                    self.operation,
                )
            return self.left.local_model_at(point)

        if self.operation == "symmetric-difference":
            if point in self.left and point not in self.right:
                return self.left.local_model_at(point)
            if point in self.right and point not in self.left:
                return self.right.local_model_at(point)
            return MetricGeometricObject._combine_local_models(
                self.left.local_model_at(point),
                self.right.local_model_at(point),
                lambda left, right: left ^ right,
                self.operation,
            )

        raise ValueError(f"Unsupported lazy operation: {self.operation!r}")


class LazyMetricMappedObject(LazyMetricObject):
    """Lazy unary operation over a metric geometric object."""

    def __init__(
        self,
        space: MetricSpace,
        operation: str,
        source: MetricGeometricObject,
        name: str = "",
        **parameters,
    ) -> None:
        """Initialize a lazy mapped-object node."""
        self.source = source
        self.parameters = parameters
        super().__init__(
            space,
            operation,
            name=name,
        )

    def __repr__(self) -> str:
        """Return a debug representation."""
        return (
            "LazyMetricMappedObject("
            f"operation={self.operation!r}, name={self.name!r})"
        )

    @property
    def children(self) -> tuple[MetricGeometricObject, ...]:
        """Return the single source child of the mapped node."""
        return (self.source,)

    @property
    def node_kind(self) -> str:
        """Return the expression-tree node kind."""
        return "unary"

    def _default_name(self) -> str:
        """Return the default mapped-object name."""
        return f"{self.operation}({self.source.name})"

    def _materialize_charted_object(self):
        """Return the charted object represented by this lazy mapped node."""
        source_object = getattr(self.source, "_charted_source_object", self.source)
        if self.operation == "project-along-direction":
            return ChartedGeometricObject.project_along_direction_onto(
                source_object,
                self.parameters["source_hyperplane"],
                self.parameters["target_hyperplane"],
                self.parameters["direction"],
                name=self.name,
            )
        if self.operation == "project-from-point":
            return ChartedGeometricObject.project_from_point_onto(
                source_object,
                self.parameters["source_hyperplane"],
                self.parameters["target_hyperplane"],
                self.parameters["center"],
                name=self.name,
            )
        if self.operation == "visible-from-direction":
            return ChartedGeometricObject.visible_from_direction(
                source_object,
                self.parameters["direction"],
                name=self.name,
            )
        if self.operation == "visible-from-point":
            return ChartedGeometricObject.visible_from_point(
                source_object,
                self.parameters["point"],
                name=self.name,
            )
        if self.operation == "image-under-smooth-map":
            return ChartedGeometricObject.image_under_smooth_map(
                source_object,
                self.parameters["forward"],
                self.parameters["preimage_on_image"],
                self.space,
                self.parameters["target_chart"],
                contains_image_point=self.parameters["contains_image_point"],
                name=self.name,
            )
        raise ValueError(f"Unsupported lazy mapped operation: {self.operation!r}")

    def _materialize_metric_object(self):
        """Return the materialized metric object for mesh/sample delegation."""
        return MetricGeometricObject.from_charted(
            self.space,
            self._materialize_charted_object(),
            name=self.name,
        )

    def _contains_lazy(self, point):
        """Evaluate point membership lazily."""
        return point in self._materialize_charted_object()

    def _local_model_lazy(self, point):
        """Evaluate a local model lazily."""
        return self._materialize_charted_object().local_model_at(point)

    def sample_points(self, resolution: int = 24):
        """Return sampled points from the materialized mapped object."""
        return self._materialize_metric_object().sample_points(resolution=resolution)

    def mesh(
        self,
        resolution: int = 64,
        bounds=None,
    ) -> ObjectMesh:
        """Return a mesh from the materialized mapped object."""
        return self._materialize_metric_object().mesh(
            resolution=resolution,
            bounds=bounds,
        )

RiemannianGeometricObject = MetricGeometricObject


class RealLineSpace(ChartedMetricSpace[float]):
    """The real line with its standard metric."""

    def __init__(self, name: str = "") -> None:
        """Initialize the standard metric real line."""
        super().__init__(
            RealLine(),
            distance=lambda left, right: abs(float(left) - float(right)),
            name=name or "R",
        )

    @property
    def space_kind(self) -> str:
        """Return the space kind identifier."""
        return "real-line"

    def to_2d(
        self,
        point: float,
        method: str = "default",
    ) -> tuple[float, float]:
        """Return a 2D visualization of a real-line point."""
        if method not in ("default", "line"):
            raise ValueError(f"Unknown 2D visualization method: {method!r}")
        value = float(point)
        if value not in self:
            raise ValueError("Point is outside the real line")
        return (value, 0.0)

    def to_3d(
        self,
        point: float,
        method: str = "default",
    ) -> tuple[float, float, float]:
        """Return a 3D visualization of a real-line point."""
        if method not in ("default", "line"):
            raise ValueError(f"Unknown 3D visualization method: {method!r}")
        value = float(point)
        if value not in self:
            raise ValueError("Point is outside the real line")
        return (value, 0.0, 0.0)

    def point(
        self,
        point: float,
        name: str = "",
    ) -> MetricGeometricObject[float]:
        """Return a singleton object in the real line."""
        return MetricGeometricObject.from_charted(
            self,
            RealPointObject(point, name=name),
            name=name,
        )

    def neighborhood_at(
        self,
        point: float,
        radius: float,
        name: str = "",
    ) -> ChartNeighborhood[float]:
        """Return a centered neighborhood in intrinsic coordinates."""
        center = float(point)
        if center not in self:
            raise ValueError("Point is outside the real line")
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        chart = self.point(center).local_model_at(center).chart
        return ChartNeighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box((-radius, radius)),
            name=name or "real-neighborhood",
        )

    def subset(
        self,
        *point_set: object,
        name: str = "",
    ) -> MetricGeometricObject[float]:
        """Return a geometric object built from a ``FloatSet``."""
        return MetricGeometricObject.from_charted(
            self,
            RealSetObject(*point_set, name=name),
            name=name,
        )


class UnitCircleSpace(ChartedMetricSpace[FloatCirclePoint]):
    """The unit circle with its standard arc-length metric."""

    def __init__(self, name: str = "") -> None:
        """Initialize the standard metric unit circle."""
        super().__init__(
            Circle(),
            distance=lambda left, right: float(
                FloatCirclePoint(left).distance_to(FloatCirclePoint(right))
            ),
            name=name or "S1",
        )

    @property
    def space_kind(self) -> str:
        """Return the space kind identifier."""
        return "unit-circle"

    def to_2d(
        self,
        point: FloatCirclePoint,
        method: str = "default",
    ) -> tuple[float, float]:
        """Return a planar embedding of a circle point."""
        if method not in ("default", "embedding"):
            raise ValueError(f"Unknown 2D visualization method: {method!r}")
        circle_point = FloatCirclePoint(point)
        if circle_point not in self:
            raise ValueError("Point is outside the unit circle")
        return circle_point.to_cartesian()

    def to_3d(
        self,
        point: FloatCirclePoint,
        method: str = "default",
    ) -> tuple[float, float, float]:
        """Return a 3D embedding of a circle point."""
        x, y = self.to_2d(point, method="embedding" if method == "default" else method)
        return (x, y, 0.0)

    def point(
        self,
        point: FloatCirclePoint,
        name: str = "",
    ) -> MetricGeometricObject[FloatCirclePoint]:
        """Return a singleton object on the unit circle."""
        return MetricGeometricObject.from_charted(
            self,
            CirclePointObject(point, name=name),
            name=name,
        )

    def neighborhood_at(
        self,
        point: FloatCirclePoint,
        radius: float,
        name: str = "",
    ) -> ChartNeighborhood[FloatCirclePoint]:
        """Return a centered intrinsic neighborhood smaller than a half-circle."""
        center = FloatCirclePoint(point)
        if center not in self:
            raise ValueError("Point is outside the unit circle")
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("Neighborhood radius must be positive")
        if radius >= math.pi:
            raise ValueError("Circle neighborhoods must have radius < pi")
        chart = self.point(center).local_model_at(center).chart
        return ChartNeighborhood(
            self,
            chart,
            center,
            EuclideanNeighborhood.box((-radius, radius)),
            name=name or "circle-neighborhood",
        )

    def subset(
        self,
        *point_set: object,
        name: str = "",
    ) -> MetricGeometricObject[FloatCirclePoint]:
        """Return a geometric object built from a ``FloatCircleSet``."""
        return MetricGeometricObject.from_charted(
            self,
            CircleSetObject(*point_set, name=name),
            name=name,
        )

    def arc(
        self,
        start: FloatCirclePoint,
        end: FloatCirclePoint,
        name: str = "",
    ) -> MetricGeometricObject[FloatCirclePoint]:
        """Return a connected arc on the unit circle."""
        return self.subset((start, end), name=name)


class EuclideanMetricSpace(ChartedMetricSpace[FloatPoint]):
    """Euclidean space with its standard metric."""

    def __init__(self, dim: int, name: str = "") -> None:
        """Initialize Euclidean space with the Euclidean distance."""
        self._dim = dim
        super().__init__(
            EuclideanSpace(dim),
            distance=lambda left, right: FloatPoint(left).distance_to(
                FloatPoint(right)
            ),
            name=name or f"R^{dim}",
        )

    @property
    def space_kind(self) -> str:
        """Return the space kind identifier."""
        return "euclidean"

    def to_2d(
        self,
        point: FloatPoint,
        method: str = "default",
    ) -> tuple[float, float]:
        """Return a 2D visualization by truncation or zero-padding."""
        if method not in ("default", "orthographic"):
            raise ValueError(f"Unknown 2D visualization method: {method!r}")
        coordinates = FloatPoint(point)
        if coordinates not in self:
            raise ValueError("Point is outside the Euclidean space")
        values = tuple(coordinates)
        if self.dim >= 2:
            return (values[0], values[1])
        return (values[0], 0.0)

    def to_3d(
        self,
        point: FloatPoint,
        method: str = "default",
    ) -> tuple[float, float, float]:
        """Return a 3D visualization by truncation or zero-padding."""
        if method not in ("default", "orthographic"):
            raise ValueError(f"Unknown 3D visualization method: {method!r}")
        coordinates = FloatPoint(point)
        if coordinates not in self:
            raise ValueError("Point is outside the Euclidean space")
        values = tuple(coordinates)
        if self.dim >= 3:
            return (values[0], values[1], values[2])
        if self.dim == 2:
            return (values[0], values[1], 0.0)
        return (values[0], 0.0, 0.0)

    def point(
        self,
        point: FloatPoint,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return a singleton object in Euclidean space."""
        return MetricGeometricObject.from_charted(
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
        """Return an axis-aligned centered neighborhood."""
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


class EuclideanRiemannianSpace(EuclideanMetricSpace):
    """Compatibility alias for the older Euclidean metric-space naming."""


class EuclideanPlaneSpace(EuclideanMetricSpace):
    """The Euclidean plane with several standard geometric objects."""

    def __init__(self, name: str = "") -> None:
        """Initialize the standard metric plane."""
        super().__init__(2, name=name or "R^2")

    def whole_plane(
        self,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return the whole plane as a geometric object."""
        return MetricGeometricObject.from_charted(
            self,
            WholePlane(name=name),
            name=name,
        )

    def ball(
        self,
        center: FloatPoint,
        radius: float,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return a closed Euclidean disk in the ambient plane."""
        return self.wrap(Ball(center, radius, name=name), name=name)

    def disk(
        self,
        center: FloatPoint,
        radius: float,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return a closed Euclidean disk in the ambient plane."""
        return self.ball(center, radius, name=name)

    def circle(
        self,
        center: FloatPoint,
        radius: float,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return the boundary circle in the ambient plane."""
        return self.wrap(Sphere(center, radius, name=name), name=name)

    def half_plane(
        self,
        normal: FloatVector,
        offset: float = 0.0,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return a closed half-plane."""
        return MetricGeometricObject.from_charted(
            self,
            HalfPlane(normal, offset=offset, name=name),
            name=name,
        )

    def angle(
        self,
        apex: FloatPoint,
        start: FloatCirclePoint,
        end: FloatCirclePoint,
        name: str = "",
    ) -> MetricGeometricObject[FloatPoint]:
        """Return a closed planar angle with interior."""
        return MetricGeometricObject.from_charted(
            self,
            PlanarAngle(apex, start, end, name=name),
            name=name,
        )


__all__ = [
    "MetricTensor",
    "MetricSpace",
    "ChartedMetricSpace",
    "MetricGeometricObject",
    "RiemannianSpace",
    "ChartedRiemannianSpace",
    "RiemannianGeometricObject",
    "RealLineSpace",
    "UnitCircleSpace",
    "EuclideanMetricSpace",
    "EuclideanRiemannianSpace",
    "EuclideanPlaneSpace",
]
