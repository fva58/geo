"""Euclidean points, vectors, and local coordinate objects."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable, Tuple

import numpy as np

from .line import Interval, Set


def _coerce_coordinate_array(value: object) -> np.ndarray:
    """Normalize a coordinate container into a 1-D float array."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise TypeError("Coordinates must form a one-dimensional sequence")
    return array


def _coerce_coordinates(args: tuple[object, ...]) -> Tuple[float, ...]:
    """Normalize constructor arguments into a tuple of float coordinates."""
    if len(args) == 1 and isinstance(args[0], Sequence) and not isinstance(
        args[0], (str, bytes)
    ):
        return tuple(_coerce_coordinate_array(args[0]).tolist())
    if len(args) == 1 and hasattr(args[0], "__array__"):
        return tuple(_coerce_coordinate_array(args[0]).tolist())
    return tuple(_coerce_coordinate_array(args).tolist())


def _check_same_dimension(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> None:
    """Require equal dimensions for coordinate-wise operations."""
    if len(left) != len(right):
        raise ValueError(
            f"Dimension mismatch: {len(left)} != {len(right)}"
        )


def _coerce_matrix(
    rows: Sequence[Sequence[object]],
) -> Tuple[Tuple[float, ...], ...]:
    """Normalize a matrix into a tuple of float rows."""
    try:
        array = np.asarray(rows, dtype=float)
    except ValueError as exc:
        raise ValueError("Matrix rows must have equal length") from exc
    if array.ndim != 2:
        raise ValueError("Matrix must be two-dimensional")
    if array.shape[0] == 0:
        raise ValueError("Matrix must not be empty")
    if array.shape[1] == 0:
        raise ValueError("Matrix rows must not be empty")
    return tuple(tuple(float(value) for value in row) for row in array)


def _matvec(matrix: Tuple[Tuple[float, ...], ...],
            vector: tuple[float, ...]) -> Tuple[float, ...]:
    """Apply a matrix to a coordinate tuple."""
    if len(matrix[0]) != len(vector):
        raise ValueError(
            f"Matrix/vector mismatch: {len(matrix[0])} != {len(vector)}"
        )
    result = np.matmul(
        np.asarray(matrix, dtype=float),
        np.asarray(vector, dtype=float),
    )
    return tuple(float(value) for value in result)


def _invert_square_matrix(
    matrix: Tuple[Tuple[float, ...], ...]
) -> Tuple[Tuple[float, ...], ...]:
    """Invert a square matrix with NumPy."""
    array = np.asarray(matrix, dtype=float)
    if array.shape[0] != array.shape[1]:
        raise ValueError("Matrix must be square")
    determinant = float(np.linalg.det(array))
    if math.isclose(determinant, 0.0, abs_tol=1e-15):
        raise ValueError("Matrix must be invertible")
    inverse = np.linalg.inv(array)
    return tuple(
        tuple(float(value) for value in row)
        for row in inverse
    )


class Vector(tuple):
    """Euclidean vector with float coordinates."""

    __slots__ = ()

    def __new__(cls, *coordinates: object) -> "Vector":
        """Create a vector from coordinates or a coordinate sequence."""
        return super().__new__(cls, _coerce_coordinates(coordinates))

    @property
    def dim(self) -> int:
        """Return the vector dimension."""
        return len(self)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"Vector{tuple(self)}"

    def __add__(self, other: "Vector") -> "Vector":
        """Add another vector."""
        _check_same_dimension(self, other)
        result = np.asarray(self, dtype=float) + np.asarray(other, dtype=float)
        return Vector(result)

    def __sub__(self, other: "Vector") -> "Vector":
        """Subtract another vector."""
        _check_same_dimension(self, other)
        result = np.asarray(self, dtype=float) - np.asarray(other, dtype=float)
        return Vector(result)

    def __neg__(self) -> "Vector":
        """Return the additive inverse."""
        return Vector(-np.asarray(self, dtype=float))

    def __mul__(self, scalar: float) -> "Vector":
        """Multiply by a scalar."""
        return Vector(np.asarray(self, dtype=float) * float(scalar))

    def __rmul__(self, scalar: float) -> "Vector":
        """Multiply by a scalar from the left."""
        return self * scalar

    def __truediv__(self, scalar: float) -> "Vector":
        """Divide by a scalar."""
        return Vector(np.asarray(self, dtype=float) / float(scalar))

    def dot(self, other: "Vector") -> float:
        """Return the Euclidean dot product."""
        _check_same_dimension(self, other)
        return float(np.dot(np.asarray(self, dtype=float),
                            np.asarray(other, dtype=float)))

    def norm(self) -> float:
        """Return the Euclidean norm."""
        return float(np.linalg.norm(np.asarray(self, dtype=float)))

    def to_tuple(self) -> Tuple[float, ...]:
        """Return coordinates as a plain tuple."""
        return tuple(self)

    @classmethod
    def zero(cls, dim: int) -> "Vector":
        """Return the zero vector in the given dimension."""
        return cls((0.0,) * dim)


class Point(tuple):
    """Euclidean point with float coordinates."""

    __slots__ = ()

    def __new__(cls, *coordinates: object) -> "Point":
        """Create a point from coordinates or a coordinate sequence."""
        return super().__new__(cls, _coerce_coordinates(coordinates))

    @property
    def dim(self) -> int:
        """Return the point dimension."""
        return len(self)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"Point{tuple(self)}"

    def __add__(self, vector: Vector) -> "Point":
        """Translate the point by a vector."""
        _check_same_dimension(self, vector)
        result = np.asarray(self, dtype=float) + np.asarray(vector, dtype=float)
        return Point(result)

    def __sub__(self, other: object) -> object:
        """Subtract a point or vector.

        - point - point -> vector
        - point - vector -> point
        """
        if isinstance(other, Point):
            _check_same_dimension(self, other)
            result = (
                np.asarray(self, dtype=float) -
                np.asarray(other, dtype=float)
            )
            return Vector(result)
        if isinstance(other, Vector):
            _check_same_dimension(self, other)
            result = (
                np.asarray(self, dtype=float) -
                np.asarray(other, dtype=float)
            )
            return Point(result)
        return NotImplemented

    def distance_to(self, other: "Point") -> float:
        """Return the Euclidean distance to another point."""
        return (self - other).norm()

    def to_tuple(self) -> Tuple[float, ...]:
        """Return coordinates as a plain tuple."""
        return tuple(self)

    @classmethod
    def origin(cls, dim: int) -> "Point":
        """Return the origin in the given dimension."""
        return cls((0.0,) * dim)


class EuclideanNeighborhood(tuple):
    """Rectangular neighborhood in Euclidean space.

    The neighborhood is modeled as a Cartesian product of line ``Set``
    coordinate constraints.
    """

    __slots__ = ()

    def __new__(cls, *coordinate_sets: object) -> "EuclideanNeighborhood":
        """Create a neighborhood from coordinate sets or sequences."""
        if len(coordinate_sets) == 1 and isinstance(
            coordinate_sets[0], Sequence
        ) and not isinstance(coordinate_sets[0], (str, bytes, Set)):
            coordinate_sets = tuple(coordinate_sets[0])
        normalized = tuple(Set(coordinate_set)
                           for coordinate_set in coordinate_sets)
        return super().__new__(cls, normalized)

    @property
    def dim(self) -> int:
        """Return the ambient dimension."""
        return len(self)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"EuclideanNeighborhood{tuple(self)}"

    def contains(self, point: Point) -> bool:
        """Check whether a point belongs to the neighborhood."""
        point = Point(point)
        if point.dim != self.dim:
            return False
        return all(coordinate in coordinate_set
                   for coordinate, coordinate_set in zip(point, self))

    def __contains__(self, point: Point) -> bool:
        """Check whether a point belongs to the neighborhood."""
        return self.contains(point)

    @classmethod
    def box(cls, *bounds: object) -> "EuclideanNeighborhood":
        """Create a rectangular neighborhood from interval bounds.

        Each bound may be:

        - ``line.Interval``
        - ``line.Set``
        - a pair ``(left, right)``
        """
        coordinate_sets = []
        for bound in bounds:
            if isinstance(bound, Set):
                coordinate_sets.append(bound)
            elif isinstance(bound, Interval):
                coordinate_sets.append(Set(bound))
            elif (isinstance(bound, Sequence) and
                  not isinstance(bound, (str, bytes)) and len(bound) == 2):
                coordinate_sets.append(
                    Set.from_single_interval(bound[0], bound[1])
                )
            else:
                raise TypeError(f"Unsupported neighborhood bound: {bound!r}")
        return cls(*coordinate_sets)

    @classmethod
    def whole(cls, dim: int) -> "EuclideanNeighborhood":
        """Return the whole Euclidean space of the given dimension."""
        return cls(*(Set(Interval(-math.inf, math.inf))
                     for _ in range(dim)))


class EuclideanChart:
    """Local chart between Euclidean point neighborhoods.

    The chart is given by explicit forward and inverse callables. Both are
    checked only on point dimensions; smoothness and locality remain semantic
    obligations of the caller.
    """

    def __init__(
        self,
        forward: Callable[[Point], Point],
        inverse: Callable[[Point], Point],
        source_dim: int,
        target_dim: int,
        domain: EuclideanNeighborhood | None = None,
        image: EuclideanNeighborhood | None = None,
    ) -> None:
        """Initialize a chart from forward and inverse maps."""
        self._forward = forward
        self._inverse = inverse
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.domain = domain
        self.image = image

    def __repr__(self) -> str:
        """Return a debug representation."""
        domain = f", domain_dim={self.domain.dim}" if self.domain else ""
        image = f", image_dim={self.image.dim}" if self.image else ""
        return (
            "EuclideanChart("
            f"source_dim={self.source_dim}, target_dim={self.target_dim}"
            f"{domain}{image})"
        )

    def __call__(self, point: Point) -> Point:
        """Apply the forward chart map."""
        point = Point(point)
        if point.dim != self.source_dim:
            raise ValueError(
                f"Source dimension mismatch: {point.dim} != {self.source_dim}"
            )
        if self.domain is not None and point not in self.domain:
            raise ValueError("Point is outside the chart domain")
        image = Point(self._forward(point))
        if image.dim != self.target_dim:
            raise ValueError(
                f"Target dimension mismatch: {image.dim} != {self.target_dim}"
            )
        if self.image is not None and image not in self.image:
            raise ValueError("Image point is outside the chart image")
        return image

    def inverse(self, point: Point) -> Point:
        """Apply the inverse chart map."""
        point = Point(point)
        if point.dim != self.target_dim:
            raise ValueError(
                f"Target dimension mismatch: {point.dim} != {self.target_dim}"
            )
        if self.image is not None and point not in self.image:
            raise ValueError("Point is outside the chart image")
        preimage = Point(self._inverse(point))
        if preimage.dim != self.source_dim:
            raise ValueError(
                f"Source dimension mismatch: {preimage.dim} != {self.source_dim}"
            )
        if self.domain is not None and preimage not in self.domain:
            raise ValueError("Inverse image point is outside the chart domain")
        return preimage

    def inverse_chart(self) -> "EuclideanChart":
        """Return the inverse chart."""
        return EuclideanChart(
            self._inverse,
            self._forward,
            self.target_dim,
            self.source_dim,
            domain=self.image,
            image=self.domain,
        )

    def compose(self, other: "EuclideanChart") -> "EuclideanChart":
        """Return the composition ``self ∘ other``."""
        if other.target_dim != self.source_dim:
            raise ValueError(
                "Chart dimensions do not compose: "
                f"{other.target_dim} != {self.source_dim}"
            )

        def forward(point: Point) -> Point:
            return self(other(point))

        def inverse(point: Point) -> Point:
            return other.inverse(self.inverse(point))

        return EuclideanChart(
            forward,
            inverse,
            other.source_dim,
            self.target_dim,
            domain=other.domain,
            image=self.image,
        )

    @classmethod
    def identity(cls, dim: int) -> "EuclideanChart":
        """Return the identity chart in the given dimension."""
        return cls(
            lambda point: point,
            lambda point: point,
            dim,
            dim,
            domain=EuclideanNeighborhood.whole(dim),
            image=EuclideanNeighborhood.whole(dim),
        )


class AffineDiffeomorphism(EuclideanChart):
    """Affine diffeomorphism ``x ↦ A x + b`` on Euclidean space."""

    def __init__(
        self,
        matrix: Sequence[Sequence[object]],
        offset: Sequence[object] | Vector,
    ) -> None:
        """Initialize an affine diffeomorphism from a matrix and offset."""
        self.matrix = _coerce_matrix(matrix)
        self.offset = Vector(offset)
        self.target_dim = len(self.matrix)
        self.source_dim = len(self.matrix[0])

        if self.offset.dim != self.target_dim:
            raise ValueError(
                f"Offset dimension mismatch: {self.offset.dim} != {self.target_dim}"
            )
        if self.source_dim != self.target_dim:
            raise ValueError("Affine diffeomorphism matrix must be square")

        self.inverse_matrix = _invert_square_matrix(self.matrix)
        self.inverse_offset = Vector(
            tuple(-value for value in _matvec(self.inverse_matrix, self.offset))
        )

        super().__init__(
            self._forward_map,
            self._inverse_map,
            source_dim=self.source_dim,
            target_dim=self.target_dim,
            domain=EuclideanNeighborhood.whole(self.source_dim),
            image=EuclideanNeighborhood.whole(self.target_dim),
        )

    def _forward_map(self, point: Point) -> Point:
        """Apply the affine forward map."""
        return Point(_matvec(self.matrix, point)) + self.offset

    def _inverse_map(self, point: Point) -> Point:
        """Apply the affine inverse map."""
        return Point(_matvec(self.inverse_matrix, point)) + self.inverse_offset

    def __repr__(self) -> str:
        """Return a debug representation."""
        return (
            "AffineDiffeomorphism("
            f"matrix={self.matrix}, offset={self.offset.to_tuple()})"
        )

    def inverse_chart(self) -> EuclideanChart:
        """Return the inverse affine diffeomorphism."""
        return AffineDiffeomorphism(
            self.inverse_matrix,
            self.inverse_offset,
        )


# Backward compatibility aliases
FloatVector = Vector
FloatPoint = Point

__all__ = [
    "Vector",
    "Point",
    "FloatVector",
    "FloatPoint",
    "EuclideanNeighborhood",
    "EuclideanChart",
    "AffineDiffeomorphism",
]
