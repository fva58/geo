"""Euclidean points, vectors, and local coordinate objects."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable, Tuple

from .floatset import FloatInterval, FloatSet


def _coerce_coordinates(args: tuple[object, ...]) -> Tuple[float, ...]:
    """Normalize constructor arguments into a tuple of float coordinates."""
    if len(args) == 1 and isinstance(args[0], Sequence) and not isinstance(
        args[0], (str, bytes)
    ):
        return tuple(float(value) for value in args[0])
    return tuple(float(value) for value in args)


def _check_same_dimension(left: tuple[float, ...], right: tuple[float, ...]) -> None:
    """Require equal dimensions for coordinate-wise operations."""
    if len(left) != len(right):
        raise ValueError(
            f"Dimension mismatch: {len(left)} != {len(right)}"
        )


def _coerce_matrix(rows: Sequence[Sequence[object]]) -> Tuple[Tuple[float, ...], ...]:
    """Normalize a matrix into a tuple of float rows."""
    matrix = tuple(tuple(float(value) for value in row) for row in rows)
    if not matrix:
        raise ValueError("Matrix must not be empty")
    row_length = len(matrix[0])
    if row_length == 0:
        raise ValueError("Matrix rows must not be empty")
    if any(len(row) != row_length for row in matrix):
        raise ValueError("Matrix rows must have equal length")
    return matrix


def _matvec(matrix: Tuple[Tuple[float, ...], ...],
            vector: tuple[float, ...]) -> Tuple[float, ...]:
    """Apply a matrix to a coordinate tuple."""
    if len(matrix[0]) != len(vector):
        raise ValueError(
            f"Matrix/vector mismatch: {len(matrix[0])} != {len(vector)}"
        )
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _invert_square_matrix(
    matrix: Tuple[Tuple[float, ...], ...]
) -> Tuple[Tuple[float, ...], ...]:
    """Invert a square matrix by Gaussian elimination."""
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Matrix must be square")

    augmented = [
        [float(value) for value in row] +
        [1.0 if i == j else 0.0 for j in range(size)]
        for i, row in enumerate(matrix)
    ]

    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        pivot_value = augmented[pivot_row][pivot_index]
        if math.isclose(pivot_value, 0.0, abs_tol=1e-15):
            raise ValueError("Matrix must be invertible")

        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_index],
            )

        pivot_value = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [
            value / pivot_value for value in augmented[pivot_index]
        ]

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                current - factor * pivot
                for current, pivot in zip(
                    augmented[row_index],
                    augmented[pivot_index],
                )
            ]

    return tuple(
        tuple(row[size:]) for row in augmented
    )


class FloatVector(tuple):
    """Euclidean vector with float coordinates."""

    __slots__ = ()

    def __new__(cls, *coordinates: object) -> "FloatVector":
        """Create a vector from coordinates or a coordinate sequence."""
        return super().__new__(cls, _coerce_coordinates(coordinates))

    @property
    def dim(self) -> int:
        """Return the vector dimension."""
        return len(self)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"FloatVector{tuple(self)}"

    def __add__(self, other: "FloatVector") -> "FloatVector":
        """Add another vector."""
        _check_same_dimension(self, other)
        return FloatVector(tuple(a + b for a, b in zip(self, other)))

    def __sub__(self, other: "FloatVector") -> "FloatVector":
        """Subtract another vector."""
        _check_same_dimension(self, other)
        return FloatVector(tuple(a - b for a, b in zip(self, other)))

    def __neg__(self) -> "FloatVector":
        """Return the additive inverse."""
        return FloatVector(tuple(-value for value in self))

    def __mul__(self, scalar: float) -> "FloatVector":
        """Multiply by a scalar."""
        return FloatVector(tuple(value * float(scalar) for value in self))

    def __rmul__(self, scalar: float) -> "FloatVector":
        """Multiply by a scalar from the left."""
        return self * scalar

    def __truediv__(self, scalar: float) -> "FloatVector":
        """Divide by a scalar."""
        return FloatVector(tuple(value / float(scalar) for value in self))

    def dot(self, other: "FloatVector") -> float:
        """Return the Euclidean dot product."""
        _check_same_dimension(self, other)
        return sum(a * b for a, b in zip(self, other))

    def norm(self) -> float:
        """Return the Euclidean norm."""
        return math.sqrt(self.dot(self))

    def to_tuple(self) -> Tuple[float, ...]:
        """Return coordinates as a plain tuple."""
        return tuple(self)

    @classmethod
    def zero(cls, dim: int) -> "FloatVector":
        """Return the zero vector in the given dimension."""
        return cls((0.0,) * dim)


class FloatPoint(tuple):
    """Euclidean point with float coordinates."""

    __slots__ = ()

    def __new__(cls, *coordinates: object) -> "FloatPoint":
        """Create a point from coordinates or a coordinate sequence."""
        return super().__new__(cls, _coerce_coordinates(coordinates))

    @property
    def dim(self) -> int:
        """Return the point dimension."""
        return len(self)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"FloatPoint{tuple(self)}"

    def __add__(self, vector: FloatVector) -> "FloatPoint":
        """Translate the point by a vector."""
        _check_same_dimension(self, vector)
        return FloatPoint(tuple(a + b for a, b in zip(self, vector)))

    def __sub__(self, other: object) -> object:
        """Subtract a point or vector.

        - point - point -> vector
        - point - vector -> point
        """
        if isinstance(other, FloatPoint):
            _check_same_dimension(self, other)
            return FloatVector(tuple(a - b for a, b in zip(self, other)))
        if isinstance(other, FloatVector):
            _check_same_dimension(self, other)
            return FloatPoint(tuple(a - b for a, b in zip(self, other)))
        return NotImplemented

    def distance_to(self, other: "FloatPoint") -> float:
        """Return the Euclidean distance to another point."""
        return (self - other).norm()

    def to_tuple(self) -> Tuple[float, ...]:
        """Return coordinates as a plain tuple."""
        return tuple(self)

    @classmethod
    def origin(cls, dim: int) -> "FloatPoint":
        """Return the origin in the given dimension."""
        return cls((0.0,) * dim)


class EuclideanNeighborhood(tuple):
    """Rectangular neighborhood in Euclidean space.

    The neighborhood is modeled as a Cartesian product of ``FloatSet``
    coordinate constraints.
    """

    __slots__ = ()

    def __new__(cls, *coordinate_sets: object) -> "EuclideanNeighborhood":
        """Create a neighborhood from coordinate sets or sequences."""
        if len(coordinate_sets) == 1 and isinstance(
            coordinate_sets[0], Sequence
        ) and not isinstance(coordinate_sets[0], (str, bytes, FloatSet)):
            coordinate_sets = tuple(coordinate_sets[0])
        normalized = tuple(FloatSet(coordinate_set)
                           for coordinate_set in coordinate_sets)
        return super().__new__(cls, normalized)

    @property
    def dim(self) -> int:
        """Return the ambient dimension."""
        return len(self)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"EuclideanNeighborhood{tuple(self)}"

    def contains(self, point: FloatPoint) -> bool:
        """Check whether a point belongs to the neighborhood."""
        point = FloatPoint(point)
        if point.dim != self.dim:
            return False
        return all(coordinate in coordinate_set
                   for coordinate, coordinate_set in zip(point, self))

    def __contains__(self, point: FloatPoint) -> bool:
        """Check whether a point belongs to the neighborhood."""
        return self.contains(point)

    @classmethod
    def box(cls, *bounds: object) -> "EuclideanNeighborhood":
        """Create a rectangular neighborhood from interval bounds.

        Each bound may be:

        - ``FloatInterval``
        - ``FloatSet``
        - a pair ``(left, right)``
        """
        coordinate_sets = []
        for bound in bounds:
            if isinstance(bound, FloatSet):
                coordinate_sets.append(bound)
            elif isinstance(bound, FloatInterval):
                coordinate_sets.append(FloatSet(bound))
            elif (isinstance(bound, Sequence) and
                  not isinstance(bound, (str, bytes)) and len(bound) == 2):
                coordinate_sets.append(
                    FloatSet.from_single_interval(bound[0], bound[1])
                )
            else:
                raise TypeError(f"Unsupported neighborhood bound: {bound!r}")
        return cls(*coordinate_sets)

    @classmethod
    def whole(cls, dim: int) -> "EuclideanNeighborhood":
        """Return the whole Euclidean space of the given dimension."""
        return cls(*(FloatSet(FloatInterval(-math.inf, math.inf))
                     for _ in range(dim)))


class EuclideanChart:
    """Local chart between Euclidean point neighborhoods.

    The chart is given by explicit forward and inverse callables. Both are
    checked only on point dimensions; smoothness and locality remain semantic
    obligations of the caller.
    """

    def __init__(
        self,
        forward: Callable[[FloatPoint], FloatPoint],
        inverse: Callable[[FloatPoint], FloatPoint],
        source_dim: int,
        target_dim: int,
        domain: EuclideanNeighborhood | None = None,
        image: EuclideanNeighborhood | None = None,
        name: str = "",
    ) -> None:
        """Initialize a chart from forward and inverse maps."""
        self._forward = forward
        self._inverse = inverse
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.domain = domain
        self.image = image
        self.name = name

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        domain = f", domain_dim={self.domain.dim}" if self.domain else ""
        image = f", image_dim={self.image.dim}" if self.image else ""
        return (
            "EuclideanChart("
            f"source_dim={self.source_dim}, target_dim={self.target_dim}"
            f"{domain}{image}{label})"
        )

    def __call__(self, point: FloatPoint) -> FloatPoint:
        """Apply the forward chart map."""
        point = FloatPoint(point)
        if point.dim != self.source_dim:
            raise ValueError(
                f"Source dimension mismatch: {point.dim} != {self.source_dim}"
            )
        if self.domain is not None and point not in self.domain:
            raise ValueError("Point is outside the chart domain")
        image = FloatPoint(self._forward(point))
        if image.dim != self.target_dim:
            raise ValueError(
                f"Target dimension mismatch: {image.dim} != {self.target_dim}"
            )
        if self.image is not None and image not in self.image:
            raise ValueError("Image point is outside the chart image")
        return image

    def inverse(self, point: FloatPoint) -> FloatPoint:
        """Apply the inverse chart map."""
        point = FloatPoint(point)
        if point.dim != self.target_dim:
            raise ValueError(
                f"Target dimension mismatch: {point.dim} != {self.target_dim}"
            )
        if self.image is not None and point not in self.image:
            raise ValueError("Point is outside the chart image")
        preimage = FloatPoint(self._inverse(point))
        if preimage.dim != self.source_dim:
            raise ValueError(
                f"Source dimension mismatch: {preimage.dim} != {self.source_dim}"
            )
        if self.domain is not None and preimage not in self.domain:
            raise ValueError("Inverse image point is outside the chart domain")
        return preimage

    def inverse_chart(self) -> "EuclideanChart":
        """Return the inverse chart."""
        inverse_name = f"{self.name}^-1" if self.name else ""
        return EuclideanChart(
            self._inverse,
            self._forward,
            self.target_dim,
            self.source_dim,
            domain=self.image,
            image=self.domain,
            name=inverse_name,
        )

    def compose(self, other: "EuclideanChart") -> "EuclideanChart":
        """Return the composition ``self ∘ other``."""
        if other.target_dim != self.source_dim:
            raise ValueError(
                "Chart dimensions do not compose: "
                f"{other.target_dim} != {self.source_dim}"
            )

        def forward(point: FloatPoint) -> FloatPoint:
            return self(other(point))

        def inverse(point: FloatPoint) -> FloatPoint:
            return other.inverse(self.inverse(point))

        if self.name and other.name:
            name = f"{self.name} o {other.name}"
        else:
            name = ""

        return EuclideanChart(
            forward,
            inverse,
            other.source_dim,
            self.target_dim,
            domain=other.domain,
            image=self.image,
            name=name,
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
            name="id",
        )


class AffineDiffeomorphism(EuclideanChart):
    """Affine diffeomorphism ``x ↦ A x + b`` on Euclidean space."""

    def __init__(
        self,
        matrix: Sequence[Sequence[object]],
        offset: Sequence[object] | FloatVector,
        name: str = "",
    ) -> None:
        """Initialize an affine diffeomorphism from a matrix and offset."""
        self.matrix = _coerce_matrix(matrix)
        self.offset = FloatVector(offset)
        self.target_dim = len(self.matrix)
        self.source_dim = len(self.matrix[0])

        if self.offset.dim != self.target_dim:
            raise ValueError(
                f"Offset dimension mismatch: {self.offset.dim} != {self.target_dim}"
            )
        if self.source_dim != self.target_dim:
            raise ValueError("Affine diffeomorphism matrix must be square")

        self.inverse_matrix = _invert_square_matrix(self.matrix)
        self.inverse_offset = FloatVector(
            tuple(-value for value in _matvec(self.inverse_matrix, self.offset))
        )
        self.name = name

        super().__init__(
            self._forward_map,
            self._inverse_map,
            source_dim=self.source_dim,
            target_dim=self.target_dim,
            domain=EuclideanNeighborhood.whole(self.source_dim),
            image=EuclideanNeighborhood.whole(self.target_dim),
            name=name,
        )

    def _forward_map(self, point: FloatPoint) -> FloatPoint:
        """Apply the affine forward map."""
        return FloatPoint(_matvec(self.matrix, point)) + self.offset

    def _inverse_map(self, point: FloatPoint) -> FloatPoint:
        """Apply the affine inverse map."""
        return FloatPoint(_matvec(self.inverse_matrix, point)) + self.inverse_offset

    def __repr__(self) -> str:
        """Return a debug representation."""
        label = f", name={self.name!r}" if self.name else ""
        return (
            "AffineDiffeomorphism("
            f"matrix={self.matrix}, offset={self.offset.to_tuple()}{label})"
        )

    def inverse_chart(self) -> EuclideanChart:
        """Return the inverse affine diffeomorphism."""
        inverse_name = f"{self.name}^-1" if self.name else ""
        return AffineDiffeomorphism(
            self.inverse_matrix,
            self.inverse_offset,
            name=inverse_name,
        )


__all__ = [
    "FloatVector",
    "FloatPoint",
    "EuclideanNeighborhood",
    "EuclideanChart",
    "AffineDiffeomorphism",
]
