Local Geometry
==============

Half-line as a geometric object with a local cone model::

    >>> from dataclasses import dataclass
    >>> from geo import (
    ...     ChartedGeometricObject,
    ...     EuclideanCone,
    ...     EuclideanNeighborhood,
    ...     FloatPoint,
    ...     LocalConeModel,
    ...     ManifoldChart,
    ... )
    >>> @dataclass(frozen=True)
    ... class LinePoint:
    ...     x: float
    >>> class LineManifold:
    ...     dim = 1
    ...     def contains(self, point):
    ...         return isinstance(point, LinePoint)
    ...     def __contains__(self, point):
    ...         return self.contains(point)
    >>> manifold = LineManifold()
    >>> chart = ManifoldChart(
    ...     lambda point: FloatPoint(point.x),
    ...     lambda coordinates: LinePoint(coordinates[0]),
    ...     dim=1,
    ...     domain_contains=manifold.contains,
    ...     image=EuclideanNeighborhood.box((-10.0, 10.0)),
    ... )
    >>> half_line = ChartedGeometricObject(
    ...     manifold,
    ...     contains=lambda point: point.x >= 0.0,
    ...     local_model=lambda point: LocalConeModel(
    ...         chart,
    ...         EuclideanCone(1, contains=lambda coordinates: coordinates[0] >= 0.0)
    ...         if point.x == 0.0 else EuclideanCone.whole(1),
    ...     ),
    ... )
    >>> model = half_line.local_model_at(LinePoint(0.0))
    >>> LinePoint(1.0) in half_line
    True
    >>> LinePoint(-1.0) in half_line
    False
    >>> FloatPoint(2.0) in model.cone
    True
    >>> FloatPoint(-2.0) in model.cone
    False

Cone in ``R^2`` induced by an arc on the unit circle::

    >>> import math
    >>> from geo import (
    ...     CircleSphereObject,
    ...     EuclideanNeighborhood,
    ...     FloatCircleSet,
    ...     FloatPoint,
    ...     SphericalCone,
    ... )
    >>> base = CircleSphereObject(
    ...     FloatCircleSet.from_single_interval(0.0, math.pi / 2.0)
    ... )
    >>> cone = SphericalCone(
    ...     base,
    ...     neighborhood=EuclideanNeighborhood.box(
    ...         (-10.0, 10.0),
    ...         (-10.0, 10.0),
    ...     ),
    ... )
    >>> FloatPoint(2.0, 3.0) in cone
    True
    >>> FloatPoint(-2.0, 3.0) in cone
    False
    >>> FloatPoint(0.0, 0.0) in cone
    True
