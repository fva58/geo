geo
===

``geo`` is an early-stage geometry package built around immutable interval
sets.

Current model
-------------

At the current stage, real numbers are modeled by Python ``float``.

This has two direct consequences:

- interval and set operations work on the discrete lattice of representable
  floating-point values;
- when a set operation removes a boundary point but still returns closed
  intervals, the implementation uses ``math.nextafter`` to move to the nearest
  representable float.

Main types
----------

Real line:

- ``FloatInterval``: one closed interval of ``float`` values;
- ``FloatSet``: normalized union of disjoint ``FloatInterval`` objects.

Circle:

- ``FloatAngle``: angle normalized to ``[0, 2π)``;
- ``FloatCirclePoint``: point on the unit circle represented by an angle;
- ``FloatCircleInterval``: connected arc on the circle;
- ``FloatCircleSet``: set of circle points built on top of ``FloatSet``.

Local differential-geometric layer:

- ``FloatPoint`` and ``FloatVector``: Euclidean coordinates in ``R^n``;
- ``EuclideanNeighborhood``: rectangular coordinate neighborhood;
- ``Map``, ``Diffeomorphism``, ``Chart``: structural protocols;
- ``ManifoldChart``, ``ChartTransition``, ``Atlas``: local coordinates on a
  manifold;
- ``EuclideanCone``, ``RadialCone``, ``SphericalCone``: cone models in
  coordinates;
- ``ChartedGeometricObject``: geometric object described by local cone models.

For arcs crossing zero, the internal representation is split into two linear
intervals: one before zero and one after zero.

Examples
--------

Real-line intervals and sets::

    >>> from geo import FloatInterval, FloatSet
    >>> interval = FloatInterval(0.0, 5.0)
    >>> diff = interval.difference(FloatInterval(2.0, 3.0))
    >>> len(diff)
    2
    >>> diff[0].contains(2.0)
    False
    >>> diff[1].contains(3.0)
    False
    >>> fset = FloatSet.from_single_interval(0.0, 1.0)
    >>> (fset | FloatSet.from_single_interval(2.0, 3.0)).to_tuple()
    ((0.0, 1.0), (2.0, 3.0))

Circle intervals::

    >>> import math
    >>> from geo import FloatCircleInterval
    >>> arc = FloatCircleInterval(3 * math.pi / 2, math.pi / 2)
    >>> arc.to_tuple()
    (4.71238898038469, 1.5707963267948966)
    >>> len(arc)
    2
    >>> list(arc)
    [(0.0, 1.5707963267948966), (4.71238898038469, 6.283185307179585)]
    >>> 0.0 in arc
    True
    >>> math.pi in arc
    False

Top-level imports::

    >>> from geo import FloatAngle, FloatCircleSet, FloatSet, real
    >>> real is float
    True
    >>> FloatAngle(2 * math.pi).value
    0.0

Local Cone Models
-----------------

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

Status
------

The package is still under active development. The current focus is:

1. stabilize the ``float``-based core;
2. keep the public API coherent;
3. extend the geometric model only after the interval foundation is stable.
