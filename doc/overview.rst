Overview
========

Current model
-------------

At the current stage, real numbers are modeled by Python ``float``.

This has two direct consequences:

- interval and set operations work on the discrete lattice of representable
  floating-point values;
- when an operation removes a boundary point but still returns closed
  intervals, the implementation uses ``math.nextafter`` to move to the nearest
  representable float.

Normalization uses the same explicit float-lattice semantics. Two intervals
are merged when they overlap or when they are adjacent with no representable
``float`` between them.

Main layers
-----------

Real line
~~~~~~~~~

- ``FloatInterval``: one closed interval of ``float`` values;
- ``FloatSet``: normalized union of disjoint ``FloatInterval`` objects.

The ``FloatSet`` constructor accepts only explicit supported forms:

- a scalar for a point;
- a ``FloatInterval`` for one interval;
- a single numeric pair ``(left, right)`` for one interval;
- sequences built from supported values.

Unsupported inputs raise ``TypeError``.

Circle
~~~~~~

- ``FloatAngle``: angle normalized to ``[0, 2π)``;
- ``FloatCirclePoint``: point on the unit circle represented by an angle;
- ``FloatCircleInterval``: connected arc on the circle;
- ``FloatCircleSet``: set of circle points built on top of ``FloatSet``.

Euclidean and local differential-geometric layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``FloatPoint`` and ``FloatVector``: Euclidean coordinates in ``R^n``;
- ``EuclideanNeighborhood``: rectangular coordinate neighborhood;
- ``Map``, ``InvertibleMap``, ``Diffeomorphism``, ``Chart``: structural
  protocols;
- ``ManifoldChart``, ``ChartTransition``, ``Atlas``: local coordinates on a
  manifold;
- ``EuclideanCone``, ``RadialCone``, ``SphericalCone``: cone models in
  coordinates;
- ``ChartedGeometricObject``: geometric object described by local cone models.

Metric-space layer
~~~~~~~~~~~~~~~~~~

- ``MetricSpace`` and ``ChartedMetricSpace``: manifolds with distance;
- ``Space``: metric space with explicit 2D/3D visualization transforms;
- ``Transform`` and ``PointTransform``: point maps between spaces;
- ``MetricGeometricObject``: geometric object in a metric space;
- ``RealLineSpace``, ``UnitCircleSpace``, ``EuclideanPlaneSpace``,
  ``SphereSpace``, ``TorusSpace``: standard spaces with ready-made objects.

The non-Euclidean spaces currently start with small native object families:

- ``SphereSpace.point_object()`` and ``SphereSpace.cap()``;
- ``TorusSpace.point_object()`` and ``TorusSpace.patch()``.

They also provide early visualization helpers:

- ``SphereSpace.sample_points()``, ``SphereSpace.mesh()``,
  ``SphereSpace.cap_mesh()``;
- ``TorusSpace.sample_points()``, ``TorusSpace.mesh()``,
  ``TorusSpace.patch_mesh()``.

Status
------

The package is still under active development. The current focus is:

1. stabilize the ``float``-based core;
2. keep the public API coherent;
3. extend the geometric model only after the interval foundation is stable.

For a practical split between the stable core and the still-experimental
higher geometry layers, see :doc:`stability`.
