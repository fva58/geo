geo
===

``geo`` is an early-stage geometry package built around immutable interval
sets and a small differential-geometric modeling layer.

Current model
-------------

At the current stage, real numbers are modeled by Python ``float``.

This has two direct consequences:

- interval and set operations work on the discrete lattice of representable
  floating-point values;
- when a set operation removes a boundary point but still returns closed
  intervals, the implementation uses ``math.nextafter`` to move to the nearest
  representable float.

Normalization follows the same model: two intervals are merged not only when
they overlap, but also when there is no representable ``float`` between them.
In particular, ``[a, b]`` and ``[nextafter(b, +inf), c]`` normalize to one
closed interval.

Main types
----------

Real line:

- ``FloatInterval``: one closed interval of ``float`` values;
- ``FloatSet``: normalized union of disjoint ``FloatInterval`` objects.

``FloatSet`` constructor inputs are intentionally explicit:

- a scalar means one point;
- ``FloatInterval`` means one interval;
- a single numeric pair ``(left, right)`` means one interval;
- sequences are flattened only when they contain supported elements;
- unsupported values raise ``TypeError`` instead of being recursively guessed.

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

Metric-space layer:

- ``MetricSpace`` and ``ChartedMetricSpace``: manifolds with distance;
- ``MetricGeometricObject``: geometric object in a metric space;
- ``RealLineSpace``, ``UnitCircleSpace``, ``EuclideanPlaneSpace``: standard
  spaces with ready-made objects.

For arcs crossing zero, the internal representation is split into two linear
intervals: one before zero and one after zero.

Documentation
-------------

Detailed documentation now lives in the Sphinx project under `doc/`.

Main pages:

- `doc/index.rst`
- `doc/overview.rst`
- `doc/stability.rst`
- `doc/user-guide.rst`
- `doc/intervals.rst`
- `doc/circle.rst`
- `doc/zoo.rst`
- `doc/geometry.rst`
- `doc/api.rst`

Examples
--------

Jupyter notebooks with demonstrations and visualizations live under
`examples/`.

Typical commands:

.. code-block:: sh

   pip install -e ".[docs]"
   pip install -e ".[examples]"
   make -C doc html
   make -C doc doctest

Status
------

The package is still under active development. The current focus is:

1. stabilize the ``float``-based core;
2. keep the public API coherent;
3. extend the geometric model only after the interval foundation is stable.

For the current stability boundary between the tested core and the more
experimental geometry layers, see `doc/stability.rst`.
