Overview
========

Core idea
---------

``geo`` is a package for modeling geometric objects and exploring them
computationally in spaces of various dimensions.

The package is centered on geometric objects rather than on a predefined
catalog of user problems. It provides ways to construct objects, combine them
into more complex ones, and apply a zoo of functions and transformations to
them.

A computation in ``geo`` may return a number, a structure, an approximation,
or another geometric object. This is essential: users should be able not only
to ask for direct results, but also to derive more informative or more
tractable objects through intersections, differences, images, projections,
visibility operations, and related constructions already supported by the
package.

When a standard function does not directly solve the user's task on the whole
object, the package should still help move the investigation forward by
producing derived objects that are easier to study. This supports an iterative
workflow in which the user refines the object under study, the region of
interest, and the line of investigation itself, including empirical
exploration in environments such as ``Jupyter Notebook``.

Computational model
---------------------------

Real numbers are modeled by Python ``float``.

This has two direct consequences:

- interval and set operations work on the discrete lattice of representable
  floating-point values;
- when an operation removes a boundary point but still returns closed
  intervals, the implementation uses ``math.nextafter`` to move to the nearest
  representable float.

Normalization uses the same explicit float-lattice semantics. Two intervals
are merged when they overlap or when they are adjacent with no representable
``float`` between them.

Layers
--------------

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

Metric-space and object layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``MetricSpace`` and ``ChartedMetricSpace``: manifolds with distance;
- ``Space``: metric space with explicit 2D/3D visualization transforms;
- ``Transform`` and ``PointTransform``: point maps between spaces;
- ``MetricGeometricObject``: geometric object in a metric space;
- ``RealLineSpace``, ``UnitCircleSpace``, ``EuclideanPlaneSpace``,
  ``SphereSpace``, ``TorusSpace``: standard spaces with ready-made objects.

The non-Euclidean spaces start with small native object families:

- ``SphereSpace.point_object()`` and ``SphereSpace.cap()``;
- ``TorusSpace.point_object()`` and ``TorusSpace.patch()``.

They also provide early visualization helpers:

- ``SphereSpace.sample_points()``, ``SphereSpace.mesh()``,
  ``SphereSpace.cap_mesh()``;
- ``TorusSpace.sample_points()``, ``TorusSpace.mesh()``,
  ``TorusSpace.patch_mesh()``.

The native non-Euclidean objects carry the same idea one level lower:

- sphere point objects and caps expose ``sample_points()`` and ``mesh()``;
- torus point objects and patches expose ``sample_points()`` and ``mesh()``.

The common metric wrapper extends object-level sampling and meshing to a
broader zoo as well:

- real-line points and subsets;
- circle points and arcs/subsets;
- wrapped Euclidean objects whose charted source already supports meshing.

On top of that, ``ObjectMesh`` provides plain export adapters for generic
wireframes, Matplotlib-style data, Plotly traces, and Three.js indexed
geometry.

The adapter layer also includes ready plotting helpers for Matplotlib and
Plotly, plus file exporters for OBJ, PLY, and glTF-friendly JSON.

Status
------

The package is under active development. The focus is:

1. keep the object-modeling and exploration story explicit in docs and code;
2. keep the stable computational subset explicit as the surface grows;
3. extend the zoo of object constructors and functions only where guarantees
   can be stated clearly.

For a practical split between the stable core and the still-experimental
higher geometry layers, see :doc:`stability`.

For the roadmap, see :doc:`roadmap`.
