geo
===

``geo`` is a package for constructing geometric objects in spaces of various
dimensions and exploring them computationally.

It does not assume the user's problem in advance. Instead, it provides a
language of objects together with functions, transformations, and derived
constructions such as intersections, projections, images, and other derived
objects supported by the current API.

When direct computation is not enough, the package is meant to support
iterative investigation: derive simpler objects, localize the study, and
continue the exploration, including empirically in ``Jupyter Notebook``.

Computational model
---------------------------

Real numbers are modeled by Python ``float``.

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

Layers
--------------

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

Metric-space and neighborhood layer:

- ``MetricSpace`` and ``ChartedMetricSpace``: manifolds with local distance;
- ``Space``: metric space with explicit 2D/3D visualization transforms;
- ``Transform`` and ``PointTransform``: point maps between spaces;
- ``MetricGeometricObject``: geometric object in a metric space;
- ``RealLineSpace``, ``UnitCircleSpace``, ``EuclideanPlaneSpace``,
  ``SphereSpace``, ``TorusSpace``: standard spaces with ready-made objects.

Non-Euclidean object families include:

- ``SphereSpace.point_object()`` and ``SphereSpace.cap()``;
- ``TorusSpace.point_object()`` and ``TorusSpace.patch()``.

Sampling and meshing include:

- ``SphereSpace.sample_points()``, ``SphereSpace.mesh()``,
  ``SphereSpace.cap_mesh()``;
- ``TorusSpace.sample_points()``, ``TorusSpace.mesh()``,
  ``TorusSpace.patch_mesh()``.

Native sampled objects include:

- sphere point objects and spherical caps with ``sample_points()`` and
  ``mesh()``;
- torus point objects and torus patches with ``sample_points()`` and
  ``mesh()``.

The common metric-object layer also exposes object-level sampling/meshing
for a broader zoo:

- real-line points and subsets;
- circle points and arcs/subsets;
- wrapped Euclidean objects that already have geometric mesh support.

`ObjectMesh` has unified export/plot adapters:

- ``wireframe_data()`` for generic edge/cell export;
- ``matplotlib_data()`` for Matplotlib-oriented plain data;
- ``plotly_data()`` for Plotly trace dictionaries;
- ``threejs_data()`` for indexed Three.js geometry buffers.

The package also provides:

- ``plot_mesh_matplotlib()`` and ``plot_mesh_plotly()`` for ready figures;
- ``obj_text()``, ``ply_text()``, ``gltf_json_data()`` on ``ObjectMesh``;
- ``write_obj()``, ``write_ply()``, and ``write_gltf_json()`` for file export.

For arcs crossing zero, the internal representation is split into two linear
intervals: one before zero and one after zero.

Architectural reading
---------------------

If you read the repository as both a programmer and a mathematician, the
preferred model is:

1. construct a geometric object in an explicit ambient space;
2. apply standard functions or transformations to it;
3. derive new objects when they are easier to study than the original one;
4. continue the investigation on those derived objects, including empirical
   work in notebooks when needed.

In that reading, unions, intersections, differences, images, projections, and
similar constructions should preferably remain symbolic lazy expressions until
a specific query asks for containment, a local model, sampling, or a finite
approximation.

Documentation
-------------

Detailed documentation now lives in the Sphinx project under `doc/`.

Main pages:

- `doc/index.rst`
- `doc/overview.rst`
- `doc/stability.rst`
- `doc/roadmap.rst`
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

Runnable Python scripts live there as well for end-to-end pipelines such
as:

- `space -> native object -> mesh`
- `metric object -> sampling/mesh`
- `mesh -> plot/export`

The notebook set follows the metric-space terminology and includes a modern
pipeline notebook covering:

- `space -> native object -> mesh`
- `mesh -> plotting/export data`

Typical commands:

.. code-block:: sh

   pip install -e ".[docs]"
   pip install -e ".[examples]"
   python examples/05_space_object_mesh_pipeline.py
   python examples/07_plot_and_export_pipeline.py
   make -C doc html
   make -C doc doctest

Status
------

The package is under active development. The focus is:

1. keep the object-modeling and exploration story explicit in docs and code;
2. keep the stable computational subset explicit as the surface grows;
3. extend the zoo of object constructors and functions only where guarantees
   can be stated clearly.

For the stability boundary between the tested core and the more
experimental geometry layers, see `doc/stability.rst`.

For the public roadmap, see `doc/roadmap.rst`.
