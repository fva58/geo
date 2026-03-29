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
- ``Space``: metric space with explicit 2D/3D visualization transforms;
- ``Transform`` and ``PointTransform``: point maps between spaces;
- ``MetricGeometricObject``: geometric object in a metric space;
- ``RealLineSpace``, ``UnitCircleSpace``, ``EuclideanPlaneSpace``,
  ``SphereSpace``, ``TorusSpace``: standard spaces with ready-made objects.

Non-Euclidean object families currently include:

- ``SphereSpace.point_object()`` and ``SphereSpace.cap()``;
- ``TorusSpace.point_object()`` and ``TorusSpace.patch()``.

Sampling and meshing currently include:

- ``SphereSpace.sample_points()``, ``SphereSpace.mesh()``,
  ``SphereSpace.cap_mesh()``;
- ``TorusSpace.sample_points()``, ``TorusSpace.mesh()``,
  ``TorusSpace.patch_mesh()``.

Native sampled objects currently include:

- sphere point objects and spherical caps with ``sample_points()`` and
  ``mesh()``;
- torus point objects and torus patches with ``sample_points()`` and
  ``mesh()``.

The common metric-object layer now also exposes object-level sampling/meshing
for a broader zoo:

- real-line points and subsets;
- circle points and arcs/subsets;
- wrapped Euclidean objects that already have geometric mesh support.

`ObjectMesh` now also has unified export/plot adapters:

- ``wireframe_data()`` for generic edge/cell export;
- ``matplotlib_data()`` for Matplotlib-oriented plain data;
- ``plotly_data()`` for Plotly trace dictionaries;
- ``threejs_data()`` for indexed Three.js geometry buffers.

On top of that, the package now provides:

- ``plot_mesh_matplotlib()`` and ``plot_mesh_plotly()`` for ready figures;
- ``obj_text()``, ``ply_text()``, ``gltf_json_data()`` on ``ObjectMesh``;
- ``write_obj()``, ``write_ply()``, and ``write_gltf_json()`` for file export.

For arcs crossing zero, the internal representation is split into two linear
intervals: one before zero and one after zero.

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

Runnable Python scripts now live there as well for end-to-end pipelines such
as:

- `space -> native object -> mesh`
- `metric object -> sampling/mesh`
- `mesh -> plot/export`

The notebook set has also been updated to the current metric-space terminology
and now includes a modern pipeline notebook covering:

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

The package is still under active development. The current focus is:

1. keep the stable subset explicit as the surface grows;
2. keep the public API coherent under the preferred ``Metric*`` vocabulary;
3. extend higher geometry only where guarantees can be stated clearly.

For the current stability boundary between the tested core and the more
experimental geometry layers, see `doc/stability.rst`.

For the current public roadmap, see `doc/roadmap.rst`.
