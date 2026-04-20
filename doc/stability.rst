Stability Guide
===============

This page describes which parts of ``geo`` are currently the safest to build
on and which parts should still be treated as evolving.

Preferred Stable Subset
-----------------------

The current preferred subset is:

- ``FloatInterval`` and ``FloatSet``
- ``FloatAngle``, ``FloatCircleInterval``, and ``FloatCircleSet``
- ``FloatPoint`` and ``FloatVector``
- ``MetricSpace`` and ``ChartedMetricSpace``
- ``MetricGeometricObject``
- ``Space`` for visualization-aware ambient spaces
- ``Transform`` and ``PointTransform`` for reusable point maps
- ``RealLineSpace``, ``UnitCircleSpace``, and ``EuclideanPlaneSpace``
- ``ObjectMesh`` export adapters and basic plotting helpers

What "preferred stable" means here:

- these names match the intended public terminology;
- the explicit ``float`` model is documented;
- boundary and validation behavior are covered by tests;
- these layers are the safest basis for object modeling and exploratory
  workflows today.

Implemented But Still Evolving
------------------------------

The package also contains working layers that should still be treated with more
care:

- cone-based local modeling beyond the tested standard cases;
- visibility operations;
- projection workflows;
- native non-Euclidean object families beyond the first sphere/torus set;
- mesh generation as a long-term API guarantee rather than just a useful
  workflow;
- file-export and plotting convenience helpers as public surface;
- notebook-oriented helpers in ``geo.interactive``.

That includes current features such as:

- ``SphereSpace`` and ``TorusSpace``;
- ``SphereSpace.point_object()`` / ``SphereSpace.cap()``;
- ``TorusSpace.point_object()`` / ``TorusSpace.patch()``;
- object-level sampling and meshing across the broader metric-object zoo;
- interactive current-space helpers like ``use_space()`` and ``plot()``.

These layers are real and tested, but their long-term public guarantees are
not yet as stable as the smaller core.

Compatibility Names
-------------------

Some older names remain available for compatibility, but they are no longer
the preferred terminology.

Examples:

- ``RiemannianSpace`` over the newer metric-space contract;
- ``ChartedRiemannianSpace`` over ``ChartedMetricSpace``;
- ``RiemannianGeometricObject`` over ``MetricGeometricObject``;
- ``EuclideanRiemannianSpace`` over ``EuclideanMetricSpace``.

When writing new code, prefer the metric-space names.

Notebook Helpers
----------------

The interactive layer in ``geo.interactive`` is intentionally a convenience
layer.

Use it when:

- you are exploring from a notebook;
- you want a default ambient space;
- you want short helpers such as ``point()``, ``disk()``, ``cap()``, or
  ``plot()``.

Do not treat it as the semantic center of the package. The explicit model is
still:

1. create a space;
2. create an object in that space;
3. apply functions or derive a new object from it;
4. sample, mesh, transform, plot, or export when that helps the
   investigation.

Practical Guidance
------------------

If you want the safest current subset of the package, stay close to:

1. ``FloatInterval`` and ``FloatSet`` on the line.
2. ``FloatCircle*`` objects on the circle.
3. ``RealLineSpace``, ``UnitCircleSpace``, and ``EuclideanPlaneSpace`` with
   ``distance()`` as the ambient-space contract.
4. ``Space`` embeddings when you need deterministic 2D/3D coordinates.
5. Standard object constructors, set-theoretic operations, projections,
   visibility, and smooth-image workflows already covered by tests.

If you move deeper into visibility, projections, native non-Euclidean object
families, or notebook convenience state, treat the API as useful but still
evolving.
