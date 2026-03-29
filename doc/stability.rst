Stability Guide
===============

This page describes which parts of ``geo`` should currently be treated as
stable enough for regular use and which parts should still be considered
experimental.

Stable Core
-----------

The current stable core is the float-based set layer and the small metric-space
layer built on top of it.

Preferred stable building blocks:

- ``FloatInterval`` and ``FloatSet``
- ``FloatAngle``, ``FloatCircleInterval``, and ``FloatCircleSet``
- ``FloatPoint`` and ``FloatVector``
- ``MetricSpace`` and ``ChartedMetricSpace``
- ``MetricGeometricObject``
- ``RealLineSpace``, ``UnitCircleSpace``, and ``EuclideanPlaneSpace``

What "stable" means here:

- these names reflect the intended public terminology;
- the explicit ``float`` model is documented;
- boundary behavior is covered by regression tests;
- unsupported constructor inputs are expected to fail predictably.

Experimental Layers
-------------------

The higher local-geometry layer should still be treated as experimental.

That currently includes:

- cone-based local modeling beyond the simplest standard objects;
- visibility operations;
- projection workflows;
- mesh generation as a geometric API guarantee;
- richer manifold and differential-geometric composition beyond the tested
  standard examples.

These features are implemented and tested on the current examples, but their
API and guarantees may still shift as the package clarifies its longer-term
scope.

Compatibility Names
-------------------

Some older names remain available for compatibility, but they are no longer the
preferred terminology.

Examples:

- ``RiemannianSpace`` is a compatibility alias over the newer metric-space
  contract;
- ``ChartedRiemannianSpace`` is a compatibility alias over
  ``ChartedMetricSpace``;
- ``RiemannianGeometricObject`` is a compatibility alias over
  ``MetricGeometricObject``;
- ``EuclideanRiemannianSpace`` is a compatibility alias over
  ``EuclideanMetricSpace``.

When writing new code, prefer the metric-space names.

Practical Guidance
------------------

If you want the safest current subset of the package, stay close to:

1. ``FloatInterval`` and ``FloatSet`` on the line.
2. ``FloatCircle*`` objects on the circle.
3. ``RealLineSpace``, ``UnitCircleSpace``, and ``EuclideanPlaneSpace`` with
   ``distance()`` as the ambient-space contract.
4. Standard geometric constructors and set-theoretic operations already covered
   by tests.

If you move into visibility, projections, or more elaborate local-model
composition, treat the API as useful but still evolving.
