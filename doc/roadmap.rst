Roadmap
=======

This page summarizes the current implementation roadmap for ``geo`` in a
public, documentation-oriented form.

Current Direction
-----------------

The project is being shaped as a geometry tool that can:

- define spaces;
- build objects inside those spaces;
- answer metric and containment questions;
- transform or embed results into 2D and 3D for visualization;
- remain practical both in explicit Python code and in notebooks.

For the more detailed execution version of this plan, see ``PLAN.md`` in the
repository root.

Completed Recently
------------------

The following items are already done and are no longer active roadmap work:

- `FloatSet` semantics were aligned with the explicit representable-float
  model.
- public `FloatSet` validation was tightened.
- chart-aware local-model composition was improved.
- the primary public ambient-space terminology moved to `Metric*`.
- `Space`, `Transform`, sphere/torus families, and mesh/export helpers were
  introduced.
- notebook-oriented interactive helpers were added.

Near Term: v0.0.2
-----------------

Goal: consolidate the current surface rather than add unrelated new geometry.

Main deliverables:

- a clear preferred public subset across the docs;
- reduced use of legacy `Riemannian*` terminology in docs and examples;
- a notebook workflow that is clearly documented as convenience, not core
  semantics.

Main work:

- keep the stability split explicit and current;
- continue migrating docs and notebooks to `Metric*` terminology;
- keep status and roadmap documents synchronized after feature work;
- add a few more examples that mix explicit `Space` objects with convenience
  helpers.

Next: v0.0.3
------------

Goal: grow higher geometry only where guarantees can be stated clearly.

Main deliverables:

- one or two additional object-family expansions with explicit guarantees;
- a clearer story for `Transform` beyond pure visualization helpers;
- a clearer split between core API and convenience API for plotting/export
  helpers.

Main work:

- expand supported object families only with tests and docs;
- define the intended scope of `Transform`;
- continue reducing reliance on compatibility aliases where practical.

Later
-----

Goal: prepare for a stronger pre-1.0 stability story.

Main themes:

- release automation;
- explicit stability promises for the preferred subset;
- deliberate scope control.

Practical Guidance
------------------

If you want the safest current subset of the package, stay closest to:

1. the `FloatSet` / `FloatCircleSet` core;
2. `MetricSpace`, `MetricGeometricObject`, and the basic standard spaces;
3. `Space` embeddings when deterministic visualization coordinates are needed;
4. the documented examples and user-guide workflows.

For the current stable-versus-evolving split, see :doc:`stability`.
