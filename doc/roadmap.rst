Roadmap
=======

This page summarizes the current implementation roadmap for ``geo`` in a
public, documentation-oriented form.

Current Direction
-----------------

The project is being reshaped as a framework for constructing geometric
objects and exploring them computationally without assuming the user's problem
in advance.

The intended workflow is:

1. construct an object in an explicit ambient space;
2. apply standard functions, transformations, and set-theoretic operations;
3. derive new objects when they are easier to study than the original one;
4. continue the investigation on those derived objects, including empirical
   exploration in notebooks when needed.

Visualization, meshing, and notebook ergonomics are therefore not side notes:
they are part of the intended investigative workflow, as long as they remain
honest about what is exact and what is exploratory.

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

Goal: make the object-modeling and exploration story explicit across the docs
and codebase.

Main deliverables:

- a clear description of the package as an object-construction and
  exploration framework;
- a split between stable object/modeling layers and more experimental helper
  layers;
- explicit documentation that object operations are expression-based and lazy;
- reduced ambiguity between `Metric*`, object functions, and visualization
  helpers.

Main work:

- keep the stability split explicit and current;
- migrate project documents away from older package identities that no longer
  match the goal;
- identify the minimal interfaces needed for object expressions and derived
  objects;
- document the current zoo of functions and object constructors more clearly;
- keep status and roadmap documents synchronized after feature work.

Next: v0.0.3
------------

Goal: expand the zoo of derived-object workflows and object functions without
pretending to solve the user's task automatically.

Main deliverables:

- a lazy expression layer for set-theoretic and mapped objects;
- one or two additional derived-object workflows with clear semantics;
- one or two additional standard functions on objects with clear guarantees;
- better notebook-oriented workflows for empirical investigation.

Main work:

- specify which current classes form the durable object/exploration core;
- define how lazy expression nodes answer containment, local-model, and
  derived-object queries;
- add tests for new derived-object workflows and object functions;
- document at least one iterative investigation workflow end to end.

Later
-----

Goal: prepare for a stronger pre-1.0 stability story.

Main themes:

- release automation;
- explicit stability promises for the preferred subset;
- deliberate scope control around the object/exploration core;
- support for richer scalar types beyond Python ``float`` where justified.

Practical Guidance
------------------

If you want the safest current subset of the package, stay closest to:

1. the `FloatSet` / `FloatCircleSet` computational core;
2. `MetricSpace`, `MetricGeometricObject`, and local-model abstractions;
3. explicit spaces and objects rather than notebook convenience state;
4. the documented examples that do not overclaim final architecture.

For the current stable-versus-evolving split, see :doc:`stability`.
