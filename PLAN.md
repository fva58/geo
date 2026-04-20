# geo Execution Plan

## Product Goal

`geo` is being shaped as a framework for constructing geometric objects and
exploring them computationally without assuming the user's problem in advance.

The target workflow is:

1. construct an object in an explicit ambient space;
2. apply standard functions, transformations, and set-theoretic operations;
3. derive new objects when they are easier to study than the original one;
4. continue the investigation on those derived objects, including empirical
   exploration in notebooks when needed.

Visualization, transforms, and notebooks are part of that exploratory
workflow, not a separate product direction.

## Snapshot

Current verified snapshot:

- `python -m unittest discover -s tests` passes with 159 tests;
- `python -m py_compile geo/*.py examples/*.py` passes.

Current headline state:

- `FloatSet` semantics and validation were tightened;
- chart-aware local-model composition was improved;
- the primary ambient-space API is now metric-first;
- lazy expression-tree nodes now represent set-theoretic and mapped derived
  objects in the metric layer;
- standard spaces include sphere and torus support;
- objects can be sampled, meshed, plotted, and exported;
- a notebook-friendly helper layer now exists for interactive workflows.

## Planning Principles

1. Keep the scalar model fixed as Python `float` for now.
2. Keep `Space` explicit and visualization-aware where that helps
   investigation.
3. Keep geometric operations symbolic and lazy until a query forces local
   evaluation.
4. Treat notebook helpers as convenience wrappers over explicit spaces.
5. Prefer one public vocabulary (`Metric*`) even when compatibility aliases
   remain available.
6. Keep status and stability documents synchronized with the code.

## Completed Recently

These items are no longer active plan work:

- `FloatInterval` / `FloatSet` adjacency semantics were aligned with the
  documented representable-float model.
- `FloatSet` input validation was tightened and recursion-based failures were
  removed.
- chart-transition-aware local-model transport was added before combining
  metric objects across charts.
- the primary public ambient-space contract moved from `Riemannian*` naming to
  `Metric*` naming.
- lazy expression-tree nodes were introduced for set-theoretic and mapped
  derived objects.
- status, roadmap, and user-facing documentation were refreshed to match the
  current package goal.

## Active Risks

### Risk 1: documentation can still drift away from the current package goal

The package now spans core algebra, metric spaces, native non-Euclidean
objects, transforms, meshing, plotting, exporting, and notebook helpers.
Those layers do not all have the same maturity, and documentation can drift if
project-state files are not kept current.

### Risk 2: compatibility aliases duplicate the mental model

`Riemannian*` names still exist beside the newer `Metric*` vocabulary. That is
useful for migration, but it keeps two conceptual APIs alive at once.

### Risk 3: interactive convenience can leak hidden state

The notebook layer improves ergonomics, but global default-space state should
not become an implicit dependency of the core model.

### Risk 4: public API shape for lazy nodes is still young

Lazy expression-tree nodes now exist and are tested, but their long-term
public contract still needs deliberate scope control.

## Roadmap Table

| Stage | Status | Purpose |
| --- | --- | --- |
| `v0.0.2` | active | Stabilize the public contract and document the supported object/exploration subset |
| `v0.0.3` | next | Expand derived-object workflows and object functions where guarantees can be stated clearly |
| `Later` | queued | Add stronger release discipline and a clearer pre-1.0 stability story |

## v0.0.2

Goal: restate the project around object construction and exploration and make
the current prototype boundaries explicit.

### Deliverables

1. A clear conceptual statement across README, status docs, and Sphinx docs.
2. Reduced dependence on outdated framing where interval sets look like the
   entire package purpose.
3. A documented split between core geometric semantics and convenience layers.
4. A documented lazy-expression story for operations on geometric objects.

### Remaining Tasks

- Audit docs for outdated framing and replace it with the object/exploration
  statement.
- Decide and document which modules are:
  - part of the computational core;
  - prototype support infrastructure;
  - compatibility-only.
- Keep `CURRENT_STATE.md`, `PLAN.md`, `CRITIQUE*`, and `doc/stability.rst`
  synchronized after each documentation or feature batch.
- Identify the minimal abstractions still missing for derived objects and lazy
  geometric operations.

### Exit Criteria

- README and Sphinx docs describe the package through object construction and
  exploration first.
- The stable-versus-evolving split is explicit and consistent in docs.
- The interactive layer is documented as thin convenience, not core semantics.
- No active doc page still presents prototype layers as the package's final
  architecture.

## v0.0.3

Goal: expand the zoo of derived-object workflows and object functions without
pretending to solve the user's task automatically.

### Deliverables

1. A lazy expression layer for set-theoretic and mapped objects.
2. One or two additional derived-object workflows with clear semantics.
3. One or two additional standard functions on objects with clear guarantees.
4. Better notebook-oriented workflows for empirical investigation.

### Candidate Scope

- Introduce expression nodes for additional derived-object workflows.
- Define how lazy expression nodes answer containment, local-model, and
  derived-object queries.
- Add one or two standard object functions with explicit guarantees.
- Demonstrate an iterative investigation workflow on a simple but nontrivial
  object.
- Continue reducing reliance on compatibility aliases where migration cost is
  low.

### Proposed Concrete Targets

- Choose one concrete exploratory workflow, for example
  `build object -> derive projection or visible part -> continue analysis on
  the result`.
- Add tests that state the guarantees for that workflow:
  - lazy expression preservation before query-time evaluation;
  - correctness of derived-object membership;
  - correctness of local models where promised;
  - compatibility with sampling/mesh workflows where promised.
- Write one focused documentation page or user-guide section for the chosen
  workflow.
- Define how `Transform` and visualization helpers relate to, but do not
  replace, the core object/exploration story.

### Exit Criteria

- The next derived-object workflow lands with explicit guarantees and tests.
- `Transform` semantics are clearer than "point map used by examples".
- The preferred public vocabulary is consistent across code and docs.
- At least one iterative investigation workflow is documented end to end.

## Later

Goal: prepare the package for a stronger pre-1.0 stability story.

### Deliverables

1. Stronger release checks around docs, tests, and public API consistency.
2. A documented stability promise for the preferred subset.
3. A clearer scope policy for future growth.

### Tasks

- Add stronger release checks around docs, tests, and public API consistency.
- Define what counts as stable enough to mention in a pre-1.0 promise.
- Decide how far the package should grow as a broad geometry tool versus a
  tighter core with higher optional layers.

### Exit Criteria

- Stability claims are backed by repeatable release checks.
- Public scope growth is intentional rather than opportunistic.
- Breaking-change expectations are clearer for the preferred subset.

## Immediate Priority Order

1. Keep stability and status documents synchronized.
2. Tighten the split between conceptual core and supporting prototype layers.
3. Define the minimal lazy-expression and derived-object interfaces.
4. Keep notebook helpers thin and explicit.
5. Expand geometry only where guarantees stay understandable.
