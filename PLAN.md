# geo Execution Plan

## Product Goal

`geo` is being built as a geometry tool that can:

- define spaces such as Euclidean spaces, spheres, and tori;
- build objects inside those spaces through constructors and set-theoretic
  operations;
- answer questions such as containment, local geometry, and distance;
- transform or embed spaces and objects into 2D and 3D for visualization;
- stay usable both from explicit Python code and from interactive notebooks.

The working architecture remains:

1. `Space`
2. `Object`
3. `Transform`
4. `Visualize`

## Snapshot

Current verified snapshot:

- `python -m unittest discover -s tests` passes with 158 tests;
- `python -m py_compile geo/*.py examples/*.py` passes.

Current headline state:

- `FloatSet` semantics and validation were tightened;
- chart-aware local-model composition was improved;
- the primary ambient-space API is now metric-first;
- standard spaces include sphere and torus support;
- objects can be sampled, meshed, plotted, and exported;
- a notebook-friendly helper layer now exists for interactive workflows.

## Planning Principles

1. Keep the scalar model fixed as Python `float` for now.
2. Keep `Space` distance-first and visualization-aware.
3. Treat notebook helpers as convenience wrappers over explicit spaces.
4. Prefer one public vocabulary (`Metric*`) even when compatibility aliases
   remain available.
5. Keep status and stability documents synchronized with the code.

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
- `Space`, `Transform`, sphere/torus object families, mesh/export helpers, and
  notebook helpers were added.
- status and critique documents were refreshed to match the current tree.

## Active Risks

### Risk 1: stability boundaries are still too soft

The package now spans core algebra, metric spaces, native non-Euclidean
objects, transforms, meshing, plotting, exporting, and notebook helpers.
Those layers do not all have the same maturity.

### Risk 2: compatibility aliases duplicate the mental model

`Riemannian*` names still exist beside the newer `Metric*` vocabulary. That is
useful for migration, but it keeps two conceptual APIs alive at once.

### Risk 3: interactive convenience can leak hidden state

The notebook layer improves ergonomics, but global default-space state should
not become an implicit dependency of the core model.

### Risk 4: docs can drift behind feature work

The package now moves quickly enough that `CURRENT_STATE.md`, `PLAN.md`,
`CRITIQUE*`, and `doc/stability.rst` need deliberate upkeep.

## Roadmap Table

| Stage | Status | Purpose |
| --- | --- | --- |
| `v0.0.2` | active | Stabilize the public contract and document the supported subset |
| `v0.0.3` | next | Grow higher geometry only where guarantees can be stated clearly |
| `Later` | queued | Add stronger release discipline and a clearer pre-1.0 stability story |

## v0.0.2

Goal: consolidate the current surface rather than add unrelated new geometry.

### Deliverables

1. A clear preferred public subset across README, stability docs, and API docs.
2. Reduced dependence on legacy `Riemannian*` terminology in docs, notebooks,
   and examples.
3. A documented notebook workflow that is clearly positioned as convenience
   sugar over explicit spaces.

### Remaining Tasks

- Audit docs and notebooks for legacy `Riemannian*` terminology and replace it
  where compatibility wording is not required.
- Decide and document which modules are:
  - preferred public API;
  - implemented but evolving;
  - compatibility-only.
- Keep `CURRENT_STATE.md`, `PLAN.md`, `CRITIQUE*`, and `doc/stability.rst`
  synchronized after each documentation or feature batch.
- Add a few more end-to-end examples that mix explicit `Space` objects with
  convenience helpers without hiding the core model.

### Exit Criteria

- README, Sphinx docs, and notebooks use `Metric*` terminology by default.
- The stable-versus-evolving split is explicit and consistent in docs.
- The interactive layer is documented as thin convenience, not core semantics.
- No active doc page still describes already-removed limitations.

## v0.0.3

Goal: grow higher geometry only where guarantees can be stated clearly.

### Deliverables

1. One or two additional object-family expansions with explicit guarantees.
2. A clearer `Transform` story beyond pure visualization helpers.
3. A clearer decision on whether plotting/export helpers are core API or
   convenience API.

### Candidate Scope

- Expand object families for supported spaces where local-model and meshing
  guarantees can be tested and documented.
- Clarify how `Transform` should grow beyond visualization helpers.
- Decide which plotting/export helpers should be considered core API and which
  should remain convenience-layer utilities.
- Continue reducing reliance on compatibility aliases where migration cost is
  low.

### Proposed Concrete Targets

- Choose one concrete geometry expansion, for example:
  - more native sphere families; or
  - more torus families; or
  - explicit wrapped-object workflows across more spaces.
- Add tests that state the guarantees for those new families:
  - containment;
  - local-model behavior at representative boundary points;
  - sampling/mesh behavior;
  - visualization/export compatibility where promised.
- Write one focused documentation page or user-guide section for the chosen
  expansion.
- Define whether `Transform` remains point-map-only or grows into a larger
  object-transport story.

### Exit Criteria

- New geometry layers land with explicit guarantees and tests.
- `Transform` semantics are clearer than "point map used by examples".
- The preferred public vocabulary is consistent across code and docs.
- At least one new expansion is documented end to end, not only implemented.

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
2. Tighten the stable/experimental boundary.
3. Continue migrating docs and examples toward `Metric*` terminology.
4. Keep notebook helpers thin and explicit.
5. Expand geometry only where guarantees stay understandable.
