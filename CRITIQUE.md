# Project Review

## Status

This review was rewritten on 2026-03-29 after the recent `MetricSpace`,
`Space`, meshing/export, and notebook-helper work.

The earlier critique is no longer accurate as written. The following issues
from the original review are now resolved:

- chart-unsafe local-model composition in metric-object Boolean operations;
- overclaiming "Riemannian" tensor semantics in the primary public API;
- `FloatSet`/`FloatInterval` adjacency semantics around `math.nextafter`;
- unsafe recursive `FloatSet` input parsing that could raise
  `RecursionError`;
- stale repository-status documents from the very early project state.

## Findings

1. Medium: the project now has a broad and useful public surface, but the
   stable/experimental boundary still needs tighter discipline. The package
   exposes interval algebra, metric spaces, native sphere/torus objects,
   transforms, meshing, export adapters, plotting helpers, and notebook
   shortcuts. That is already a large product surface for a pre-1.0 geometry
   package, and not every layer has the same maturity guarantees.
   Files:
   - `doc/stability.rst`
   - `README.rst`
   - `geo/__init__.py`

2. Medium: compatibility naming still duplicates the conceptual model.
   Keeping `Riemannian*` aliases is pragmatic, but it also leaves two
   parallel vocabularies in the public API. That increases onboarding cost and
   makes it easier for examples and downstream code to drift back to the older
   terminology.
   Files:
   - `geo/__init__.py`
   - `geo/riemannian.py`
   - `doc/api.rst`

3. Medium: the new notebook convenience layer uses process-global mutable
   state for the current default space. This is a reasonable tradeoff for
   notebooks, but it should remain clearly documented as convenience state,
   not as part of the semantic core. Otherwise hidden ambient context can make
   examples, tests, and larger applications harder to reason about.
   Files:
   - `geo/interactive.py`
   - `doc/user-guide.rst`

4. Low: project-state documents still need ongoing synchronization after large
   feature pushes. The original critique documents became stale because the
   package moved quickly; the same can happen again unless these files are
   treated as maintained engineering artifacts rather than one-time notes.
   Files:
   - `CURRENT_STATE.md`
   - `PLAN.md`
   - `doc/stability.rst`

## Architecture

- The core idea is stronger now than in the original review: the package has a
  coherent `float`-based set foundation, a distance-first ambient-space layer,
  and a growing visualization/export story.

- The product direction is now much clearer: define spaces, construct objects
  inside them, answer metric questions, and transform or embed them into 2D/3D
  for visualization.

- The main remaining risk is no longer "broken invariants in obvious core
  operations". It is scope control. The package now has enough working layers
  that each new addition should justify where it sits on the
  stable-versus-experimental boundary.

- The notebook helper layer is useful and aligned with the product goal, but
  it should stay thin. Interactive convenience should wrap the core model, not
  become a second implicit programming model.

## Roadmap

1. Keep the stable subset explicit and current in `README`, `doc/stability`,
   and release notes.
2. Gradually reduce reliance on legacy `Riemannian*` compatibility names in
   docs, examples, and tests.
3. Keep notebook helpers as a convenience layer with minimal hidden state and
   clear escape hatches back to explicit `Space` objects.
4. Add stronger release automation around docs, tests, and public API checks so
   status drift is caught early.
5. Expand object families and transforms only where the package can explain the
   resulting guarantees clearly.

## Verification

- `python -m unittest discover -s tests` -> passed, 158 tests
- `python -m py_compile geo/*.py examples/*.py` -> passed

The current package is in a materially better state than the original review
described. The remaining critique is mostly about scope discipline, API
clarity, and maintaining that improved state as the surface grows.
