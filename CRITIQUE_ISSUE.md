# Keep the Package Surface Aligned With Its Stability Guarantees

## Summary

The package has moved well beyond the state described in the original critique.
Core `FloatSet` semantics, validation, metric-space naming, and chart-aware
local-model composition have all been improved.

The main concern is now different: the project has accumulated a broad public
surface quickly, and its stable-versus-experimental boundary needs to stay
explicit as that surface grows.

## Main Problems

### 1. Scope is growing faster than the stability story

The package now exposes:

- float-based interval and circle algebra;
- metric spaces and ambient-space objects;
- `Space` embeddings into 2D and 3D;
- native sphere and torus object families;
- meshing, export adapters, and plotting helpers;
- notebook-oriented convenience helpers.

That is already a serious public API surface for a pre-1.0 geometry package.
The code is much healthier than before, but docs and release notes need to
keep telling users which parts are the safest to build on.

Relevant files:
- `README.rst`
- `doc/stability.rst`
- `geo/__init__.py`

### 2. Legacy compatibility names still duplicate the mental model

The package now prefers `MetricSpace` and `MetricGeometricObject`, but
`Riemannian*` aliases remain available. That is a reasonable migration tactic,
yet it also leaves two vocabularies in circulation.

Relevant files:
- `geo/riemannian.py`
- `geo/__init__.py`
- `doc/api.rst`

### 3. Notebook convenience uses global mutable default-space state

The new interactive layer is useful, but it introduces process-global ambient
context. That is fine as a convenience feature, as long as it remains clearly
documented as optional sugar rather than core semantics.

Relevant files:
- `geo/interactive.py`
- `doc/user-guide.rst`

### 4. Status documents can drift again unless maintained deliberately

The previous critique became stale because the code moved faster than the
status notes. The same failure mode will repeat unless `CURRENT_STATE.md`,
`PLAN.md`, and `doc/stability.rst` are treated as maintained artifacts.

Relevant files:
- `CURRENT_STATE.md`
- `PLAN.md`
- `doc/stability.rst`

## Why This Matters

The project now has enough working capability that its next risk is not
"nothing works". The risk is that users will not know which guarantees are
hard, which are compatibility shims, and which layers are still evolving.

That matters more as soon as the package is used from notebooks, examples, or
third-party code, where convenience APIs and experimental layers can easily
look more stable than they really are.

## Suggested Direction

1. Keep the stable subset explicit and current.
2. Continue moving docs and examples toward metric-space terminology.
3. Keep the interactive default-space layer thin and obviously optional.
4. Add stronger release/documentation synchronization so status drift is caught
   early.
5. Expand the higher geometry surface only when the package can explain the
   resulting guarantees clearly.

## Verification

Local checks at update time:

- `python -m unittest discover -s tests` passed with 158 tests
- `python -m py_compile geo/*.py examples/*.py` passed

This is now a scope-and-clarity issue, not the earlier core-invariant failure
issue.
