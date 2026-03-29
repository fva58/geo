# geo Development Plan

## Current State

The package is no longer in the state described by the older planning notes.
The repository currently contains:

- a working `float`-based interval and set core;
- working circle geometry modules;
- Euclidean, manifold, geometric, and metric-space layers;
- a non-trivial public API in `geo/__init__.py`;
- passing tests under `unittest`.

At review time, the following checks passed:

- `python -m unittest discover -s tests`
- `python -m py_compile geo/*.py`

The main problem is no longer "the package does not import". The main problem
is that some higher-level abstractions still overclaim relative to the
invariants actually enforced by the code.

## Planning Principles

1. Keep the scalar model fixed as Python `float` for now.
2. Stabilize the interval/set foundation before extending higher geometry.
3. Do not broaden the metric layer beyond its actual guarantees.
4. Keep documentation synchronized with the actual repository state.

## Main Risks

### Risk 1: foundational set semantics are underspecified

The package claims to work on the discrete lattice of representable `float`
values, but adjacency and normalization semantics still need a sharper
contract.

### Risk 2: public validation is too permissive

Some constructors accept overly broad iterable inputs. That creates ambiguous
parsing rules and unsafe failure modes.

### Risk 3: local geometric composition is not chart-safe

Set-theoretic operations on geometric objects currently assume compatible local
coordinates without enforcing the required chart transition logic.

### Risk 4: terminology can still overclaim

Some compatibility names still carry stronger historical terminology than the
current distance-based contract.

## Release Roadmap

## v0.0.2

Goal: harden the `float` core and clean up the public contract of the package
foundation.

### Scope

- interval/set algebra;
- public input validation;
- circle behavior that depends on interval normalization;
- state and contract documentation.

### Tasks

- Define the exact adjacency rule for intervals in the representable-float
  model.
- Make `FloatInterval.union()` match that rule.
- Make `FloatSet.merge()` and normalization use the same rule.
- Add tests for intervals separated by exactly one `math.nextafter` step.
- Add circle-set tests that rely on the same boundary semantics.
- Replace recursive "accept almost any iterable" parsing in `FloatSet` with
  explicit supported forms.
- Ensure unsupported inputs fail with `TypeError` or `ValueError`, never with
  recursion-based crashes.
- Add negative tests for strings, malformed nested iterables, and ambiguous
  interval-like values.
- Update `CURRENT_STATE.md` to reflect the actual repository state.
- Document the float-based model and its endpoint compromises precisely.

### Done When

- interval/set behavior matches a written contract;
- invalid public inputs fail predictably;
- circle operations remain consistent with the same core semantics;
- project status documents no longer contradict the code.

## v0.0.3

Goal: make higher geometry stricter before adding more features.

### Scope

- local chart composition;
- set-theoretic operations on geometric objects;
- metric-space API discipline and naming cleanup.

### Tasks

- Define what it means for two local cone models to be compatible.
- Introduce chart-transition-based transport before combining local models from
  different charts.
- Add tests where equal objects are represented through different charts.
- Add tests for boundary local models after union and intersection.
- Consolidate `MetricSpace` and `ChartedMetricSpace` as the preferred public
  API.
- Keep older "Riemannian" names in compatibility mode only where needed.
- Move examples, docs, and tests to the metric-space terminology.

### Done When

- geometric-object composition is chart-aware;
- metric semantics are aligned with the names used in the API;
- regression tests cover the current architectural failure modes.

## Later

Goal: expand only after the package contract is stable.

### Scope

- scope control;
- product positioning;
- engineering automation.

### Tasks

- Identify which modules are stable and which are experimental.
- Mark experimental layers clearly in documentation.
- Avoid extending visibility, projection, and mesh features until the interval
  and chart-composition foundations are harder.
- Add stronger engineering automation around linting, tests, and release
  verification.
- Re-evaluate whether the package should remain focused on interval/set
  geometry with a thin local-geometry layer or continue toward a broader
  manifold/metric-geometry toolkit.

### Done When

- the package has an explicit stability story;
- new features are added only on top of tested invariants;
- scope growth follows a clear product direction instead of opportunistic
  expansion.

## Immediate Priority Order

1. Stabilize the interval/set contract in the explicit `float` model.
2. Tighten validation and eliminate unsafe parsing behavior.
3. Align circle behavior with the same core semantics.
4. Fix or constrain chart-based composition of geometric objects.
5. Bring the metric-space layer and its naming into agreement.
6. Keep planning and status documents synchronized with reality.

## Short Execution Plan

### Iteration 1

- Finalize the written interval adjacency contract.
- Fix `FloatInterval`/`FloatSet` normalization behavior.
- Add boundary and regression tests for one-step `nextafter` gaps.

### Iteration 2

- Tighten `FloatSet` input parsing and validation.
- Add negative tests for unsupported public inputs.
- Refresh `CURRENT_STATE.md` and related status notes.

### Iteration 3

- Define chart compatibility rules for local models.
- Fix or restrict geometric-object boolean composition.
- Add regression tests for mixed-chart scenarios.

### Iteration 4

- Consolidate `distance()` as the primary ambient-space contract.
- Keep older terminology only as compatibility aliases.
- Move public examples and docs to metric-space naming.

### Iteration 5

- Reassess which modules should be documented as stable.
- Prepare the next pre-release only after the above invariants are covered by
  tests.
