# geo Development Plan

## Current State

The package has a partially working `float`-based core for sets on the real
line and an unfinished module for circular geometry.

For the current stage of development, the model of real numbers is fixed as
`float`. The package should be stabilized in that form before any attempt to
generalize scalar backends.

### What already works

- `geo.floatset` contains a usable immutable representation of intervals and
  normalized unions of intervals.
- `tests/test_floatset.py` covers the current behavior of `FloatInterval` and
  `FloatSet`.
- `python -m unittest discover -s tests` passes.

### What is blocking further development

- `geo.floatcircle` does not compile and cannot be treated as a stable base.
- `geo.real` imports a name that does not exist.
- `FloatSet` input normalization does not correctly distinguish interval pairs
  from sequences of points.
- `FloatInterval.difference()` and `FloatInterval.complement()` are not
  mathematically consistent with set difference on the real line.
- Public package metadata and documentation are still minimal.

## Mathematical Decisions To Fix First

Before adding new geometry, the package needs an explicit contract.

### Decision 1: domain model

The package currently models subsets of machine `float` values.

This choice affects the correctness of `difference`, `complement`, and every
operation that relies on `math.nextafter`. The implementation and the
documentation should say this explicitly.

### Decision 2: boundary semantics

Choose one of the following:

1. Keep only closed intervals.
2. Introduce open and half-open boundaries.

Even in the `float` model, endpoint semantics must be written clearly, because
set difference and complement depend on whether neighboring representable values
are included or excluded.

### Decision 3: circle point model

Choose the canonical representation of a point on `S^1`:

1. Angle in `[0, 2pi)`.
2. Cartesian pair `(x, y)` on the unit circle.

The alternative representation can still be supported by conversion methods,
but one must be primary.

## Development Roadmap

### Phase 1: stabilize the real-line `float` core

Goal: make `geo.floatset` mathematically and programmatically consistent under
the explicit assumption that real numbers are represented by Python `float`.

Tasks:

- Fix `FloatInterval.complement()` runtime errors.
- Fix `geo.real` imports and exported aliases.
- Redesign `FloatSet` input parsing so `(a, b)` means an interval when intended.
- Revisit `difference` and `complement` semantics for the `float` model.
- Document the role of `math.nextafter` in endpoint handling.
- Add tests for:
  - complements,
  - tuples as intervals,
  - infinities,
  - point intervals,
  - exact endpoint behavior.

Done when:

- `geo.floatset` and `geo.real` import cleanly.
- Behavior matches the written specification.
- Tests cover both normal cases and boundary cases.

### Phase 2: rewrite circular geometry on top of the stabilized core

Goal: replace the current draft in `geo.floatcircle` with a coherent design.

Recommended approach:

- Represent a circle interval as an oriented arc.
- Convert wrapped arcs to one or two linear intervals on `[0, 2pi)`.
- Reuse the real-line set operations internally.
- Normalize results back into a circular representation.

Tasks:

- Rewrite `FloatAngle` as a small, well-defined primitive.
- Define `FloatCirclePoint` consistently with the chosen mathematical model.
- Reimplement `FloatCircleInterval`.
- Reimplement `FloatCircleSet`.
- Add tests for:
  - normalization modulo `2pi`,
  - full circle,
  - single-point arcs,
  - arcs crossing zero,
  - set operations on circular intervals.

Done when:

- `python -m py_compile geo/*.py` passes.
- The circular API has a documented invariant set.
- Circle tests live in `tests/` and pass under `unittest`.

### Phase 3: define and clean the public API

Goal: make the package usable as a library rather than a code sketch.

Tasks:

- Fill `geo/__init__.py` with the supported public exports.
- Add module-level documentation and examples.
- Write a real `README.rst`.
- Add package metadata in `pyproject.toml`:
  - `requires-python`,
  - classifiers,
  - keywords,
  - project URLs.
- Decide on versioning and compatibility policy.

Done when:

- A user can understand the package from the README alone.
- Import paths are intentional and documented.
- The package can be built and installed with a meaningful metadata set.

### Phase 4: raise engineering quality

Goal: make future mathematical work cheaper and safer.

Tasks:

- Move all ad hoc scripts into `tests/` or example files.
- Add linting to the regular workflow.
- Add coverage checks for the core modules.
- Consider CI for build, import, lint, and tests.

Done when:

- Every supported module is tested.
- Regressions are caught automatically.
- There is no gap between the documented API and tested API.

### Phase 5: extend toward geometry in Riemannian spaces

Goal: build higher-level geometry only after the foundation is stable.

Suggested order:

1. Euclidean points and vectors.
2. Local charts and coordinate maps.
3. Geometric subsets in local coordinates.
4. Riemannian structures and operations.

This phase should start only after phases 1 to 3 are complete.

## Immediate Priority Order

1. Fix the mathematical contract for intervals and sets in the `float` model.
2. Repair `geo.floatset` and simplify `geo.real` to match that contract.
3. Rewrite `geo.floatcircle` instead of patching the current draft.
4. Expand tests before adding new geometry.
5. Finalize packaging and public documentation.

## Short Execution Plan

### Iteration 1

- Fix imports and runtime errors.
- Simplify `geo.real` for the explicit `float` model.
- Document endpoint semantics.

### Iteration 2

- Correct tuple parsing and real-line semantics.
- Write missing tests for real-line operations.

### Iteration 3

- Replace `geo.floatcircle` with a minimal correct implementation.
- Add circle tests in `tests/`.

### Iteration 4

- Publish a coherent public API.
- Complete README and package metadata.
- Prepare the first meaningful pre-release version.
