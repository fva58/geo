# Current State Snapshot

Date: 2026-03-29

## Repository State

- The repository contains a working `float`-based interval/set core.
- `geo.floatcircle` is present and importable.
- `geo/__init__.py` exposes a non-trivial public API.
- The repository also contains Euclidean, manifold, geometric, and
  metric-space-oriented layers.
- A new `Space` protocol now models spaces with both `distance()` and explicit
  2D/3D visualization transforms.
- A minimal `Transform` layer now models reusable point maps between spaces.
- Early standard point spaces now include `SphereSpace` and `TorusSpace`.
- The preferred public terminology is now metric-space-based, while older
  "Riemannian" names remain as compatibility aliases.
- The working tree may be dirty during active development; this snapshot does
  not assume a clean repository.

## Verification Snapshot

### Tests

Command:

```sh
python -m unittest discover -s tests
```

Result:

- Passed
- 133 tests

### Bytecode compilation

Command:

```sh
python -m py_compile geo/*.py
```

Result:

- Passed

## Package Status

### Stable or partially stable

- `geo.floatset` is usable and covered by tests, but its adjacency and
  normalization semantics now have an explicit written contract and regression
  coverage.
- `geo.floatcircle` is importable and covered by tests.
- `geo.euclidean`, `geo.manifold`, and parts of `geo.geometric` are implemented
  and exercised by the current test suite.
- The metric-space layer now uses `distance()` as its primary public contract.
- The ambient-space layer now also exposes a visualization-aware `Space`
  protocol.
- The package now has explicit visualization transforms into Euclidean 2D and
  3D.
- The package metadata and README are now populated.

### Risky or still underspecified

- Some compatibility names still use older "Riemannian" terminology even
  though the actual contract is metric-space-based.
- The package still needs a clearer stability boundary between core modules and
  higher experimental geometry layers.
- `SphereSpace` and `TorusSpace` currently model point geometry and
  visualization embeddings, but they do not yet offer the same ready-made
  object families available in the Euclidean plane.

## Active Architectural Decision

For the current development stage, the package assumes:

- the scalar model is Python `float`;
- interval and set operations are defined on representable floating-point
  values;
- endpoint adjustments may use `math.nextafter`;
- broadening the scalar model is lower priority than stabilizing the current
  `float` contract;
- the ambient-space abstraction is distance-first rather than tensor-first;
- spaces may expose non-isometric 2D/3D visualization embeddings in addition
  to intrinsic distance;
- higher-level geometry should be constrained by the guarantees of the core,
  not the other way around.

## Immediate Focus

1. Stabilize `FloatInterval` and `FloatSet` semantics in the explicit `float`
   model.
2. Keep public validation and failure modes explicit and predictable.
3. Align `FloatCircle*` behavior with the same interval/set contract.
4. Continue tightening chart-based composition of geometric objects.
5. Grow the `Space` and `Transform` APIs around explicit standard spaces and
   visualization transforms.
