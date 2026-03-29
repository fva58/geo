# Current State Snapshot

Date: 2026-03-29

## Repository State

- The repository contains a working `float`-based interval/set core.
- `geo.floatcircle` is present and importable.
- `geo/__init__.py` exposes a non-trivial public API.
- The repository also contains Euclidean, manifold, geometric, and
  Riemannian-oriented layers.
- The working tree is not clean: there are existing documentation-related
  changes in `doc/` and generated images.

## Verification Snapshot

### Tests

Command:

```sh
python -m unittest discover -s tests
```

Result:

- Passed
- 112 tests

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
  normalization semantics still need a sharper written contract.
- `geo.floatcircle` is importable and covered by tests.
- `geo.euclidean`, `geo.manifold`, and parts of `geo.geometric` are implemented
  and exercised by the current test suite.
- The package metadata and README are now populated.

### Risky or still underspecified

- `FloatSet` input parsing is still too permissive and can fail unsafely on
  unsupported iterables.
- The interval/set core still needs a precise contract for adjacency in the
  representable-float model.
- Boolean operations on higher-level geometric objects are not yet strict
  enough about chart compatibility.
- The current "Riemannian" layer does not yet enforce invariants strong enough
  to justify the terminology unconditionally.

## Active Architectural Decision

For the current development stage, the package assumes:

- the scalar model is Python `float`;
- interval and set operations are defined on representable floating-point
  values;
- endpoint adjustments may use `math.nextafter`;
- broadening the scalar model is lower priority than stabilizing the current
  `float` contract;
- higher-level geometry should be constrained by the guarantees of the core,
  not the other way around.

## Immediate Focus

1. Stabilize `FloatInterval` and `FloatSet` semantics in the explicit `float`
   model.
2. Tighten public validation and predictable failure modes.
3. Align `FloatCircle*` behavior with the same interval/set contract.
4. Fix or restrict chart-based composition of geometric objects.
5. Clarify or harden the metric layer before extending the Riemannian API.
