# Current State Snapshot

Date: 2026-03-29

## Repository State

- The repository contains a working `float`-based interval/set core with
  explicit adjacency semantics based on representable `float` values.
- `geo.floatcircle` is present and aligned with the same core interval/set
  contract.
- The preferred public ambient-space terminology is now
  `MetricSpace` / `ChartedMetricSpace` / `MetricGeometricObject`.
- Older `Riemannian*` names remain available as compatibility aliases.
- The repository now also contains a visualization-aware `Space` layer, a
  `Transform` layer, mesh/export/plot helpers, runnable examples, and a small
  notebook convenience layer in `geo.interactive`.
- Standard spaces now include:
  - `RealLineSpace`
  - `UnitCircleSpace`
  - `EuclideanPlaneSpace`
  - `SphereSpace`
  - `TorusSpace`
- Standard object workflows now include:
  - set operations inside explicit ambient spaces;
  - native sphere caps and torus patches;
  - object-level sampling and meshing;
  - plotting/export adapters for `ObjectMesh`;
  - notebook-friendly current-space helpers.

## Verification Snapshot

### Tests

Command:

```sh
python -m unittest discover -s tests
```

Result:

- Passed
- 158 tests

### Bytecode compilation

Command:

```sh
python -m py_compile geo/*.py examples/*.py
```

Result:

- Passed

## Package Status

### Preferred stable or semi-stable building blocks

- `geo.floatset`
- `geo.floatcircle`
- `geo.euclidean`
- `geo.manifold`
- `MetricSpace` and `ChartedMetricSpace`
- `MetricGeometricObject`
- `Space`
- `Transform` and `PointTransform`
- `RealLineSpace`
- `UnitCircleSpace`
- `EuclideanPlaneSpace`
- `ObjectMesh` export adapters and basic plotting helpers

### Implemented but still evolving

- richer local-cone composition beyond the tested standard cases;
- visibility and projection workflows;
- native non-Euclidean object families beyond the first sphere/torus set;
- file-export and plotting convenience layers as long-term API surface;
- notebook-oriented default-space helpers in `geo.interactive`.

## Active Architectural Direction

For the current stage, the project is best understood as:

1. define spaces;
2. build objects inside those spaces;
3. answer metric and containment questions;
4. transform or embed the results into 2D and 3D for visualization.

This implies the following design choices:

- the scalar model is Python `float`;
- interval and set operations are defined on representable floating-point
  values;
- endpoint adjustments may use `math.nextafter`;
- the ambient-space abstraction is distance-first rather than tensor-first;
- spaces may expose non-isometric 2D/3D visualization embeddings in addition
  to intrinsic distance;
- notebook helpers are convenience wrappers over explicit spaces, not a second
  semantic core.

## Immediate Focus

1. Keep the stability boundary explicit as the public surface grows.
2. Reduce reliance on legacy `Riemannian*` compatibility names in docs and
   examples.
3. Keep `geo.interactive` thin and clearly optional.
4. Expand higher geometry only where guarantees can be described clearly.
5. Keep project-state documents synchronized with the code after each feature
   batch.
