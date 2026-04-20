# Current State Snapshot

Date: 2026-04-20

## Repository State

- The repository contains a working `float`-based interval/set core with
  explicit adjacency semantics based on representable `float` values.
- `geo.floatcircle` is present and aligned with the same core interval/set
  contract.
- The preferred public ambient-space terminology is now
  `MetricSpace` / `ChartedMetricSpace` / `MetricGeometricObject`.
- Older `Riemannian*` names remain available as compatibility aliases.
- Set-theoretic operations on metric objects now build explicit lazy
  expression-tree nodes.
- Projection, visibility, and smooth-image workflows now also build lazy
  mapped-object nodes instead of eager ad-hoc wrapper objects.
- The repository also contains a visualization-aware `Space` layer, a
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
  - lazy expression trees for derived objects;
  - projection, visibility, and smooth-image derived objects;
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
- 159 tests

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
- `LazyMetricObject`
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
- the exact long-term public shape of lazy expression-tree APIs;
- visibility, projection, and smooth-image workflows as durable public API;
- native non-Euclidean object families beyond the first sphere/torus set;
- file-export and plotting convenience layers as long-term API surface;
- notebook-oriented default-space helpers in `geo.interactive`.

## Active Architectural Direction

For the current stage, the project is best understood as:

1. construct geometric objects in explicit ambient spaces;
2. apply standard functions, transformations, and set-theoretic operations;
3. derive new objects when they are easier to study than the original one;
4. continue the investigation on those derived objects, including empirical
   exploration in notebooks when needed.

This implies the following design choices:

- the scalar model is Python `float`;
- interval and set operations are defined on representable floating-point
  values;
- endpoint adjustments may use `math.nextafter`;
- the preferred semantic center is the geometric object rather than a
  predefined catalog of user problems;
- object operations should remain symbolic as long as possible through lazy
  expression-tree nodes;
- spaces may expose non-isometric 2D/3D visualization embeddings because
  visualization is part of the intended exploratory workflow;
- notebook helpers are convenience wrappers over explicit spaces, not a second
  semantic core.

## Immediate Focus

1. Keep project-state documents synchronized with the current object/exploration
   story.
2. Clarify the durable API around lazy expression-tree nodes and derived
   objects.
3. Keep `geo.interactive` thin and clearly optional.
4. Expand the zoo of object constructors and functions only where guarantees
   can be described clearly.
5. Continue reducing documentation drift after each feature batch.
