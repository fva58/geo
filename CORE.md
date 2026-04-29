# Core Architecture

`geo` is an early-stage geometry package. The project does not optimize for
backward compatibility. The primary criterion is a mathematically coherent
core.

## Kernel

The kernel consists of six kinds of entities:

- `Point`: a representation of a point in a space.
- `Manifold`: a set of points with local Euclidean charts.
- `Chart`: a local coordinate map between manifold points and `R^n`.
- `Metric`: an intrinsic distance on a manifold.
- `Subset`: a geometric subset of a manifold with local cone data.
- `Embedding`: an extrinsic map into Euclidean coordinates for visualization or
  computation.

These are logically distinct. A class may implement more than one role, but
the API should not confuse them.

## Standard Point Models

The standard spaces should be understood through their point models:

- `R^n`: a point is `FloatPoint(x_1, ..., x_n)`.
- `S^1`: a point is `FloatCirclePoint(theta)`.
- `T^n = (S^1)^n`: a point is `TorusPoint(theta_1, ..., theta_n)`.
- `S^n`: a point is `SpherePoint(v)` where `v` is any nonzero vector in
  `R^(n+1)`, normalized to the fixed radius.

This is the unifying design rule:

- Euclidean space is given by coordinates.
- Torus is given by one circle point per axis.
- Sphere is given by a nonzero ambient vector modulo radial scaling.

## Layering

The codebase should be read in layers:

1. `euclidean.py`, `floatcircle.py`, `manifold.py`: local mathematical
   primitives.
2. `riemannian.py`, `space.py`: intrinsic spaces and standard models.
3. `geometric.py`: subsets and local cone models.
4. visualization and IO helpers: meshes, plotting, exports.

Interactive convenience is not part of the core model.

## What Is Not Core

The following are explicitly non-core:

- notebook-style global default space state;
- compatibility aliases that duplicate one concept under several names;
- mixing intrinsic metric semantics with visualization choices in the same
  public abstraction unless the distinction is explicit.

## Refactoring Direction

The current refactoring direction is:

- keep one primary vocabulary, not parallel `Metric*` and `Riemannian*`
  vocabularies;
- expose a small top-level API centered on spaces, points, charts, subsets,
  and embeddings;
- treat interactive helpers as optional adapters, not as defining semantics;
- prefer mathematically canonical point models over ad hoc constructor
  convenience.
