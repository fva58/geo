# Stabilize Core Invariants Before Extending Higher Geometry Layers

## Summary

The package has a promising core idea around immutable interval/set objects on
top of Python `float`, but the current implementation exposes higher-level
geometry abstractions whose guarantees are weaker than the public API suggests.

The main concern is not that the code is completely broken. Tests pass and the
package imports. The concern is that some foundational invariants are either
not enforced or are implemented in a way that only works on happy-path
examples.

## Main Problems

### 1. Chart-unsafe composition of local models

Boolean operations on `RiemannianGeometricObject` combine local cone models as
if both operands already lived in the same coordinate chart. The current code
checks only the dimension, not chart compatibility or the needed transition.

That can produce incorrect local geometry for objects that are equal as sets
but represented through different charts.

Relevant code:
- `geo/riemannian.py:197`
- `geo/riemannian.py:212`
- `geo/riemannian.py:218`

### 2. "Riemannian" metrics are not enforced to be Riemannian

`ChartedRiemannianSpace` accepts symmetric degenerate tensors. As a result, the
package can represent semimetrics while exposing a Riemannian API.

Relevant code:
- `geo/riemannian.py:41`
- `geo/riemannian.py:52`
- `geo/riemannian.py:149`

### 3. Float-set normalization is weaker than the documented model

The README says the package works on the discrete lattice of representable
floats, but adjacent intervals separated by one `nextafter` step are still
kept disjoint. That makes the normalization logic weaker than the documented
mathematical model.

Relevant code:
- `README.rst:10`
- `README.rst:14`
- `geo/floatset.py:183`
- `geo/floatset.py:381`
- `geo/floatcircle.py:261`

### 4. Input validation fails unsafely in `FloatSet`

Unsupported iterable inputs can recurse until `RecursionError` instead of
raising a controlled `TypeError`.

Relevant code:
- `geo/floatset.py:343`
- `geo/floatset.py:348`

### 5. Repository state documents are stale

`CURRENT_STATE.md` no longer matches the real state of the repository. That is
small compared to the code issues, but it makes the engineering status harder
to trust.

Relevant code:
- `CURRENT_STATE.md:22`
- `CURRENT_STATE.md:35`
- `CURRENT_STATE.md:49`

## Why This Matters

The current package is strongest at the interval/set layer. The further it
goes into manifold and Riemannian abstractions, the more important it becomes
to enforce exact invariants instead of relying on documentation and intention.

If those invariants are not enforced, the package risks becoming broad before
it becomes reliable.

## Suggested Direction

1. Fix and specify the interval/set foundation first.
2. Tighten validation and failure modes in public constructors.
3. Make chart transitions explicit in local-model composition.
4. Either enforce positive-definite metrics or soften the naming of the
   current metric layer.
5. Keep project-state documentation synchronized with the actual tree.

## Verification

Local checks at review time:

- `python -m unittest discover -s tests` passed with 112 tests
- `python -m py_compile geo/*.py` passed

So this is not a "project does not work" issue. It is a "core invariants and
scope discipline need tightening" issue.
