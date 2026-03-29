# Project Review

## Findings

1. High: boolean operations on `RiemannianGeometricObject` combine local cone
   models as if both objects were already expressed in the same chart. The
   implementation only checks chart dimension, then evaluates both cones on
   the same coordinate point and returns the result in the left chart. This
   can produce incorrect local geometry for equal set-theoretic objects
   described in different charts.
   Files:
   - `geo/riemannian.py:197`
   - `geo/riemannian.py:212`
   - `geo/riemannian.py:218`

2. High: `ChartedRiemannianSpace` does not enforce positive-definite metrics.
   `_coerce_metric_tensor` checks only shape and symmetry, and `norm()` accepts
   degenerate tensors as long as the quadratic form is not negative. That
   means the package currently accepts structures that are not Riemannian
   metrics under the advertised terminology.
   Files:
   - `geo/riemannian.py:41`
   - `geo/riemannian.py:52`
   - `geo/riemannian.py:149`

3. Medium: the `FloatInterval`/`FloatSet` normalization logic does not fully
   match the documented "discrete lattice of representable floats" model.
   Adjacent intervals separated by exactly one `nextafter` step are kept
   disjoint even when there is no representable float between them. That makes
   the foundational normalization weaker than the README claims.
   Files:
   - `README.rst:10`
   - `README.rst:14`
   - `geo/floatset.py:183`
   - `geo/floatset.py:381`
   - `geo/floatcircle.py:261`

4. Medium: invalid input handling in `FloatSet` fails unsafely. Recursive
   translation of arbitrary iterables can end in `RecursionError` for values
   like `FloatSet("ab")` instead of raising a normal `TypeError`.
   Files:
   - `geo/floatset.py:343`
   - `geo/floatset.py:348`

5. Low: repository status documents are stale enough to misrepresent the
   actual state of the codebase. `CURRENT_STATE.md` says `floatcircle` is not
   importable, `README.rst` is empty, `geo/__init__.py` is empty, and there
   are 25 tests; none of that matches the current tree.
   Files:
   - `CURRENT_STATE.md:22`
   - `CURRENT_STATE.md:35`
   - `CURRENT_STATE.md:49`
   - `README.rst:1`
   - `geo/__init__.py:1`

## Bugs

- `RiemannianGeometricObject.union/intersection/difference/symmetric_difference`
  rely on `_combine_local_models`, but that helper ignores chart transitions.
  The result is only valid if both operands already use compatible local
  coordinates, which is not enforced.

- `ChartedRiemannianSpace.metric_tensor()` accepts symmetric degenerate
  matrices, so the class can represent pseudo- or semi-metrics while exposing
  a Riemannian API.

- `FloatSet._translate()` recursively descends into unsupported iterables and
  can blow the stack instead of producing a controlled validation error.

## Architecture

- The core idea is solid: immutable interval and set objects over Python
  `float` can be a useful and coherent package foundation.

- The current scope is too wide for the maturity of the core. The project
  already exposes intervals, circle sets, local charts, cone models,
  visibility, projections, mesh generation, and a Riemannian layer.

- The strongest part of the package is still the interval/set foundation. The
  higher-level geometry layers look experimental and are not yet protected by
  invariants strong enough to justify the full mathematical terminology used
  in the API.

- The main positioning risk is the "Riemannian" label. Right now the package
  does not yet enforce positive-definite metrics or correct chart-aware local
  composition, so the name promises more than the implementation guarantees.

## Roadmap

1. Stabilize `FloatInterval` and `FloatSet` so their algebra exactly matches
   the representable-float model described in the documentation.

2. Tighten public input validation and make failure modes predictable
   (`TypeError`/`ValueError` instead of recursion-based crashes).

3. Introduce explicit compatibility rules for local models and chart
   transitions before composing geometric objects across charts.

4. Either harden the metric layer to real Riemannian invariants
   (positive-definiteness, clearer tangent semantics) or temporarily weaken
   the naming until those guarantees exist.

5. Keep project-state documents synchronized with the actual codebase and test
   suite, otherwise they stop being useful as engineering artifacts.

## Verification

- `python -m unittest discover -s tests` -> passed, 112 tests
- `python -m py_compile geo/*.py` -> passed

These checks confirm that the current happy path works, but they do not cover
the invariant-level issues listed above.
