# Critique Follow-Up Checklist

## v0.0.2

### Core Algebra

- [ ] Define the exact interval adjacency semantics for the representable-float
      model.
- [ ] Update `FloatInterval.union()` to match that definition.
- [ ] Update `FloatSet.merge()` and normalization so it uses the same rule.
- [ ] Add tests for intervals separated by exactly one `math.nextafter` step.
- [ ] Add tests for circle-set operations that depend on the same adjacency
      semantics.

### Validation

- [ ] Replace recursive "accept almost any iterable" parsing in `FloatSet`
      with explicit supported input forms.
- [ ] Make unsupported public inputs fail with `TypeError` or `ValueError`,
      never `RecursionError`.
- [ ] Add negative tests for strings, nested invalid iterables, and malformed
      interval-like inputs.

### Documentation

- [ ] Update `CURRENT_STATE.md` so it reflects the current repository state.
- [ ] Keep test counts and import/build status synchronized with reality.
- [ ] Document the exact mathematical compromises of the float-based model.

### Regression Guard

- [ ] Add tests reproducing the invalid-input recursion scenario.
- [ ] Run `python -m unittest discover -s tests` after each fix batch.

## v0.0.3

### Local Geometry

- [ ] Define what it means for two local cone models to be compatible.
- [ ] Refuse set-theoretic composition of local models when chart compatibility
      cannot be established.
- [ ] Introduce chart-transition-based transport before combining local models
      from different charts.
- [ ] Add tests where equal objects are represented through different charts.
- [ ] Add tests that verify boundary local models after union/intersection.

### Metric Layer

- [ ] Decide whether `ChartedRiemannianSpace` should enforce positive-definite
      metrics or intentionally support weaker structures.
- [ ] If the answer is "Riemannian", validate positive-definiteness.
- [ ] If the answer is "weaker metric object", rename classes or documentation
      so the API does not overclaim.
- [ ] Add tests for rejecting degenerate and indefinite metric tensors.

### Regression Guard

- [ ] Add tests reproducing the chart-composition bug scenario.
- [ ] Add tests reproducing the degenerate-metric acceptance scenario.
- [ ] Run `python -m unittest discover -s tests` after each fix batch.

## Later

### Scope Control

- [ ] Identify which modules are considered stable today.
- [ ] Mark experimental layers explicitly in docs if they are not yet stable.
- [ ] Avoid extending visibility/projection/mesh features until the interval
      and chart-composition foundations are harder.

### Documentation

- [ ] Document which guarantees are semantic conventions versus enforced
      runtime invariants.

### Product Direction

- [ ] Re-evaluate whether the package should stay centered on interval/set
      geometry with a small local-geometry layer, or continue toward a broader
      manifold/Riemannian toolkit.
