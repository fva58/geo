# Critique Follow-Up Checklist

## Already Resolved

- [x] Align `FloatInterval`/`FloatSet` adjacency semantics with the documented
      representable-float model.
- [x] Add regression tests for `math.nextafter`-adjacent intervals and related
      circle behavior.
- [x] Replace unsafe recursive `FloatSet` parsing with explicit public input
      validation.
- [x] Make unsupported `FloatSet` inputs fail with controlled `TypeError` or
      `ValueError`.
- [x] Introduce chart-transition-aware transport before combining local models
      across charts.
- [x] Reframe the primary ambient-space API around `MetricSpace` instead of
      overclaiming strict Riemannian tensor semantics.
- [x] Refresh stale repository-status documents from the early project state.

## Near Term

### Stability Discipline

- [ ] Update `doc/stability.rst` so it matches the current feature set,
      especially sphere/torus object families and notebook helpers.
- [ ] Keep `CURRENT_STATE.md`, `PLAN.md`, and critique documents synchronized
      after each feature batch.
- [ ] Decide which modules are "preferred public API" versus "implemented but
      still evolving".

### Naming Cleanup

- [ ] Reduce the use of legacy `Riemannian*` compatibility names in docs,
      examples, and tests.
- [ ] Decide whether some compatibility aliases should eventually move to a
      deprecation path.

### Interactive Layer

- [ ] Keep `geo.interactive` explicitly documented as a notebook convenience
      layer, not as the core semantic model.
- [ ] Add a few more notebook-oriented examples that mix `use_space()`,
      `plot()`, and explicit `Space` objects without hiding too much state.

## Medium Term

### Product Surface

- [ ] Expand object families only where the package can explain the local-model
      and meshing guarantees clearly.
- [ ] Clarify how transforms should grow beyond visualization helpers into
      richer space-to-space workflows.
- [ ] Decide how far file export and plotting helpers should live inside core
      versus convenience layers.

### Engineering Automation

- [ ] Add stronger release checks for docs and public API synchronization.
- [ ] Add a lightweight policy for updating test counts, examples, and status
      docs when public features land.

## Later

### Versioning and Stability

- [ ] Formalize a stability promise for pre-1.0 releases.
- [ ] Define what should count as a breaking change across the stable subset.
- [ ] Reassess whether the package should keep growing as a broad geometry tool
      or preserve a tighter core with optional higher layers.
