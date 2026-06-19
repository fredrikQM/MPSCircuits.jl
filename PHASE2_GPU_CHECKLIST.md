# MPSCircuits Phase 2 GPU Checklist

This file tracks what is complete and defines the next implementation target:
unify runtime configuration into a single object to eliminate kwarg bloat (backend policy, optimization order, and hyperparameters), with Phase C diagnostics remaining optional follow-on work.

## Progress Tracker

- [x] Step 1: Make CUDA optional at package level
- [x] Step 2: Centralize backend data movement
- [x] Step 3: Apply conversion at coarse boundaries
- [x] Step 4: Enforce device consistency in gate application
- [x] Step 5: GPU-enable heavy tensor-network paths
- [x] Step 6: Keep tiny dense kernels on CPU first
- [x] Step 7: Add backend behavior and parity tests
- [x] Step 8: Add benchmark harness and chi sweeps
- [x] Step 9: Implement Phase B progress/event hooks
- [ ] Step 10: Introduce unified runtime configuration object (next)
- [ ] Step 11: Document user-facing workflow
- [ ] Step 12: Optional Phase C extended diagnostics

## Current Baseline (Completed)

- CUDA extension loading and backend transfer policy are functional.
- GPU FP32/FP64 execution is working with precision propagation fixes.
- CPU-only precision propagation regression was fixed in core backend code.
- Benchmark harness supports protocol comparisons, phase timing, and chi sweeps.
- Test suite passes on CPU-only setup.

## Next: Unified Runtime Configuration Object

### Objective

Replace scattered kwargs with one cohesive runtime configuration object that captures:

- backend execution policy (cpu/gpu/auto, fallback, transfer policy, precision)
- optimization policy (protocol choice, optimization order, iteration schedule)
- numerical hyperparameters (tolerance, max_bond_dim, working_cutoff, etc.)
- observability/progress policy (progress tracker, emission cadence, optional diagnostics)

This should preserve readability in high-level algorithm functions and avoid further keyword argument growth.

### Design Constraints

- Keep algorithmic methods line-by-line readable at a mathematical level.
- Maintain full backward compatibility initially via thin keyword wrappers.
- Keep defaults explicit and centralized.
- Ensure config is serializable/printable for reproducibility in benchmarks and notebooks.

### Proposed API Shape

- Add a config type in core source, for example:
  - `RuntimeConfig`
- Suggested nested fields:
  - `backend::BackendRuntimeConfig`
  - `optimization::OptimizationRuntimeConfig`
  - `numerics::NumericsRuntimeConfig`
  - `observability::ObservabilityRuntimeConfig`
- Entry points accept:
  - `config::RuntimeConfig=RuntimeConfig()`

### Migration Plan

- Phase 10A: define config structs + defaults + validation.
- Phase 10B: update `compile_mps_circuit` and `replace_gates!` internals to consume config fields.
- Phase 10C: keep existing kwargs as compatibility shim that constructs `RuntimeConfig`.
- Phase 10D: update benchmarks/notebooks/tests to pass config explicitly where useful.
- Phase 10E: deprecate redundant kwargs after migration is stable.

### Acceptance Criteria

- High-level compiler/procrustes functions no longer carry long kwarg lists.
- Existing callsites keep working through compatibility wrappers.
- Benchmark and notebook workflows can emit a compact config summary for run reproducibility.
- No measurable performance regression from config indirection in hot paths.

## Completed: Phase B Progress/Event Hooks

### Objective

Add low-overhead, disable-able event hooks so optimization progress can be rendered live in notebooks (layer progress, optimization counters, bond-dimension diagnostics, and timing).

### Design Constraints

- Hooks must be optional and off by default.
- Hot-loop overhead must be negligible when hooks are disabled.
- Event payloads should use lightweight immutable NamedTuple fields.
- Avoid string formatting in hot paths; renderer handles formatting externally.

### API Plan

- Top-level compile entry points accept a single tracker object:
  - `progress::AbstractProgressTracker=NoProgressTracker()`
- Callback and throttling behavior are encapsulated in `CallbackProgressTracker`.
- High-level methods emit lightweight progress events through tracker helper calls.

### Event Schema (Phase B)

Emit the following events with consistent keys:

- `:run_start`
  - protocol, backend, precision, n_layers_max, n_iterations_per_layer
- `:layer_start`
  - layer, total_layers, current_circuit_len, mps_work_maxlinkdim
- `:layer_generated`
  - layer, entangling_layer_len, mps_work_maxlinkdim, flag_disentangled
- `:layer_opt_start`
  - layer, layer_opt_total_steps, run_opt_completed_steps, run_opt_est_total_steps
- `:opt_step_done`
  - layer, iter_in_layer, iter_total_for_layer, gate_index, gate_count,
    layer_step_done, layer_step_total,
    run_step_done, run_step_est_total,
    ket_maxlinkdim, bra_maxlinkdim, env_norm
- `:layer_done`
  - layer, circuit_len, layer_elapsed_s
- `:run_done`
  - total_layers_done, total_opt_steps_done, total_elapsed_s, final_circuit_len

Notes:

- `run_step_est_total` may be an estimate (dynamic total) unless a prepass is later added.
- `env_norm` is optional if expensive; compute only when enabled in config.

### Code Insertion Points

- `src/compilers.jl`
  - emit run and layer boundary events
  - track global counters and elapsed times
- `src/procrustes.jl`
  - emit per-optimization-step events from `replace_gates!`
  - capture `ket`/`bra` maxlinkdim diagnostics

### Performance Safeguards

- Guard all hook code with a single boolean check per event point.
- Throttle emissions via `emit_every` (for example emit every N gate updates).
- Keep optional expensive diagnostics behind flags:
  - `include_env_norm=false`
  - `include_tensor_types=false`

### Acceptance Criteria (Phase B)

- Notebook callback receives live events during `IterativeDecomposeOptimizeAll`.
- Can show per-layer and whole-run optimization progress without modifying algorithm behavior.
- Disabling callback restores baseline runtime within noise for representative benchmark cases.

## Optional Next: Phase C Extended Diagnostics

Phase C is intentionally optional and should be started after Phase B is stable.

### Candidate Additions

- CUDA memory snapshots (`memory_status`) at configurable cadence.
- Transfer counters by boundary type (CPU->GPU, GPU->CPU).
- Per-phase synchronization timing and rolling ETA.
- Warning events for fallback/coercion and mixed-device corrections.
- Notebook tqdm-style renderer helper built on `AbstractProgressTracker` events.

### Exit Criteria (Phase C)

- Demo notebook can render a richer diagnostics dashboard.
- Added diagnostics remain opt-in and low overhead when disabled.

## Documentation Follow-up (Step 11)

- Document progress hook API and notebook rendering examples.
- Document recommended configs for debug vs production.
- Add troubleshooting notes for missing events and disabled hooks.

## Risks to Watch

- Hidden cost from over-frequent event emission.
- Expensive diagnostics accidentally enabled in production runs.
- Inconsistent event schema across compiler/procrustes call sites.
- Test fixtures that accidentally hit unrelated edge cases.

---

## Notes for Future You

- Prefer one shared algorithm path and backend-specialized helpers.
- Keep CUDA optional for package users.
- Optimize only after correctness and backend parity are stable.
