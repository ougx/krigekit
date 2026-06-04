# Changelog

## 0.1.0 (unreleased)

Initial release.

### Features

- Ordinary and simple kriging (point and block)
- Co-kriging with Linear Model of Coregionalisation
- Universal kriging / KED (external drift)
- Sequential Gaussian Simulation (SGSIM)
- Space-time kriging — sum-metric and product-sum ST covariance models
- Spatially Varying Anisotropy (per-block variogram, `varying_vgm` mode)
- Cross-validation (leave-one-out)
- Kriging weight storage and reuse (`store_weight` / `use_old_weight`)
- OpenMP parallelism with per-`solve()` thread count control
- `set_vgm(append=False)` to replace variogram model on a reused object

### Multi-event universal kriging (MEUK)

New classes for MEUK (Tonkin et al. 2016, *Advances in Water Resources* 87:92–105):

- **`MEUKFortran`** — Fortran-backed implementation.  Wraps the existing
  co-kriging engine with `unbias=0` and an augmented drift matrix that
  reproduces MEUK's block design matrix (Eq. 11 of the paper).  One
  `solve()` call handles all *m* events simultaneously.  Includes a
  two-level result cache (Python dict + between-solve Fortran factorisation
  reuse) so that repeated predictions on the same grid require zero
  additional Fortran work.

- **`MEUK`** — Pure Python/NumPy backend.  Implements the block-structured
  solver of Eq. 13–16 directly via `scipy.linalg`.  Numerically identical to
  `MEUKFortran` (differences ≤ 10⁻¹²); no compiled library required.

Both backends:
- Accept per-event `local_drift` (event-specific coefficients) and
  `global_drift` (shared coefficient) arrays.
- Cache results keyed on `(event_id, pred_coords, local_drift, global_drift)`.
- Expose `predict(target_event, ...)` (single-event) and
  `predict_all(pred_coords, pred_local_drifts, pred_global_drifts)` (all
  events in one call).
- Invalidate the cache automatically when `set_variogram` or `add_event` is
  called; `clear_cache()` resets manually.

The `pred_global_drifts` argument of `predict_all` accepts:
- `None` — no global drift
- `ndarray (ngrid, r)` — same values broadcast to all events
- `dict {event_id: ndarray}` — per-event values (needed when the global
  covariate magnitude varies per event, e.g. pumping rates)

### Persistent between-solve factorisation cache (Fortran)

The Cholesky factorisation of the kriging covariance matrix **K** can now be
preserved across successive `solve()` calls via an opt-in flag `pf_cache=True`
passed to the constructor (or `krige_initialize`).  When enabled, the factored
matrices (`L`, `K⁻¹F`, Schur complement) are stored after the first solve and
reused on subsequent calls with unchanged observations and variogram — saving
the $O(N^3)$ factorisation for the cost of an $O(N^2 p)$ array copy.

**Architecture** — all persistent-factor interaction is outside the parallel
block loop:

- *Before the loop* — each thread pre-warms its private `ctx%cache` from
  `self%pf` via `copy_all`.  Matching blocks then hit the existing intra-solve
  cache in `assemble_linear_system` and never enter a CRITICAL section.
- *Inside the loop* — no pf logic; the hot path is completely free of locks.
- *After the loop* — the first thread with a valid `ctx%cache` writes it to
  `self%pf` inside a single `!$OMP CRITICAL(pf_save)`.

The persistent factor (`self%pf`) and the per-thread intra-solve cache
(`ctx%cache`) are both instances of the same `t_factor_cache` derived type,
sharing `alloc`, `matches`, `save_key`, `copy_to`, and `copy_all` methods.

Cache invalidation is automatic:
- `set_obs` — coordinates may change K; sets `pf%valid = .false.`
- `set_vgm` — variogram changes K; sets `pf%valid = .false.`
- `update_obs_value` — values affect only the RHS, not K; cache preserved

The cache is **disabled by default** (`pf_cache=False`).  Enable it only when
you plan to call `solve()` multiple times on the same observation grid.

New C API functions:
- `krige_get_factor_info(handle, npp, p, valid)` — query dimensions and validity
- `krige_get_factor_matrices(handle, npp, p, L, kinv_drift, schur)` — copy matrices

New Python method:
- `Kriging.get_factor()` — returns a dict with keys `valid`, `npp`, `p`,
  `L`, `kinv_drift`, `schur` (all as NumPy arrays when `valid=True`)

### Structured result array (`get_result_array`)

`Kriging.get_result_array()` returns a NumPy structured array (one row per
block) combining block coordinates, estimates/simulations, and variances in a
single object.

| Scenario | Fields |
|---|---|
| Kriging, `nvar=1` | `x, y [,z], estimate, variance` |
| Kriging, `nvar>1` | `x, y, est_v1, …, est_v{nvar}, var_v1, …` |
| SGSIM, `nvar=1` | `x, y, sim_1, …, sim_{nsim}, variance` |
| SGSIM, `nvar>1` | `x, y, v1_s1, …, v{nvar}_s{nsim}, var_v1, …` |

```python
k.solve()
ra = k.get_result_array()
ra.dtype.names          # ('x', 'y', 'estimate', 'variance')
ra['estimate']          # 1-D array, shape (nblocks,)

import pandas as pd
df = pd.DataFrame(ra)   # convert to DataFrame directly
```

New C API function:
- `krige_get_block_coord(handle, ndim_c, nblocks, out)` — copies
  `block%coord(1:ndim, 1:nblocks)` into a caller-allocated buffer

### Internal / correctness changes

- `set_obs_drift` must be called (or re-called) after each `set_obs` when
  `ndrift > 0`; `set_obs` zeros the external drift rows on each call.
  Using `update_obs_value` is the correct API when only values change.
- `set_search` must be called after `set_obs`; it caps `obs%nmax` at `n`
  (the actual observation count).  Skipping `set_search` after a repeated
  `set_obs` would leave `obs%nmax = HUGE(int)`, causing integer overflow in
  `prepare()`.

### Co-kriging improvements

- **`std_ck` flag** — selects the co-kriging unbiasedness formulation when
  `nvar > 1` and `unbias = 1`:
  - `std_ck=True` (default): standard co-kriging with separate per-variable
    constraints (Σwᵢ = 1 for the target variable, Σwⱼ = 0 for secondary
    variables).  Results match gstat/ISATIS.
  - `std_ck=False`: Isaaks & Srivastava formulation — single combined
    constraint (Σw = 1 across all variables) plus a local-mean correction
    applied post-solve.  Matches legacy GSLIB behaviour.

- **Unified drift array** — `obs%drift` and `block%drift` are now 3-D:
  - `obs%drift  [ndrift+naug, 1,    nobs]`  — F-matrix column (same for all targets)
  - `block%drift [ndrift+naug, nvar, nblock]` — f₀ RHS (varies per target variable)

  External drift rows (1:`ndrift`) and unbiasedness indicator/RHS rows
  (`ndrift+1:ndrift+naug`) are stored in the same array.  This eliminates all
  branching from the matrix-assembly loop and resolves the long-standing
  "TODO: need separate drift for each variable" in `assemble_rhs`.

- **Auto-allocation of drift arrays** — `set_obs()` and `set_grid()` now
  allocate and fill the unbiasedness indicator / RHS rows automatically.
  `set_obs_drift()` and `set_grid_drift()` write into the pre-allocated arrays
  instead of allocating.

- **`set_grid_drift(drift, ivar=None)`** — `ivar` selects which target
  variable's RHS receives this drift (`None` or `ivar < 0` broadcasts to all).
  Replaces the old broadcast-only signature.

### Internal / correctness fixes

- Fixed a bug in `initialize_kriging_ctx` where `pmax` (sizing the Cholesky
  factor-cache arrays `factor_kinv_drift` and `factor_schur`) was not scaled
  by `nvar` for standard co-kriging, causing out-of-bounds writes at runtime.
- `naug` is now defined as the number of unbiasedness constraint rows only
  (not including `ndrift`).  Total augmented rows = `ndrift + naug`.  All
  internal matsize/pmax computations updated accordingly.
