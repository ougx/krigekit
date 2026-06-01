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

### Persistent between-solve factorisation cache (Fortran)

The Cholesky factorisation of the kriging covariance matrix **K** is now
preserved on the `t_kriging` object after each `solve()` call (`pf_L`,
`pf_kinv_drift`, `pf_schur`).  On subsequent `solve()` calls with unchanged
observations and variogram, `kriging_setup` is skipped and the cached factors
are copied into the thread-private context — saving the $O(N^3)$
factorisation for the cost of an $O(N^2 p)$ array copy.

Cache invalidation is automatic:
- `set_obs` — coordinates may change K; sets `pf_valid = .false.`
- `set_vgm` — variogram changes K; sets `pf_valid = .false.`
- `update_obs_value` — values affect only the RHS, not K; cache preserved

Within a single `solve()` call, the existing intra-solve block-to-block cache
(same neighbour set → skip `kriging_setup`) remains highest priority.  The new
between-call cache is used as a fallback when the intra-solve cache is cold.

New C API functions:
- `krige_get_factor_info(handle, npp, p, valid)` — query dimensions and validity
- `krige_get_factor_matrices(handle, npp, p, L, kinv_drift, schur)` — copy matrices

New Python method:
- `Kriging.get_factor()` — returns a dict with keys `valid`, `npp`, `p`,
  `L`, `kinv_drift`, `schur` (all as NumPy arrays when `valid=True`)

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
