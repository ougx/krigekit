# Changelog

## 0.1.0 (unreleased)

Initial release.

### Breaking changes

#### ST search time coordinate — linear scaling replaces variogram transform

`set_search` (Fortran/C API) and `SpaceTimeKriging.set_search` (Python) now use
a **linear** time-to-search-coordinate mapping:

```
t_kd = t * time_at
```

Previously the time axis was mapped through a saturating variogram function
(`signed_time_coord = sign(f_time_vgm_st(vtype, nugget, sill, at, t), t)`).
That transform saturated for absolute time values `|t| >> time_at`, collapsing
all observations to the same KD-tree coordinate and causing infinite recursion
(stack overflow) for structured datasets (e.g. fixed monitoring stations with
repeated observation times).

The new linear mapping is:
- **Monotone and unbounded** — no saturation artefacts.
- **Model-consistent** — for the sum-metric model `h_ST = sqrt(h_S^2 + (at·Δt)^2)`,
  the L2 distance in the `(x, y, z, t·at)` search space equals `h_ST` exactly.
- **`maxdist` now operates in km-equivalent units** (same as `h_ST`), not in
  variogram-value space.

**Removed parameters**: `time_transform / time_vtype`, `time_nugget`, `time_sill`
from `set_search` / `krige_st_set_search`.  Only `time_at` (the temporal scale,
same value as in `set_st_model`) remains.  Pass `time_at=at` to keep search and
variogram scales consistent.

**Implementation note** — gfortran does not correctly set the `present()` flag
for optional arguments passed through a CLASS polymorphic dispatch (vtable call).
The CAPI workaround pre-writes `obs%time_at` before calling `set_search`;
`set_search` then reads that pre-stored value as its effective default, so `time_at`
is used for both the KD-tree coordinate build and the subsequent distance computations.

#### Duplicate observation coordinate check

`set_obs` now validates observation coordinates before storing them.  If any two
observations for the same variable share identical coordinate tuples (all
coordinate components equal), it reports a clear error before `set_search` can
build a KD-tree on invalid input:

```
ERROR: Duplicate coordinate found! Station <i> and Station <j> share identical coordinates.
Common cause: multiple observations at the same location and time.
Remove or aggregate duplicate observations before calling set_obs.
```

The degenerate-split guard previously patched into the tree builder
(`if (m >= u .or. m < l) m = (l+u-1)/2`) has been removed.  It was a
band-aid for a problem that is now prevented before it reaches the tree.

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

The Cholesky factorisation of the kriging covariance matrix **K** can now be
preserved across successive `solve()` calls via an opt-in flag `pf_cache=True`
passed to the constructor (or `krige_initialize`).  When enabled, the factored
matrices (`L`, `K⁻¹F`, Schur complement) are stored after the first solve and
reused on subsequent calls with unchanged observations and variogram — saving
the $O(N^3)$ factorisation for the cost of an $O(N^2 p)$ array copy.

**Architecture** — persistent-factor interaction is limited to read-only
pre-warming plus one after-loop save:

- *Before the loop* — each thread pre-warms its private `ctx%cache` from
  `self%pf` via `copy_all`.  Matching blocks then hit the existing intra-solve
  cache in `assemble_linear_system` and never enter a CRITICAL section.
- *Inside the loop* — no persistent-cache write occurs on the hot path.  Fresh
  factorisations update only the prepared factors and mark the current
  thread-local `ctx%matA`/`ctx%rhsB` as matching those factors; hcache hits
  remain factor-only.
- *After the loop* — the first thread whose `ctx%cache` still has a valid
  assembled system copies the factors and `ctx%matA`/`ctx%rhsB` to `self%pf`
  inside a single `!$OMP CRITICAL(pf_save)`.

The persistent factor (`self%pf`) and the per-thread intra-solve cache
(`ctx%cache`) are both instances of the same `t_factor_cache` derived type,
sharing `alloc`, `matches`, `save_key`, `copy_to`, and `copy_all` methods.

An additional per-thread multi-slot cache (`ctx%hcache`) retains recently
prepared factorisations during a single `solve()` call.  This catches repeated
neighbour systems even when they are not consecutive blocks.  The multi-slot
cache stores only the prepared factor matrices (`L`, `K^{-1}F`, and the Schur
factor), not the assembled `matA`/`rhsB` snapshots used for inspection.  It is
bounded by `factor_cache_size` slots and a per-thread memory cap; lookup uses a
small bucket table (`hash -> bucket -> linked slot list`) so only the matching
bucket is scanned instead of every cached slot.  Each hash candidate is still
verified with the full neighbour-set key before reuse, so collisions cannot
reuse an incorrect factorisation.  Replacement is global least-recently-used
across the slots, with replaced entries unlinked from their old bucket and
reinserted into the new bucket.

Cache invalidation is automatic:
- `set_obs` — coordinates may change K; sets `pf%valid = .false.`
- `set_vgm` — variogram changes K; sets `pf%valid = .false.`
- `update_obs_value` — values affect only the RHS, not K; cache preserved

The cache is **disabled by default** (`pf_cache=False`).  Enable it only when
you plan to call `solve()` multiple times on the same observation grid.

New C API functions:
- `krige_get_factor_info(handle, npp, p, valid)` — query dimensions and validity
- `krige_get_factor_matrices(handle, npp, p, L, kinv_drift, schur)` — copy matrices
- `krige_get_factor_system(handle, npp, p, nvar, matA, rhsB)` — copy the
  assembled LHS/RHS snapshot used to build the persistent factor

New Python method:
- `Kriging.get_factor()` — returns a dict with keys `valid`, `npp`, `p`,
  `L`, `kinv_drift`, `schur`, `matA`, and `rhsB` (all as NumPy arrays when
  `valid=True`)

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

- The Fortran solver now uses a shared inheritance framework.  `kriging_base.F90`
  defines the abstract `t_kriging_base`, common data containers, the unified
  per-thread context, the shared `solve` template, weight storage, and factor
  caches.  `t_kriging` in `kriging.F90` and `t_kriging_st` in `kriging_st.F90`
  now extend that base type and implement only the spatial/ST-specific hooks
  such as search, covariance assembly, variance calculation, and variogram
  formatting.
- The CAPI handle registry moved into `kriging_capi_common.F90` and stores
  polymorphic `class(t_kriging_base)` pointers.  The spatial and ST CAPI modules
  downcast from the shared registry to their concrete types.
- `solve(ncache=...)` now controls the per-thread multi-slot factor cache for a
  single solve call.  Use `ncache=0` to disable the hcache, `ncache=1` for a
  one-slot comparison, and the default `None` to keep the compiled/object
  default.
- `set_obs_drift` must be called (or re-called) after each `set_obs` when
  `ndrift > 0`; `set_obs` zeros the external drift rows on each call.
  Using `update_obs_value` is the correct API when only values change.
- `set_obs` rejects duplicate observation coordinate tuples for each variable.
  For space-time observations, the time coordinate is part of the tuple.
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
