# Changelog

## 0.2.8 (unreleased)

## 0.2.7

### Added - Variogram Extensions

- Added statistical binning heuristics (`sturges`, `fd`, `scott`, `kmeans`) for `avg_vgm()` and `VariogramModel.calc_average()`.
- Added robust variogram estimators (`dowd`, `genton`) based on Median Absolute Deviation and $Q_n$ scale estimators to resist outlier influence.
- Added custom distance metrics for `raw_vgm()` via the `metric` argument, supporting non-Euclidean spatial coordinate systems.
- Added goodness-of-fit metrics calculation (RMSE, MSE, MAE, R2) to `fit_vgm()` and `VariogramModel.fit(return_metrics=True)`.

### Changed - Negative cross-variogram validation

Cross-variogram validation now permits negative nugget and structured sill
entries for cross pairs while retaining nonnegative requirements for direct
variograms. This supports valid PSD LMC matrices in which fine-texture
indicators are negatively correlated with AEM resistivity.

## 0.2.6

### Changed - Variogram analysis module architecture

Variogram analysis is now separated into kernel, geometry, empirical,
fitting, plotting, marginal-model, space-time-model, and multivariable-system
modules. `SpaceTimeVariogramModel` composes spatial and temporal
`VariogramModel` instances and owns only full space-time clouds and coupling
fits, mirroring the Fortran `vgm_struct_st` composition.

The existing `krigekit.variogram` import path remains a compatibility facade.
Legacy space-time calls on `VariogramModel` are forwarded to a lazily created
`SpaceTimeVariogramModel`; new code should instantiate the space-time class
directly.

`avg_vgm()` and the model `calc_average()` workflow now accept explicit
variable-width bin edges through `h_width` and `t_width`, while preserving
their existing scalar fixed-width behavior. This supports fine short-lag bins
and progressively wider bins where spatial or temporal pairs are sparse.

### New - Product structures in space-time marginal variograms

`SpaceTimeKriging.set_vgm()` and `set_vgm_temporal()` now accept
`product=True`, giving spatial and temporal marginals the same product-nesting
semantics as `Kriging.set_vgm()`. This allows valid quasi-periodic covariances
such as a long-term Gaussian decay multiplied by a hole-effect structure.
`VariogramModel` adds
`to_temporal_specs()` and `apply_temporal_to()` to replay fitted temporal
models into the space-time engine.

The new groundwater-level gallery example fits separate spatial and temporal
marginals from `obs_gwlevel.csv`, including a weak annual cycle whose amplitude
decays over the long-term Gaussian range. It also kriges half-year groundwater
level series for H0049 and H0001 from 1980 through 2020 with uncertainty bands,
plus leave-one-well-out reconstructions for sparse wells H0017 and H1477. A
2008.5 grid snapshot compares contemporaneous-only spatial kriging against
all-years space-time kriging with a reproducible 20% snapshot holdout.

`SpaceTimeVariogramModel.fit_spacetime_sum_metric()` now fits marginal amplitude
refinements, one joint sill per spatial structure, and the joint temporal scale
from a full space-time lag surface. `to_sum_metric_kriging_specs()` converts the
result directly to `SpaceTimeKriging` inputs.

### New - CCl4 product-sum variogram fitting example

`SpaceTimeVariogramModel` provides `fit_spacetime_product_sum()`,
`calc_spacetime_variogram()`, `set_spacetime_params()`, and
`to_spacetime_kriging_specs()` for a class-based fitting, adjustment, and
kriging-transfer workflow.

Spatial lag geometry is shared through `calc_lag_vectors()` and
`calc_anisotropic_lag()`. Empirical variogram functions accept an `anisotropy`
mapping, preserve physical `distance`, and add `anisotropic_distance` for
fitting. `SpaceTimeVariogramModel.calc_spacetime_variogram_between()` applies the same
geometry when evaluating coordinate and time pairs.

Added `st_variogram_fitting_ctet.py`, which uses that workflow to calculate the
empirical space-time variogram for `test_data/ctet.csv`, perform constrained
multistart product-sum fitting, diagnose weakly identified boundary solutions,
compare the automatic fit with the manually adjusted production model, and
print the engine-ready covariance parameters.

The existing CCl4 kriging example now explicitly identifies its transform as a
uniform quantile transform rather than a normal-score transform.

## 0.2.5

### New — `VariogramSystem.set_markov_cross`

Builds an indicator/covariate cross-variogram by the **Markov Model 1 (MM1)**
collocated-cokriging assumption: the cross adopts the secondary variable's
structure scaled by the collocated correlation,
`b_ps = rho * sqrt(b_pp * b_ss)`, which is positive semidefinite by
construction. This complements `fit_lmc()` — use `fit_lmc()` for co-sampled
multivariate data, and `set_markov_cross()` for a sparse primary (e.g. well
logs) cokriged with a dense secondary covariate (e.g. AEM), where a joint LMC
fit would collapse the cross-covariance. `corr=None` estimates the correlation
from collocated observations. See the *Multivariable systems* section of the
variogram-fitting guide.

## 0.2.4

### New — `plot_map3d` / `plot_vgm_map3d` overhaul

The 3D variogram fence map received several improvements:

**World-axis-aligned fences (new default)**

Fences are now aligned with the X/Y/Z world axes by default
(`rotate_fences=False`):

- **Fence A** — horizontal XY plane (shows azimuth anisotropy)
- **Fence B** (`n_fences ≥ 2`, default) — vertical XZ (East–West) plane (shows dip)
- **Fence C** (`n_fences ≥ 3`) — vertical YZ (North–South) plane

This makes the anisotropy angles readable directly from the plot axes.
Pass `rotate_fences=True` to restore the previous behaviour where fences
are rotated to the fitted model's principal planes.

**Anisotropy axis lines**

When `rotate_fences=False` and model angles are available, a red line is
drawn on each fence showing the major axis projected onto that fence plane:

- XY fence → azimuth direction (horizontal)
- XZ fence → East–West dip component
- YZ fence → North–South dip component (when `n_fences=3`)

The legend label reads `major (az=…°, dip=…°)` and includes plunge when
`n_fences=3`.  Lines lie IN their fence plane — they do not float in 3D
space — making the angles unambiguous to read.

**Z-order fix**

Previously, three separate `plot_surface` calls were used — one per fence.
Matplotlib's painter algorithm sorted depth per-collection, so one fence
would always render on top of another regardless of viewing angle.  All
fence polygons are now merged into a single `Poly3DCollection` so
depth-sorting applies across all three fences together.

**Rendering gap fix**

Adjacent polygons in a `Poly3DCollection` left consistent sub-pixel gaps
due to independent rasterization.  Fixed by drawing polygon edges in the
same colour as their face (`edgecolors=face_color, linewidths=0.3`).

**`fill_nan=False` parameter**

Optional nearest-neighbour fill of empty lag bins before rendering.
Off by default; useful when data are sparse and bins are patchy.

### Changed - Score transforms support kriging

`set_nscore` and `set_uscore` now work for ordinary/simple kriging
(`nsim=0`) as well as SGSIM.  Observation values are transformed before solve,
and kriging estimates/variances are back-transformed to data-unit moments after
solve with Gaussian quadrature.

Added a dedicated data-transform user guide covering normal-score transforms,
uniform quantile transforms, tail handling, declustering weights, and the
transform/back-transform helper APIs.

### New - C API score transform helpers

Added C API helpers for applying the active marginal transform after
`set_nscore` or `set_uscore`:

- `krige_transform_value_to_score(handle, ivar, n, value, score)`
- `krige_transform_score_to_value(handle, ivar, n, score, value)`

Python exposes the same functionality as
`Kriging.transform_value_to_score(...)`,
`Kriging.transform_score_to_value(...)`, and the alias
`Kriging.back_transform_score(...)`.

## 0.2.3

### Changed — auto-tuned `nthread` / `ncache` in `solve()`

`solve()` now shrinks its thread count and factor-cache size when the problem
structure proves the extra resources cannot help:

- **`ncache` → 1** when every variable's neighbour set is fixed
  (`need_search` false, i.e. all obs fit within `nmax`) and the local
  range/nugget modifiers are uniform across blocks.  Every block then assembles
  the identical kriging system, which the single always-on slot already reuses,
  so the multi-slot hash cache is redundant memory.
- **`ncache` → 0** when `varying_vgm` is set, because factor caching is
  disabled in that mode and no slot is ever reused.
- **`nthread` ≤ number of blocks**, since spawning more OpenMP workers than
  blocks only adds overhead.

Both adjustments only ever *shrink* the requested values, so they never change
results or lower the achievable cache-hit rate.  Cross-validation and SGSIM
keep the full requested cache (their neighbour sets vary block to block).

**Fortran / C API additions:**

- `normal_score` module (`nscore.f90`) — `t_nscore` transform table with
  `build` / `forward` / `back` and linear/power/hyperbolic tail models
- `krige_set_nscore(handle, ivar, zmin, zmax, ltail, utail, ltpar, utpar, nwt, wt)`

See {doc}`../user_guide/sgsim` for the full workflow.

### New - Uniform quantile transform for SGSIM

{py:meth}`~krigekit.Kriging.set_uscore` enables the same empirical quantile
transform as `set_nscore`, but maps observations to uniform CDF scores in
`[0, 1]` instead of standard-normal scores. Simulated values are interpreted as
probabilities and back-transformed to data units through the shared quantile
table. `set_quantile` is provided as a Python alias.

It requires `nsim > 0`; fit the variogram on the uniform scores (for example,
use sill `1/12` for a fully uniform marginal). Tail extrapolation, bounds, and
declustering weights use the same arguments as `set_nscore`.

**Fortran / C API additions:**

- `t_nscore%forward_uniform` / `t_nscore%back_uniform`
- `krige_set_uscore(handle, ivar, zmin, zmax, ltail, utail, ltpar, utpar, nwt, wt)`

### Fixed — SGSIM singularity when grid nodes coincide with observations

A grid node placed exactly on an observation produced a previously-simulated
node co-located with a hard datum, putting two identical rows in the kriging
matrix (singular) and yielding `NaN`.  The SGSIM neighbour search now drops a
previously-simulated block when it coincides — in space, and for space-time also
in time — with a hard observation already in the neighbourhood, since the datum
already conditions that location.  Such nodes now reproduce the observed value
instead of failing.  A diagonal-regularisation retry was also added to the
linear solver as a last-resort guard against any residual singular system.

## 0.2.2

### New — Normal-score transform for SGSIM

{py:meth}`~krigekit.Kriging.set_nscore` enables a normal-score (Gaussian
anamorphosis) transform for sequential Gaussian simulation.  The conditioning
data are mapped to normal scores through a weighted empirical CDF (declustering
weights optional), the simulation runs in Gaussian space, and the realisations
are back-transformed to data units with GSLIB-style tail extrapolation
(linear / power / hyperbolic) bounded by `zmin` / `zmax`.

The transform is implemented in the Fortran **engine**, behind the C API, so
every client language obtains the same transform and bit-reproducible
realisations.  It requires `nsim > 0`; fit the variogram on the normal scores
(unit sill).

```python
k = Kriging(nsim=20, seed=42)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=30)
k.set_nscore(ivar=1)   # optional: zmin, zmax, ltail, utail, ltpar, utpar, weights
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=40.0)
k.set_grid(coord=grid_coord)
k.set_sim()
k.set_search(ivar=1)
k.solve()
sims, _ = k.get_results()   # realisations back-transformed to data units
```

### Documentation and internal

- Performance-tuning guide for `nthread` / `ncache` and the factorisation cache
  (see {doc}`../user_guide/performance`).
- Internal: a private `check_solved` guard in the C API before results are
  returned; GitHub Actions auto-test workflow.

## 0.2.1 — 2026-06-12

PyPI packaging release — README and metadata updates only; no functional
changes from 0.2.0.

## 0.2.0 — 2026-06-08

First public release on PyPI, renamed from `pyKriging` to `krigekit`.  (Version
0.1.0 was an internal development version and was never tagged or released.)

### New — Multiple Indicator Kriging and SIS

#### `IndicatorKriging` class

New {py:class}`~krigekit.IndicatorKriging` class implementing Multiple Indicator
Kriging (MIK) and Sequential Indicator Simulation (SIS) for categorical variables.
Extends `Kriging` — all setup, solve, and results methods are inherited.

**Constructor parameters:**

- `ncat` — number of categories K (indicator variables ivar = 1..K)
- `nvar` — total co-kriging variables; defaults to `ncat` (pure MIS); set
  `nvar = ncat + M` to add M secondary continuous co-variates
- All other kwargs passed through to `Kriging`

**New methods:**

- `set_categorical_obs(coord, categories, category_labels, nmax, maxdist)` —
  converts raw category labels into K binary indicator datasets in one call,
  automatically computing `I_k = (categories == label_k)` for each indicator
- `set_indicator_vgm(vtype, nugget, sill, a_major, ...)` — sets all K² variogram
  pairs in one call; the `cross` parameter selects the cross-sill strategy:
  - `"same"` *(default)* — single shared sill for all pairs
  - `"proportional"` — auto sills = p_k (1 − p_k); cross sills = √(s_k · s_l);
    LMC positive-definite; requires `proportions`
  - `"independent"` — cross sills = 0 (K separate ordinary-kriging systems)

**Fortran/C API additions:**

- `krige_ind_create(handle)` — allocates a `t_kriging_indicator` object
- `krige_ind_set_ncat(handle, ncat)` — separates indicator count from total
  variable count for co-kriging MIS

**Usage example:**

```python
from krigekit import IndicatorKriging

ik = IndicatorKriging(ncat=4, ndim=2, nsim=50, seed=42)
ik.set_categorical_obs(coord=obs_coord, categories=obs_cats,
                       category_labels=["A", "B", "C", "D"], nmax=20)
ik.set_indicator_vgm(vtype="sph", nugget=0.02, sill=0.19,
                     a_major=500, a_minor1=80, azimuth=90,
                     cross="proportional",
                     proportions=np.array([0.18, 0.23, 0.21, 0.38]))
ik.set_grid(coord=grid_coord)
ik.set_sim()
for k in range(1, 5):
    ik.set_search(ivar=k, anis1=80/500, azimuth=90)
ik.solve()
sims, _ = ik.get_results()        # shape (ngrid, 4, 50) — one-hot
cat_idx = np.argmax(sims, axis=1) # shape (ngrid, 50)
```

See the {doc}`../auto_examples/s_sis_lithofacies` gallery example for a
full lithofacies SIS with side-by-side strategy comparison.

### New — `solver_stats` property

Both `Kriging` and `SpaceTimeKriging` now expose a `solver_stats` property
that returns counts from the most recent `solve()` call:

```python
k.solve()
print(k.solver_stats)
# {'chol_ok': 9950, 'ssytrf_fact': 1, 'ssytrf_reuse': 49}
```

| Key | Meaning |
|---|---|
| `chol_ok` | Blocks solved via Cholesky (fast path) |
| `ssytrf_fact` | Bunch-Kaufman LDL^T factorizations (O(n³), once per neighbourhood) |
| `ssytrf_reuse` | Blocks solved via cached SSYTRF (O(n²)) |

A non-zero `ssytrf_fact` means the kriging matrix was not positive-definite
for at least one neighbourhood (singular or near-duplicate observations).

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
