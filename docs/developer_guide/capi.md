# C API reference (`kriging_capi.F90`, `kriging_st_capi.f90`)

The C API is an ISO C Binding wrapper around the spatial `t_kriging` and
space-time `t_kriging_st` Fortran types.  Both inherit from
`t_kriging_base`; see [Fortran architecture](architecture.md) for the shared
solve framework.

Every public method is exposed as a C-callable function that takes an opaque
64-bit integer **handle** instead of the Fortran derived type.  Handles are
registry slot indices, not raw pointers.

All functions return `integer(c_int) ierr`: **0 = success**, non-zero = error.
Retrieve the error message with `krige_get_last_error`.

## Design conventions

| Convention | Detail |
|---|---|
| Handle type | `int64` (shared polymorphic registry slot index, not a raw pointer) |
| Boolean flags | `int32`: `0` = false, `1` = true |
| Strings | Null-terminated C char arrays; converted internally with `c2fstr` |
| Array layout | Fortran column-major `(ndim, nobs)` — Python transposes before calling |
| Array sizes | Passed explicitly alongside every pointer; no assumed-size except `pointweight` in `krige_set_grid_block` |
| Optional args | Python always supplies concrete values; no sentinel / `has_*` logic needed here |

---

## Fortran object model

The spatial and ST CAPIs share the same handle infrastructure in
`kriging_capi_common.F90`.  That registry stores `class(t_kriging_base)`
pointers, so a slot can hold either concrete type:

```text
class(t_kriging_base)
  +-- type(t_kriging)       spatial API, kriging_capi.F90
  +-- type(t_kriging_st)    ST API, kriging_st_capi.f90
```

Each API module retrieves the base pointer and then downcasts it with
`select type` before calling concrete-only methods.  Shared methods, including
persistent-factor accessors, live on `t_kriging_base` and are available to both
spatial and ST objects.

## Lifecycle

### `krige_create`

```c
int krige_create(int64_t *handle)
```

Allocates a new `t_kriging` object and returns its registry slot as an opaque
handle.  Python stores this handle and passes it to every subsequent call.

### `krige_destroy`

```c
int krige_destroy(int64_t *handle)
```

Finalises and deallocates the object; zeros the handle so stale use is caught
early.

---

## Initialization

### `krige_initialize`

```c
int krige_initialize(int64_t handle,
    int ndim, int nvar, int ndrift,
    int unbias, int nsim,
    int anisotropic_search, int weight_correction,
    int use_old_weight, int store_weight,
    int cross_validation, int write_mat,
    int neglect_error, int varying_vgm, int std_ck, int verbose,
    int pf_cache,
    const char *weight_file,
    const double bounds[2],
    int seed)
```

Must be called once after `krige_create` and before any other function.

| Parameter | Type | Description |
|---|---|---|
| `ndim` | `int` | Spatial dimensions: 2 or 3 |
| `nvar` | `int` | Number of variables (1 = kriging, >1 = co-kriging) |
| `ndrift` | `int` | Number of external drift functions (0 = none) |
| `unbias` | `int` | 1 = ordinary kriging; 0 = simple kriging |
| `nsim` | `int` | 0 = kriging only; >0 = number of SGSIM realisations |
| `anisotropic_search` | `int` | 0/1 use anisotropic search ellipse |
| `weight_correction` | `int` | 0/1 clip negative weights and re-normalise |
| `use_old_weight` | `int` | 0/1 read weights from `weight_file` |
| `store_weight` | `int` | 0/1 write weights to `weight_file` |
| `cross_validation` | `int` | 0/1 leave-one-out cross-validation mode |
| `write_mat` | `int` | 0/1 write kriging matrix to CSV (debug) |
| `neglect_error` | `int` | 0/1 write NaN instead of stopping on singular matrix |
| `varying_vgm` | `int` | 0/1 use a different variogram per block (SVA mode) |
| `std_ck` | `int` | 0/1 co-kriging unbiasedness formulation (only used when `nvar > 1` and `unbias = 1`). **1** = standard co-kriging: separate per-variable constraints (Σw₁=1, Σw₂=0); matches gstat/ISATIS. **0** = Isaaks & Srivastava: single combined constraint (Σw₁+Σw₂=1) plus local-mean correction. Default: 1. |
| `verbose` | `int` | 0/1 print progress messages |
| `pf_cache` | `int` | 0/1 enable the persistent between-solve factorisation cache.  When 1, the Cholesky factorisation of K is stored after the first `krige_solve` and reused on subsequent calls when observations and variogram are unchanged.  Default: 0 (disabled). |
| `weight_file` | `char*` | Path for weight file; empty string when unused |
| `bounds` | `double[2]` | `[lower, upper]` clipping bounds on the estimate |
| `seed` | `int` | Random seed for SGSIM (0 = use clock) |

---

## Observations

### `krige_set_obs`

```c
int krige_set_obs(int64_t handle,
    int ivar, int nobs, int ndim_c,
    const double *coord,    // [ndim_c × nobs], Fortran column-major
    const double *value,    // [nobs]
    const double *variance, // [nobs]  measurement error variance; 0 if unknown
    int nmax, double maxdist, double sk_mean)
```

Sets coordinates, values, and per-observation measurement variance for one
variable.  Pass `INT_MAX` for `nmax` and `DBL_MAX` for `maxdist` to use all
observations within unlimited range.  `sk_mean` is only used when `unbias=0`.
Duplicate coordinate tuples within the same variable are rejected.  For spatial
kriging, all `ndim_c` coordinate rows must match to count as a duplicate.

### `krige_set_obs_drift`

```c
int krige_set_obs_drift(int64_t handle,
    int ivar, int ndrift_c, int nobs,
    const double *drift)    // [ndrift_c × nobs], Fortran column-major
```

Sets external drift values at observation locations.  Must be called after
`krige_set_obs` for the same `ivar`, and only when `ndrift > 0`.

### `krige_update_obs_value`

```c
int krige_update_obs_value(int64_t handle,
    int ivar, int nobs,
    const double *value)    // [nobs]
```

Replaces observation values in-place without touching coordinates or the
KD-tree.  Use with `use_old_weight` to re-estimate with new data without
recomputing search neighbourhoods or the LHS factorisation.

---

## Variogram

### `krige_reset_vgm`

```c
int krige_reset_vgm(int64_t handle, int ivar, int jvar)
```

Clears all nested structures for the `(ivar, jvar)` pair (and its mirror
`(jvar, ivar)` for cross-variograms) across every block.  Call before
`krige_set_vgm` when reusing an object with a different variogram model.

### `krige_set_vgm`

```c
int krige_set_vgm(int64_t handle,
    int ivar, int jvar,
    const char *vtype,      // null-terminated: "sph" "exp" "gau" "pow" "lin" "hol" "bsq" "cir" "nug"
    double nugget, double sill,
    double a_major, double a_minor1, double a_minor2,
    double azimuth, double dip, double plunge)
```

Appends one nested structure for the `(ivar, jvar)` variable pair.  Call
multiple times to build a composite model.  For co-kriging the LMC constraint
`b12² ≤ b11 × b22` must hold per structure.

### `krige_set_vgm_block`

```c
int krige_set_vgm_block(int64_t handle,
    int ivar, int jvar, int ib,
    const char *vtype,
    double nugget, double sill,
    double a_major, double a_minor1, double a_minor2,
    double azimuth, double dip, double plunge)
```

Same as `krige_set_vgm` but targets a single block `ib` (1-based).  Requires
`varying_vgm=1` in `krige_initialize` and `krige_set_grid` to have been called
first (block count must be known).

---

## Grid

### `krige_set_grid`

```c
int krige_set_grid(int64_t handle,
    int ngrid, int ndim_c,
    const double *coord,        // [ndim_c × ngrid], Fortran column-major
    const double *rangescale,   // [ngrid]  variogram range scaling; pass 1.0 when unused
    const double *localnugget)  // [ngrid]  additional per-block nugget; pass 0.0 when unused
```

Sets estimation targets for point kriging.  Use `krige_set_grid_block` for
block kriging or `krige_set_grid_cv` for cross-validation.

### `krige_set_grid_block`

```c
int krige_set_grid_block(int64_t handle,
    int block_type,
    int ngrid, int ndim_c,
    const double *coord,        // [ndim_c × ngrid]
    int nblock,
    const int *nblockpnt,       // [nblock]  sub-nodes per block
    const double *pointweight,  // [sum(nblockpnt)]
    const double *blocksize,    // [ndim_c × nblock]
    const double *rangescale,   // [nblock]
    const double *localnugget)  // [nblock]
```

Sets estimation targets for block kriging.

| `block_type` | Meaning |
|---|---|
| `-4` | Gaussian quadrature sub-nodes (auto-generated) |
| `> 0` | User-supplied sub-nodes |

`pointweight` length equals `sum(nblockpnt)`; Fortran derives it via
`sum(nblockpnt)` so no separate `npw` argument is needed.

### `krige_set_grid_cv`

```c
int krige_set_grid_cv(int64_t handle)
```

Configures leave-one-out cross-validation mode.  Fortran derives the grid
from the observation coordinates automatically.  Call instead of
`krige_set_grid` when `cross_validation=1`.

### `krige_set_grid_drift`

```c
int krige_set_grid_drift(int64_t handle,
    int ivar,
    int ndrift_c, int nblocks,
    const double *drift)    // [ndrift_c × nblocks], Fortran column-major
```

Sets external drift values at block locations.  Must be called after any
`krige_set_grid*` variant, and only when `ndrift > 0`.

| Parameter | Description |
|---|---|
| `ivar` | Target-variable index (1-based) whose RHS receives this drift. Pass `ivar < 0` to broadcast the same drift to **all** target variables — the common case when external drift is the same regardless of which variable is being estimated. |
| `ndrift_c` | Number of drift functions (= `ndrift` from `krige_initialize`) |
| `nblocks` | Number of blocks (`block%n`, **not** `grid%n` for block kriging) |
| `drift` | Drift values `[ndrift_c × nblocks]`, Fortran column-major |

> **Note on `ivar` semantics.** In `krige_set_obs_drift`, `ivar` identifies the
> **source** variable (whose observations form the F-matrix column).  Here it
> identifies the **target** variable (which estimation's RHS uses this drift).
> These are opposite ends of the kriging system — both use 1-based indexing and
> `< 0` is not valid for `krige_set_obs_drift`.

---

## Simulation (SGSIM)

### `krige_set_sim`

```c
int krige_set_sim(int64_t handle,
    int nblocks,
    const int    *randpath,  // [nblocks]  random visiting order
    int nsim_c, int nvar_c,
    const double *sample)    // [nsim_c × nvar_c × nblocks]
```

Supplies the random visiting path and pre-drawn standard-normal samples for
SGSIM.  Python generates both arrays before calling.  Call after
`krige_set_grid` and before `krige_set_search`.  Only needed when `nsim > 0`.

---

## Search

### `krige_set_search`

```c
int krige_set_search(int64_t handle,
    int ivar,
    double anis1, double anis2,
    double azimuth, double dip, double plunge)
```

Builds the KD-tree and configures the search ellipse for variable `ivar`.
Call once per variable after all observations are loaded.

| Parameter | Description |
|---|---|
| `anis1` | Horizontal anisotropy ratio (minor/major); 1.0 = isotropic |
| `anis2` | Vertical anisotropy ratio (vertical/major); 1.0 = isotropic |
| `azimuth` | Major axis azimuth (degrees, clockwise from North) |
| `dip` | Dip angle (degrees, positive downward) |
| `plunge` | Plunge angle (degrees) |

---

## Solve

### `krige_prepare`

```c
int krige_prepare(int64_t handle)
```

Pre-allocates result arrays and sets up the block loop.  Called automatically
by `krige_solve`; exposed separately for benchmarking.

### `krige_solve`

```c
int krige_solve(int64_t handle, int nthread, int ncache)
```

Runs the kriging or SGSIM block loop.  `nthread = 0` uses the OpenMP runtime
default; `nthread > 0` caps the thread count for this call only.  `ncache = -1`
keeps the object's current hcache slot default, `ncache = 0` disables the
multi-slot hcache for this solve, and `ncache > 0` sets the per-thread hcache
slot count for this solve only.  Results are available via the getters
immediately after this returns.

---

## Result getters

### `krige_get_nblocks`

```c
int krige_get_nblocks(int64_t handle, int *n)
```

Returns the number of estimation blocks.

### `krige_get_nsim`

```c
int krige_get_nsim(int64_t handle, int *n)
```

Returns the number of simulations (1 for plain kriging).

### `krige_get_block_coord`

```c
int krige_get_block_coord(int64_t handle,
    int ndim_c, int nblocks,
    double *out)    // [ndim_c × nblocks], Fortran column-major
```

Copies the block centroid coordinates into a caller-allocated buffer.  The
output is filled as `out(ndim_c, nblocks)` in Fortran column-major order.
Python allocates a `(ndim_c, nblocks)` Fortran-order array and transposes to
`(nblocks, ndim_c)` for standard row-major convention.

For SGSIM, blocks are reordered back to the original (non-randomised) index
order inside `krige_solve`, so the coordinates returned here always correspond
to `krige_get_estimate_all` positions at the same block index.

### `krige_get_estimate`

```c
int krige_get_estimate(int64_t handle,
    int nsim_c, int nblocks,
    double *out)    // [nblocks × nsim_c], C row-major
```

Copies the primary-variable estimate for all simulations.  Output is
`out[ib, isim]` for block `ib` (0-based) and simulation `isim` (0-based).

### `krige_get_estimate_all`

```c
int krige_get_estimate_all(int64_t handle,
    int nblocks, int nvar_c, int nsim_c,
    double *out)    // [nblocks × nvar_c × nsim_c]
```

Copies estimates for all variables and simulations.  Output convention:
`out[ib, kvar, isim]` — block index first, matching the `(nobs, ndim)` coord
convention.

### `krige_get_variance`

```c
int krige_get_variance(int64_t handle,
    int nblocks,
    double *out)    // [nblocks]
```

Copies the primary-variable kriging variance.

### `krige_get_variance_all`

```c
int krige_get_variance_all(int64_t handle,
    int nblocks, int nvar_c,
    double *out)    // [nblocks × nvar_c × nvar_c]
```

Copies the full conditional covariance matrix at every block.
`out[ib, iv, jv]` is the covariance between variables `iv+1` and `jv+1` at
block `ib+1`; the diagonal `out[ib, k, k]` is variable `k+1`'s kriging
variance.

---

## Persistent factorisation cache

Between-solve factorisation caching allows the Cholesky factorisation of the
kriging covariance matrix **K** to be reused across successive `krige_solve`
calls when observation coordinates and the variogram have not changed.

The cache is populated automatically after the first successful `krige_solve`
and invalidated by `krige_set_obs` or `krige_set_vgm`.  These two functions
let Python query the cached matrices for inspection or debugging.

Internally there are three cache layers:

- `ctx%cache`: one single-entry cache per worker thread.  It handles immediate
  repeated systems and is pre-warmed from the persistent cache when available.
- `ctx%hcache`: one bounded multi-slot cache per worker thread for repeated
  systems within a single solve.  It stores only recent prepared factor
  matrices (`L`, `K^{-1}F`, and the Schur factor) in LRU slots and indexes them
  with a bucket table (`hash -> bucket -> linked slot list`), so lookup scans
  only the matching bucket.  Every hash candidate is verified against the full
  neighbour key with `fcache_matches`, so collisions are safe.
- `self%pf`: the optional persistent cache shared by the kriging object across
  solves when `pf_cache=1`.  This cache also stores the assembled `matA`/`rhsB`
  snapshot exposed by `krige_get_factor_system`.

`self%pf` is saved after the parallel block loop from a thread whose current
`ctx%matA`/`ctx%rhsB` still match its thread-local factors.  hcache hits do not
update the assembled system, so they are not used to populate `matA`/`rhsB` for
inspection.

The multi-slot cache size is controlled in Fortran by `factor_cache_size`
(default 64 slots per thread) and by a per-thread byte cap
(`MAX_HCACHE_BYTES`).  The byte cap limits `L`, `kinv_drift`, and `schur`
storage for each OpenMP worker context.

For cache-path testing, the Python `solve(ncache=...)` argument and C API
`krige_solve(..., ncache)` argument override this slot count for one solve
call.  Use `ncache=0` to disable the multi-slot hcache, `ncache=1` for a
one-slot hcache, or `ncache=None` in Python / `ncache=-1` in C to keep the
compiled default.  The Makefile `HCACHE` variable sets that compiled default:
`make HCACHE=0` disables the multi-slot hcache by default, `make HCACHE=1`
builds a one-slot default, and bare `make` uses the normal 64-slot default.
These controls do not disable the single-entry `ctx%cache` or the optional
persistent `self%pf` cache.

### `krige_get_factor_info`

```c
int krige_get_factor_info(int64_t handle,
    int *npp_out,    // number of neighbours (rows/cols of K)
    int *p_out,      // drift + unbiasedness columns (Schur size)
    int *valid_out)  // 1 = valid; 0 = not yet computed or invalidated
```

Returns the dimensions and validity of the cached factorisation.
Call this first to obtain `npp` and `p` before allocating buffers for
`krige_get_factor_matrices`.

### `krige_get_factor_matrices`

```c
int krige_get_factor_matrices(int64_t handle,
    int npp, int p,
    double *L_out,       // [npp × npp]       upper-tri Cholesky of K
    double *kinv_out,    // [npp × max(1,p)]   K^{-1} F
    double *schur_out)   // [max(1,p) × max(1,p)]  Cholesky of F'K^{-1}F
```

Copies the three persistent factor matrices into caller-allocated arrays.
`npp` and `p` must match the values returned by `krige_get_factor_info`.

All arrays are in Fortran column-major order.  The solver uses `uplo='U'`
(LAPACK `spotrf`), so:

- **`L_out`** — upper triangle is the Cholesky factor U; `K = U' U`.
  The lower triangle retains the original K values and should be ignored.
- **`kinv_out`** — `K^{-1} F` where F is the full drift matrix.
- **`schur_out`** — upper triangle is the Cholesky factor of `F' K^{-1} F`.

### `krige_get_factor_system`

```c
int krige_get_factor_system(int64_t handle,
    int npp, int p, int nvar,
    double *matA_out,    // [npp+p x npp+p] assembled LHS before factorization
    double *rhsB_out)    // [nvar x npp+p] assembled RHS before solving
```

Copies the assembled linear system that produced the persistent factor.  This is
for inspection and debugging; `npp`, `p`, and `nvar` must match the kriging
object and the dimensions returned by `krige_get_factor_info`.

#### Invalidation rules

| Trigger | Effect on persistent cache |
|---|---|
| `krige_set_obs` | `pf_valid = false` (coordinates change K) |
| `krige_set_vgm` | `pf_valid = false` (variogram changes K) |
| `krige_update_obs_value` | **No effect** — values do not enter K |

#### Python wrapper

```python
f = kriging_object.get_factor()
# f['valid']       bool
# f['npp']         int — size of K
# f['p']           int — size of Schur complement
# f['L']           ndarray (npp, npp)       — upper-triangular factor
# f['kinv_drift']  ndarray (npp, max(1,p))  — K^{-1} F
# f['schur']       ndarray (max(1,p), max(1,p))
# f['matA']        ndarray (npp+p, npp+p)   — assembled LHS
# f['rhsB']        ndarray (nvar, npp+p)    — assembled RHS
```

---

## Weight store

### `krige_free_weight_store`

```c
int krige_free_weight_store(int64_t handle)
```

Frees the in-memory weight store.

### `krige_get_weight_nnear`

```c
int krige_get_weight_nnear(int64_t handle,
    int ngroups_c, int nblock_c,
    int *out)    // [ngroups_c × nblock_c]
```

Copies the neighbour-count array.  `ngroups = nvar` for kriging,
`ngroups = 2 * nvar` for SGSIM.

### `krige_get_weight_inear`

```c
int krige_get_weight_inear(int64_t handle,
    int nmax_c, int ngroups_c, int nblock_c,
    int *out)    // [nmax_c × ngroups_c × nblock_c]
```

Copies the 1-based observation-index array for every neighbour slot.
Padded slots are zero.

### `krige_get_weight_data`

```c
int krige_get_weight_data(int64_t handle,
    int nmax_c, int ngroups_c, int nvar_c, int nblock_c,
    double *out)    // [nmax_c × ngroups_c × nvar_c × nblock_c]
```

Copies kriging weights.  Padded slots are zero.

### `krige_get_weight_var`

```c
int krige_get_weight_var(int64_t handle,
    int nvar_c, int nblock_c,
    double *out)    // [nvar_c × nvar_c × nblock_c]
```

Copies the stored kriging variances (shape matches `krige_get_variance_all`
transposed to Fortran order).

### `krige_set_weights`

```c
int krige_set_weights(int64_t handle,
    int nmax_c, int ngroups_c, int nvar_c, int nblock_c,
    const int    *nnear_in,   // [ngroups_c × nblock_c]
    const int    *inear_in,   // [nmax_c × ngroups_c × nblock_c]
    const double *weight_in,  // [nmax_c × ngroups_c × nvar_c × nblock_c]
    const int    *order_in,   // [nblock_c]
    const double *var_in)     // [nvar_c × nvar_c × nblock_c]
```

Loads pre-computed weights into the store and sets `use_old_weight = true`
so that the next `krige_solve` applies them directly without solving the
kriging system.

---

## Utilities

### `krige_get_last_error`

```c
int krige_get_last_error(char *buffer, int nbuf)
```

Copies the last error message (null-terminated) into `buffer`.  `nbuf` is the
buffer capacity in bytes.  Returns 0 even when an error exists — the error
string itself carries the information.

### `krige_to_str`

```c
int64_t krige_to_str(int64_t handle)
```

Returns a pointer to a null-terminated Fortran character array containing a
human-readable summary of the kriging object.  Returns 0 on error.  **Do not
free the returned pointer** — it is owned by the Fortran object.

### `krige_get_max_threads` / `krige_get_num_threads`

```c
void krige_get_max_threads(int *n)
void krige_get_num_threads(int *n)
```

Query the OpenMP thread count.  Both return 1 when the library is compiled
without OpenMP (`--no-openmp`).

---

## Space-time API (`kriging_st_capi.f90`)

All ST entry points are prefixed `krige_st_` and share the same handle
registry as the spatial API.  Differences from the spatial API are noted below;
methods not listed here have the same signature as their `krige_` counterparts
(e.g. `krige_st_solve`, `krige_st_prepare`, `krige_st_get_estimate`, …).

### `krige_st_set_obs`

As with `krige_set_obs`, duplicate observation coordinate tuples are rejected.
For ST data, the duplicate key includes spatial coordinates and time.

```c
int krige_st_set_obs(int64_t handle,
    int ivar, int nobs,
    const double *coord,    // [4 × nobs], Fortran column-major; rows 1:3 = spatial, row 4 = time
    const double *value,    // [nobs]
    const double *variance, // [nobs]
    int nmax, double maxdist, double sk_mean)
```

Identical to `krige_set_obs` except that `coord` has 4 rows (3 spatial + 1 time).
`maxdist` is in **km-equivalent units** — the same space as `h_ST`.

### `krige_st_set_grid`

```c
int krige_st_set_grid(int64_t handle,
    int ngrid,
    const double *coord,  // [3 × ngrid], Fortran column-major (spatial only)
    const double *time)   // [ngrid]
```

Accepts spatial coordinates and times as separate arrays (unlike `krige_set_grid`
which takes a single `ndim`-row coord array).

### `krige_st_set_st_model`

```c
int krige_st_set_st_model(int64_t handle,
    const char *model,       // "sum_metric" or "product_sum"
    const char *transform,   // variogram type for f_time: "lin", "exp", "sph", …
    double at,               // joint temporal scale (same units as time coordinate)
    double time_nugget,      // temporal variogram nugget
    double time_sill,        // temporal variogram sill
    double k_ps)             // product-sum k (ignored for sum_metric)
```

Sets global ST model parameters shared by all variogram entries.

### `krige_st_set_search`

```c
int krige_st_set_search(int64_t handle,
    int ivar,
    double time_at,    // temporal scale: t_kd = t * time_at  (km-equivalent)
    double anis1,      // spatial minor/major anisotropy ratio
    double anis2,      // spatial vertical/major anisotropy ratio
    double azimuth,    // major-axis azimuth (degrees, clockwise from North)
    double dip,        // dip angle (degrees)
    double plunge)     // plunge angle (degrees)
```

Builds the 4D KD-tree for variable `ivar`.  The time axis is stored as
`t_kd = t * time_at` so that the L2 distance in the `(x, y, z, t·time_at)`
space equals the sum-metric distance:

```
h_ST = sqrt(h_S^2 + (time_at * dt)^2)
```

Pass `time_at` equal to `at` from `krige_st_set_st_model` to keep search and
variogram scales consistent.  `maxdist` set in `krige_st_set_obs` is then a
radius in km-equivalent (h_ST) units.

---

## Handle registry (internals)

Python receives a 1-based slot index (`int64`) rather than a raw Fortran
pointer.  This avoids passing non-C-interoperable derived-type addresses
through ctypes.

| Function | Description |
|---|---|
| `store_obj` | Finds the first free slot; grows the registry (doubles) if full |
| `get_obj` | Validates the slot index and recovers the typed Fortran pointer |
| `release_obj` | Nullifies the slot on `krige_destroy`; does not compact the array |

The registry starts at 16 slots and doubles on demand, keeping old slot
numbers stable for existing Python handles.
