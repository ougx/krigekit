# C API reference (`kriging_capi.F90`)

The C API is an ISO C Binding wrapper around the `t_kriging` Fortran type.
Every public method is exposed as a C-callable function that takes an opaque
64-bit integer **handle** instead of the Fortran derived type.

All functions return `integer(c_int) ierr`: **0 = success**, non-zero = error.
Retrieve the error message with `krige_get_last_error`.

## Design conventions

| Convention | Detail |
|---|---|
| Handle type | `int64` (registry slot index, not a raw pointer) |
| Boolean flags | `int32`: `0` = false, `1` = true |
| Strings | Null-terminated C char arrays; converted internally with `c2fstr` |
| Array layout | Fortran column-major `(ndim, nobs)` — Python transposes before calling |
| Array sizes | Passed explicitly alongside every pointer; no assumed-size except `pointweight` in `krige_set_grid_block` |
| Optional args | Python always supplies concrete values; no sentinel / `has_*` logic needed here |

---

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
    int neglect_error, int varying_vgm, int verbose,
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
| `verbose` | `int` | 0/1 print progress messages |
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
    int ndrift_c, int nblocks,
    const double *drift)    // [ndrift_c × nblocks], Fortran column-major
```

Sets external drift values at block locations.  Must be called after any
`krige_set_grid*` variant, and only when `ndrift > 0`.

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
int krige_solve(int64_t handle, int nthread)
```

Runs the kriging or SGSIM block loop.  `nthread = 0` uses the OpenMP runtime
default; `nthread > 0` caps the thread count for this call only.  Results are
available via the getters immediately after this returns.

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
