# C API reference (`kriging_capi.F90`, `kriging_capi_common.F90`, `kriging_st_capi.f90`)

This page documents the exported C ABI listed in
`src/krigekit/kriging.def`.  The ABI wraps the spatial `t_kriging`,
indicator `t_kriging_indicator`, and space-time `t_kriging_st` Fortran types.

Every status-returning function returns `int ierr`: `0` means success and any
non-zero value means failure.  On failure, call `krige_get_last_error` for the
message.  Object handles are opaque 64-bit registry slot indices, not raw
pointers.

## Conventions

| Convention | Detail |
|---|---|
| Handle type | `int64_t`; pass by value except create/destroy, which take `int64_t *`. |
| Boolean flags | `int`: `0=false`, `1=true`. |
| Strings | Null-terminated C strings. |
| Point arrays | C/PInvoke callers pass row-major `(npoint, ndim)` memory, e.g. `[x0,y0,x1,y1,...]`. The Fortran dummy arguments are declared as column-major `(ndim, npoint)`, which is the same raw memory layout. |
| Other arrays | Shapes below are caller-facing flat buffer shapes unless marked otherwise. |
| Indexing | Public variable, block, and category indices are 1-based unless a result buffer description explicitly says 0-based. |
| Optional arrays | Some simulation pointers may be `NULL`; most other array arguments must point to valid storage with the documented extent. |

## Error And Diagnostics

### `krige_get_last_error`

```c
int krige_get_last_error(char *buffer, int nbuf)
```

Copies the last error message into `buffer` as a null-terminated string.

### `krige_solver_stats`

```c
void krige_solver_stats(int64_t handle, int out[5])
```

Writes counters from the last solve:

```text
out[0] = failed blocks
out[1] = fresh Cholesky factorizations
out[2] = cached Cholesky reuses
out[3] = fresh SSYTRF factorizations
out[4] = cached SSYTRF reuses
```

## Spatial Lifecycle

### `krige_create`

```c
int krige_create(int64_t *handle)
```

Allocates a spatial kriging object.

### `krige_ind_create`

```c
int krige_ind_create(int64_t *handle)
```

Allocates an indicator kriging/SIS object.  Use the returned handle with the
shared `krige_*` setup, solve, and result functions.

### `krige_destroy`

```c
int krige_destroy(int64_t *handle)
```

Finalizes and deallocates a spatial or indicator object and sets `*handle = 0`.

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

Initializes a spatial or indicator object.  Use `unbias=1` for ordinary
kriging and `unbias=0` for simple kriging.  Plain kriging uses `nsim=0`;
SGSIM/SIS uses `nsim > 0`.

### `krige_ind_set_ncat`

```c
int krige_ind_set_ncat(int64_t handle, int ncat)
```

Sets the number of indicator categories after `krige_ind_create` and
`krige_initialize`.  `ncat` must satisfy `1 <= ncat <= nvar`.  If
`ncat < nvar`, the remaining variables are secondary continuous covariates.

## Observations And Transforms

### `krige_set_obs`

```c
int krige_set_obs(int64_t handle,
    int ivar, int nobs, int ndim_c,
    const double *coord,     // [nobs x ndim_c]
    const double *value,     // [nobs]
    const double *variance,  // [nobs]
    double sk_mean)
```

Sets coordinates, values, and measurement-error variance for one variable.
`sk_mean` is only used for simple kriging (`unbias=0`).

### `krige_update_obs_value`

```c
int krige_update_obs_value(int64_t handle,
    int ivar, int nobs,
    const double *value)     // [nobs]
```

Replaces values without rebuilding coordinates or the KD-tree.

### `krige_set_obs_drift`

```c
int krige_set_obs_drift(int64_t handle,
    int ivar, int ndrift_c, int nobs,
    const double *drift)     // [nobs x ndrift_c]
```

Sets external drift values at observation locations.  Call after
`krige_set_obs` for the same source variable.

### `krige_set_nscore`

```c
int krige_set_nscore(int64_t handle,
    int ivar,
    double zmin, double zmax,
    int ltail, int utail,
    double ltpar, double utpar,
    int nwt,
    const double *wt)        // [nwt], ignored when nwt=0
```

Builds a normal-score transform for `ivar` from the current observations and
configures automatic back-transform after solving.

### `krige_set_uscore`

```c
int krige_set_uscore(int64_t handle,
    int ivar,
    double zmin, double zmax,
    int ltail, int utail,
    double ltpar, double utpar,
    int nwt,
    const double *wt)        // [nwt], ignored when nwt=0
```

Builds a uniform-CDF score transform for `ivar`.

Tail codes are `1=linear`, `2=power`, and `4=hyperbolic`.

### `krige_transform_value_to_score`

```c
int krige_transform_value_to_score(int64_t handle,
    int ivar, int n,
    const double *value,     // [n]
    double *score)           // [n]
```

Transforms data values through the active score transform.

### `krige_transform_score_to_value`

```c
int krige_transform_score_to_value(int64_t handle,
    int ivar, int n,
    const double *score,     // [n]
    double *value)           // [n]
```

Back-transforms scores to data values.

## Spatial Variograms

### `krige_reset_vgm`

```c
int krige_reset_vgm(int64_t handle, int ivar, int jvar)
```

Clears nested structures for a variable pair.

### `krige_set_vgm`

```c
int krige_set_vgm(int64_t handle,
    int ivar, int jvar,
    const char *vtype,
    double nugget, double sill,
    double a_major, double a_minor1, double a_minor2,
    double azimuth, double dip, double plunge,
    int is_product)
```

Adds a nested variogram structure.  `is_product=0` adds a structure;
`is_product=1` multiplies it with the preceding structure in covariance space.

### `krige_set_vgm_block`

```c
int krige_set_vgm_block(int64_t handle,
    int ivar, int jvar, int ib,
    const char *vtype,
    double nugget, double sill,
    double a_major, double a_minor1, double a_minor2,
    double azimuth, double dip, double plunge,
    int is_product)
```

Adds a variogram structure for block `ib`.  Requires `varying_vgm=1` and a
previous grid setup.

## Spatial Grid, Simulation, Search, And Gradients

### `krige_set_grid`

```c
int krige_set_grid(int64_t handle,
    int ngrid, int ndim_c,
    const double *coord,        // [ngrid x ndim_c]
    const double *rangescale,   // [ngrid]
    const double *localnugget)  // [ngrid]
```

Sets point-kriging target locations.

### `krige_set_grid_block`

```c
int krige_set_grid_block(int64_t handle,
    int block_type,
    int ngrid, int ndim_c,
    const double *coord,        // [ngrid x ndim_c]
    int nblock,
    const int *nblockpnt,       // [nblock]
    const double *pointweight,  // [sum(nblockpnt)]
    const double *blocksize,    // [nblock x ndim_c]
    const double *rangescale,   // [nblock]
    const double *localnugget)  // [nblock]
```

Sets block-kriging targets.  `block_type=-4` uses Gaussian quadrature
sub-nodes; positive values use caller-supplied sub-nodes.

### `krige_set_grid_cv`

```c
int krige_set_grid_cv(int64_t handle)
```

Configures leave-one-out cross-validation targets from observations.

### `krige_set_grid_drift`

```c
int krige_set_grid_drift(int64_t handle,
    int ivar,
    int ndrift_c, int nblocks,
    const double *drift)        // [nblocks x ndrift_c]
```

Sets drift values at target blocks.  `ivar < 0` broadcasts to all target
variables.

### `krige_set_sim`

```c
int krige_set_sim(int64_t handle,
    int nblocks,
    const int *randpath,        // [nblocks], or NULL
    int nsim_c, int nvar_c,
    const double *sample)       // [nblocks x nvar_c x nsim_c], or NULL
```

Configures spatial SGSIM/SIS random path and samples.  `NULL` pointers let
Fortran generate the data.  Call after grid setup and before search setup.

### `krige_set_grad`

```c
int krige_set_grad(int64_t handle,
    int ivar, int ngrad, int ndim_c,
    const double *coord1,       // [max(ngrad,1) x ndim_c]
    const double *coord2,       // [max(ngrad,1) x ndim_c]
    const double *grad_val,     // [max(ngrad,1)]
    const double *variance,     // [max(ngrad,1)]
    int ndrift_c,
    const double *drift_ext)    // [max(ngrad,1) x max(ndrift_c,1)]
```

Sets gradient observation pairs for variable `ivar`.  Each pair constrains
`Z(coord1) - Z(coord2)`.  Use `ngrad=0` to reset gradient constraints.

### `krige_set_search`

```c
int krige_set_search(int64_t handle,
    int ivar,
    double anis1, double anis2,
    double azimuth, double dip, double plunge,
    int nmax, double maxdist,
    int sector_search)
```

Builds the KD-tree and configures the search neighborhood.  Negative `nmax` or
`maxdist` keeps the existing value.

## Solve And Results

The solve and result functions are shared by spatial, indicator, and space-time
objects.  For plain kriging, `krige_get_nsim` returns `0`; pass
`max(1, nsim)` to result getters that require an `nsim_c` extent.

### `krige_prepare`

```c
int krige_prepare(int64_t handle)
```

Pre-allocates result arrays and sets up the block loop.  `krige_solve` calls it
automatically.

### `krige_solve`

```c
int krige_solve(int64_t handle, int nthread, int ncache)
```

Runs the kriging or simulation loop.  `nthread=0` uses the OpenMP default.
`ncache=-1` keeps the object/default cache setting, `0` disables the shared
factor cache for this solve, and positive values set the shared cache slot
count for this solve.

### `krige_get_nblocks`

```c
int krige_get_nblocks(int64_t handle, int *n)
```

Returns the number of target blocks.

### `krige_get_nsim`

```c
int krige_get_nsim(int64_t handle, int *n)
```

Returns configured simulations: `0` for plain kriging, `>0` for simulation.

### `krige_get_block_coord`

```c
int krige_get_block_coord(int64_t handle,
    int ndim_c, int nblocks,
    double *out)               // [nblocks x ndim_c]
```

Copies block centroid coordinates.

### `krige_get_estimate`

```c
int krige_get_estimate(int64_t handle,
    int nsim_c, int nblocks,
    double *out)               // [nblocks x nsim_c]
```

Copies primary-variable estimates.  Use `nsim_c=max(1, krige_get_nsim(...))`.

### `krige_get_estimate_all`

```c
int krige_get_estimate_all(int64_t handle,
    int nblocks, int nvar_c, int nsim_c,
    double *out)               // [nblocks x nvar_c x nsim_c]
```

Copies estimates for all variables and simulations.

### `krige_get_variance`

```c
int krige_get_variance(int64_t handle,
    int nblocks,
    double *out)               // [nblocks]
```

Copies primary-variable kriging variance.

### `krige_get_variance_all`

```c
int krige_get_variance_all(int64_t handle,
    int nblocks, int nvar_c,
    double *out)               // [nblocks x nvar_c x nvar_c]
```

Copies the full conditional covariance matrix for every block.

### `krige_to_str`

```c
int64_t krige_to_str(int64_t handle)
```

Returns a pointer to an object-owned null-terminated summary string.  Do not
free the returned pointer.

## Persistent Factorization Cache

### `krige_get_factor_info`

```c
int krige_get_factor_info(int64_t handle,
    int *npp_out,
    int *p_out,
    int *valid_out)
```

Returns cached factor dimensions and whether the persistent factor is valid.

### `krige_get_factor_matrices`

```c
int krige_get_factor_matrices(int64_t handle,
    int npp, int p,
    double *L_out,             // [npp x npp], Fortran matrix storage
    double *kinv_out,          // [npp x max(1,p)], Fortran matrix storage
    double *schur_out)         // [max(1,p) x max(1,p)], Fortran matrix storage
```

Copies persistent factor matrices.  The upper triangle of `L_out` is the
Cholesky factor `U` such that `K = U' U`.

### `krige_get_factor_system`

```c
int krige_get_factor_system(int64_t handle,
    int npp, int p, int nvar,
    double *matA_out,          // [(npp+p) x (npp+p)], Fortran matrix storage
    double *rhsB_out)          // [nvar x (npp+p)], Fortran matrix storage
```

Copies the assembled system that produced the persistent factor.

## Weight Store

### `krige_free_weight_store`

```c
int krige_free_weight_store(int64_t handle)
```

Frees the in-memory weight store.

### `krige_get_weight_dims`

```c
int krige_get_weight_dims(int64_t handle,
    int *nmax_out,
    int *ngroups_out,
    int *nblock_out)
```

Returns the allocated weight-store dimensions.

### `krige_get_weight_nnear`

```c
int krige_get_weight_nnear(int64_t handle,
    int ngroups_c, int nblock_c,
    int *out)                 // [ngroups_c x nblock_c]
```

Copies neighbor counts.

### `krige_get_weight_inear`

```c
int krige_get_weight_inear(int64_t handle,
    int nmax_c, int ngroups_c, int nblock_c,
    int *out)                 // [nmax_c x ngroups_c x nblock_c]
```

Copies 1-based observation indices for every neighbor slot.

### `krige_get_weight_data`

```c
int krige_get_weight_data(int64_t handle,
    int nmax_c, int ngroups_c, int nvar_c, int nblock_c,
    double *out)              // [nmax_c x ngroups_c x nvar_c x nblock_c]
```

Copies kriging weights.

### `krige_get_weight_var`

```c
int krige_get_weight_var(int64_t handle,
    int nvar_c, int nblock_c,
    double *out)              // [nvar_c x nvar_c x nblock_c]
```

Copies stored kriging variances.

### `krige_set_weights`

```c
int krige_set_weights(int64_t handle,
    int nmax_c, int ngroups_c, int nvar_c, int nblock_c,
    const int *nnear_in,      // [ngroups_c x nblock_c]
    const int *inear_in,      // [nmax_c x ngroups_c x nblock_c]
    const double *weight_in,  // [nmax_c x ngroups_c x nvar_c x nblock_c]
    const int *order_in,      // [nblock_c]
    const double *var_in)     // [nvar_c x nvar_c x nblock_c]
```

Loads precomputed weights and enables weight reuse for the next solve.

## Spatial Thread Queries

### `krige_get_max_threads`

```c
void krige_get_max_threads(int *n)
```

Returns the OpenMP maximum thread count, or `1` when built without OpenMP.

### `krige_get_num_threads`

```c
void krige_get_num_threads(int *n)
```

Returns the OpenMP team size observed inside a parallel region, or `1` when
built without OpenMP.

## Space-Time Lifecycle

Space-time objects use the `krige_st_*` creation and setup functions below.
Shared solve, result, transform, factor, and weight-store operations still use
the unprefixed `krige_*` names (`krige_solve`, `krige_get_estimate`, and so on).

### `krige_st_create`

```c
int krige_st_create(int64_t *handle)
```

Allocates a space-time kriging object.

### `krige_st_destroy`

```c
int krige_st_destroy(int64_t *handle)
```

Finalizes and deallocates a space-time object and sets `*handle = 0`.

### `krige_st_initialize`

```c
int krige_st_initialize(int64_t handle,
    int nvar, int ndrift, int unbias, int nsim,
    int anisotropic_search, int weight_correction,
    int use_old_weight, int store_weight,
    int cross_validation, int write_mat,
    int neglect_error, int verbose,
    const char *weight_file,
    const double bounds[2],
    int seed)
```

Initializes a space-time object.  Spatial dimension is fixed internally at 3;
observation coordinates include a fourth time coordinate.

## Space-Time Observations, Gradients, And Model

### `krige_st_set_obs`

```c
int krige_st_set_obs(int64_t handle,
    int ivar, int nobs, int ndim,
    const double *coord,     // [nobs x (ndim+1)], last column is time
    const double *value,     // [nobs]
    const double *variance,  // [nobs]
    double sk_mean)
```

Sets space-time observations.  `ndim` is the spatial dimension; currently use
`ndim=3`, so `coord` has four values per point: `x,y,z,t`.

### `krige_st_set_grad`

```c
int krige_st_set_grad(int64_t handle,
    int ivar, int ngrad, int ndim,
    const double *coord1,       // [max(ngrad,1) x (ndim+1)]
    const double *coord2,       // [max(ngrad,1) x (ndim+1)]
    const double *grad_val,     // [max(ngrad,1)]
    const double *variance,     // [max(ngrad,1)]
    int ndrift_c,
    const double *drift_ext)    // [max(ngrad,1) x max(ndrift_c,1)]
```

Sets space-time gradient pairs.  Use `ngrad=0` to reset gradients.

### `krige_st_set_st_model`

```c
int krige_st_set_st_model(int64_t handle,
    const char *model,
    const char *transform,
    double at,
    double time_nugget,
    double time_sill,
    double k_ps)
```

Sets the global ST model.  `model` is `"sum_metric"` or `"product_sum"`;
`transform` is a variogram type for the temporal transform.

## Space-Time Variograms

### `krige_st_reset_vgm`

```c
int krige_st_reset_vgm(int64_t handle, int ivar, int jvar)
```

Clears ST variogram structures for a variable pair.

### `krige_st_set_vgm`

```c
int krige_st_set_vgm(int64_t handle,
    int ivar, int jvar,
    const char *vtype,
    double nugget, double sill,
    double a_major, double a_minor1, double a_minor2,
    double azimuth, double dip, double plunge,
    int is_product)
```

Adds a spatial marginal variogram structure to the ST model.

### `krige_st_set_vgm_temporal`

```c
int krige_st_set_vgm_temporal(int64_t handle,
    int ivar, int jvar,
    const char *vtype,
    double nugget,
    double sill,
    double at_k,
    int is_product)
```

Adds a temporal marginal variogram structure.

### `krige_st_set_vgm_joint_sills`

```c
int krige_st_set_vgm_joint_sills(int64_t handle,
    int ivar, int jvar,
    int n,
    const double *sills)       // [n]
```

Sets joint sills for an ST variable pair.

## Space-Time Grid, Simulation, And Search

### `krige_st_set_grid`

```c
int krige_st_set_grid(int64_t handle,
    int ngrid,
    const double *coord,        // [ngrid x 3]
    const double *time,         // [ngrid]
    const double *rangescale,   // [ngrid]
    const double *localnugget)  // [ngrid]
```

Sets point-kriging ST targets with spatial coordinates and time supplied
separately.

### `krige_st_set_grid_block`

```c
int krige_st_set_grid_block(int64_t handle,
    int nblocks,
    const double *coord,        // [nblocks x 3], block centroids
    const double *time,         // [nblocks]
    const int *nblockpnt,       // [nblocks]
    int npnts_total,
    const double *blockcoord,   // [npnts_total x 3], subnodes
    const double *blocktime,    // [npnts_total]
    const double *pointweight,  // [npnts_total]
    const double *rangescale,   // [nblocks]
    const double *localnugget)  // [nblocks]
```

Sets ST block-kriging targets.

### `krige_st_set_grid_drift`

```c
int krige_st_set_grid_drift(int64_t handle,
    int ndrift_c, int nblocks,
    const double *drift)        // [nblocks x ndrift_c]
```

Sets ST grid drift values and broadcasts them to all target variables.

### `krige_st_set_sim`

```c
int krige_st_set_sim(int64_t handle,
    int nblocks,
    const int *randpath,        // [nblocks], or NULL
    int nsim_c,
    const double *sample)       // [nblocks x nsim_c], or NULL
```

Configures ST simulation random path and samples.  `NULL` pointers let Fortran
generate values.  Caller-supplied samples fill the primary variable; additional
variables are zero-filled internally.

### `krige_st_set_search`

```c
int krige_st_set_search(int64_t handle,
    int ivar,
    double time_at,
    double anis1, double anis2,
    double azimuth, double dip, double plunge,
    int nmax,
    double maxdist,
    int sector_search)
```

Builds the ST KD-tree and configures the search neighborhood.  `time_at` scales
time into distance units for the KD-tree.  Negative `nmax` or `maxdist` keeps
the existing value.

## Space-Time Thread Queries

### `krige_st_get_max_threads`

```c
void krige_st_get_max_threads(int *n)
```

Returns the OpenMP maximum thread count, or `1` when built without OpenMP.

### `krige_st_get_num_threads`

```c
void krige_st_get_num_threads(int *n)
```

Returns the OpenMP team size observed inside a parallel region, or `1` when
built without OpenMP.
