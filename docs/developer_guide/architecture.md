# Fortran architecture

The Fortran solver is organized around an abstract base type plus two concrete
specializations:

```text
t_kriging_base          src/libkriging/kriging_base.F90
  |
  +-- t_kriging         src/libkriging/kriging.F90
  |
  +-- t_kriging_st      src/libkriging/kriging_st.F90
```

`t_kriging_base` owns the common state and the shared solve framework.  The
spatial and space-time types inherit from it and provide the pieces that depend
on covariance geometry, neighbour search, and result variance.

## Base layer

`kriging_base.F90` contains the code that is independent of the concrete
covariance type:

- Common options, dimensions, observation/grid/block pointers, gradients,
  weight storage, and the persistent factor cache.
- Shared data containers: `t_data`, `t_obsgrid`, `t_grid`, `t_blockgrid`, and
  `t_grad`.
- The unified per-thread workspace, `t_kriging_ctx`, used by both spatial and
  space-time solves.
- Common API methods such as `initialize`, `set_obs`, `set_grid_*`, `set_sim`,
  weight storage, persistent-factor accessors, and string formatting.
- The non-overridable `solve` template method and shared linear-system solve.

The base type uses deferred procedures for the behavior that must remain
type-specific:

- `init`
- `prepare`
- `search_neighbors`
- `assemble_lhs`
- `assemble_rhs`
- `calc_variance`
- `tostr_vgm`
- `finalize`

This keeps the block loop, cache logic, drift handling, weight storage, SGSIM
bookkeeping, and persistent-factor inspection in one implementation while still
letting each concrete type define its own covariance and search model.

## Spatial kriging

`t_kriging` in `kriging.F90` is the spatial specialization.  It stores spatial
`vgm_struct` variograms and implements:

- Spatial grid setup and optional per-block variogram allocation.
- Spatial search-tree construction in `set_search`.
- Spatial neighbour search.
- Spatial covariance assembly for the LHS and RHS.
- Spatial conditional variance.

Common arrays such as `obs`, `grid`, `block`, `grad`, and `pf` live on the base
type; `t_kriging` adds only the spatial variogram state.

## Space-time kriging

`t_kriging_st` in `kriging_st.F90` is the space-time specialization.  It stores
`vgm_struct_st` variograms and ST model parameters, including the temporal
transform used by the ST search coordinate.

Space-time coordinates use the same shared data containers as spatial kriging:
`ndim` remains the spatial dimension, while `nlag = ndim + 1` and
`coord(nlag,:)` stores native time.  `set_search` builds an `nlag`-dimensional
KD-tree after transforming the time coordinate for search.  Covariance assembly
uses `coord(1:ndim,:)` for spatial lag and `coord(nlag,:)` for temporal lag.

## Solve template

Both concrete types call the same `t_kriging_base%solve` method:

```text
pre_solve
prepare                      concrete hook
parallel block loop
  estimate_block
    assemble_linear_system
      search_neighbors       concrete hook
      assemble_lhs/rhs       concrete hooks
    solve_linear_system      shared factorization/cache path
    calc_variance            concrete hook
post-loop persistent-cache save
```

The cache layers are therefore shared by spatial and space-time kriging:

- `ctx%cache`: one single-entry prepared-factor cache per worker thread.
- `ctx%hcache`: one bounded multi-slot prepared-factor cache per worker thread.
- `self%pf`: optional persistent cache on the kriging object, saved after the
  parallel loop.

The hot loop caches prepared factors only.  The assembled `matA` and `rhsB`
snapshots for inspection are copied to `self%pf` only during the post-loop save.

## C API registry

`kriging_capi_common.F90` provides the shared CAPI handle registry.  Registry
slots store `class(t_kriging_base)` pointers, so one registry can hold either a
`t_kriging` or a `t_kriging_st` object.

The spatial CAPI (`kriging_capi.F90`) and ST CAPI (`kriging_st_capi.f90`) remain
thin wrappers.  Each wrapper retrieves the base pointer from the shared registry
and uses `select type` to downcast to the concrete type expected by that API.
This avoids duplicated registry/error-string utilities while preserving typed
entry points for spatial and ST callers.
