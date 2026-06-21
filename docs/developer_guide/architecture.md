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
- `self%hcache`: one bounded, set-associative prepared-factor cache shared by
  all worker threads during a solve.
- `self%pf`: optional persistent cache on the kriging object, saved after the
  parallel loop.

The hot loop caches prepared factors only.  The assembled `matA` and `rhsB`
snapshots for inspection are copied to `self%pf` only during the post-loop save.

### Shared four-way hcache

`self%hcache` is a fixed-size, four-way set-associative cache allocated before
the OpenMP block loop and freed when `solve` exits.  Its principal arrays are:

```text
slot(way, bucket)     four t_factor_entry values per bucket
clock(bucket)         monotonically increasing local LRU timestamp
lock(bucket)          one OpenMP lock per bucket
```

The following example shows a 16-slot pool.  Each column is an independent
bucket and therefore an independent synchronization domain:

```text
                       self%hcache: 16 slots

                 bucket 1      bucket 2      bucket 3      bucket 4
               +----------+  +----------+  +----------+  +----------+
way 1          | entry 1  |  | entry 1  |  | entry 1  |  | entry 1  |
way 2          | entry 2  |  | entry 2  |  | entry 2  |  | entry 2  |
way 3          | entry 3  |  | entry 3  |  | entry 3  |  | entry 3  |
way 4          | entry 4  |  | entry 4  |  | entry 4  |  | entry 4  |
               +----------+  +----------+  +----------+  +----------+
LRU state      clock(1)       clock(2)       clock(3)       clock(4)
lock scope     lock(1)        lock(2)        lock(3)        lock(4)

hash A -------------> bucket 1: search/evict only these four ways
hash B -------------------------------------> bucket 3: operates concurrently
```

The number of buckets is `floor(effective_slots / 4)`.  Positive requested
sizes below four are promoted to four; larger non-multiples of four and sizes
reduced by the memory cap are rounded down to complete buckets.

`fhcache_hash_key` mixes the prepared system dimensions, local range and nugget
modifiers, per-group neighbour counts, and sorted neighbour indices.  The
bucket is selected by:

```text
bucket = 1 + modulo(hash, nbucket)
```

The complete lookup and insertion path is:

```text
                         worker thread
                              |
                              v
                    build current system key
                              |
                              v
                  +-------------------------+
                  | thread-local ctx%cache? |
                  +-------------------------+
                       | hit          | miss
                       v              v
                 reuse factors    hash the key
                                      |
                                      v
                              select one bucket
                                      |
                              blocking bucket lock
                                      |
                                      v
                         scan at most four ways
                            | hit          | miss
                            v              v
                  verify full key       unlock
                  copy to ctx%cache        |
                  update local LRU         v
                  unlock               factorize
                                            |
                                            v
                                  non-blocking insert lock
                                    | busy        | acquired
                                    v             v
                              skip insert    refresh same key,
                                             use empty way, or
                                             evict bucket LRU
```

The thread-local cache handles the hottest repeated-system case without shared
synchronization.  The shared cache is consulted only after that local check
misses.

Only the four entries in that bucket are searched.  A stored hash is a filter,
not proof of identity: `fcache_matches` verifies the complete key before any
factor is reused.  On a hit, the entry's prepared Cholesky or SSYTRF factors
are copied to the thread-private `ctx%cache`, and `last_used` is updated from
the bucket-local clock.

Insertion first refreshes an identical resident key.  Otherwise it selects the
first invalid way or the way with the smallest `last_used` value.  This is LRU
within one bucket; there is deliberately no global eviction list or global
lock.

Parallel synchronization is asymmetric:

- Lookup uses `omp_set_lock` and waits for the selected bucket.  Waiting can be
  cheaper than redundantly factorizing a system already being published.
- Insert uses `omp_test_lock`.  If the bucket is busy, insertion is skipped.
  The current block already has valid thread-local factors, so skipping changes
  only potential future reuse.
- Single-threaded execution bypasses lock calls.

Two workers can still miss and factorize the same system concurrently before
either publishes it.  This is benign duplication, not a data race.  All reads,
writes, LRU updates, refreshes, and evictions of shared entries occur while the
corresponding bucket lock is held.

Each shared entry stores only the cache key and prepared factors.  It does not
store the assembled `matA` or `rhsB`; those remain thread-local, while the
optional `self%pf` retains the inspectable assembled-system snapshot across
solve calls.

## C API registry

`kriging_capi_common.F90` provides the shared CAPI handle registry.  Registry
slots store `class(t_kriging_base)` pointers, so one registry can hold either a
`t_kriging` or a `t_kriging_st` object.

The spatial CAPI (`kriging_capi.F90`) and ST CAPI (`kriging_st_capi.f90`) remain
thin wrappers.  Each wrapper retrieves the base pointer from the shared registry
and uses `select type` to downcast to the concrete type expected by that API.
This avoids duplicated registry/error-string utilities while preserving typed
entry points for spatial and ST callers.

## Python variogram architecture

The Python variogram analysis code follows the same conceptual boundaries as
the Fortran variogram types, but it uses composition for space-time models:

```text
_VariogramModelBase
  +-- VariogramModel                 one spatial or temporal marginal
  +-- SpaceTimeVariogramModel        full ST cloud and coupling state
        |-- spatial: VariogramModel
        +-- temporal: VariogramModel

VariogramSystem                       multivariable direct/cross models
```

This matches `vgm_struct_st` in `variogram_st.f90`, which contains `cs` and
`ct` members rather than extending `vgm_struct`.

The implementation is split by responsibility:

| Python module | Responsibility | Fortran analogue |
|---|---|---|
| `variogram_kernels.py` | Kernel correlation/variogram functions | `corefunc_fn` |
| `variogram_component.py` | One flat theoretical component (`VgmComponent`) | `vgm_component` |
| `variogram_structure.py` | Nested/product theoretical model for one pair (`VgmStructure`) | `vgm_struct` |
| `variogram_structure_st.py` | Theoretical space-time coupling (`VgmStructureST`, `cs`/`ct`) | `vgm_struct_st` |
| `variogram_geometry.py` | Rotations and anisotropic lag distance | `vgm_aniso`, `rotation.f90` |
| `variogram_empirical.py` | Pair clouds, averaging, and directional analysis | Python analysis layer |
| `variogram_fitting.py` | Generic weighted marginal fitting | Python analysis layer |
| `variogram_plotting.py` | Variogram curves and maps | Python analysis layer |
| `variogram_base.py` | Observation and empirical-cache workflow | Shared model state |
| `variogram_model.py` | Marginal analysis workflow (owns a `VgmStructure`) | `vgm_struct` analysis |
| `variogram_st.py` | Product-sum and sum-metric coupling | `vgm_struct_st` |
| `variogram_system.py` | Multivariable/LMC workflow | Variogram matrix by variable pair |
| `variogram_observation.py` | `ObservationSet` (data + search config) | Observation/search state |
| `variogram_accessors.py` | Internal 1-based `vgm[i, j]` / `obs[i]` accessors | Variogram matrix / observation indexing |

`krigekit.variogram` is a compatibility facade, so existing helper and class
imports remain valid. New code may import the public classes directly
from `krigekit`.

### Theoretical model layer (`VgmComponent`, `VgmStructure`)

`VgmComponent` and `VgmStructure` are the *purely theoretical* model objects:
they own model type, sills, nugget, and anisotropy and evaluate covariance and
variogram values, but hold no observations, empirical clouds, or fit state.
`VgmStructure` owns an ordered list of `VgmComponent`s and applies the same
product-group rule as the engine -- a `product=True` component multiplies the
preceding one in covariance space, and each group is summed.

The Python side keeps the component **flat** -- the same field layout as
`Kriging.set_vgm` -- and `VgmComponent.to_flat_dict()` / `from_flat_dict()` are
the authoritative bridge to the C API; the Fortran engine regroups the
anisotropy fields into its `vgm_aniso` type after transfer.

`VariogramModel` owns a single `VgmStructure` (`model.structure`) and delegates
all theoretical evaluation (`covariance`, `variogram`, `calc_covariance`,
`calc_variogram`) and engine transfer to it -- it keeps no parallel component
list, and the legacy `_VgmComponent` has been removed.

`VgmStructureST` (`variogram_structure_st.py`) is the space-time analogue of
`VgmStructure`: a purely theoretical object holding a spatial structure (`cs`)
and a temporal structure (`ct`) plus the coupling parameters of a
product-sum or sum-metric model. It evaluates the joint semivariogram
(`calc_variogram(spatial_lag, temporal_lag)`) and emits engine specs
(`to_kriging_specs`), mirroring `vgm_struct_st`'s `cs`/`ct` composition rather
than extending `vgm_struct`. `SpaceTimeVariogramModel` owns one
(`model.structure`) and delegates all space-time theoretical evaluation and
engine-spec generation to it: the analysis wrapper composes the two
`VariogramModel` marginals and runs the empirical/fitting workflow, rebuilds the
owned `VgmStructureST` whenever its coupling parameters change, and a parameter
override (`calc_spacetime_variogram(..., params=...)`) builds a transient
structure from the same stored marginal types and anisotropy.

The space-time coupling methods live **only** on `SpaceTimeVariogramModel` -- the
old `VariogramModel.__getattr__` chain that lazily forwarded `*_spacetime_*`
calls to a hidden compatibility wrapper has been removed, so a marginal model no
longer silently grows space-time state. Construct a `SpaceTimeVariogramModel`
explicitly to fit a coupling.

Index convention: variable indices in a system (`obs[ivar]`, `vgm[ivar, jvar]`)
are 1-based labels matching the engine, while component indices within a
structure (`set_anisotropy(index=...)`) are 0-based Python positions.

### The `vgm[i, j]` accessor

`VariogramSystem.vgm` (a `_VgmAccessor` from `variogram_accessors.py`) presents
the per-pair structures as a symmetric matrix: `vgm[i, j]` and `vgm[j, i]`
return the same `VgmStructure`, indices are 1-based (0/negative/non-integer
raise a teaching error), and a fixed `nvar` is a strict upper bound while
`nvar=None` grows the variable count to the largest referenced index.

A pair is **materialized** once it has a stored structure and **configured**
only once that structure has at least one component. Reading `vgm[i, j]`
materializes an empty structure but does *not* configure it, so it never
becomes an LMC input or a transfer target; selection uses
`vgm.configured_pairs()`, never materialization. The system stores the
`VgmStructure` objects directly, so `vgm[i, j]` *is* the canonical per-pair
store; LMC/empirical fitting (`fit_pair`, `fit_lmc`, `set_markov_cross`) and
`apply` operate on it. The old per-pair `VariogramModel` wrapper and the
`system.models` dict are gone (`fit_pair` builds a transient `VariogramModel`
only to reuse the curve-fitting machinery, writing the fitted components back
into the canonical structure).

`system.obs[ivar]` (a `_ObservationAccessor`) mirrors this for single indices,
returning an `ObservationSet` that carries observation data plus the
per-variable neighbour-search configuration. The same 1-based and
materialized-vs-configured rules apply: an observation is configured only once
it has coordinates and values, and empirical/LMC pair selection uses
`obs.configured_indices()`. `set_search(...)` records search settings on the
set for transfer to the engine; neighbour selection itself stays in Fortran.

`VariogramSystem.apply(kriging, observations=True, variograms=True)` is
transactional: `validate_for` checks target `nvar`/`ndim`/`ndrift`, configured
pair indices, required auto-variograms for cross pairs, and structure ranges
*before* any mutation, so a failure leaves the target unchanged. It then
transfers observations in ascending variable order (drift immediately after
each base observation) and configured variograms in canonical upper-triangle
order, delegating to `ObservationSet.apply_to` and `VgmStructure.apply_to`.
`apply_observations` / `apply_variograms` expose each half independently.

### Unified fitting (`FitResult`)

Every fit entry point returns a `FitResult` (`variogram_fitting.py`) rather than
mutating-and-returning `self` or a bare params tuple. `FitResult` carries the
fitted `target`, the flat `params`, the optimizer `cov`/`optimizer`,
goodness-of-fit `metrics`, the fitted-point count `nobs`, and `param_labels`
(one `(component, vtype, param)` triple per parameter). `FitResult.summary()`
renders a labelled table with per-parameter `value`, `variance`, `std_err`, and
a two-sided Wald `p_value` (Student-t with `nobs - nparams` degrees of freedom);
the p-values are heuristic because binned variogram points are not independent.

The same contract spans every level:

| Entry point | Fit |
|---|---|
| `VgmStructure.fit` | isotropic sills/ranges/nugget for one pair |
| `VariogramModel.fit` / `.fit_anisotropy` | marginal isotropic / directional ranges |
| `VariogramSystem.fit_pair` / `.fit_lmc` | per-pair and joint LMC coregionalization |
| `SpaceTimeVariogramModel.fit(model=...)` | `"product_sum"` or `"sum_metric"` coupling |

Constrained/weighted joint fits (LMC, space-time) populate `optimizer` and
labelled `params` but leave `variance`/`p_value` as `NaN`, since a curve-fit
covariance is not meaningful under those constraints.
