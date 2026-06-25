# Kriging weight reuse

Kriging is a linear estimator: once the neighbour sets and kriging weights have
been computed for a fixed grid, variogram, search setup, and observation
geometry, the same weights can be applied to new observation values.  Weight
reuse avoids rebuilding and refactoring the kriging system when only the data
values change.

Use weight reuse for workflows such as:

- repeated estimates on the same grid for many realisations or scenarios;
- sensitivity runs where observation values change but coordinates do not;
- saving kriging weights once, then replaying them later from a file;
- inspecting the selected neighbours and weights used for each block.

## Store weights in memory

Pass `store_weight=True` to the constructor, solve normally, then call
`get_weights()`.

```python
import numpy as np
from krigekit import Kriging

rng = np.random.default_rng(7)
obs_coord = rng.uniform(0, 100, (50, 2))
obs_value = rng.normal(10.0, 2.0, 50)
grid_coord = np.mgrid[0:101:10, 0:101:10].reshape(2, -1).T

k = Kriging(ndim=2, nvar=1, store_weight=True)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.05, sill=0.95, a_major=40.0)
k.set_grid(coord=grid_coord)
k.set_search(ivar=1, nmax=16)
k.solve()

est, var = k.get_results()
weights = k.get_weights()
```

For a single variable, `weights` contains:

| Key | Shape | Meaning |
|---|---|---|
| `nnear` | `(nblock, ngroups)` | Active neighbour count per block/group |
| `inear` | `(nblock, ngroups, nmax)` | 1-based observation indices |
| `weight` | `(nblock, ngroups, nmax)` | Kriging weights |
| `variance` | `(nblock,)` | Stored kriging variance |

Entries beyond `nnear[ib, ig]` are zero-padded.  `inear` uses Fortran's
1-based indexing, so subtract 1 before indexing Python arrays.

```python
ib = 0
nn = weights["nnear"][ib, 0]
idx = weights["inear"][ib, 0, :nn] - 1
wts = weights["weight"][ib, 0, :nn]

manual_estimate = float(np.dot(wts, obs_value[idx]))
```

For co-kriging, `weight` has shape `(nblock, nvar, ngroups, nmax)`.

## Reuse weights in memory

Create a second object with the same grid, observation coordinates, search
setup, and variogram model.  Load the stored dictionary with `set_weights()`
before calling `solve()`.

```python
new_obs_value = obs_value + 1.5

k2 = Kriging(ndim=2, nvar=1, use_old_weight=True)
k2.set_obs(ivar=1, coord=obs_coord, value=new_obs_value)
k2.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.05, sill=0.95, a_major=40.0)
k2.set_grid(coord=grid_coord)
k2.set_search(ivar=1, nmax=16)
k2.set_weights(weights)
k2.solve()

est2, var2 = k2.get_results()
```

The estimates are recomputed from the new observation values and the old
weights.  The variance is restored from `weights["variance"]` when that key is
present.  If you pass a dictionary without `variance`, the replayed variance is
zero-filled.

## Store weights to a file

To persist weights across Python sessions, pass both `store_weight=True` and a
`weight_file` path.  The file is written during `solve()`.

```python
weight_file = "ok_weights.fac"

k = Kriging(ndim=2, nvar=1, store_weight=True, weight_file=weight_file)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.05, sill=0.95, a_major=40.0)
k.set_grid(coord=grid_coord)
k.set_search(ivar=1, nmax=16)
k.solve()
```

Read the saved weights back with `use_old_weight=True`.

```python
k_replay = Kriging(ndim=2, nvar=1, use_old_weight=True, weight_file=weight_file)
k_replay.set_obs(ivar=1, coord=obs_coord, value=new_obs_value)
k_replay.set_grid(coord=grid_coord)
k_replay.solve()

est_replay, var_replay = k_replay.get_results()
```

When replaying from a file, the variogram and search definitions are read from
the stored weights, so the replay object only needs the observation values and
the grid.

## What must stay the same

The stored weights are only valid for the geometry and model that produced
them.  Keep these unchanged between the store and replay run:

- observation coordinates and variable ordering;
- grid coordinates and block definitions;
- `nmax`, search settings, and sector-search choices;
- variogram model, including anisotropy and any SVA modifiers;
- co-kriging variable count and ordering;
- SGSIM path and random samples, if replaying simulations.

Observation values may change.  That is the main use case.

## SGSIM and co-simulation

Weight reuse also works with SGSIM and joint co-simulation.  In simulation
mode, the weight store includes the random visiting order and the simulation
groups in addition to observation groups.  To reproduce a simulation replay,
use either the same explicit `randpath` and `sample` arrays, or construct both
objects with the same seed and call `set_sim()` in the same order.

```python
k1 = Kriging(ndim=2, nvar=1, nsim=10, seed=42, store_weight=True)
# set_obs, set_grid, set_vgm ...
k1.set_sim()
k1.set_search(ivar=1)
k1.solve()
w = k1.get_weights()

k2 = Kriging(ndim=2, nvar=1, nsim=10, seed=42, use_old_weight=True)
# same setup ...
k2.set_sim()
k2.set_search(ivar=1)
k2.set_weights(w)
k2.solve()
```

## Memory management

The in-memory store can be released explicitly:

```python
k.free_weight_store()
```

After freeing it, `get_weights()` will raise an error until another
`store_weight=True` solve allocates a new store.

## Common pitfalls

- Calling `get_weights()` without `store_weight=True` raises an error.
- `store_weight=True` and `use_old_weight=True` are mutually exclusive on the
  same object.
- `set_weights()` is incompatible with an object constructed with
  `store_weight=True`; create a replay object instead.
- The stored `inear` indices are 1-based.
- Reused weights are not a substitute for refitting after observation
  coordinates, grid geometry, search settings, or variogram parameters change.

## See also

- [Performance tuning](performance.md) - factor caching and OpenMP tuning
- [Ordinary kriging](ordinary_kriging.md) - base workflow
- [Co-kriging](cokriging.md) - multi-variable workflow
- [Sequential Gaussian simulation](sgsim.md) - simulation workflow
- [API reference](../api/index.md) - full `Kriging` class documentation
