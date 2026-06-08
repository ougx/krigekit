# Co-kriging

Co-kriging estimates a **primary variable** at unsampled locations by jointly
conditioning on observations of one or more **secondary variables**.  The
secondary variables are cheaper or denser to measure and are spatially
correlated with the primary.

A common example: estimating clay fraction (primary, sparse borehole data)
using airborne electromagnetic resistivity (secondary, dense grid coverage).

---

## Minimal example

```python
import numpy as np
from pykriging import Kriging

# --- observations ---
obs1_coord = ...   # primary variable coords, shape (n1, ndim)
obs1_value = ...   # primary variable values,  shape (n1,)
obs2_coord = ...   # secondary variable coords, shape (n2, ndim)
obs2_value = ...   # secondary variable values,  shape (n2,)
grid_coord = ...   # estimation grid,            shape (ngrid, ndim)

k = Kriging(ndim=3, nvar=2, std_ck=True)

k.set_obs(ivar=1, coord=obs1_coord, value=obs1_value, nmax=50)
k.set_obs(ivar=2, coord=obs2_coord, value=obs2_value, nmax=50)

k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.00, sill=0.12, a_major=5000)
k.set_vgm(ivar=2, jvar=2, vtype="sph", nugget=0.00, sill=0.068, a_major=5000)
k.set_vgm(ivar=1, jvar=2, vtype="sph", nugget=0.05, sill=0.04, a_major=5000)

k.set_grid(coord=grid_coord)

k.set_search(ivar=1)
k.set_search(ivar=2)

k.solve()
est, var = k.get_results()   # est shape: (ngrid, nvar)
primary_est = est[:, 0]      # primary variable estimate
```

---

## Variogram models

Co-kriging requires a variogram for each variable pair — call `set_vgm` once
per pair:

| Call | What it models |
|---|---|
| `set_vgm(1, 1, ...)` | Auto-variogram of the primary variable |
| `set_vgm(2, 2, ...)` | Auto-variogram of the secondary variable |
| `set_vgm(1, 2, ...)` | Cross-variogram (spatial co-variation between variables) |

The cross-variogram must satisfy the **Linear Model of Coregionalisation (LMC)**
constraint for each nested structure *k*:

```
b₁₂ₖ² ≤ b₁₁ₖ × b₂₂ₖ
```

If the constraint is violated the covariance matrix is not positive
semi-definite and the kriging system may fail.  pyKriging does not enforce
this automatically; use `neglect_error=True` to continue past failures and
inspect the resulting `NaN` blocks.

---

## Unbiasedness formulation: `std_ck`

The `std_ck` flag controls how the unbiasedness constraint is imposed in the
augmented kriging system.  It only matters when `nvar > 1` and `unbias = 1`
(ordinary co-kriging).

### `std_ck=True` — standard co-kriging (default)

The augmented system for estimating the primary variable has **separate**
constraints per variable:

```
[ C₁₁  C₁₂  1   0 ] [ w₁ ]   [ c₀₁ ]
[ C₂₁  C₂₂  0   1 ] [ w₂ ] = [ c₀₂ ]
[  1ᵀ   0ᵀ  0   0 ] [ μ₁ ]   [  1  ]
[  0ᵀ   1ᵀ  0   0 ] [ μ₂ ]   [  0  ]
```

This enforces Σw₁ = 1 and Σw₂ = 0 independently.  The estimate is invariant
to shifts in the secondary variable's global mean.  **Results match gstat and
ISATIS.**

### `std_ck=False` — Isaaks & Srivastava

A single combined constraint is used: Σw₁ + Σw₂ = 1.  A local-mean
correction is applied post-solve to compensate for differing means between
variables.  This matches legacy GSLIB behaviour.

> **Singularity note.** Standard co-kriging uses an `nvar × nvar` Schur
> complement in the solver, whereas the Isaaks formulation uses a scalar (1×1)
> that is always positive.  Standard co-kriging is therefore more susceptible
> to near-singular systems, especially when a variable has no neighbours within
> the search radius at a given grid point.  If you encounter many `NaN` blocks,
> try a larger `nmax`, a wider `maxdist`, or switch to `std_ck=False` as a
> diagnostic.

---

## Retrieving results

`get_results()` returns `(est, var)` for all variables simultaneously:

```python
est, var = k.get_results()
# est[ib, ivar-1]             — estimate at block ib for variable ivar
# var[ib, ivar-1, jvar-1]     — conditional covariance between ivar and jvar at block ib
# var[ib, ivar-1, ivar-1]     — kriging variance for variable ivar
```

---

## Heterotopic data

The two variable datasets do **not** need to be co-located.  Each variable has
its own search tree and neighbour set.  A grid point that has no secondary
neighbours within its search radius will solve using only primary observations
(the secondary constraint row contributes no data).

---

## Matching gstat output

To reproduce `gstat`/`predict.gstat` results:

```r
# R / gstat
g <- gstat(NULL, "var1", z~1, locations=~x+y+z, data=obs1, model=m1, nmax=50)
g <- gstat(g,    "var2", z~1, locations=~x+y+z, data=obs2, model=m2, nmax=50)
g <- gstat(g, id=c("var1","var2"), model=mc)
pred <- predict(g, grid)
```

```python
# Python / pyKriging
k = Kriging(ndim=3, nvar=2, std_ck=True)   # std_ck=True matches gstat
k.set_obs(ivar=1, coord=obs1[["x","y","z"]].values, value=obs1.z, nmax=50)
k.set_obs(ivar=2, coord=obs2[["x","y","z"]].values, value=obs2.z, nmax=50)
k.set_vgm(ivar=1, jvar=1, **m1_params)
k.set_vgm(ivar=2, jvar=2, **m2_params)
k.set_vgm(ivar=1, jvar=2, **mc_params)
k.set_grid(coord=grid[["x","y","z"]].values)
k.set_search(ivar=1)
k.set_search(ivar=2)
k.solve()
est, var = k.get_results()
# est[:, 0] ≈ pred["var1.pred"]
```

---

## Universal co-kriging (KED)

When `ndrift > 0`, call `set_obs_drift` for each variable's observations and
`set_grid_drift` for the estimation grid:

```python
k = Kriging(ndim=3, nvar=2, ndrift=1, std_ck=True)

k.set_obs(ivar=1, coord=obs1_coord, value=obs1_value, nmax=50)
k.set_obs_drift(ivar=1, drift=obs1_elevation)   # shape (n1, 1)

k.set_obs(ivar=2, coord=obs2_coord, value=obs2_value, nmax=50)
k.set_obs_drift(ivar=2, drift=obs2_elevation)   # shape (n2, 1)

k.set_vgm(...)   # set variograms as usual

k.set_grid(coord=grid_coord)
# ivar=None broadcasts the same drift to all target variables
k.set_grid_drift(drift=grid_elevation, ivar=None)   # shape (ngrid, 1)

k.set_search(ivar=1)
k.set_search(ivar=2)
k.solve()
```

If the external drift has a **different effect on each variable**, pass separate
arrays with an explicit `ivar`:

```python
k.set_grid_drift(drift=grid_elevation_var1, ivar=1)
k.set_grid_drift(drift=grid_elevation_var2, ivar=2)
```

---

## See also

- [Variogram models](../variogram_models.rst) — model types, nesting, anisotropy
- [Array conventions](../array_conventions.md) — coordinate and result shapes
- [C API reference](../developer_guide/capi.md) — `krige_initialize`, `krige_set_grid_drift`
- [API reference](../api/index.md) — full `Kriging` class documentation
