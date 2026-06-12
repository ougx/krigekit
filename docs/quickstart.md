# Quick start

This page shows two paths to ordinary kriging: the one-shot convenience
function and the full class interface.  Both produce identical results.

## One-shot convenience function

The fastest way to get an estimate and kriging variance:

```python
import numpy as np
from krigekit import ordinary_kriging

obs_coord  = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
obs_value  = np.array([1.0, 2.0, 3.0, 4.0, 2.5])
grid_coord = np.mgrid[0:1.1:0.25, 0:1.1:0.25].reshape(2, -1).T  # 25 points

est, var = ordinary_kriging(
    obs_coord, obs_value, grid_coord,
    vgm_spec=dict(vtype="sph", nugget=0.0, sill=1.0, a_major=1.0, a_minor1=0.8),
    nmax=5,
)

print(est.shape, var.shape)   # (25,) (25,)
```

`est` is the kriging estimate at each grid point; `var` is the kriging variance.

## Full class interface

The `Kriging` class gives you full control over every step:

```python
import numpy as np
from krigekit import Kriging

obs_coord  = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
obs_value  = np.array([1.0, 2.0, 3.0, 4.0, 2.5])
grid_coord = np.mgrid[0:1.1:0.25, 0:1.1:0.25].reshape(2, -1).T

k = Kriging(ndim=2, nvar=1)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=5)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=1.0, a_minor1=0.8)
k.set_grid(coord=grid_coord)
k.set_search(ivar=1)
k.solve()

est, var = k.get_results()          # separate arrays
ra = k.get_result_array()           # structured array: x, y, estimate, variance
```

### Workflow order

```
Kriging(ndim, nvar)
  └─ set_obs(ivar, coord, value, nmax)      # load observations
  └─ set_vgm(ivar, jvar, vtype, ...)        # define variogram model
  └─ set_grid(coord)                        # define estimation targets
  └─ set_search(ivar)                       # build spatial search index
  └─ solve()                                # run kriging
  └─ get_results()                          # retrieve est, var as separate arrays
  └─ get_result_array()                     # retrieve as a structured NumPy array
```

:::{note}
For **Spatially Varying Anisotropy** (`varying_vgm=True`), `set_grid` must be
called before `set_vgm` because the number of blocks must be known first.
In standard mode the order of `set_vgm` and `set_grid` is flexible.
:::

## What's next?

- [Array conventions](array_conventions.md) — coordinate shapes and result layouts
- [Variogram models](variogram_models.rst) — types, parameters, nested structures
- [Ordinary kriging user guide](user_guide/ordinary_kriging.md) — anisotropy, reuse, and more
- [API reference](api/index.md) — full method signatures
