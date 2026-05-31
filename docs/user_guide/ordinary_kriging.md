# Ordinary kriging

Ordinary kriging (OK) estimates an unknown value at a target location as a
weighted average of nearby observations, with weights chosen so that the
estimator is unbiased and has minimum variance.

## Minimal example

```python
import numpy as np
from pykriging import ordinary_kriging

rng = np.random.default_rng(42)
obs_coord  = rng.uniform(0, 100, (50, 2))
obs_value  = rng.normal(5.0, 1.0, 50)
grid_coord = np.mgrid[0:101:5, 0:101:5].reshape(2, -1).T  # 21×21 = 441 points

est, var = ordinary_kriging(
    obs_coord, obs_value, grid_coord,
    vgm_spec=dict(vtype="sph", nugget=0.1, sill=0.9, a_major=40.0),
    nmax=20,
)
# est.shape → (441,)   var.shape → (441,)
```

## Simple kriging

Simple kriging (SK) treats the mean as known rather than estimating it from
the data.  Pass `unbias=0` and a `sk_mean`:

```python
from pykriging import Kriging

k = Kriging(ndim=2, nvar=1, unbias=0)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=20,
          sk_mean=float(obs_value.mean()))
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.1, sill=0.9, a_major=40.0)
k.set_grid(coord=grid_coord)
k.set_search(ivar=1)
k.solve()
est, var = k.get_results()
```

## Anisotropic variogram

Specify different ranges along each axis.  `azimuth` rotates the major axis
clockwise from North (degrees):

```python
from pykriging import Kriging

k = Kriging(ndim=2, nvar=1)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=20)
k.set_vgm(ivar=1, jvar=1,
          vtype="sph",
          nugget=0.05, sill=0.95,
          a_major=80.0, a_minor1=30.0,  # 8:3 anisotropy ratio
          azimuth=30.0)                  # NNE–SSW orientation
k.set_grid(coord=grid_coord)
k.set_search(ivar=1)
k.solve()
est, var = k.get_results()
```

## Limiting the search neighbourhood

`nmax` controls the maximum number of neighbours used per kriging system.
A smaller `nmax` is faster but may reduce accuracy in sparse areas.
`maxdist` adds a hard distance cutoff:

```python
k.set_obs(ivar=1, coord=obs_coord, value=obs_value,
          nmax=15, maxdist=50.0)
```

## Result clipping

Clip estimates to a physically meaningful range with `bounds`:

```python
k = Kriging(bounds=[0.0, 100.0])
```

Values outside the range are clamped after kriging, which avoids negative
estimates for strictly positive quantities (e.g. porosity, concentration).

## Reusing a Kriging object

You can call `set_obs`, `set_vgm`, `set_grid`, `set_search`, and `solve`
again on the same object to estimate a new dataset without recreating it.
Use `append=False` on `set_vgm` to replace the previous variogram model:

```python
k = Kriging(ndim=2, nvar=1)

for obs_c, obs_v in dataset_sequence:
    k.set_obs(ivar=1, coord=obs_c, value=obs_v, nmax=20)
    k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=40.0,
              append=False)     # ← reset variogram each iteration
    k.set_grid(coord=grid_coord)
    k.set_search(ivar=1)
    k.solve()
    est, var = k.get_results()
```

## Cross-validation

Leave-one-out cross-validation reuses the same workflow, switching the grid
to CV mode:

```python
k = Kriging(ndim=2, nvar=1)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=20)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.1, sill=0.9, a_major=40.0)
k.set_grid_cv(ivar=1)           # estimation targets = leave-one-out positions
k.set_search(ivar=1)
k.solve()
cv_est, cv_var = k.get_results()
```

`cv_est[i]` is the kriging prediction at observation `i` using all other
observations.

## See also

- [Variogram models](../variogram_models.md) — model types, nesting, anisotropy
- [Array conventions](../array_conventions.md) — coordinate and result shapes
- [API reference](../api/index.md) — full `Kriging` class documentation
