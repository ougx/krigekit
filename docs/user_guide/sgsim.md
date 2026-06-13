# Sequential Gaussian simulation

Sequential Gaussian simulation (SGSIM) generates conditional realisations of a
Gaussian random field.  Unlike kriging — which returns a single smoothed
estimate and a variance — simulation produces multiple equiprobable maps that
honour the data and reproduce the variogram, making it suitable for uncertainty
quantification and as input to flow / transport models.

## Minimal example

Set `nsim > 0` on the constructor and call `set_sim()` before `solve()`:

```python
import numpy as np
from krigekit import Kriging

rng = np.random.default_rng(0)
obs_coord  = rng.uniform(0, 100, (60, 2))
obs_value  = rng.normal(5.0, 1.0, 60)
grid_coord = np.mgrid[0:101:2, 0:101:2].reshape(2, -1).T

k = Kriging(nsim=20, seed=42)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=30)
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=40.0)
k.set_grid(coord=grid_coord)
k.set_sim()
k.set_search(ivar=1)
k.solve()
sims, _ = k.get_results()          # shape (ngrid, 20)
```

Each `sims[:, i]` is one realisation.  Realisations honour the data (a node that
coincides with an observation reproduces it) and reproduce the input covariance
model; different seeds give independent ensembles, while the **same seed
reproduces the realisations bit-for-bit** across platforms.

A one-call convenience function is also available:

```python
from krigekit import sequential_gaussian_simulation

sims = sequential_gaussian_simulation(
    obs_coord, obs_value, grid_coord,
    vgm_spec=dict(vtype="sph", sill=1.0, a_major=40.0),
    nsim=20, nmax=30, seed=42,
)
```

## Normal-score transform

SGSIM assumes a multiGaussian model, but environmental variables are often
strongly non-Gaussian (concentrations, hydraulic conductivity, percentages).
The standard practice is to transform the data to **normal scores**, simulate in
Gaussian space, and back-transform the realisations to data units.

krigekit performs this transform **inside the engine** (behind the C API), so it
is applied consistently from every client language and the realisations stay
bit-reproducible.  Enable it with `set_nscore()` after `set_obs()`:

```python
k = Kriging(nsim=20, seed=42)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=30)
k.set_nscore(ivar=1)                       # data -> normal scores
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=40.0)  # unit-sill nscore variogram
k.set_grid(coord=grid_coord)
k.set_sim()
k.set_search(ivar=1)
k.solve()
sims, _ = k.get_results()                  # realisations back-transformed to data units
```

Two things to keep in mind:

- **Fit the variogram on the normal scores** (unit sill), not on the raw data —
  the simulation runs in Gaussian space.
- `set_nscore` requires `nsim > 0`; it is a simulation transform.

### Tail extrapolation and bounds

The back-transform maps each simulated score through the data's empirical CDF.
For scores beyond the smallest / largest datum it extrapolates into the tails,
bounded by `zmin` / `zmax`.  These default to the data minimum / maximum (no
extrapolation beyond the observed range):

```python
k.set_nscore(
    ivar=1,
    zmin=0.0, zmax=200.0,             # physical bounds for the tails
    ltail="linear",                   # lower-tail model
    utail="hyperbolic", utpar=1.5,    # heavier upper tail (positive data only)
)
```

| Tail model | Notes |
|---|---|
| `"linear"` *(default)* | straight line between the extreme datum and `zmin` / `zmax` |
| `"power"` | `ltpar` / `utpar` shape parameter controls curvature |
| `"hyperbolic"` | upper tail only; requires strictly positive data |

### Declustering weights

If the data are spatially clustered, pass per-observation declustering weights
so the transform reproduces the *declustered* distribution:

```python
k.set_nscore(ivar=1, weights=decluster_weights)   # length == nobs
```

## See also

- {doc}`../auto_examples/s_ok2d_sgsim` — runnable SGSIM gallery example
- [Performance tuning](performance.md) — `nthread`, `ncache`, the factor cache
- [Array conventions](../array_conventions.md) — coordinate and result shapes
- [API reference](../api/index.md) — full `Kriging` class documentation
