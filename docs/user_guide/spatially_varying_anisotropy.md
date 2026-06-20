# Spatially varying anisotropy

Spatially varying anisotropy (SVA) lets the variogram behaviour change from
one estimation block to the next.  Use it when the continuity direction,
effective range, or local uncertainty is not constant across the domain.

KrigeKit supports three levels of spatial variation:

- `rangescale`: a per-block multiplier on the effective variogram range.
- `localnugget`: a per-block additional nugget term.
- `varying_vgm=True` plus `set_vgm_block`: a full per-block variogram model,
  including block-specific ranges and rotation angles.

## Range scale

`rangescale` is the lightest-weight SVA option.  It scales the lag vector
before variogram evaluation:

```text
h_scaled = h / rangescale[ib]
```

Values greater than 1 make observations behave as if they are closer together,
which increases the effective range and usually lowers kriging variance.
Values between 0 and 1 shorten the effective range.

```python
import numpy as np
from krigekit import Kriging

rng = np.random.default_rng(42)
obs_coord = rng.uniform(0, 100, (80, 2))
obs_value = rng.normal(0.0, 1.0, 80)

x, y = np.mgrid[0:101:5, 0:101:5]
grid_coord = np.column_stack([x.ravel(), y.ravel()])

# Longer continuity in the eastern half of the grid.
rangescale = np.where(grid_coord[:, 0] > 50.0, 2.0, 0.75)

k = Kriging(ndim=2, nvar=1)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=24)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.05, sill=0.95, a_major=35.0)
k.set_grid(coord=grid_coord, rangescale=rangescale)
k.set_search(ivar=1)
k.solve()

est, var = k.get_results()
```

`rangescale` must have one value per point-kriging node.  For block kriging,
pass one value per block, not one value per quadrature sub-node.

## Local nugget

`localnugget` adds local uncertainty at each estimation block.  It is useful
for locations where the target support or measurement quality varies over
space, for example a zone where the mapped variable is known to be less
reliable.

```python
localnugget = np.zeros(len(grid_coord))
localnugget[grid_coord[:, 1] > 75.0] = 0.10

k = Kriging(ndim=2, nvar=1)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=24)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.05, sill=0.95, a_major=35.0)
k.set_grid(coord=grid_coord, localnugget=localnugget)
k.set_search(ivar=1)
k.solve()

est, var = k.get_results()
```

You can use `rangescale` and `localnugget` together:

```python
k.set_grid(
    coord=grid_coord,
    rangescale=rangescale,
    localnugget=localnugget,
)
```

Defaults are `rangescale=1.0` and `localnugget=0.0` for every block, so
omitting both gives ordinary global-variogram kriging.

## Per-block variograms

For full SVA, construct the object with `varying_vgm=True`, call `set_grid`,
then call `set_vgm_block` for each block.  Block indices are 1-based.

```python
k = Kriging(ndim=2, nvar=1, varying_vgm=True)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=24)
k.set_grid(coord=grid_coord)

for ib, (gx, gy) in enumerate(grid_coord, start=1):
    if gx < 50.0:
        # Western domain: NW-SE continuity.
        azimuth = 315.0
        a_major = 45.0
        a_minor = 18.0
    else:
        # Eastern domain: NE-SW continuity.
        azimuth = 45.0
        a_major = 70.0
        a_minor = 25.0

    k.set_vgm_block(
        ib=ib,
        ivar=1,
        jvar=1,
        vtype="sph",
        nugget=0.05,
        sill=0.95,
        a_major=a_major,
        a_minor1=a_minor,
        azimuth=azimuth,
    )

k.set_search(ivar=1)
k.solve()
est, var = k.get_results()
```

Call `set_vgm_block` multiple times for the same block to build a nested
model.  Product covariance groups are not supported for per-block variograms.

## Block kriging

Both `rangescale` and `localnugget` are also available on `set_grid_block`.
The arrays are sized by block:

```python
k.set_grid_block(
    coord=block_points,
    block_type=1,
    nblockpnt=nblockpnt,
    pointweight=pointweight,
    rangescale=block_rangescale,      # shape (nblock,)
    localnugget=block_localnugget,    # shape (nblock,)
)
```

## Choosing an SVA mode

| Need | Use |
|---|---|
| Regional range gets longer or shorter | `rangescale` |
| Local estimation uncertainty changes | `localnugget` |
| Continuity direction rotates over space | `varying_vgm=True` and `set_vgm_block` |
| Full per-block variogram parameters | `varying_vgm=True` and `set_vgm_block` |

`rangescale` and `localnugget` are usually the best starting point because they
reuse one global variogram model and only add per-block modifiers.  Use
per-block variograms when the anisotropy orientation or nested structures
really need to vary by location.

## Practical checks

- Keep `rangescale` positive.
- Make `rangescale` and `localnugget` arrays the same length as the number of
  point grid nodes, or the number of blocks for block kriging.
- For `set_vgm_block`, call `set_grid` first and use 1-based block indices.
- Start with smooth spatial fields for SVA parameters; abrupt changes can
  create sharp changes in variance and weights.

## See also

- {doc}`../auto_examples/s_ok2d_sva` - runnable SVA gallery example
- [Ordinary kriging](ordinary_kriging.md) - basic `Kriging` workflow
- [Variogram models](../variogram_models.rst) - model types and anisotropy
- [Performance tuning](performance.md) - `nthread`, `ncache`, and factor reuse
- [API reference](../api/index.md) - full `Kriging` class documentation
