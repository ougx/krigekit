# Multiple Indicator Kriging and SIS

{py:class}`~krigekit.IndicatorKriging` implements **Multiple Indicator Kriging
(MIK)** for probability estimation and **Sequential Indicator Simulation (SIS)**
for stochastic categorical simulation.

## Concepts

### Indicator variables

For K mutually exclusive categories, each sample location is encoded as K
binary indicator variables:

```
I_k(x) = 1   if category k is observed at x
I_k(x) = 0   otherwise
```

The K indicators sum to 1 at every point (`Σ I_k = 1`).  Kriging each I_k
yields an estimate of the local conditional probability `P(category = k | data)`.

### Theoretical variogram sills

The indicator variance for category k is `p_k (1 − p_k)`, where `p_k` is the
proportion of category k in the data.  The theoretical cross-variogram sill
between I_k and I_l is `−p_k · p_l` (negative, because the categories are
mutually exclusive).  In practice, positive cross-sill approximations are used
and `post_solve` normalisation produces a valid probability simplex.

## Estimation (MIK)

```python
import numpy as np
from krigekit import IndicatorKriging

ik = IndicatorKriging(ncat=3, ndim=2)

ik.set_categorical_obs(
    coord=obs_coord,       # (nobs, 2) array
    categories=obs_labels, # string or integer category per sample
    category_labels=["A", "B", "C"],
    nmax=20,
)

ik.set_indicator_vgm(
    vtype="sph", nugget=0.02, sill=0.20,
    a_major=500.0, a_minor1=100.0,
    azimuth=0.0,
)

ik.set_grid(coord=grid_coord)

for k in range(1, 4):
    ik.set_search(ivar=k, anis1=100.0/500.0)

ik.solve()

probs, var = ik.get_results()  # probs.shape == (ngrid, 3)
del ik
```

`probs[:, k]` is the estimated probability of category k at each grid node,
normalised to sum to 1.

## Simulation (SIS)

Pass `nsim > 0` and call {py:meth}`~krigekit.IndicatorKriging.set_sim` before
solving.  Each realisation visits grid nodes in a random sequential order and
draws a category by inverting the local conditional CDF.

```python
ik = IndicatorKriging(ncat=3, ndim=2, nsim=50, seed=42)

ik.set_categorical_obs(coord=obs_coord, categories=obs_labels,
                       category_labels=["A", "B", "C"], nmax=20)
ik.set_indicator_vgm(vtype="sph", nugget=0.02, sill=0.20,
                     a_major=500.0, a_minor1=100.0)
ik.set_grid(coord=grid_coord)
ik.set_sim()                  # must be called after set_grid

for k in range(1, 4):
    ik.set_search(ivar=k, anis1=100.0/500.0)

ik.solve()

sims, _ = ik.get_results()   # shape (ngrid, 3, 50) — one-hot encoded
cat_idx = np.argmax(sims, axis=1)  # (ngrid, 50) — integer category index
del ik
```

## Cross-variogram strategies

{py:meth}`~krigekit.IndicatorKriging.set_indicator_vgm` sets all K² variogram
pairs in one call.  The `cross` parameter controls how off-diagonal (cross)
sills are derived:

| `cross` | Auto sill | Cross sill | When to use |
|---|---|---|---|
| `"same"` *(default)* | `sill` for all k | `sill` | Simplest; good starting point |
| `"proportional"` | `p_k (1 − p_k)` | `√(s_k · s_l)` | LMC-valid; needs `proportions` |
| `"independent"` | `p_k (1 − p_k)` or `sill` | 0 | Most conservative |

### Uniform sill

```python
ik.set_indicator_vgm(vtype="sph", nugget=0.02, sill=0.19,
                     a_major=500, a_minor1=80, azimuth=90,
                     cross="same")
```

### Proportional sills (LMC)

Auto sills calibrated to the indicator variance; cross sills set to their
geometric mean so the coregionalisation matrix is positive-definite.

```python
props = np.array([0.18, 0.23, 0.21, 0.38])   # observed p_k per category
ik.set_indicator_vgm(vtype="sph", nugget=0.02, sill=0.19,
                     a_major=500, a_minor1=80, azimuth=90,
                     cross="proportional", proportions=props)
```

### Independent (no cross-coupling)

Cross-variogram sills are set to zero — equivalent to running K separate
ordinary kriging systems.

```python
ik.set_indicator_vgm(vtype="sph", nugget=0.02, sill=0.19,
                     a_major=500, a_minor1=80, azimuth=90,
                     cross="independent", proportions=props)
```

## Co-kriging MIS

Secondary continuous variables can be added by setting `nvar = ncat + M`:

```python
ik = IndicatorKriging(ncat=3, nvar=4, ndim=2)  # 3 indicators + 1 secondary

ik.set_categorical_obs(coord=obs_coord, categories=obs_labels,
                       category_labels=["A", "B", "C"], nmax=20)
ik.set_obs(ivar=4, coord=sec_coord, value=sec_val)  # secondary variable

for iv in range(1, 5):
    for jv in range(1, 5):
        ik.set_vgm(ivar=iv, jvar=jv, vtype="sph", sill=0.2,
                   a_major=500, a_minor1=100)

ik.set_grid(coord=grid_coord)
for k in range(1, 5):
    ik.set_search(ivar=k, anis1=100.0/500.0)

ik.solve()
probs, var = ik.get_results()   # shape (ngrid, 3) — secondary excluded
```

The secondary variable contributes to kriging weights but is excluded from
the CDF draw and probability normalisation.

## Variogram orientation

In krigekit the default variogram major axis is aligned with the **Y** axis.
For horizontal stratigraphy (long range along X), pass the same `azimuth` to
**both** `set_indicator_vgm` and `set_search`.  Passing it only to `set_search`
leaves the variogram ellipse pointing the wrong way and produces vertical patches
in the simulated images.

```python
AZIMUTH = 90.0   # rotate major axis from Y → X

ik.set_indicator_vgm(..., azimuth=AZIMUTH)        # variogram ellipse
for k in range(1, ncat + 1):
    ik.set_search(ivar=k, anis1=anis1, azimuth=AZIMUTH)  # search ellipse
```

## Gallery example

The {doc}`../auto_examples/s_sis_lithofacies` gallery example demonstrates
MIS on a 2-D lithofacies outcrop dataset, comparing the uniform-sill and
proportional-sill strategies side-by-side.
