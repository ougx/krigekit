# Multiple Indicator Kriging and SIS

{py:class}`~krigekit.IndicatorKriging` implements **Multiple Indicator Kriging
(MIK)** for probability estimation and **Sequential Indicator Simulation (SIS)**
for stochastic categorical simulation.

Indicator encoding and the K×K coregionalization are built with
{py:class}`~krigekit.IndicatorVariogramSystem` and transferred to the engine
with its `apply()`; `IndicatorKriging` then allocates the solver, solves, and
post-processes probabilities.

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
from krigekit import IndicatorKriging, IndicatorVariogramSystem

system = IndicatorVariogramSystem(categories=["A", "B", "C"])
system.set_categorical_obs(
    obs_coord,             # (nobs, 2) array
    obs_labels,            # string or integer category per sample
)
system.set_indicator_vgm(
    vtype="sph", nugget=0.02,
    a_major=500.0, a_minor1=100.0, azimuth=0.0,
    sill_strategy="theoretical", cross_strategy="closure",
)

ik = IndicatorKriging(ncat=3, ndim=2)
system.apply(ik)               # transfers indicators + K² structures

ik.set_grid(coord=grid_coord)
for k in range(1, 4):
    ik.set_search(ivar=k, anis1=100.0/500.0, nmax=20)

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
system = IndicatorVariogramSystem(categories=["A", "B", "C"])
system.set_categorical_obs(obs_coord, obs_labels)
system.set_indicator_vgm(vtype="sph", nugget=0.02,
                         a_major=500.0, a_minor1=100.0,
                         sill_strategy="theoretical", cross_strategy="closure")

ik = IndicatorKriging(ncat=3, ndim=2, nsim=50, seed=42)
system.apply(ik)
ik.set_grid(coord=grid_coord)
ik.set_sim()                  # must be called after set_grid

for k in range(1, 4):
    ik.set_search(ivar=k, anis1=100.0/500.0, nmax=20)

ik.solve()

sims, _ = ik.get_results()   # shape (ngrid, 3, 50) — one-hot encoded
cat_idx = np.argmax(sims, axis=1)  # (ngrid, 50) — integer category index
del ik
```

## Cross-variogram strategies

{py:meth}`~krigekit.IndicatorVariogramSystem.set_indicator_vgm` configures all
K² variogram pairs in one call.  Two orthogonal options select the
coregionalization: `sill_strategy` (diagonal/auto sills) and `cross_strategy`
(off-diagonal/cross sills):

| `sill_strategy` | Auto sill |
|---|---|
| `"theoretical"` *(default)* | `p_k (1 − p_k)` from proportions |
| `"uniform"` | `sill` for every category |

| `cross_strategy` | Cross sill | When to use |
|---|---|---|
| `"closure"` *(default)* | `−p_k p_l` | Recommended; closed (`B·1=0`) and PSD |
| `"proportional"` | `√(s_k · s_l)` | LMC-valid per nested structure |
| `"independent"` | 0 | Most conservative; K separate systems |
| `"uniform"` | `sill` | Simplest approximation |

`fit(method="closure")` then refits the coregionalization from empirical
variograms while enforcing both positive semidefiniteness and closure; see the
{doc}`variogram_fitting` guide.

### Closure (recommended)

Auto sills `p_k(1 − p_k)` and cross sills `−p_k p_l` reproduce the closed
indicator covariance `diag(p) − p pᵀ`.

```python
system.set_indicator_vgm(vtype="sph", nugget=0.02,
                         a_major=500, a_minor1=80, azimuth=90,
                         sill_strategy="theoretical", cross_strategy="closure")
```

### Uniform sill

```python
system.set_indicator_vgm(vtype="sph", nugget=0.02, sill=0.19,
                         a_major=500, a_minor1=80, azimuth=90,
                         sill_strategy="uniform", cross_strategy="uniform")
```

### Proportional sills (LMC)

Auto sills calibrated to the indicator variance; cross sills set to their
geometric mean so the coregionalisation matrix is positive-definite.

```python
props = np.array([0.18, 0.23, 0.21, 0.38])   # observed p_k per category
system.set_indicator_vgm(vtype="sph", nugget=0.02,
                         a_major=500, a_minor1=80, azimuth=90,
                         sill_strategy="theoretical", cross_strategy="proportional",
                         proportions=props)
```

### Independent (no cross-coupling)

Cross-variogram sills are set to zero — equivalent to running K separate
ordinary kriging systems.

```python
system.set_indicator_vgm(vtype="sph", nugget=0.02,
                         a_major=500, a_minor1=80, azimuth=90,
                         sill_strategy="theoretical", cross_strategy="independent")
```

## Co-kriging MIS

Secondary continuous variables can be added by setting `nvar = ncat + M`:

The indicator block (variables 1..K) is built with the system and applied; the
secondary variable and its cross-models are set on the engine directly.

```python
system = IndicatorVariogramSystem(categories=["A", "B", "C"])
system.set_categorical_obs(obs_coord, obs_labels)
system.set_indicator_vgm(vtype="sph", a_major=500, a_minor1=100,
                         sill_strategy="theoretical", cross_strategy="closure")

ik = IndicatorKriging(ncat=3, nvar=4, ndim=2)  # 3 indicators + 1 secondary
system.apply(ik)                                # indicator block (ivar 1..3)
ik.set_obs(ivar=4, coord=sec_coord, value=sec_val)  # secondary variable

ik.set_vgm(ivar=4, jvar=4, vtype="sph", sill=1.0, a_major=500, a_minor1=100)
for k in range(1, 4):                           # indicator–secondary cross-models
    ik.set_vgm(ivar=k, jvar=4, vtype="sph", sill=0.1, a_major=500, a_minor1=100)

ik.set_grid(coord=grid_coord)
for k in range(1, 5):
    ik.set_search(ivar=k, anis1=100.0/500.0, nmax=20)

ik.solve()
probs, var = ik.get_results()   # shape (ngrid, 3) — secondary excluded
```

The secondary variable contributes to kriging weights but is excluded from
the CDF draw and probability normalisation.

## Variogram orientation

In KrigeKit the default variogram major axis is aligned with the **Y** axis.
For horizontal stratigraphy (long range along X), pass the same `azimuth` to
**both** the system's `set_indicator_vgm` and the engine's `set_search`.  Passing
it only to `set_search` leaves the variogram ellipse pointing the wrong way and
produces vertical patches in the simulated images.

```python
AZIMUTH = 90.0   # rotate major axis from Y → X

system.set_indicator_vgm(..., azimuth=AZIMUTH)    # variogram ellipse
for k in range(1, ncat + 1):
    ik.set_search(ivar=k, anis1=anis1, azimuth=AZIMUTH)  # search ellipse
```

## Gallery example

The {doc}`../auto_examples/s_sis_lithofacies` gallery example demonstrates
MIS on a 2-D lithofacies outcrop dataset, comparing the uniform-sill and
proportional-sill strategies side-by-side.
