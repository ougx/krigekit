# Space-time kriging

Space-time kriging estimates an unknown field at target locations **and** times
simultaneously, exploiting correlation in both dimensions.  It is the right
choice when your observations are irregularly distributed across space and time
and you want a single model that respects both spatial and temporal structure
(e.g. groundwater plumes, air quality surfaces, climate fields).

## Coordinate convention

Observation and grid arrays follow the combined `(nobs, ndim+1)` layout: the
first `ndim` columns are spatial coordinates `(x [, y [, z]])` and the last
column is time in any consistent unit (decimal years, days, etc.).

```python
# 3-D spatial + 1-D temporal:  shape (nobs, 4)
obs_coord = np.column_stack([df["x"], df["y"], df["z"], df["t"]])

# 2-D spatial + 1-D temporal: shape (nobs, 3)
obs_coord = np.column_stack([df["x"], df["y"], df["t"]])
```

Alternatively, pass spatial and temporal arrays separately to `set_obs` /
`set_grid` via the `time` keyword:

```python
k.set_obs(ivar=1, coord=xy_array, time=t_array, value=obs_value)
k.set_grid(coord=grid_xy,         time=grid_t)
```

## Quick start — one-shot function

For a single variable and a simple workflow, `spacetime_kriging` wraps the
full `SpaceTimeKriging` class in one call:

```python
import numpy as np
from krigekit import spacetime_kriging

# observations: (x, y, t) – 2-D spatial + time
obs_coord = np.column_stack([obs_x, obs_y, obs_t])   # (nobs, 3)

est, var = spacetime_kriging(
    obs_coord  = obs_coord,
    obs_value  = obs_value,
    grid_coord = grid_xy,            # (ngrid, 2)  spatial only
    grid_time  = grid_t,             # (ngrid,)
    spatial_spec  = dict(vtype="sph", nugget=0.05, sill=0.95, a_major=2000.0),
    temporal_spec = dict(vtype="exp", nugget=0.0,  sill=1.0,  at_k=5.0),
    joint_sills   = [0.3],           # one per spatial structure (sum-metric)
    model="sum_metric",
    at=5.0,                          # joint temporal scale = a_t
    nmax=30,
)
```

**Grid coordinate convention:** when `grid_time` is supplied separately,
`grid_coord` must contain only spatial coordinates:

```python
grid_coord.shape == (ngrid, ndim)
grid_time.shape  == (ngrid,)
```

Do **not** append time as an additional column in `grid_coord` when using the
`grid_time` argument.

## Minimal example

This small example is intended to test the basic API and coordinate convention.
It uses synthetic observations, so the numerical result is not scientifically
meaningful.

```python
import numpy as np
from krigekit import spacetime_kriging

rng = np.random.default_rng(1234)

# 50 observations in 2-D space plus time
obs_x = rng.uniform(0.0, 1000.0, 50)
obs_y = rng.uniform(0.0, 1000.0, 50)
obs_t = rng.uniform(0.0, 10.0, 50)
obs_coord = np.column_stack([obs_x, obs_y, obs_t])

obs_value = (
    np.sin(obs_x / 300.0)
    + np.cos(obs_y / 250.0)
    + 0.2 * obs_t
)

# One target point at one prediction time
grid_xy = np.array([[500.0, 500.0]])
grid_t  = np.array([5.0])

est, var = spacetime_kriging(
    obs_coord=obs_coord,
    obs_value=obs_value,
    grid_coord=grid_xy,
    grid_time=grid_t,
    spatial_spec=dict(vtype="sph", nugget=0.01, sill=0.99, a_major=400.0),
    temporal_spec=dict(vtype="exp", nugget=0.01, sill=0.99, at_k=3.0),
    joint_sills=[0.3],
    model="sum_metric",
    at=3.0,
    nmax=20,
)

print(est[0], var[0])
```

## Full workflow

`SpaceTimeKriging` gives full control and is required for cross-validation,
co-kriging, SGSIM, or the product-sum model:

```python
from krigekit import SpaceTimeKriging

k = SpaceTimeKriging(nvar=1)
k.set_st_model("sum_metric", transform="exp", at=5.0,
               time_nugget=0.0, time_sill=0.5)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=30)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.05, sill=0.95,
          a_major=2000.0, a_minor1=2000.0, a_minor2=400.0)
k.set_vgm_joint_sills(1, 1, 0.3)    # sum-metric only
k.set_grid(coord=grid_xy, time=grid_t)
k.set_search(ivar=1, time_at=5.0)
k.solve()
est, var = k.get_results(copy=True)
del k
```

## Space-time covariance models

### Sum-metric model

The sum-metric model defines a joint space-time distance:

$$h_{ST} = \sqrt{h_s^2 + (\text{time\_at} \cdot h_t)^2}$$

and builds the covariance from three terms — a purely spatial variogram
$\gamma_S(h_s)$, a purely temporal variogram $\gamma_T(h_t)$, and a joint
variogram evaluated at the metric distance $h_{ST}$:

$$\gamma(h_s, h_t) = \gamma_S(h_s) + \gamma_T(h_t) + \gamma_{ST}(h_{ST})$$

```python
k.set_st_model("sum_metric", transform="exp", at=5.0,
               time_nugget=0.0, time_sill=0.5)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=0.8,
          a_major=2000.0, a_minor1=2000.0, a_minor2=500.0)
k.set_vgm_joint_sills(1, 1, 0.3)   # joint sill for the ST component
```

`transform` sets the variogram type used for the joint ST component;
`at` is its practical range in `time_at`-equivalent units; `time_sill` scales
its contribution.  Call `set_vgm_joint_sills` once after all `set_vgm` calls,
passing one float per spatial nested structure.

**Important:** `at` defines the temporal scale of the joint space-time variogram
used by the sum-metric model.  `time_at` in `set_search()` is a neighbour-search
scaling factor that converts temporal separation into equivalent spatial
distance for KD-tree searches.  For the sum-metric model, it is usually
appropriate to set `time_at = at`, but the two parameters serve different
purposes.

### Product-sum model

The product-sum model (De Cesare et al., 2001) is more flexible and avoids
the artificial isometry of the joint metric.  In variogram form:

$$\gamma(h_s, h_t) = a\,\tilde\gamma_S(h_s) + b\,\tilde\gamma_T(h_t)
                   + p\,\tilde\gamma_S(h_s)\,\tilde\gamma_T(h_t)$$

where $\tilde\gamma_S$ and $\tilde\gamma_T$ are the marginal variograms
normalised to sill = 1.  Validity requires $a, b > 0$, $p \le 0$, and
$a + b + p > 0$ (positive total sill).

KrigeKit stores the product-sum model in covariance form:

$$C(h_s, h_t) = k_\text{ps}\,C_S(h_s)\,C_T(h_t) + C_S(h_s) + C_T(h_t)$$

Converting from fitted variogram parameters $(a, b, p)$ to KrigeKit's
covariance parameters:

$$C_S(0) = a + p, \qquad C_T(0) = b + p, \qquad
k_\text{ps} = -\frac{p}{C_S(0)\cdot C_T(0)}$$

These equations should be checked against the exact fitted product-sum
parameterisation used in your calibration workflow.  The conversion is included
here because KrigeKit stores the model in covariance form, while many fitting
workflows estimate product-sum parameters in variogram form.

Because $p \le 0$, the coupling coefficient $k_\text{ps} \ge 0$ — a larger
value means observations that are close in **both** space and time are more
correlated than either marginal alone predicts (appropriate for coherent,
slowly evolving phenomena such as groundwater plumes).

```python
# Fitted variogram parameters
a_ps, b_ps, p_ps = 0.10, 0.06, -0.005
a_s, a_t = 5000.0, 9.0         # spatial range (m), temporal range (yr)

sill_s   = a_ps + p_ps         # = 0.095  spatial covariance sill
sill_t   = b_ps + p_ps         # = 0.055  temporal covariance sill
k_ps_val = -p_ps / (sill_s * sill_t)   # = 0.957

k = SpaceTimeKriging(nvar=1)
k.set_st_model("product_sum", k_ps=k_ps_val)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=50)
k.set_vgm(ivar=1, jvar=1, vtype="sph",
          nugget=0.0005, sill=sill_s,
          a_major=a_s, a_minor1=a_s, a_minor2=a_s / 5)    # 5× vertical anisotropy
k.set_vgm_temporal(ivar=1, jvar=1, vtype="gau",
                   nugget=0.0005, sill=sill_t, at_k=a_t)
```

`set_vgm_joint_sills` is **not** used with `product_sum`; the coupling is
handled entirely by `k_ps`.

### Product structures in spatial and temporal marginals

Both marginal setters support `product=True` with the same convention as
`Kriging.set_vgm()`:

```python
# Spatial Gaussian envelope multiplied by a 3-D-valid periodic structure.
k.set_vgm(1, 1, vtype="gau", sill=0.8, a_major=5000.0)
k.set_vgm(
    1, 1,
    vtype="cyc",
    sill=1.0,
    a_major=1000.0,
    product=True,
)

# Temporal Gaussian envelope multiplied by an annual hole effect.
k.set_vgm_temporal(1, 1, vtype="gau", sill=0.4, at_k=30.0)
k.set_vgm_temporal(
    1, 1,
    vtype="hol",
    sill=1.0,
    at_k=0.5,
    product=True,
)
```

The first member of each marginal cannot be a product structure. Product
nesting affects the spatial marginal $C_S$ or temporal marginal $C_T$.
Use a kernel valid for the spatial dimensionality; for example, `hol` is not
positive definite in 3-D space, while `cyc` is valid.
For the sum-metric model, `set_vgm_joint_sills()` still defines separate
additive joint terms for each stored spatial component; it does not multiply
the joint terms according to the spatial product groups.

## Temporal search scale

The KD-tree neighbour search operates in the combined
$(x, y, z, t \cdot \text{time\_at})$ space.  `time_at` in `set_search`
converts the time axis into the same length units as the spatial axes, so that
the L2 distance in search space correctly prioritises nearby observations.

For the **sum-metric** model, pass the joint temporal scale `at` directly:

```python
k.set_search(ivar=1, time_at=at)
```

For the **product-sum** model, match the rate at which each marginal loses
covariance per unit displacement — spatially $C_S(0)/a_s$ and temporally
$C_T(0)/a_t$ — and equate them:

$$\text{time\_at} = \frac{a_s}{a_t} \cdot \frac{C_S(0)}{C_T(0)}$$

```python
time_at_search = (a_s / a_t) * (sill_s / sill_t)
k.set_search(ivar=1, time_at=time_at_search)
```

The sill-ratio correction down-weights time when the temporal variogram
explains proportionally less variance than the spatial one.

## Kriging matrix regularisation — nugget

Near-singular kriging systems commonly arise when observations are highly
clustered in space and/or time, producing nearly identical covariance rows.
Repeated measurements at the same well (same $x, y, z$) but different times are
a common example.  With a Gaussian
temporal variogram whose practical range $a_t$ is long relative to the
sampling interval $\delta t$, two consecutive monthly samples satisfy:

$$C(0,\,\delta t) \approx C(0,\,0) \quad \text{for } \delta t \ll a_t$$

Their rows in the kriging matrix are nearly identical, so the system assigns
enormous oscillating weights ($\pm 10^2$) that cancel but make estimates
numerically unreliable.

Adding a small nugget $\eta$ to both the spatial and temporal variograms lifts
the matrix diagonal while leaving off-diagonal entries unchanged, restoring
full rank:

```python
NUGGET = 0.0005    # ~0.3 % of total sill — chosen by leave-one-out CV
k.set_vgm(         ..., nugget=NUGGET, ...)
k.set_vgm_temporal(..., nugget=NUGGET, ...)
```

**Choosing the nugget via cross-validation:**  a large nugget minimises overall
RMSE but systematically under-predicts peak values (negative bias on the
high-concentration tail); a small nugget honours peaks at the cost of slightly
higher overall RMSE.  Use leave-one-out CV scored on the tail separately:

```python
k = SpaceTimeKriging(nvar=1, cross_validation=True)
# ... set_obs, set_vgm, etc. ...
k.set_grid_cv()
k.solve()
cv_est, _ = k.get_results(copy=True)
```

## Normal-score transform for skewed data

When the distribution is highly skewed (e.g. contaminant concentrations
spanning three orders of magnitude), apply a normal-score transform before
kriging:

```python
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution="normal", random_state=0)
df["nscore"] = qt.fit_transform(df[["value"]]).ravel()

# krige in normal-score space ...
# back-transform after kriging:
conc = qt.inverse_transform(est.reshape(-1, 1)).ravel()
conc = np.maximum(conc, 0.0)   # clip negatives for strictly positive quantities
```

**Note:** Quantile-based back-transformation can compress extreme values outside
the range well represented by the observations.  Always evaluate prediction bias
on the upper tail during cross-validation if accurate peak reconstruction is
important.

Keep the fitted `qt` object for back-transformation after kriging.  The
transform is applied to observations only; the fitted variogram is for the
normal-score values.

## Fitting the product-sum variogram

Use `VariogramModel` to calculate the experimental lag surface and fit the
three scalar parameters $(a, b, p)$ and marginal ranges $(a_s, a_t)$ by
weighted least-squares:

```python
from krigekit import SpaceTimeVariogramModel

model = SpaceTimeVariogramModel()
model.set_obs(spatial_coord, transformed_value, times=observation_time)
model.calc_experimental(
    cutoff=5500.0,
    t_cutoff=13.0,
    anisotropy={"anis1": 1.0, "anis2": 0.2},
)
model.calc_average(
    h_col="anisotropic_distance",
    h_width=500.0,
    t_col="time_lag",
    t_width=0.5,
)

model.fit_spacetime_product_sum(
    spatial_vtype="sph",
    temporal_vtype="gau",
    starts=[
        (0.10, 0.06, -0.005, 5000.0, 9.0),
        (0.12, 0.04, -0.020, 3000.0, 10.0),
    ],
    bounds=[
        (0.0, 0.30), (0.0, 0.20), (-0.20, 0.0),
        (1000.0, 8000.0), (1.0, 13.0),
    ],
    weight_cap_quantile=0.90,
)

a_ps, b_ps, p_ps, a_s, a_t = model.spacetime_params_
predicted_gamma = model.calc_spacetime_variogram(hs, ht)
```

`distance` in the raw cloud remains the physical Euclidean lag used by
variogram maps and diagnostics. `anisotropic_distance` is the equivalent
major-axis lag calculated after applying the engine-compatible azimuth, dip,
plunge, and minor/major ratios. Consequently, the fitted spatial range is the
major range. Do not pre-scale the coordinate array when this option is used.

For evaluation at coordinate pairs, use
`model.calc_spacetime_variogram_between(coord0, coord1, time0, time1)`.
`calc_spacetime_variogram(hs, ht)` remains useful when `hs` is already a scalar
physical or anisotropy-adjusted spatial lag.

The fit uses pair counts as weights. `weight_cap_quantile` limits the influence
of exceptionally populous bins so that the short-lag region does not overwhelm
the rest of the surface. Inspect the fitted surface and all multistart results
in `model.spacetime_fit_results_`; the lowest objective alone does not establish
that all five parameters are well identified.

After cross-validation, manually adjusted parameters can be stored on the same
object and converted to the covariance form expected by `SpaceTimeKriging`:

```python
model.set_spacetime_params((0.10, 0.06, -0.005, 5000.0, 9.0))
specs = model.to_spacetime_kriging_specs()

k.set_st_model(specs["model"], k_ps=specs["k_ps"])
k.set_vgm(ivar=1, istruct=1, **specs["spatial_spec"])
k.set_vgm_temporal(ivar=1, istruct=1, **specs["temporal_spec"])
k.set_search(ivar=1, time_at=specs["time_at"])
```

The constraints `a + p > 0` and `b + p > 0` ensure that the converted
covariance marginal sills

$$
C_S(0)=a+p,\qquad C_T(0)=b+p
$$

are both positive. Checking only `a + b + p > 0` is insufficient for the
covariance conversion used by `SpaceTimeKriging`.

The `st_variogram_fitting_ctet.py` gallery example applies these constraints to
`test_data/ctet.csv` with capped pair-count weights and multiple starting
points. It also demonstrates a common failure mode: the lowest-residual
solution collapses the temporal covariance sill and hits a range bound because
the five product-sum parameters are weakly identified by the available lag
surface. The example therefore compares the automatic fit with the rounded,
cross-validation-informed production model used by `st_ok3d_ctet.py`.

Kernel choice for the temporal marginal:

- **Gaussian** (`gau`) — smooth parabolic behaviour near $h_t = 0$; appropriate
  for phenomena that evolve continuously over years (e.g. slow-moving plumes).
- **Exponential** (`exp`) — linear near-origin; better for fields with abrupt
  short-term changes.
- **Spherical** (`sph`) — reaches its sill at a finite range; use when temporal
  correlation is negligible beyond a clear cutoff.

Use leave-one-out cross-validation to confirm the kernel choice
(see `_diag_at_kernel.py` in the examples for a systematic sweep).

### Long-term decay with an annual cycle

Repeated groundwater-level measurements often show a slow multi-year loss of
correlation plus a weaker annual cycle. Estimate the temporal marginal from
**within-well pairs only**. Pairing observations from different wells mixes
spatial differences into the temporal variogram.

For `obs_gwlevel.csv`, first convert hydraulic head to depth to water,
`dem10 - sl_lev_va`, to remove the broad topographic trend. The temporal
semivariance is unchanged because `dem10` is constant within a well.

$$
C_T(h) = A\,G(h;a_T) + B\,G(h;a_T)\cos(2\pi h),
$$

where

$$
G(h;a_T)=\exp\left[-3.0625\left(\frac{h}{a_T}\right)^2\right].
$$

The corresponding semivariogram is

$$
\gamma_T(h)=\eta+A[1-G(h;a_T)]
 +B[1-G(h;a_T)\cos(2\pi h)].
$$

This is an additive Gaussian background plus a second Gaussian multiplied by
an annual hole-effect structure:

```python
temporal = VariogramModel()
temporal.set_vgm("gau", nugget=eta, sill=A, a_major=a_t)
temporal.set_vgm("gau", sill=B, a_major=a_t)
temporal.set_vgm("hol", sill=1.0, a_major=0.5, product=True)

temporal.apply_temporal_to(k, ivar=1, jvar=1)
```

For `hol`, the covariance is $\cos(\pi h/a)$, so its period is $2a$.
Therefore `a_major=0.5` gives a one-year period when time is measured in
years.

Do not use a single `gau * cyc` product for a weak seasonal signal without
checking its amplitude. The current `cyc` kernel has a fixed normalized
half-period correlation of $\exp(-2)\approx0.135$, which imposes a much
stronger oscillation than this groundwater dataset supports. The additive
background plus smaller `gau * hol` product estimates the seasonal amplitude
through `B` while remaining positive definite.

See `examples/space_time/st_variogram_fitting_gwlevel.py` for the complete fit
and half-year kriged hydrographs for wells H0049 and H0001 from 1980 through
2020. The example retains observations at the target wells, adds a small
observation-error variance to regularize repeated measurements at identical
coordinates, and plots 95% kriging intervals. It also includes a stricter
leave-one-well-out reconstruction for H0017 and H1477. Each sparse well has
only one withheld observation but 17 neighboring wells with at least 30
observations within 30,000 ft.

The example also compares a gridded snapshot at `timeindex=2008.5`. A seeded
20% holdout removes 65 of the 325 contemporaneous measurements. Spatial
kriging uses only the remaining 2008.5 data, while space-time kriging may also
use other dates, including other measurements from the withheld wells. In this
demonstration the holdout RMSE decreases from about `26.63 ft` for the spatial
baseline to `1.73 ft` for space-time kriging. This evaluates reconstruction of
a sparsely sampled **date**; it is intentionally different from
leave-one-well-out validation.

## Cross-validation

Leave-one-out cross-validation follows the same workflow as for ordinary
kriging — switch the grid to CV mode and leave all other calls unchanged:

```python
k = SpaceTimeKriging(nvar=1, cross_validation=True)
k.set_st_model("product_sum", k_ps=k_ps_val)
k.set_obs(ivar=1, coord=obs_coord, value=V, nmax=50)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=NUGGET, sill=sill_s,
          a_major=a_s, a_minor1=a_s, a_minor2=a_s / Z_SCALE)
k.set_vgm_temporal(ivar=1, jvar=1, vtype="gau",
                   nugget=NUGGET, sill=sill_t, at_k=a_t)
k.set_grid_cv()                 # predict at observation locations
k.set_search(ivar=1, time_at=time_at_search)
k.solve()
cv_est, _ = k.get_results(copy=True)
```

`cv_est[i]` is the estimate at observation `i` using all other observations.
Score in the original (back-transformed) units to detect bias in the tails:

```python
cc = np.maximum(qt.inverse_transform(cv_est.reshape(-1, 1)).ravel(), 0.0)
d  = cc - conc_true
rmse    = np.sqrt((d**2).mean())
bias_hi = d[conc_true > 1000].mean()   # under-prediction of peaks?
```

## Common pitfalls

### Mixing coordinate formats

When using separate spatial and temporal arrays,

```python
k.set_obs(coord=xy, time=t)
```

the coordinate array must contain only spatial dimensions.  When using a
combined coordinate array,

```python
coord = np.column_stack([x, y, t])
```

the time column is already embedded and should not also be passed separately.
The same rule applies to `set_grid()` and the one-shot `spacetime_kriging()`
function.

### Forgetting `set_vgm_temporal`

For the product-sum model, both spatial and temporal marginal variograms must be
defined:

```python
k.set_vgm(...)
k.set_vgm_temporal(...)
```

### Using `set_vgm_joint_sills` with product-sum

`set_vgm_joint_sills()` is used only for the sum-metric model.  For the
product-sum model, the coupling is controlled by `k_ps` in `set_st_model()`.

### Singular matrices

If kriging weights become extremely large or oscillatory, first test a small
nugget before increasing the search neighbourhood size.  Increasing `nmax` can
make a near-singular local system worse when the added neighbours are highly
redundant.

### Search scaling

`time_at` affects only neighbour selection.  It does not modify the fitted
variogram or covariance model itself.

## See also

- [Variogram models](../variogram_models.rst) — model types, nesting, anisotropy
- [Ordinary kriging](ordinary_kriging.md) — base workflow shared with ST kriging
- [API reference](../api/index.md) — `SpaceTimeKriging`, `spacetime_kriging`
- Full worked example: `examples/st_ok3d_ctet.py` (CCl4 plume)
