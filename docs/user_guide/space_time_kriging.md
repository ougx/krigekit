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

krigekit stores the product-sum model in covariance form:

$$C(h_s, h_t) = k_\text{ps}\,C_S(h_s)\,C_T(h_t) + C_S(h_s) + C_T(h_t)$$

Converting from fitted variogram parameters $(a, b, p)$ to krigekit's
covariance parameters:

$$C_S(0) = a + p, \qquad C_T(0) = b + p, \qquad
k_\text{ps} = -\frac{p}{C_S(0)\cdot C_T(0)}$$

These equations should be checked against the exact fitted product-sum
parameterisation used in your calibration workflow.  The conversion is included
here because krigekit stores the model in covariance form, while many fitting
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

qt = QuantileTransformer(random_state=0)
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

Fit the three scalar parameters $(a, b, p)$ and marginal ranges $(a_s, a_t)$
by weighted least-squares over the experimental variogram bins:

```python
from scipy.optimize import minimize

def gamma_ps(hs, ht, a, b, p, a_s, a_t):
    gs = sph(hs, a_s)     # normalised spatial marginal
    gt = gauss(ht, a_t)   # normalised temporal marginal (Gaussian)
    return a * gs + b * gt + p * gs * gt

def wls(params):
    a, b, p, a_s, a_t = params
    if a <= 0 or b <= 0 or a_s <= 0 or a_t <= 0:
        return 1e12
    if p > 0 or a + b + p <= 0:
        return 1e12
    pred = gamma_ps(hs_bins, ht_bins, a, b, p, a_s, a_t)
    return float(np.sum(n_pairs * (pred - gamma_bins) ** 2))

res = minimize(wls, x0=[0.11, 0.025, -0.01, 2500, 6.5],
               method="Nelder-Mead",
               options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 20000})
a_ps, b_ps, p_ps, a_s, a_t = res.x
```

Weighting by pair count (`n_pairs`) gives greater influence to statistically
stable bins containing more data pairs.  Because short-lag bins often contain
many more pairs than long-lag bins, inspect the fitted surface visually to ensure
longer-range structure is not under-represented.

Kernel choice for the temporal marginal:

- **Gaussian** (`gau`) — smooth parabolic behaviour near $h_t = 0$; appropriate
  for phenomena that evolve continuously over years (e.g. slow-moving plumes).
- **Exponential** (`exp`) — linear near-origin; better for fields with abrupt
  short-term changes.
- **Spherical** (`sph`) — reaches its sill at a finite range; use when temporal
  correlation is negligible beyond a clear cutoff.

Use leave-one-out cross-validation to confirm the kernel choice
(see `_diag_at_kernel.py` in the examples for a systematic sweep).

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
- Full worked example: `examples/st_ok3d_ctet.py` (CCl4 plume, Hanford Site)
