# Variogram analysis and fitting

`VariogramModel` is the Python-side workflow object for exploratory variogram
analysis.  It stores observations, empirical variogram clouds, averaged bins,
fit parameters, and nested model structures before the model is applied to a
`Kriging` object.

Use it when you want to:

- calculate raw and averaged experimental variograms,
- inspect anisotropy with directional averages or a variogram map,
- fit one or more nested structures with optional weights,
- manually adjust a fitted model before kriging,
- reuse the same model definition with `Kriging.set_vgm()`.

## Basic workflow

```python
from krigekit import Kriging, VariogramModel

model = VariogramModel()
model.set_obs(obs_coord, obs_value)

# A template with one spherical structure.
model.set_vgm(
    vtype="sph",
    nugget=0.05,
    sill=0.95,
    a_major=500.0,
)

raw = model.calc_experimental(cutoff=2000.0, verbose=False)
avg = model.calc_average(h_width=100.0)

model.fit(weight_col=("variogram", "count"), inplace=True)
model.plot()

k = Kriging()
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=24)
k.set_grid(coord=grid_coord)
model.apply_to(k, ivar=1, jvar=1)
k.set_search(ivar=1)
k.solve()
estimate, variance = k.get_results()
```

`calc_experimental()` stores the raw variogram cloud on the model.
`calc_average()` bins that cloud by lag distance.  `fit()` updates the
structure parameters using the current averaged table unless a table is passed
explicitly.

The cached workflow state is available through sklearn-style attributes:

| Attribute | Meaning |
|---|---|
| `raw_variogram_` | Pairwise empirical variogram cloud |
| `avg_variogram_` | Lag-binned experimental variogram |
| `params_` | Last fitted flat parameter vector |
| `pcov_` | Parameter covariance returned by SciPy |
| `fitted_model_` | Copy of the model at the last successful fit |

The internal aliases `_raw`, `_avg`, `_params`, and `_pcov` are also present,
but user code should prefer the public trailing-underscore names.

## Nested structures

Call `set_vgm()` once per structure.  The keyword names match
`Kriging.set_vgm()` except that `ivar` and `jvar` are omitted.

```python
model = VariogramModel()
model.set_obs(obs_coord, obs_value)

model.set_vgm(vtype="nug", nugget=0.04, sill=0.0, a_major=1.0)
model.set_vgm(vtype="sph", nugget=0.0, sill=0.35, a_major=250.0)
model.set_vgm(vtype="exp", nugget=0.0, sill=0.55, a_major=900.0)

model.calc_experimental(cutoff=2500.0, verbose=False)
model.calc_average(h_width=100.0)
model.fit(inplace=True)
```

For ordinary one-dimensional lag fitting, the flat fit vector is ordered as
`sill, a_major` for each non-nugget structure, followed by a single trailing
`nugget` when `fit_nugget=True`.

The model can be copied into kriging in either form:

```python
model.apply_to(k, ivar=1, jvar=1)
```

or:

```python
for spec in model.to_kriging_specs(replace=True):
    k.set_vgm(ivar=1, jvar=1, **spec)
```

## Weighted fitting

Weighted fitting is useful because averaged bins do not carry equal
information.  Bins based on many observation pairs are usually more stable than
bins based on a few pairs.

```python
model.fit(weight_col=("variogram", "count"), inplace=True)
```

`weight_col` gives larger influence to larger weights.  With the default
averaged table, the pair count is stored under the multi-index column
`("variogram", "count")`.

Alternatively, use SciPy-style standard deviations:

```python
model.fit(sigma_col="sigma", inplace=True)
```

Do not pass `weights` or `weight_col` together with `sigma` or `sigma_col`.

## Manual adjustment

Numerical optimizers often produce a useful starting point rather than the
final geological model.  After fitting, adjust the flat parameter vector with
`set_params()`:

```python
model.fit(inplace=True)
model.set_params([0.35, 250.0, 0.55, 900.0, 0.04])
```

For fields outside the flat fit vector, such as anisotropy angles or fixed
minor ranges, adjust a structure directly:

```python
model.set_structure_params(0, a_minor1=120.0, azimuth=35.0)
```

If all structures share the same orientation and minor-to-major ratio,
`set_anisotropy()` is shorter:

```python
model.set_anisotropy(anis1=0.35, azimuth=35.0)
```

## Directional anisotropy

When the major-axis orientation is known or estimated visually, fit major and
minor ranges by averaging the empirical cloud along that fixed orientation.

```python
model = VariogramModel()
model.set_obs(obs_coord, obs_value)
model.set_vgm(
    vtype="sph",
    sill=0.8,
    nugget=0.05,
    a_major=1000.0,
    a_minor1=300.0,
    azimuth=90.0,
)

model.calc_experimental(cutoff=3000.0, calc_angle=True, verbose=False)

directional = model.calc_directional_average(
    h_width=100.0,
    cutoff=2500.0,
    angle_tol=15.0,
)

model.fit_anisotropy(
    directional,
    p0=(0.8, 1000.0, 300.0, 0.05),
    weight_col="count",
    inplace=True,
)
```

`fit_anisotropy()` keeps `azimuth`, `dip`, and `plunge` fixed.  For each
structure it fits `sill`, `a_major`, and `a_minor1`; in 3-D it can also fit
`a_minor2` when requested.

## Sum-metric space-time coupling

After fitting spatial and temporal marginals separately, calculate a full
space-time lag surface and fit the coupling with
`fit_spacetime_sum_metric()`:

```python
joint = VariogramModel()
joint.set_obs(obs_xy, obs_value, times=obs_time)
joint.calc_experimental(
    cutoff=120_000.0,
    t_cutoff=20.0,
    maxobs=2500,
    seed=2026,
    verbose=False,
)
joint.calc_average(
    h_width=5000.0,
    t_col="time_lag",
    t_width=0.5,
)

joint.fit_spacetime_sum_metric(
    spatial_model,
    temporal_model,
    transform="lin",
    p0=(1.0, 1.0, 100.0, 20.0),
)
specs = joint.to_sum_metric_kriging_specs()
```

The flat parameter order is `spatial_scale`, `temporal_scale`, one
`joint_sill` per spatial structure, and `at`. The marginal scales reconcile
models fitted on the spatial and temporal boundaries with the interior
space-time surface. Fixing `time_sill=1` avoids redundancy between
`time_sill` and `at` for the linear temporal metric transform.

The groundwater-level gallery example demonstrates this workflow, including
transfer of the scaled marginals, joint sill, and temporal metric scale to
`SpaceTimeKriging`.

### Fitting the short 3-D axis

The shortest 3-D range, usually `a_minor2`, is often the least stable fitted
parameter.  It needs close-lag pairs aligned with a narrow direction, and those
pairs can be sparse even when the total number of observation pairs is large.

Two settings help:

- Prefer `h_bins` over a single fixed `h_width` for 3-D directional fitting.
  When `h_width=None`, `calc_directional_average()` computes a separate
  effective bin width for each axis as `max_projected_lag / h_bins`.  The short
  `minor2` axis therefore gets narrower lag bins than the major axis, instead
  of being represented by only a few coarse bins.
- Balance fitting weights by axis.  Raw pair-count weights can let the
  better-populated major and `minor1` directions dominate the least-squares
  objective.  Normalize counts within each axis so each directional curve has
  comparable influence.

```python
model.calc_experimental(cutoff=36.0, calc_angle=True, verbose=False)

directional = model.calc_directional_average(
    h_bins=18,        # per-axis effective h_width
    cutoff=36.0,      # long enough to see the major range
    angle_tol=20.0,   # tighter directions reduce cross-axis mixing
)

directional["axis_weight"] = (
    directional["count"]
    / directional.groupby("axis", observed=True)["count"].transform("sum")
)

model.fit_anisotropy(
    directional,
    include_minor2=True,
    fit_nugget=False,
    weight_col="axis_weight",
    inplace=True,
)
```

The 3-D gallery example uses this pattern.  In that synthetic case the fit
improves from an overestimated short range to approximately
`a_major = 29.7`, `a_minor1 = 11.6`, and `a_minor2 = 8.4` for a true model of
`30`, `12`, and `8`.

## Variogram map

For two-dimensional data, `plot_map()` displays the raw variogram cloud in lag
space.  This is useful before fitting anisotropy because it shows whether the
chosen major direction agrees with the experimental continuity.

```python
model.calc_experimental(cutoff=3000.0, calc_angle=True, verbose=False)
model.plot_map(cutoff=2500.0)
model.plot_map(angle_aniso="estimate", cutoff=2500.0)
```

`angle_aniso="estimate"` overlays an automatically estimated orientation.
Use that as an exploratory aid, then set an explicit `azimuth` before fitting
the production model.

For three-dimensional data, use `plot_map3d()`.  It draws up to three
orthogonal fence sections through the lag-space origin, coloured by the
average semivariogram value in each lag bin.

By default (`rotate_fences=False`) the fences align with the world X/Y/Z
axes so that anisotropy angles can be read directly off the axes:

- **Fence A** (always) — horizontal XY plane; shows the azimuth pattern.
- **Fence B** (`n_fences ≥ 2`, default) — vertical XZ (East–West) section;
  shows the dip.
- **Fence C** (`n_fences ≥ 3`) — vertical YZ (North–South) section.

When model angles are supplied, a red line is drawn on each fence showing
the projected major axis — azimuth direction on the XY fence, dip
component on the vertical fences — so the fitted orientation can be
compared against the empirical map.

```python
model.calc_experimental(cutoff=3000.0, verbose=False)

# Two world-axis fences (default) with model-angle overlay.
model.plot_map3d(cutoff=2500.0)

# Estimate the orientation from the cloud if no model is fitted yet.
model.plot_map3d(angle_aniso="estimate", cutoff=2500.0)

# Three fences (adds North–South vertical section and plunge to label).
model.plot_map3d(cutoff=2500.0, n_fences=3)

# Rotate fences to the model's principal planes instead.
model.plot_map3d(cutoff=2500.0, rotate_fences=True)

# For sparse 3-D clouds, fill empty in-range display bins from nearest occupied bins.
model.plot_map3d(cutoff=2500.0, fill_nan=True)
```

All fence polygons are rendered in a single `Poly3DCollection`, so
depth-sorting is correct when rotating the interactive plot.
By default, empty bins are left empty so the plot shows sampling support.
`fill_nan=True` is a display-only nearest-neighbour fill for smoother example
figures; it is constrained to the cutoff or maximum lag radius and does not
change the raw or averaged variogram data.

Pass `fill_nan=True` when data are sparse and the fence has many empty
lag bins.

## Multivariable systems

Use `VariogramSystem` when fitting direct and cross variograms for cokriging.
It mirrors the kriging API by carrying `ivar` and `jvar` through the workflow.

```python
from krigekit import VariogramSystem

system = VariogramSystem(nvar=2)
system.set_obs(ivar=1, coord=coord_1, value=value_1)
system.set_obs(ivar=2, coord=coord_2, value=value_2)

system.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=500.0)
system.set_vgm(ivar=2, jvar=2, vtype="sph", sill=0.6, a_major=500.0)
system.set_vgm(ivar=1, jvar=2, vtype="sph", sill=0.4, a_major=500.0)

system.calc_experimental(ivar=1, jvar=1, cutoff=2000.0, verbose=False)
system.calc_experimental(ivar=2, jvar=2, cutoff=2000.0, verbose=False)
system.calc_experimental(ivar=1, jvar=2, cutoff=2000.0, verbose=False)
system.calc_average(h_width=100.0)

fitted_system, result = system.fit_lmc(fit_ranges=True, fit_nugget=True)
fitted_system.apply_to(k)
```

`fit_pair()` fits one pair independently.  For cokriging, prefer `fit_lmc()`
because it fits the requested pairs together while enforcing positive
semidefinite sill matrices for each nested structure.

### Markov-model cross-variograms (sparse primary + dense secondary)

`fit_lmc()` fits the cross-variogram from data, which is correct when both
variables are well sampled.  But when a **sparse primary** (e.g. categorical
well logs) is cokriged with a **dense secondary** covariate (e.g. an airborne
geophysical survey), the primary usually has little structured variance, so a
valid (positive-semidefinite) LMC drives the cross-covariance toward zero and
the covariate can no longer inform the primary.

For that case use `set_markov_cross()`, which builds the cross by the **Markov
Model 1** assumption: the cross adopts the secondary's structure, scaled by the
collocated correlation,

```{math}
b_{ps}^{(k)} = \rho \, \sqrt{b_{pp}^{(k)} \, b_{ss}^{(k)}}
```

which is positive semidefinite by construction for ``|rho| <= 1`` — no clamping
needed.

```python
system = VariogramSystem(nvar=2)
system.set_obs(ivar=1, coord=well_xy, value=indicator)   # sparse primary
system.set_obs(ivar=2, coord=aem_xy,  value=covariate)   # dense secondary

system.set_vgm(ivar=1, jvar=1, vtype="exp", nugget=0.05, sill=0.20, a_major=6500.0)
system.set_vgm(ivar=2, jvar=2, vtype="exp", nugget=0.02, sill=0.05, a_major=6500.0)

# cross from the collocated correlation (the cross adopts variable 2's structure)
system.set_markov_cross(primary=1, secondary=2, corr=0.8)
system.apply_to(k)
```

Pass `corr=None` to estimate the correlation from collocated observations (the
two variables must then share coordinates); otherwise compute it from the
collocated subset and pass it explicitly.  Markov Model 2 is not yet
implemented.

In short: use `fit_lmc()` for co-sampled multivariate data, and
`set_markov_cross()` for sparse-primary / dense-secondary collocated cokriging
(Almeida & Journel, 1994; Goovaerts, 1997).

## Example gallery

See the gallery example
`examples/s_variogram_fitting.py` for a complete indicator variogram workflow:
it computes the empirical cloud, fits a two-structure anisotropic model for the
`lithofacies.csv` data, rounds the fitted parameters, and applies the model to
ordinary kriging.
