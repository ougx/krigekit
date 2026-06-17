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

## Example gallery

See the gallery example
`examples/s_variogram_fitting.py` for a complete indicator variogram workflow:
it computes the empirical cloud, fits a two-structure anisotropic model for the
`lithofacies.csv` data, rounds the fitted parameters, and applies the model to
ordinary kriging.
