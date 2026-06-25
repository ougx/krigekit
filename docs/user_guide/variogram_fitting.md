# Variogram analysis and fitting

`VariogramModel` is the Python-side workflow object for exploratory variogram
analysis.  It stores observations, empirical variogram clouds, averaged bins,
fit parameters, and nested model structures before the model is applied to a
`Kriging` object.

Use `SpaceTimeVariogramModel` when fitting a full space-time lag surface. It
contains separate `spatial` and `temporal` `VariogramModel` marginals and owns
the product-sum or sum-metric coupling parameters:

```python
from krigekit import SpaceTimeVariogramModel, VariogramModel

spatial = VariogramModel()
temporal = VariogramModel()
joint = SpaceTimeVariogramModel(spatial=spatial, temporal=temporal)
```

Use it when you want to:

- calculate raw and averaged experimental variograms,
- inspect anisotropy with directional averages or a variogram map,
- fit one or more nested structures with optional weights,
- manually adjust a fitted model before kriging,
- reuse the same model definition with `Kriging.set_vgm()`.

Use `VariogramSystem` for direct and cross variograms involving multiple
variables:

| Class | Role |
|---|---|
| `VariogramModel` | One spatial or temporal marginal |
| `SpaceTimeVariogramModel` | Full space-time surface and coupling |
| `VariogramSystem` | Multivariable direct/cross variograms and LMC fitting |

## Basic workflow

When you want the fitted object to transfer both observations and variogram
parameters to kriging, use `VariogramSystem` even for a single variable:

```python
from krigekit import Kriging, VariogramSystem

model = VariogramSystem(nvar=1)
model.set_obs(ivar=1, coord=obs_coord, value=obs_value)

# A template with one spherical structure.
model.set_vgm(
    1,
    1,
    vtype="sph",
    nugget=0.05,
    sill=0.95,
    a_major=500.0,
)

raw = model.calc_experimental(cutoff=2000.0, verbose=False)
avg = model.calc_average(h_width=100.0)

model.fit(method="pair", ivar=1, weight_col=("variogram", "count"), inplace=True)

k = Kriging()
k.set_grid(coord=grid_coord)
model.apply(k)
k.set_search(ivar=1, nmax=24)
k.solve()
estimate, variance = k.get_results()
```

`calc_experimental()` stores the raw variogram cloud on the model.
`calc_average()` bins that cloud by lag distance. `fit(method="pair", ...)`
uses the cached averaged table unless a table is passed explicitly, and
`inplace=True` replaces the template structure with the fitted values.
`VariogramSystem.apply()` validates the target `Kriging` object, then transfers
the stored observations and fitted variogram structures.  `set_search()` still
belongs to the kriging object because it configures the engine-side neighbor
search used during solving.

### When anisotropy is applied

`VariogramModel.set_vgm()` stores theoretical-model anisotropy, but it does
**not** automatically transform `calc_experimental()`.  These operations use
anisotropy differently:

| Operation | Uses anisotropy from `set_vgm()`? | Effect |
|---|---:|---|
| `calc_experimental()` | No | Physical pair distances and an isotropic cutoff unless `anisotropy=...` is passed explicitly |
| `calc_average()` | No automatic choice | Bins the selected `h_col`; default is physical `distance` |
| `calc_directional_average()` | Yes, orientation only | Uses stored azimuth/dip/plunge to define major and minor directions |
| `calc_covariance()` / `calc_variogram()` | Yes | Evaluates coordinate pairs with each structure's stored ranges and rotation |
| `plot_map()` / `plot_map3d()` | Yes, for overlays/rotated views | The empirical values remain data-derived |
| `apply_to()` / `to_kriging_specs()` | Yes | Transfers stored anisotropy to the kriging engine |

The raw empirical semivariance `0.5 * (z_i - z_j)²` is data-derived and does
not itself depend on a variogram model.  To apply one geometric transform
during empirical pair selection, pass ratios and angles explicitly:

```python
raw = model.calc_experimental(
    cutoff=2000.0,
    anisotropy={
        "anis1": 0.30,    # minor1 / major range
        "anis2": 0.15,    # minor2 / major range; relevant in 3-D
        "azimuth": 75.0,
        "dip": 10.0,
        "plunge": 0.0,
    },
    verbose=False,
)
```

This produces two lag columns:

- `distance`: physical Euclidean distance, unchanged.
- `anisotropic_distance`: equivalent major-axis distance after rotation and
  minor/major scaling.

The `cutoff` is applied to `anisotropic_distance`, but ordinary averaging still
uses physical distance by default.  Select the transformed lag explicitly:

```python
avg_physical = model.calc_average(h_width=100.0)
avg_transformed = model.calc_average(
    h_col="anisotropic_distance",
    h_width=100.0,
)
```

The separation is intentional.  A nested model can contain multiple structures
with different ranges or rotations, so there may be no single stored
anisotropy that can unambiguously define empirical pair selection or binning.
`VariogramSystem.set_vgm()` follows the same rule for direct and cross clouds:
its stored pair-model anisotropy is not implicitly passed to
`VariogramSystem.calc_experimental()`.
For anisotropy discovery, the usual workflow is to calculate the physical
cloud with `calc_angle=True`, inspect directional maps/averages, and only then
store the selected orientation and ranges with `set_vgm()` or
`set_anisotropy()`.

`SpaceTimeVariogramModel` is the exception: spatial anisotropy configured with
`set_spacetime_anisotropy()` is part of its joint space-time distance
definition and is forwarded to its empirical calculation unless explicitly
overridden.

### Parallel empirical calculation

Pair generation is O(n²) and is usually the dominant cost for a large
empirical variogram cloud.  `calc_experimental()` and the lower-level
`raw_vgm()`, `raw_cross_vgm()`, and `cross_vgm()` functions accept `n_jobs`:

```python
raw = model.calc_experimental(
    cutoff=2000.0,
    n_jobs=None,     # adaptive default: 1 to 4 worker threads
    verbose=False,
)

# Or choose an explicit limit:
raw = model.calc_experimental(cutoff=2000.0, n_jobs=4, verbose=False)
```

For a finite Cartesian cutoff, a SciPy `cKDTree` first produces only spatial
candidate pairs.  This avoids scanning the complete O(n²) triangle when the
cutoff is small relative to the study area.  An anisotropic cutoff uses a
conservative Euclidean search radius, followed by the exact anisotropic test,
so no valid pair is lost.

The candidate indices are divided into bounded vectorized chunks (about
32,000 pairs each).  Without a usable KD-tree cutoff, upper-triangle index
chunks are generated incrementally instead; the complete `np.triu_indices`
arrays are never materialized.  Worker threads calculate lag vectors,
distances, semivariances, time filters, and angles for separate chunks.
Results are collected in row-major pair order, so `n_jobs=1` and `n_jobs>1`
produce the same rows and values.

`n_jobs=None` is the adaptive default.  The decision is made after `maxobs`
subsampling and, where applicable, KD-tree cutoff candidate generation:

- Up to 262,144 candidate pairs: one worker.
- Above that threshold: approximately one worker per 262,144 candidate pairs,
  capped at four and at the usable physical-core count.

Thus a large `nobs` with a tight cutoff can remain single-threaded when the
KD-tree finds few candidate pairs, while a smaller dense cloud can use several
workers.  Set `n_jobs=1` to force sequential execution, a positive integer for
an explicit limit, or `n_jobs=-1` to use all usable physical cores.  Explicit
values are also capped at that physical-core limit.  Physical cores are
detected with `psutil.cpu_count(logical=False)` and constrained by the current
process CPU affinity, which respects container and job-scheduler restrictions.

Threading helps most when the retained or unfiltered cloud contains enough
pairs to fill several chunks.  With a tight finite cutoff, KD-tree candidate
generation can make the sequential path so fast that extra threads add little.
For small clouds, scheduling and allocation overhead can outweigh the gain.
A custom Python distance callback may also scale poorly because callback code
can remain limited by the Python GIL.

Parallelism reduces elapsed time but not the O(n²) output size.  Use `maxobs`
when the complete pair cloud would exceed available memory:

```python
raw = model.calc_experimental(
    cutoff=2000.0,
    maxobs=10_000,
    seed=1234,
    n_jobs=-1,
    verbose=False,
)
```

Before allocating pair results, the empirical functions estimate the final
numeric DataFrame size from the effective pair count and requested columns.
For finite Cartesian cutoffs this uses the exact KD-tree candidate count;
otherwise it uses the full triangular or rectangular pair count after
`maxobs` sampling.  The calculation raises `MemoryError` when the estimate
exceeds 80% of currently available RAM:

```python
raw = model.calc_experimental(
    cutoff=2000.0,
    max_memory_fraction=0.6,  # use a stricter 60% limit
)
```

Pass `max_memory_fraction=None` to override the guard for a deliberately large
calculation.  The estimate is an upper bound when a time cutoff or exact
anisotropic cutoff will remove some spatial candidates.

Preallocating the pandas DataFrame would not remove the fundamental memory
cost: each retained pair still needs indices, lag components, distance,
semivariance, and optional time/angle columns.  The implementation instead
builds bounded calculation chunks and allocates the final columns only after
filtering.  `maxobs` and a finite spatial cutoff remain the effective ways to
control memory.

### Custom Distance Metrics

For non-Euclidean coordinates, `calc_experimental()` supports custom distance metrics:

```python
def manhattan(u, v):
    return np.sum(np.abs(u - v))

# The metric is forwarded to scipy.spatial.distance.cdist
raw = model.calc_experimental(metric=manhattan, verbose=False)
```

### Statistical Binning and Robust Estimators

For irregular pair densities, pass explicit bin edges instead of one fixed
width. This keeps narrow bins where short-lag detail matters and wider bins
where pairs become sparse:

```python
avg = model.calc_average(
    h_width=[0.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
)
```

Alternatively, `calc_average()` supports automated statistical binning heuristics
via string keywords, which determine the optimal number of bins or widths based
on the lag distribution:

```python
avg = model.calc_average(h_width="fd")      # Freedman-Diaconis rule (robust to outliers)
avg = model.calc_average(h_width="sturges") # Sturges' rule (optimal for normal data)
avg = model.calc_average(h_width="scott")   # Scott's rule (minimizes MSE)
avg = model.calc_average(h_width="kmeans", h_bins=15) # Density-based k-means clustering
```

To handle outliers (such as localized extreme sample values) that skew traditional
variance calculations, you can specify a robust estimator:

```python
avg = model.calc_average(estimator="dowd")   # Dowd's Median Absolute Deviation
avg = model.calc_average(estimator="genton") # Genton's scale estimator (Qn proxy)
```

Space-time clouds can use independent variable-width spatial and temporal
bins:

```python
avg = model.calc_average(
    h_width=[0.0, 100.0, 250.0, 500.0, 1000.0, 2500.0],
    t_col="time_lag",
    t_width=[0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)
```

Each array gives bin edges. Intervals are left-closed and right-open, while
the last interval includes its final edge. Pairs outside the supplied edge
range are omitted. Scalar `h_width` and `t_width` values retain fixed-width
behavior.

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

model.set_vgm(vtype="sph", nugget=0.04, sill=0.35, a_major=250.0)
model.set_vgm(vtype="exp", nugget=0.0, sill=0.55, a_major=900.0)

model.calc_experimental(cutoff=2500.0, verbose=False)
model.calc_average(h_width=100.0)
model.fit(inplace=True)
```

For ordinary one-dimensional lag fitting, the flat fit vector is ordered as
`sill, a_major` for every stored structure, followed by a single trailing
`nugget` when `fit_nugget=True`. Store that nugget on the first structure;
normally there is no need to add a separate `vtype="nug"` structure.

Product covariance groups use the same convention as the kriging API:

```python
temporal = VariogramModel()
temporal.set_vgm("gau", sill=0.4, a_major=30.0)
temporal.set_vgm("hol", sill=1.0, a_major=0.5, product=True)
```

The hole-effect covariance is multiplied with the preceding Gaussian
covariance instead of added. Product members use the same flat fitting order as
additive structures.

The model can be copied into kriging in either form:

```python
model.apply_to(k, ivar=1, jvar=1)
```

or:

```python
for spec in model.to_kriging_specs(replace=True):
    k.set_vgm(ivar=1, jvar=1, **spec)
```

## Anisotropy detection

Detecting preferred directions before fitting ranges prevents fitting the
isotropic average of what are actually two or three distinct ranges.  The
standard sequence is:

1. Compute the raw cloud with `calc_angle=True`.
2. Inspect a variogram map or run `estimate_aniso_angle()`.
3. Lock `azimuth` (and `dip` for 3-D) in the model template.
4. Proceed to directional range fitting.

### Recommended anisotropy workflow

The plotting methods do not update the model automatically.  By default,
`plot_map()` and `plot_map3d()` show the orientation already stored on the
first model structure (`angle_aniso="model"`).  Pass
`angle_aniso="estimate"` or `estimate=True` to estimate an orientation from
the empirical cloud and display it.  Then extract the numeric estimate,
inspect it, and store it explicitly.

The corresponding Python functions for each step are:

| Step | Purpose | Python function |
|---|---|---|
| 1 | Store observations | `model.set_obs()` |
| 2 | Create an initial model template | `model.set_vgm()` |
| 3 | Calculate the physical empirical cloud | `model.calc_experimental(calc_angle=True)` |
| 4 | Inspect the empirical map | `model.plot_map()` or `model.plot_map3d()` |
| 5 | **Fit and apply the anisotropy orientation** | **`model.fit_aniso_angle()`** |
| 6 | Calculate major/minor directional curves | `model.calc_directional_average()` |
| 7 | Fit directional ranges (orientation fixed) | `model.fit_anisotropy()` |
| 8 | Validate the fitted orientation and ranges | `model.plot_map(angle_aniso="model", ellipse_aniso="model")` or `model.plot_map3d(angle_aniso="model")` |
| 9 | Transfer the final model to kriging | `model.apply_to()` |

**Fit the orientation first.** Step 5 (`model.fit_aniso_angle()`) estimates the
anisotropy orientation from the empirical cloud and writes `azimuth` / `dip` /
`plunge` into the structures, so the directional binning (step 6) and the
range fit (step 7) use the correct axes.  A 3-D cloud uses the multi-started,
model-based profile fit; a 2-D cloud uses the fast PCA azimuth.  It must run
**before** `fit_anisotropy`.  For manual control you can instead read
`estimate_aniso_angle()` (or `fit_aniso_angle()`) yourself and apply the angles
with `model.set_anisotropy()`, as the 2-D example below shows.

#### Two-dimensional example

```python
from krigekit import Kriging, VariogramModel
from krigekit.variogram import estimate_aniso_angle

model = VariogramModel()
model.set_obs(obs_coord, obs_value)

# Initial template.  At this stage the angle and minor range are provisional.
model.set_vgm(
    vtype="sph",
    nugget=0.05,
    sill=0.95,
    a_major=1000.0,
    a_minor1=1000.0,
    azimuth=0.0,
)

# Build the physical lag cloud.  Stored set_vgm anisotropy is intentionally
# not applied to this exploratory calculation.
raw = model.calc_experimental(
    cutoff=3000.0,
    calc_angle=True,
    verbose=False,
)

# Inspect the world-coordinate map.  This displays the stored model angle
# (currently 0 degrees) but does not estimate or modify anything.
model.plot_map(angle_aniso="model", cutoff=2500.0)

# Ask the plotting helper to estimate and display the empirical major axis.
# This is visual only: the model remains unchanged.
model.plot_map(angle_aniso="estimate", cutoff=2500.0)

# Obtain the numeric estimate so it can be inspected and stored explicitly.
# In 2-D this returns the azimuth and the minor/major ratio (anis1).
(azimuth,), (ratio_guess,) = estimate_aniso_angle(raw)
print(f"estimated azimuth={azimuth:.1f}, ratio={ratio_guess:.3f}")

# Lock the selected orientation into the model.  You may use the estimated
# ratio as a starting value or replace it with a geologically informed value.
model.set_anisotropy(
    azimuth=azimuth,
    ratio_minor1=ratio_guess,
)

# Build empirical curves along the stored major and minor axes.
directional = model.calc_directional_average(
    h_bins=15,
    cutoff=2500.0,
    angle_tol=15.0,
)

# Fit sill, major range, minor range, and nugget while keeping azimuth fixed.
model.fit_anisotropy(
    directional,
    p0=(0.95, 1000.0, 300.0, 0.05),
    weight_col="count",
    inplace=True,
)

# Validate the final model against the unchanged empirical cloud.
model.plot_map(
    angle_aniso="model",
    ellipse_aniso="model",
    cutoff=2500.0,
)

# Transfer the fitted structures, including anisotropy, to kriging.
k = Kriging(ndim=2, nvar=1)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value)
k.set_grid(coord=grid_coord)
model.apply_to(k, ivar=1, jvar=1)
k.set_search(ivar=1, nmax=24)
k.solve()
```

#### Three-dimensional differences

For 3-D data, `model.fit_aniso_angle()` fits all three angles
(`azimuth` / `dip` / `plunge`) with the multi-started, model-based profile fit
and writes them into the structures:

```python
model.calc_experimental(
    cutoff=3000.0,
    calc_angle=True,
    verbose=False,
)

# Visual estimate only; does not modify model.structure.
model.plot_map3d(angle_aniso="estimate", cutoff=2500.0)

# Fit and apply the 3-D orientation (and seed the minor ranges from the fitted
# anisotropy ratios) -- run this before the directional range fit.
model.fit_aniso_angle()

directional = model.calc_directional_average(
    h_bins=15,
    cutoff=2500.0,
    angle_tol=15.0,
    include_minor2=True,
)

model.fit_anisotropy(
    directional,
    p0=(0.95, 1000.0, 400.0, 200.0, 0.05),
    include_minor2=True,
    weight_col="count",
    inplace=True,
)

# World-axis fences with the fitted orientation overlaid.
model.plot_map3d(angle_aniso="model", cutoff=2500.0)

# Or rotate the fences into the fitted model's principal planes.
model.plot_map3d(
    angle_aniso="model",
    rotate_fences=True,
    cutoff=2500.0,
)
```

The automated angle estimate is a starting point, not an automatic model
decision.  Check the map for sparse bins, short-lag noise, multiple continuity
directions, and geological plausibility before calling `set_anisotropy()`.

### Variogram map (2-D)

`plot_map()` bins the raw cloud by signed lag offset (`d0`, `d1`) and renders
the result as a pcolormesh.  The direction of minimum semivariance is the
major (maximum continuity) axis.

```python
model.calc_experimental(cutoff=3000.0, calc_angle=True, verbose=False)

# Plain map.
model.plot_map(cutoff=2500.0)

# Overlay an automatically estimated orientation.
model.plot_map(angle_aniso="estimate", cutoff=2500.0)
```

`angle_aniso="estimate"` calls `estimate_aniso_angle()` internally and overlays
the result.  Use the map to sanity-check the estimate, then set an explicit
`azimuth` in the model template before fitting.

### Variogram map (3-D)

`plot_map3d()` draws up to three orthogonal fence sections through the
lag-space origin, coloured by average semivariance.  By default
(`rotate_fences=False`) the fences align with the world X/Y/Z axes so that
model angles can be read directly off the plot:

- **Fence A** (always) — horizontal XY plane; shows the azimuth pattern.
- **Fence B** (`n_fences ≥ 2`, default) — vertical XZ (East–West) section;
  shows the dip.
- **Fence C** (`n_fences ≥ 3`) — vertical YZ (North–South) section.

When model angles are supplied, a red line is projected onto each fence
showing the major-axis direction so the fitted orientation can be compared
against the empirical map.

```python
model.calc_experimental(cutoff=3000.0, verbose=False)

# Two world-axis fences (default) with model-angle overlay.
model.plot_map3d(cutoff=2500.0)

# Estimate the orientation from the cloud if no model is fitted yet.
model.plot_map3d(angle_aniso="estimate", cutoff=2500.0)

# Three fences (adds North–South vertical section).
model.plot_map3d(cutoff=2500.0, n_fences=3)

# Rotate fences to the model's principal planes instead.
model.plot_map3d(cutoff=2500.0, rotate_fences=True)

# For sparse 3-D clouds, fill empty in-range bins from nearest occupied bins.
model.plot_map3d(cutoff=2500.0, fill_nan=True)
```

All fence polygons are rendered in a single `Poly3DCollection` so depth-sorting
is correct when rotating the interactive plot.  `fill_nan=True` is a
display-only nearest-neighbour fill constrained to the cutoff radius; it does
not change the underlying data.

### Automated direction estimation

Two estimators are available, both returning the angles plus the minor/major
anisotropy ratios:

- `estimate_aniso_angle()` -- a fast weighted PCA of the near-origin lag cloud.
  A good seed; best on dense/gridded clouds.  Choose `r_max` near **half the
  shortest range** (too small is dominated by the sampling lattice on grids).
- `fit_aniso_angle()` -- a slower, multi-started **model-based** profile fit.
  More robust on scattered, strongly anisotropic clouds where the PCA can be
  tens of degrees wrong; itself biased by a regular sampling lattice.

```python
from krigekit.variogram import estimate_aniso_angle, fit_aniso_angle

model.calc_experimental(cutoff=3000.0, calc_angle=True, verbose=False)
raw = model.raw_variogram_

# 2-D: returns ((azimuth_deg,), (anis1,))
(azimuth,), (anis1,) = estimate_aniso_angle(raw)
print(f"Major axis azimuth: {azimuth:.1f} deg,  ratio: {anis1:.2f}")

# 3-D: returns ((azimuth_deg, dip_deg, plunge_deg), (anis1, anis2))
(azimuth, dip, plunge), (anis1, anis2) = estimate_aniso_angle(raw, dim3d=True)
(azimuth, dip, plunge), (anis1, anis2) = fit_aniso_angle(raw)   # robust refinement
```

Usually you do not call these directly -- `model.fit_aniso_angle()` runs the
right one for the cloud's dimension and applies the result.  The returned
`azimuth` / `dip` / `plunge` can also be fed into `set_vgm()` or
`set_anisotropy()`:

```python
model.set_vgm(
    vtype="sph",
    sill=0.8,
    nugget=0.05,
    a_major=1000.0,
    a_minor1=300.0,
    azimuth=azimuth,   # from estimate_aniso_angle
)
```

The estimate uses a short-lag neighbourhood (default: 25th-percentile of lag
distances) so that only the near-origin, high-continuity pairs influence the
PCA.  Pass `r_max` explicitly to control the neighbourhood radius.  The
estimate is exploratory — always cross-check it against the variogram map before
locking the angle in the production model.

## Directional anisotropy fitting

Once the orientation is established from the map or automated estimate, fix
`azimuth` (and `dip` for 3-D) in the model template and fit major and minor
ranges from directional averages.

```python
model = VariogramModel()
model.set_obs(obs_coord, obs_value)
model.set_vgm(
    vtype="sph",
    sill=0.8,
    nugget=0.05,
    a_major=1000.0,
    a_minor1=300.0,
    azimuth=90.0,       # locked from detection step
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
`a_minor2` when `include_minor2=True`.

## Fitting the short 3-D axis

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

Do not pass `weights` or `weight_col` together with `sigma_col`.

### Goodness-of-Fit Metrics

You can calculate goodness-of-fit metrics to evaluate how well the theoretical model
captures the experimental variogram. Pass `return_metrics=True` to `fit()`:

```python
fitted, cov, metrics = model.fit(return_metrics=True)

print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"R-squared: {metrics['R2']:.4f}")
```

The metrics dictionary includes Mean Squared Error (`MSE`), Root Mean Squared Error (`RMSE`), Mean Absolute Error (`MAE`), and the coefficient of determination (`R2`).
These metrics are calculated over the averaged variogram bins used during the fit.

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
model.set_anisotropy(ratio_minor1=0.35, azimuth=35.0)
```

`anis1` and `anis2` remain accepted aliases, but `ratio_minor1` and
`ratio_minor2` are clearer in new variogram-analysis code.

## Space-time coupling

Fit the spatial and temporal boundary marginals as ordinary `VariogramModel`
objects, then compose them in a `SpaceTimeVariogramModel` for the full
two-dimensional lag surface.

### Sum-metric coupling

After fitting spatial and temporal marginals separately, calculate a full
space-time lag surface and fit the coupling with
`fit(model="sum_metric")`, which returns a `FitResult`:

```python
from krigekit import SpaceTimeVariogramModel

joint = SpaceTimeVariogramModel(
    spatial=spatial_model,
    temporal=temporal_model,
)
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

result = joint.fit(
    model="sum_metric",
    transform="lin",
    p0=(1.0, 1.0, 100.0, 20.0),
    weight_cap_quantile=0.90,
)
print(result.summary())          # labelled spatial_scale/temporal_scale/joint_sill/at
specs = joint.to_sum_metric_kriging_specs()
```

The flat parameter order is `spatial_scale`, `temporal_scale`, one
`joint_sill` per spatial structure, and `at`. The marginal scales reconcile
models fitted on the spatial and temporal boundaries with the interior
space-time surface. Fixing `time_sill=1` avoids redundancy between
`time_sill` and `at` for the linear temporal metric transform. Capping pair
counts prevents a few densely populated bins from dominating the surface fit.

The groundwater-level gallery example demonstrates this workflow, including
transfer of the scaled marginals, joint sill, and temporal metric scale to
`SpaceTimeKriging`.

### Product-sum coupling

For a product-sum model, calculate and average the full space-time cloud, then
fit the coefficients and marginal ranges.  When the spatial data are
anisotropic, pass the spatial anisotropy to `calc_experimental()` so that the
stored cloud uses the equivalent major-axis lag for spatial binning:

```python
joint = SpaceTimeVariogramModel()
joint.set_obs(obs_xyz, transformed_value, times=obs_time)
joint.calc_experimental(
    cutoff=5500.0,
    t_cutoff=13.0,
    anisotropy={"anis1": 1.0, "anis2": 0.2},
    verbose=False,
)
joint.calc_average(
    h_col="anisotropic_distance",
    h_width=500.0,
    t_col="time_lag",
    t_width=0.5,
)

joint.fit(
    model="product_sum",
    spatial_vtype="sph",
    temporal_vtype="gau",
    starts=[
        (0.10, 0.06, -0.005, 5000.0, 9.0),
        (0.12, 0.04, -0.020, 3000.0, 10.0),
    ],
    bounds=[
        (0.0, 0.30),
        (0.0, 0.20),
        (-0.20, 0.0),
        (1000.0, 8000.0),
        (1.0, 13.0),
    ],
    weight_cap_quantile=0.90,
)
specs = joint.to_spacetime_kriging_specs()
```

The fitted vector is `(a, b, p, spatial_range, temporal_range)`. Valid
covariance conversion requires `p <= 0`, `a + p > 0`, and `b + p > 0`.
Inspect `spacetime_fit_results_` because product-sum parameters can be weakly
identified even when the fitted surface looks reasonable.

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

# With no pair given, compute the cloud for every pair that has a model set.
system.calc_experimental(cutoff=2000.0, verbose=False)
system.calc_average(h_width=100.0)

result = system.fit_lmc(fit_ranges=True, fit_nugget=True)
result.target.apply(k)        # result is a FitResult; .target is the fitted system
```

`fit_lmc()` returns a `FitResult`: `result.target` is the fitted `VariogramSystem`
and `result.optimizer` is the SciPy result.  `system.fit(method="lmc")` and
`system.fit(method="pair", ivar=...)` are equivalent facades over `fit_lmc()` and
`fit_pair()`.  `fit_pair()` fits one pair independently.  For cokriging, prefer `fit_lmc()`
because it fits the requested pairs together while enforcing positive
semidefinite sill matrices for each nested structure.

Negative cross-nuggets and cross-sills are valid when the complete
coregionalization matrix is positive semidefinite. For example, an indicator
for fine texture can have negative cross-covariance with resistivity while both
direct variograms remain nonnegative. `fit_lmc()` enforces matrix validity;
do not clamp individual cross entries to zero based only on their sign.

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
system.apply(k)
```

Pass `corr=None` to estimate the correlation from collocated observations (the
two variables must then share coordinates); otherwise compute it from the
collocated subset and pass it explicitly.  Markov Model 2 is not yet
implemented.

In short: use `fit_lmc()` for co-sampled multivariate data, and
`set_markov_cross()` for sparse-primary / dense-secondary collocated cokriging
(Almeida & Journel, 1994; Goovaerts, 1997).

### Indicator coregionalization (closure)

For mutually exclusive, exhaustive categories (`I_1 + ... + I_K = 1`), use
`IndicatorVariogramSystem`.  It encodes raw labels into `K` indicator variables,
builds the `K × K` coregionalization from category proportions, and fits a
coregionalization that is both positive semidefinite **and** closed — every
matrix has rows summing to zero, matching the constraint `B · 1 = 0` implied by
the indicators summing to one.

```python
from krigekit import IndicatorVariogramSystem

system = IndicatorVariogramSystem(categories=["sand", "silt", "clay"])
system.set_categorical_obs(coord, categories)          # encode + record proportions

# Initial K×K block: theoretical autos p_k(1-p_k), closed cross sills -p_k p_l.
system.set_indicator_vgm(vtype="sph", a_major=500.0,
                         sill_strategy="theoretical", cross_strategy="closure")

system.calc_average(h_width=50.0)                       # empirical curves for all pairs
result = system.fit(method="closure")                  # PSD + closure-preserving LMC
result.target.validate_closure()                       # assert row sums ~ 0
result.target.apply(indicator_kriging)                 # transfer to IndicatorKriging
```

`fit(method="closure")` parameterizes each nested coregionalization matrix as
`B = Q L Lᵀ Qᵀ`, where the contrast basis `Q = contrast_basis(K)` spans the space
orthogonal to the all-ones vector, so positive semidefiniteness and closure hold
at every fitted lag.  See {doc}`indicator_kriging` for the full estimation/
simulation workflow and the available `sill_strategy`/`cross_strategy` options.

## Example gallery

Relevant examples:

- `examples/variogram/s_variogram_fitting.py`: nested 2-D indicator variogram.
- `examples/variogram/s_variogram_fitting_3d.py`: 3-D directional anisotropy.
- `examples/variogram/st_variogram_fitting_gwlevel.py`: marginal and
  sum-metric coupling fitting.
- `examples/variogram/st_variogram_fitting_ctet.py`: product-sum fitting and
  manual adjustment.
