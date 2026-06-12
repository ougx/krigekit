# Universal Kriging and Kriging with External Drift

Universal kriging and kriging with external drift are used when the spatial variable of interest is not stationary in its mean. Instead of assuming a constant unknown mean, these methods model the mean as a deterministic trend while still using a covariance or variogram model to describe spatially correlated residual variation.

In `krigekit`, these methods are intended for cases where the target variable varies systematically with location or with one or more auxiliary predictors. Typical examples include groundwater head controlled by elevation, soil properties related to terrain indices, air temperature related to altitude, or contaminant concentration influenced by distance to a source.

This page explains the conceptual difference between ordinary kriging, universal kriging, and kriging with external drift, then gives practical guidance for preparing inputs, selecting drift terms, and interpreting results.

## When to use this method

Use universal kriging or kriging with external drift when the target variable has a visible large-scale trend.

For example, ordinary kriging may be insufficient if:

* values increase or decrease systematically from west to east;
* values are strongly related to elevation;
* the local mean changes with distance from a river, shoreline, fault, source area, or urban center;
* a secondary raster or model output provides useful information about the expected spatial pattern;
* residuals look spatially correlated after removing a deterministic trend.

In these cases, the observed value can be understood as:

```{math}
Z(\mathbf{s}) = m(\mathbf{s}) + \varepsilon(\mathbf{s})
```

where:

* $Z(\mathbf{s})$ is the variable observed at location $\mathbf{s}$;
* $m(\mathbf{s})$ is the deterministic mean or drift;
* $\varepsilon(\mathbf{s})$ is a spatially correlated residual with mean zero.

The goal is to estimate $Z(\mathbf{s}_0)$ at an unsampled location $\mathbf{s}_0$, while accounting for both the deterministic drift and the spatial covariance of the residuals.

## Ordinary kriging, universal kriging, and KED

The three methods differ mainly in how they represent the mean.

### Ordinary kriging

Ordinary kriging assumes an unknown but locally constant mean:

```{math}
Z(\mathbf{s}) = \mu + \varepsilon(\mathbf{s})
```

The mean $\mu$ is not known, but it is assumed to be constant within the local search neighborhood.

Ordinary kriging is appropriate when there is no strong deterministic trend, or when the search neighborhood is small enough that the mean can reasonably be treated as locally constant.

### Universal kriging

Universal kriging assumes that the mean is a function of spatial coordinates:

```{math}
Z(\mathbf{s}) = \sum_{k=0}^{p} \beta_k f_k(\mathbf{s}) + \varepsilon(\mathbf{s})
```

where $f_k(\mathbf{s})$ are known drift functions, such as:

```{math}
1,\ x,\ y,\ x^2,\ xy,\ y^2
```

The coefficients $\beta_k$ are unknown and estimated implicitly through the kriging system.

Universal kriging is useful when the trend can be represented directly from coordinates, such as a linear regional gradient or a smooth polynomial surface.

### Kriging with external drift

Kriging with external drift uses one or more auxiliary variables to describe the trend:

```{math}
Z(\mathbf{s}) = \sum_{k=0}^{p} \beta_k q_k(\mathbf{s}) + \varepsilon(\mathbf{s})
```

where $q_k(\mathbf{s})$ are external drift variables known at both observation locations and prediction locations.

Examples of external drift variables include:

* elevation;
* slope;
* distance to river;
* remote sensing index;
* hydraulic conductivity zone indicator;
* land-use class indicator;
* output from a deterministic model;
* another spatial variable measured more densely than the target variable.

KED is useful when the auxiliary variable has a meaningful relationship with the target variable and is available over the full prediction grid.

## Input requirements

Universal kriging and KED require the usual kriging inputs plus drift information.

At minimum, the following are needed:

```python
obs_coord      # observation coordinates, shape (nobs, ndim)
obs_value      # observed target values, shape (nobs,)
grid_coord     # prediction coordinates, shape (ngrid, ndim)
vgm_spec       # variogram or covariance specification for residuals
```

For universal kriging, drift terms can be generated from coordinates:

```python
obs_drift      # drift design matrix at observation locations, shape (nobs, ndrift)
grid_drift     # drift design matrix at prediction locations, shape (ngrid, ndrift)
```

For KED, drift terms usually come from auxiliary data:

```python
obs_drift      # external drift values at observation locations
grid_drift     # external drift values at prediction locations
```

The number and order of drift columns must match between `obs_drift` and `grid_drift`.

For example, if the drift terms are intercept, elevation, and distance to river, then both matrices should use the same column order:

```text
column 0: intercept
column 1: elevation
column 2: distance_to_river
```

## Basic example: universal kriging with a linear coordinate drift

The simplest universal kriging model uses an intercept plus linear coordinate terms:

```{math}
m(x, y) = \beta_0 + \beta_1 x + \beta_2 y
```

The drift design matrix can be created from the coordinates:

```python
import numpy as np

def linear_xy_drift(coord):
    """
    Build a linear drift matrix from 2D coordinates.

    Parameters
    ----------
    coord : ndarray of shape (n, 2)
        Coordinate array with columns x and y.

    Returns
    -------
    drift : ndarray of shape (n, 3)
        Drift matrix with columns [1, x, y].
    """
    x = coord[:, 0]
    y = coord[:, 1]

    return np.column_stack([
        np.ones(coord.shape[0]),
        x,
        y,
    ])
```

Then use the same drift function for observations and prediction locations:

```python
obs_drift = linear_xy_drift(obs_coord)
grid_drift = linear_xy_drift(grid_coord)
```

A typical universal kriging call may look like this:

```python
from krigekit import universal_kriging

est, var = universal_kriging(
    obs_coord=obs_coord,
    obs_value=obs_value,
    grid_coord=grid_coord,
    obs_drift=obs_drift,
    grid_drift=grid_drift,
    vgm_spec=vgm_spec,
    nmax=24,
)
```

Here, `est` contains the kriging estimates and `var` contains the kriging variances.

Depending on the exact `krigekit` API, the function name or argument names may differ. The important principle is that the drift basis must be provided at both observation and prediction locations.

## Basic example: KED with elevation as external drift

Suppose the target variable is groundwater head, and elevation is available at all observation points and all grid cells.

The drift matrix can be built as:

```python
def elevation_drift(elevation):
    """
    Build a drift matrix using elevation as external drift.

    Parameters
    ----------
    elevation : ndarray of shape (n,)
        Elevation values.

    Returns
    -------
    drift : ndarray of shape (n, 2)
        Drift matrix with columns [1, elevation].
    """
    return np.column_stack([
        np.ones(elevation.shape[0]),
        elevation,
    ])
```

Prepare drift values:

```python
obs_drift = elevation_drift(obs_elevation)
grid_drift = elevation_drift(grid_elevation)
```

Then run kriging with external drift:

```python
from krigekit import universal_kriging

est, var = universal_kriging(
    obs_coord=obs_coord,
    obs_value=obs_value,
    grid_coord=grid_coord,
    obs_drift=obs_drift,
    grid_drift=grid_drift,
    vgm_spec=vgm_spec,
    nmax=24,
)
```

Although this example uses the same function name as universal kriging, the method is conceptually KED because the drift basis uses an external variable rather than only coordinate terms.

## Multiple external drift variables

KED can use multiple auxiliary variables.

For example:

```python
obs_drift = np.column_stack([
    np.ones(nobs),
    obs_elevation,
    obs_distance_to_river,
    obs_slope,
])

grid_drift = np.column_stack([
    np.ones(ngrid),
    grid_elevation,
    grid_distance_to_river,
    grid_slope,
])
```

Then pass the drift matrices to the kriging routine:

```python
est, var = universal_kriging(
    obs_coord=obs_coord,
    obs_value=obs_value,
    grid_coord=grid_coord,
    obs_drift=obs_drift,
    grid_drift=grid_drift,
    vgm_spec=vgm_spec,
    nmax=32,
)
```

Use multiple drift variables only when they are physically meaningful and available consistently over the prediction domain.

Adding many drift terms can make the kriging system unstable, especially when local neighborhoods are small or when drift variables are highly correlated.

## The kriging system

Universal kriging and KED add unbiasedness constraints to the ordinary kriging system.

For a prediction location $\mathbf{s}_0$, the estimator is:

```{math}
\hat{Z}(\mathbf{s}_0) = \sum_{i=1}^{n} \lambda_i Z(\mathbf{s}_i)
```

The weights $\lambda_i$ are chosen so that the estimator is unbiased for the specified drift functions:

```{math}
\sum_{i=1}^{n} \lambda_i f_k(\mathbf{s}_i) = f_k(\mathbf{s}_0)
```

for each drift term $k$.

The resulting system can be written as:

```{math}
\begin{bmatrix}
C & F \\
F^T & 0
\end{bmatrix}
\begin{bmatrix}
\lambda \\
\mu
\end{bmatrix}
=
\begin{bmatrix}
c_0 \\
f_0
\end{bmatrix}
```

where:

* $C$ is the covariance matrix among neighboring observations;
* $F$ is the drift matrix at neighboring observation locations;
* $c_0$ is the covariance vector between observations and the prediction location;
* $f_0$ is the drift vector at the prediction location;
* $\lambda$ are kriging weights;
* $\mu$ are Lagrange multipliers for the drift constraints.

The covariance or variogram model should describe the residual spatial variation after accounting for the drift.

## Choosing drift terms

The most important modeling decision is the drift specification.

A good drift variable should satisfy three conditions:

1. It should be related to the target variable.
2. It should be known at both observation and prediction locations.
3. It should represent broad-scale structure rather than short-range noise.

Common choices include:

| Situation                    | Possible drift terms                               |
| ---------------------------- | -------------------------------------------------- |
| Regional gradient            | $1, x, y$                                          |
| Curved regional trend        | $1, x, y, x^2, xy, y^2$                            |
| Elevation control            | $1, \text{elevation}$                              |
| Terrain influence            | $1, \text{elevation}, \text{slope}, \text{aspect}$ |
| River or coastline effect    | $1, \text{distance to river}$                      |
| Urban or land-use influence  | land-use indicators                                |
| Model-assisted interpolation | deterministic model output                         |

Avoid adding drift terms simply because they are available. A weak or noisy drift can make predictions worse.

## Variogram modeling for residuals

For universal kriging and KED, the variogram should ideally describe residuals, not the raw target variable.

A practical workflow is:

1. Fit a preliminary trend model using the selected drift variables.
2. Compute residuals:

```{math}
r(\mathbf{s}) = Z(\mathbf{s}) - \hat{m}(\mathbf{s})
```

3. Estimate the experimental variogram of the residuals.
4. Fit a variogram model to the residual variogram.
5. Use that variogram model in universal kriging or KED.

For example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=False)
model.fit(obs_drift, obs_value)

trend_obs = model.predict(obs_drift)
residual = obs_value - trend_obs
```

Then estimate and fit the residual variogram using your preferred variogram workflow.

This residual-based approach is usually more appropriate than fitting a variogram directly to the raw values when a strong trend is present.

## Search neighborhood requirements

Universal kriging and KED require enough neighboring data points to support the drift constraints.

If there are `ndrift` drift terms, then each local neighborhood must contain more than `ndrift` observations. In practice, it should contain substantially more.

For example:

| Drift terms                 | Minimum observations | Suggested `nmax` |
| --------------------------- | -------------------: | ---------------: |
| Intercept only              |                   2+ |            12–24 |
| Intercept, x, y             |                   4+ |            16–32 |
| Intercept, elevation        |                   3+ |            16–32 |
| Intercept, x, y, x², xy, y² |                   7+ |            32–64 |

If too few neighbors are used, the kriging matrix may become singular or unstable.

Practical suggestions:

```python
nmax = 24      # reasonable starting point for simple drift
nmax = 32      # safer for multiple drift variables
nmax = 48      # useful for polynomial drift or noisy data
```

Also consider setting a minimum number of neighbors:

```python
nmin = 8
```

If the local neighborhood has fewer than `nmin` observations, the prediction can be skipped or marked as missing.

## Scaling drift variables

External drift variables should usually be scaled before use, especially if they have very different magnitudes.

For example, coordinates may be in meters, elevation may range from 100 to 2000, and distance to river may range from 0 to 100000. Large differences in scale can cause numerical instability.

A simple standardization is:

```python
def standardize_train_apply(obs_x, grid_x):
    mean = np.nanmean(obs_x)
    std = np.nanstd(obs_x)

    if std == 0:
        raise ValueError("Cannot standardize a constant drift variable.")

    obs_scaled = (obs_x - mean) / std
    grid_scaled = (grid_x - mean) / std

    return obs_scaled, grid_scaled
```

Use the observation statistics to scale both observation and grid drift values:

```python
obs_elev_s, grid_elev_s = standardize_train_apply(
    obs_elevation,
    grid_elevation,
)

obs_drift = np.column_stack([
    np.ones(nobs),
    obs_elev_s,
])

grid_drift = np.column_stack([
    np.ones(ngrid),
    grid_elev_s,
])
```

Do not standardize observation and grid drift values separately. They must be transformed using the same mean and standard deviation.

## Handling missing drift values

KED requires external drift values at both observation and prediction locations.

If a prediction location has missing drift values, then the prediction cannot be computed reliably at that location.

Common strategies include:

* mask out grid cells with missing drift;
* fill missing external drift using a separate interpolation method;
* restrict prediction to the valid drift domain;
* use a simpler drift model where auxiliary data are incomplete.

For observation locations, missing drift values are more serious. Observations with missing drift values should usually be removed from the KED run.

Example:

```python
valid_obs = np.isfinite(obs_value) & np.all(np.isfinite(obs_drift), axis=1)

obs_coord_valid = obs_coord[valid_obs]
obs_value_valid = obs_value[valid_obs]
obs_drift_valid = obs_drift[valid_obs]
```

For grid locations:

```python
valid_grid = np.all(np.isfinite(grid_drift), axis=1)

est = np.full(grid_coord.shape[0], np.nan)
var = np.full(grid_coord.shape[0], np.nan)

est_valid, var_valid = universal_kriging(
    obs_coord=obs_coord_valid,
    obs_value=obs_value_valid,
    grid_coord=grid_coord[valid_grid],
    obs_drift=obs_drift_valid,
    grid_drift=grid_drift[valid_grid],
    vgm_spec=vgm_spec,
    nmax=24,
)

est[valid_grid] = est_valid
var[valid_grid] = var_valid
```

## Interpreting kriging variance

The kriging variance from universal kriging or KED measures interpolation uncertainty under the specified covariance and drift model.

It is affected by:

* observation spacing;
* variogram range and sill;
* nugget effect;
* search neighborhood;
* drift constraints;
* local geometry of observations;
* availability and quality of drift variables.

It should not be interpreted as including all forms of model uncertainty. In particular, it usually does not fully account for uncertainty in the choice of drift variables or variogram model.

A low kriging variance does not necessarily mean the drift model is correct. It only means the estimate is well constrained under the assumptions of the selected model.

## Cross-validation

Cross-validation is strongly recommended.

A typical leave-one-out or k-fold workflow should check:

* mean error;
* mean absolute error;
* root mean squared error;
* standardized error;
* spatial pattern of residuals;
* whether residuals remain correlated with external drift variables.

Useful diagnostic questions include:

* Does KED improve prediction error compared with ordinary kriging?
* Are errors reduced in areas where the drift variable is informative?
* Are residuals still spatially patterned?
* Are predictions biased at high or low drift values?
* Does the method extrapolate poorly outside the observed drift range?

A simple comparison table is useful:

| Method                 | ME | MAE | RMSE | Notes            |
| ---------------------- | -: | --: | ---: | ---------------- |
| Ordinary kriging       |    |     |      | baseline         |
| Universal kriging      |    |     |      | coordinate trend |
| KED: elevation         |    |     |      | external drift   |
| KED: elevation + slope |    |     |      | multiple drift   |

The best model is not always the one with the most drift variables. Prefer the simplest model that improves prediction and remains physically interpretable.

## Avoiding extrapolation problems

KED can behave poorly when prediction locations have external drift values outside the range observed in the sample data.

For example, if observation elevations range from 200 to 600 meters, but prediction grid elevations range from 50 to 1200 meters, KED may extrapolate the trend beyond the support of the data.

Check drift ranges before prediction:

```python
def print_range(name, obs_x, grid_x):
    print(f"{name}:")
    print(f"  observation range: {np.nanmin(obs_x):.3f} to {np.nanmax(obs_x):.3f}")
    print(f"  grid range:        {np.nanmin(grid_x):.3f} to {np.nanmax(grid_x):.3f}")

print_range("elevation", obs_elevation, grid_elevation)
```

If the grid drift range extends far beyond the observed range, consider:

* collecting more observations;
* limiting the prediction domain;
* using a simpler drift model;
* transforming the drift variable;
* checking predictions carefully in extrapolation zones.

## Relationship to regression kriging

KED is closely related to regression kriging, but the two methods are not identical in implementation.

In regression kriging, the workflow is usually:

1. Fit a regression model from auxiliary variables.
2. Compute residuals.
3. Krige the residuals.
4. Add the regression prediction and kriged residual.

In KED, the drift coefficients are estimated implicitly inside the kriging system through unbiasedness constraints.

In many practical cases, regression kriging and KED produce similar results, especially when the same drift variables and residual covariance model are used. However, they are not always numerically identical.

KED is often preferred when the drift should be handled directly within the kriging system. Regression kriging is often more flexible when using nonlinear machine learning models for the trend.

## Practical workflow

A recommended workflow is:

1. Plot the data and identify possible large-scale trends.
2. Select candidate drift variables.
3. Check whether drift variables are available at both observations and grid locations.
4. Scale or transform drift variables if needed.
5. Fit a preliminary trend model.
6. Compute residuals.
7. Fit a residual variogram.
8. Run universal kriging or KED.
9. Compare against ordinary kriging using cross-validation.
10. Inspect residual maps and uncertainty maps.

Example outline:

```python
# 1. Build drift matrices
obs_drift = np.column_stack([
    np.ones(nobs),
    obs_elevation_scaled,
])

grid_drift = np.column_stack([
    np.ones(ngrid),
    grid_elevation_scaled,
])

# 2. Fit or provide a residual variogram model
vgm_spec = {
    "model": "sph",
    "nugget": 0.05,
    "sill": 1.00,
    "range": 2500.0,
}

# 3. Run KED
est, var = universal_kriging(
    obs_coord=obs_coord,
    obs_value=obs_value,
    grid_coord=grid_coord,
    obs_drift=obs_drift,
    grid_drift=grid_drift,
    vgm_spec=vgm_spec,
    nmax=32,
)

# 4. Reshape for mapping, if using a structured grid
est_map = est.reshape(ny, nx)
var_map = var.reshape(ny, nx)
```

## Common problems

### Singular kriging matrix

Possible causes:

* too few neighbors;
* too many drift terms;
* duplicate observation locations;
* highly correlated drift variables;
* constant drift variable within local neighborhoods;
* variogram parameters that produce an ill-conditioned covariance matrix.

Possible fixes:

* increase `nmax`;
* reduce the number of drift terms;
* remove duplicate points;
* standardize drift variables;
* avoid high-order polynomial drift;
* use a small nugget for numerical stability.

### KED performs worse than ordinary kriging

Possible causes:

* weak relationship between drift and target variable;
* poor-quality external drift raster;
* drift variable available at grid cells but not well aligned with point observations;
* variogram fitted to raw values rather than residuals;
* extrapolation beyond observed drift range;
* too many drift terms.

Possible fixes:

* compare scatterplots of target values against drift variables;
* use residual variograms;
* simplify the drift model;
* check cross-validation results;
* inspect prediction maps for artifacts.

### Predictions show unrealistic large-scale pattern

Possible causes:

* external drift dominates the kriging estimate;
* drift variable has artifacts or discontinuities;
* prediction grid contains drift values outside the observation range;
* polynomial coordinate drift is too flexible.

Possible fixes:

* standardize and inspect drift variables;
* use simpler drift terms;
* mask unreliable regions;
* compare with ordinary kriging;
* run cross-validation by spatial blocks.

## Recommended defaults

For a first KED run:

```python
nmax = 24
nmin = 8
drift_terms = ["intercept", "one_external_variable"]
```

Use a simple external drift first, such as elevation or model output. Add additional drift variables only after checking cross-validation results.

For a first universal kriging run:

```python
drift_terms = ["intercept", "x", "y"]
nmax = 24
```

Avoid quadratic drift until the linear trend has been tested.

## Summary

Universal kriging and kriging with external drift extend ordinary kriging by allowing the local mean to vary systematically across space.

Use universal kriging when the trend can be represented by coordinate functions. Use KED when an external auxiliary variable explains part of the spatial pattern.

The most important practical points are:

* provide drift values at both observation and prediction locations;
* use the same drift columns in the same order for both;
* fit the variogram to residuals when a strong trend exists;
* use enough neighbors to support the drift constraints;
* standardize external drift variables when needed;
* compare results against ordinary kriging using cross-validation.

These methods are powerful, but they should be used carefully. A good external drift can improve predictions substantially. A poor or overfit drift can make predictions less reliable than ordinary kriging.
