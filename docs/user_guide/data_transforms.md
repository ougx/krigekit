# Data transforms

Many environmental and geoscience variables are strongly non-Gaussian:
concentrations, hydraulic conductivity, transmissivity, grade, permeability,
and percentages often vary by orders of magnitude.  Kriging those values in raw
units can make the variogram and local estimates overly sensitive to high-end
outliers.

krigekit can transform observation values inside the Fortran engine, solve in
score space, and back-transform results to data units.  The same transform is
available to every C-API client and works for ordinary/simple kriging
(`nsim=0`) and SGSIM (`nsim > 0`).

## Normal-score transform

Use `set_nscore()` to map the observed data distribution to standard-normal
scores:

```python
from krigekit import Kriging

k = Kriging(nsim=0)  # use nsim > 0 for SGSIM
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=30)
k.set_nscore(ivar=1)
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=40.0)
k.set_grid(coord=grid_coord)
k.set_search(ivar=1)
k.solve()
estimate, variance = k.get_results()
```

Fit the variogram in normal-score space, not raw data space.  For a fully
normal-scored marginal, the total sill is usually near `1.0`.

## Uniform quantile transform

Use `set_uscore()` to map observations to empirical CDF scores in
continuous uniform distribution `[0, 1]`:

```python
k = Kriging(nsim=0)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=30)
k.set_uscore(ivar=1)
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0 / 12.0, a_major=40.0)
k.set_grid(coord=grid_coord)
k.set_search(ivar=1)
k.solve()
estimate, variance = k.get_results()
```

`set_quantile()` is an alias for `set_uscore()`.  Fit the variogram in
uniform-score space.  A perfectly uniform marginal has variance `1/12`, so that
is a useful starting sill when the transformed variable is close to uniform.

## Back-transform behavior

For SGSIM, each simulated score realization is directly back-transformed
through the empirical quantile table.

For kriging (`nsim=0`), krigekit treats the score-space result as a conditional
normal distribution and uses Gauss-Hermite quadrature to report data-unit
moments:

```{math}
Y \mid \text{data} \sim N(\mu_Y, \sigma_Y^2), \qquad Z = T^{-1}(Y)
```

where `Y` is the kriged score variable and `T^{-1}` is the empirical
back-transform table built by `set_nscore()` or `set_uscore()`.  The reported
estimate is

```{math}
E[Z \mid \text{data}]
= \int T^{-1}(y)\,\phi(y;\mu_Y,\sigma_Y^2)\,dy
\approx \sum_i w_i\,T^{-1}\!\left(\mu_Y + \sigma_Y x_i\right),
```

where `x_i` and `w_i` are Gauss-Hermite nodes and weights expressed in the
*probabilists'* convention — that is, the quadrature rule for the standard
normal weight {math}`\phi(x)`, so the weights satisfy {math}`\sum_i w_i = 1`
and the nodes enter as {math}`\mu_Y + \sigma_Y x_i` (no {math}`\sqrt{2}` or
{math}`1/\sqrt{\pi}` factors).  krigekit uses the five-node rule
{math}`x_i \in \{0, \pm 1.3556, \pm 2.8570\}`.  The reported variance is

```{math}
\operatorname{Var}(Z \mid \text{data})
\approx \sum_i w_i\,
\left[T^{-1}\!\left(\mu_Y + \sigma_Y x_i\right)\right]^2
- \left(E[Z \mid \text{data}]\right)^2.
```

For two transformed variables, cross-covariance is approximated by bivariate
quadrature under the score-space joint normal model:

```{math}
\begin{bmatrix} y_{1,ij} \\ y_{2,ij} \end{bmatrix}
=
\begin{bmatrix} \mu_{Y_1} \\ \mu_{Y_2} \end{bmatrix}
+ L
\begin{bmatrix} x_i \\ x_j \end{bmatrix},
\qquad
LL^\mathsf{T} = \Sigma_Y.
```

```{math}
\operatorname{Cov}(Z_1, Z_2 \mid \text{data})
\approx
\sum_i\sum_j w_i w_j\,
T_1^{-1}(y_{1,ij})\,T_2^{-1}(y_{2,ij})
- E[Z_1 \mid \text{data}]\,E[Z_2 \mid \text{data}].
```

- `estimate` is the back-transformed conditional mean.
- `variance` is the back-transformed conditional variance.
- Cross-covariances involving transformed variables are also quadrature
  transformed under the score-space joint normal approximation.

This is usually preferable to simply back-transforming the score-space mean,
because nonlinear transforms can shift the mean and variance.

```{note}
The conditional-normal model is exact only for the **normal-score** transform,
where Gaussian working space is the modelling assumption.  For the **uniform
quantile** transform the kriged variable lives on a {math}`[0, 1]` scale, so
treating its conditional distribution as Gaussian is a moment-matching
approximation: quadrature nodes can fall outside {math}`[0, 1]` and are clamped
by the tail bounds, which can bias the back-transformed mean for cells whose
estimate sits near 0 or 1 with large kriging variance.  This affects only
kriging (`nsim=0`); SGSIM back-transforms each realization directly through the
quantile table and is unaffected.  Use `set_nscore()` if you need a
well-justified conditional mean and variance in data units.
```

## Transform and back-transform values

After calling `set_nscore()` or `set_uscore()`, you can transform arbitrary
values with the same table:

```python
score = k.transform_value_to_score(raw_values, ivar=1)
raw   = k.transform_score_to_value(score, ivar=1)
```

The C API exposes the same operations:

```c
krige_transform_value_to_score(handle, ivar, n, value, score);
krige_transform_score_to_value(handle, ivar, n, score, value);
```

## Bounds, tails, and weights

The empirical transform table is built from the current observations.  Values
outside the observed score range are handled by tail extrapolation and bounded
by `zmin` / `zmax`.  By default those bounds are the observed minimum and
maximum.

```python
k.set_nscore(
    ivar=1,
    zmin=0.0,
    zmax=200.0,
    ltail="linear",
    utail="hyperbolic",
    utpar=1.5,
)
```

| Tail model | Notes |
|---|---|
| `"linear"` | Straight line between the extreme datum and `zmin` / `zmax` |
| `"power"` | `ltpar` / `utpar` controls curvature |
| `"hyperbolic"` | Upper tail only; requires positive data |

If observations are spatially clustered, pass declustering weights so the
transform reproduces the declustered distribution:

```python
k.set_nscore(ivar=1, weights=decluster_weights)
k.set_uscore(ivar=1, weights=decluster_weights)
```

Only one marginal transform can be active per variable, so call either
`set_nscore()` or `set_uscore()` for a given `ivar`.

## Practical guidance

Use `set_nscore()` when you want a Gaussian working scale, especially for
SGSIM or heavily skewed continuous variables.

Use `set_uscore()` when a rank/quantile scale is more natural or when you want
the variogram to describe correlation in cumulative-probability space.

Always fit and interpret the variogram in the transformed score space.  The
reported kriging estimate and variance are returned in data units, but the
model you provide still belongs to the transformed variable.
