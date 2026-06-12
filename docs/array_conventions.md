# Array conventions

krigekit uses a consistent set of array shapes across all methods.
Understanding these conventions prevents the most common errors.

## Coordinate arrays

All coordinate arrays follow **(n, ndim)** layout — rows are points, columns
are spatial dimensions.  This matches NumPy, pandas, and scikit-learn.

```python
obs_coord   # shape (nobs,  ndim)   — observation locations
grid_coord  # shape (ngrid, ndim)   — estimation targets
drift       # shape (nobs,  ndrift) — external drift variables
```

The Python wrapper **transposes internally** to the Fortran core's
`(ndim, nobs)` column-major layout.  You never need to transpose manually.

**Example:**

```python
import numpy as np

# 50 observations in 2-D
obs_coord = np.random.rand(50, 2)  # ✓  shape (50, 2)

# 100 × 100 estimation grid in 2-D
x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid_coord = np.column_stack([x.ravel(), y.ravel()])  # ✓  shape (10000, 2)
```

## Value arrays

Observation values are always 1-D:

```python
obs_value  # shape (nobs,)  — one value per observation
```

## Result arrays

`get_results()` returns `(est, var)`.  The shapes depend on how the
`Kriging` object was configured:

| Configuration | `est` shape | `var` shape |
|---|---|---|
| Kriging, `nvar=1`, `nsim=1` | `(nblock,)` | `(nblock,)` |
| Co-kriging, `nvar>1`, `nsim=1` | `(nblock, nvar)` | `(nblock, nvar, nvar)` |
| SGSIM, `nvar=1`, `nsim>1` | `(nblock, nsim)` | `(nblock,)` |
| SGSIM, `nvar>1`, `nsim>1` | `(nblock, nvar, nsim)` | `(nblock, nvar, nvar)` |

For co-kriging `var` is the full conditional covariance matrix at each block:
`var[ib, k, k]` is variable `k`'s kriging variance at block `ib`.

### `get_estimate_all` and `get_variance_all`

These methods always return the full multi-variable, multi-simulation arrays
regardless of `squeeze` settings:

```python
est_all = k.get_estimate_all()   # shape (nblock, nvar, nsim)
var_all = k.get_variance_all()   # shape (nblock, nvar, nvar)
```

## Data types

krigekit accepts any numeric dtype for input arrays and converts internally:

- Coordinates → `float64`
- Values → `float64`
- Results → `float64`

Passing `float32` or integer arrays is safe; a copy will be made.

## Fortran internals (for contributors)

The Fortran core stores results as `value(isim, ivar, iblock)` and
conditional covariance as `variance(ivar, jvar, iblock)`.  The Python
getters return the transposed dimension order shown in the table above.
Use `get_results(copy=True)` when a C-contiguous copy is preferred for
downstream NumPy/Pandas workflows.
