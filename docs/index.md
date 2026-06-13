# krigekit

**krigekit** is a Python interface to a high-performance Fortran kriging and
Sequential Gaussian Simulation engine, parallelised with OpenMP.

::::{grid} 2
:::{grid-item-card} Quick start
:link: quickstart
:link-type: doc
Get from installation to your first kriging map in five minutes.
:::
:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Full reference for every class, method, and convenience function.
:::
:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc
Task-oriented walkthroughs for ordinary kriging, co-kriging, SGSIM, and more.
:::
:::{grid-item-card} Examples
:link: auto_examples/index
:link-type: doc
Gallery of runnable scripts — ordinary kriging, co-kriging, SGSIM, and more.
:::
::::

## What krigekit does

| Capability | Notes |
|---|---|
| Ordinary and simple kriging | Point and block support |
| Co-kriging | Multiple variables, Linear Model of Coregionalisation |
| Universal kriging / KED | External drift variables |
| Sequential Gaussian Simulation | Reproducible random paths, multi-realisation, optional normal-score transform |
| Space-time kriging | Sum-metric and product-sum ST covariance models |
| Spatially Varying Anisotropy | Per-block variogram (SVA mode) |
| Multiple Indicator Kriging / SIS | Categorical variables; uniform, proportional, or independent cross-variogram strategies |
| Cross-validation | Leave-one-out |
| Kriging weight reuse | Store and replay weights for fast value updates |
| OpenMP parallelism | Thread count controllable per `solve()` call |

## Why krigekit?

krigekit is designed for workflows that need **Python usability with a
compiled-Fortran backend**.  Its Fortran core handles large grids, SGSIM
realisation paths, space-time systems, and OpenMP scheduling in a single
library.  The Python layer is a thin ctypes wrapper — no heavy dependencies,
no JIT compilation step.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
auto_examples/index
variogram_models
user_guide/index
api/index
array_conventions
developer_guide/index
changelog
```

