# API reference

The full API is auto-generated from the source docstrings.

```{toctree}
:maxdepth: 1

../autoapi/krigekit/index
../autoapi/krigekit/kriging/index
../autoapi/krigekit/kriging_st/index
../autoapi/krigekit/kriging_indicator/index
```

## Summary

### Convenience functions

One-shot wrappers that create, configure, solve, and return results in a
single call.  Suitable for simple workflows; use the class API directly for
full control.

| Function | Description |
|---|---|
| {py:func}`krigekit.ordinary_kriging` | Ordinary (or simple) kriging for one variable |
| {py:func}`krigekit.cokriging` | Co-kriging for two or more variables |
| {py:func}`krigekit.sequential_gaussian_simulation` | SGSIM — multiple realisations |
| {py:func}`krigekit.spacetime_kriging` | Space-time kriging (one variable) |
| {py:func}`krigekit.spacetime_cokriging` | Space-time co-kriging |

### Classes

| Class | Description |
|---|---|
| {py:class}`krigekit.Kriging` | Full kriging workflow: OK, SK, co-kriging, KED, SGSIM, SVA |
| {py:class}`krigekit.SpaceTimeKriging` | Space-time extension with ST covariance models |
| {py:class}`krigekit.IndicatorKriging` | Multiple Indicator Kriging (MIK) and Sequential Indicator Simulation (SIS) |
