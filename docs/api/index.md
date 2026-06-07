# API reference

The full API is auto-generated from the source docstrings.

```{toctree}
:maxdepth: 1

../autoapi/pykriging/index
../autoapi/pykriging/kriging/index
../autoapi/pykriging/kriging_st/index
../autoapi/pykriging/kriging_indicator/index
```

## Summary

### Convenience functions

One-shot wrappers that create, configure, solve, and return results in a
single call.  Suitable for simple workflows; use the class API directly for
full control.

| Function | Description |
|---|---|
| {py:func}`pykriging.ordinary_kriging` | Ordinary (or simple) kriging for one variable |
| {py:func}`pykriging.cokriging` | Co-kriging for two or more variables |
| {py:func}`pykriging.sequential_gaussian_simulation` | SGSIM — multiple realisations |
| {py:func}`pykriging.spacetime_kriging` | Space-time kriging (one variable) |
| {py:func}`pykriging.spacetime_cokriging` | Space-time co-kriging |

### Classes

| Class | Description |
|---|---|
| {py:class}`pykriging.Kriging` | Full kriging workflow: OK, SK, co-kriging, KED, SGSIM, SVA |
| {py:class}`pykriging.SpaceTimeKriging` | Space-time extension with ST covariance models |
| {py:class}`pykriging.IndicatorKriging` | Multiple Indicator Kriging (MIK) and Sequential Indicator Simulation (SIS) |
