# Changelog

## 0.1.0 (unreleased)

Initial release.

### Features

- Ordinary and simple kriging (point and block)
- Co-kriging with Linear Model of Coregionalisation
- Universal kriging / KED (external drift)
- Sequential Gaussian Simulation (SGSIM)
- Space-time kriging — sum-metric and product-sum ST covariance models
- Spatially Varying Anisotropy (per-block variogram, `varying_vgm` mode)
- Cross-validation (leave-one-out)
- Kriging weight storage and reuse (`store_weight` / `use_old_weight`)
- OpenMP parallelism with per-`solve()` thread count control
- `set_vgm(append=False)` to replace variogram model on a reused object
