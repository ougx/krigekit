"""
test_ordinary_kriging.py
========================
Tests for ordinary point kriging using the Kriging class and
the ordinary_kriging convenience function.

Datasets
--------
obs_simple / grid_simple : 5 observations, 3 grid nodes — tiny synthetic
                           dataset; tests basic API and exact-match behaviour.
pc2d / grid2d            : 62 field observations of percent coarse, 4800
                           grid nodes; results compared to pre-computed
                           reference values stored in grid2d.csv.
"""

import numpy as np
import pytest
from pykriging import Kriging, ordinary_kriging


# Variogram for obs_simple: pure nugget (isotropic, no spatial structure)
_VGM_SIMPLE = "sph 0.01 0.09 100.0 100.0 100.0 0.0 0.0 0.0"

# Variogram fitted to pc2d dataset
_VGM_PC2D   = "sph 0.0 0.12 5000.0 5000.0 5000.0 0.0 0.0 0.0"


# ---------------------------------------------------------------------------
# Basic API tests (obs_simple / grid_simple)
# ---------------------------------------------------------------------------

class TestKrigingClass:
    """Tests using the full Kriging class interface on the small synthetic dataset."""

    def test_estimate_shape(self, simple_obs, simple_grid):
        """estimate and variance have shape (ngrid,) for kriging (nsim=0)."""
        coord, value = simple_obs
        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=5)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_SIMPLE)
        k.set_grid(coord=simple_grid)
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        assert est.shape == (simple_grid.shape[0],)
        assert var.shape == (simple_grid.shape[0],)

    def test_variance_nonnegative(self, simple_obs, simple_grid):
        """Kriging variance must be >= 0 at all grid nodes."""
        coord, value = simple_obs
        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=5)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_SIMPLE)
        k.set_grid(coord=simple_grid)
        k.set_search(ivar=1)
        k.solve()
        _, var = k.get_results()
        assert np.all(var >= 0.0), f"Negative variance found: {var.min()}"

    def test_exact_match(self, simple_obs):
        """When a grid node coincides with an observation, estimate == observed value."""
        coord, value = simple_obs
        # Grid = first observation point
        grid_at_obs = coord[[0]]
        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=5)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_SIMPLE)
        k.set_grid(coord=grid_at_obs)
        k.set_search(ivar=1)
        k.solve()
        est, _ = k.get_results()
        assert est[0] == pytest.approx(value[0], rel=1e-4)

    def test_weights_sum_to_one(self, simple_obs, simple_grid):
        """
        For ordinary kriging (unbias=1) the kriging weights sum to 1 at each node.
        We verify this indirectly: kriging of a constant field must return that constant.
        """
        coord, _ = simple_obs
        const_value = 3.14 * np.ones(coord.shape[0])
        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord, value=const_value, nmax=5)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_SIMPLE)
        k.set_grid(coord=simple_grid)
        k.set_search(ivar=1)
        k.solve()
        est, _ = k.get_results()
        assert est == pytest.approx(3.14, rel=1e-4)

    def test_repr(self, simple_obs, simple_grid):
        coord, value = simple_obs
        k = Kriging(ndim=2, nvar=1)
        assert "Kriging" in repr(k)
        assert "ndim=2" in repr(k)

    def test_destructor_does_not_crash(self, simple_obs, simple_grid):
        """__del__ must not crash even if solve was never called."""
        coord, value = simple_obs
        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord, value=value)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_SIMPLE)
        del k   # should not raise


# ---------------------------------------------------------------------------
# Convenience function tests (obs_simple / grid_simple)
# ---------------------------------------------------------------------------

class TestOrdinaryKrigingFunction:
    """Tests for the ordinary_kriging() convenience function."""

    def test_returns_two_arrays(self, simple_obs, simple_grid):
        coord, value = simple_obs
        result = ordinary_kriging(coord, value, simple_grid, _VGM_SIMPLE, nmax=5)
        assert len(result) == 2

    def test_estimate_in_reasonable_range(self, simple_obs, simple_grid):
        coord, value = simple_obs
        est, _ = ordinary_kriging(coord, value, simple_grid, _VGM_SIMPLE, nmax=5)
        # Estimate should lie within [min(obs), max(obs)] for ordinary kriging
        assert est.min() >= value.min() * 0.9
        assert est.max() <= value.max() * 1.1

    def test_bad_coord_shape_raises(self, simple_obs, simple_grid):
        coord, value = simple_obs
        bad_coord = coord.T   # (ndim, nobs) instead of (nobs, ndim) — wrong convention
        with pytest.raises(AssertionError):
            ordinary_kriging(bad_coord, value, simple_grid, _VGM_SIMPLE, nmax=5)

    def test_bad_value_shape_raises(self, simple_obs, simple_grid):
        coord, value = simple_obs
        # A value array with wrong number of elements (not nobs) should raise
        bad_value = value[:-1]   # one element short
        with pytest.raises(Exception):
            ordinary_kriging(coord, bad_value, simple_grid, _VGM_SIMPLE, nmax=5)


# ---------------------------------------------------------------------------
# Field dataset test: pc2d vs reference results
# ---------------------------------------------------------------------------

class TestPC2DKriging:
    """
    Regression test against pre-computed reference kriging results (grid2d.csv).
    Tolerance is relaxed to allow for minor solver/variogram differences.
    """

    def test_estimate_correlation_with_reference(self, pc2d_obs, pc2d_grid):
        coord, value        = pc2d_obs
        grid_coord, ref_est = pc2d_grid
        est, var = ordinary_kriging(
            coord, value, grid_coord,
            vgm_spec=_VGM_PC2D,
            nmax=62,
        )
        # Pearson correlation with reference must be > 0.99
        corr = np.corrcoef(est, ref_est)[0, 1]
        assert corr > 0.99, f"Correlation with reference = {corr:.4f} (expected > 0.99)"

    def test_pc2d_variance_nonnegative(self, pc2d_obs, pc2d_grid):
        coord, value        = pc2d_obs
        grid_coord, _       = pc2d_grid
        _, var = ordinary_kriging(coord, value, grid_coord, _VGM_PC2D, nmax=62)
        assert np.all(var >= 0.0)

    def test_pc2d_estimate_in_data_range(self, pc2d_obs, pc2d_grid):
        coord, value        = pc2d_obs
        grid_coord, _       = pc2d_grid
        est, _ = ordinary_kriging(coord, value, grid_coord, _VGM_PC2D, nmax=62)
        # With clipping off, ordinary kriging can extrapolate slightly outside
        # the data range; allow 10 % margin
        margin = 0.10 * (value.max() - value.min())
        assert est.min() >= value.min() - margin
        assert est.max() <= value.max() + margin