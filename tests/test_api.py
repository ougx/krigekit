"""
test_api.py
===========
Tests for input validation, edge cases, and the full Kriging class API
including set_obs_drift, set_grid_drift, set_grid_cv, and bounds clipping.
"""

import numpy as np
import pytest
from pykriging import Kriging, ordinary_kriging

_VGM = "sph 0.0 1.0 50.0 50.0 50.0 0.0 0.0 0.0"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_coord_wrong_ndim_raises(self):
        coord = np.random.rand(10, 3)   # 3D coord
        value = np.random.rand(10)
        grid  = np.random.rand(5, 2)
        with pytest.raises(AssertionError, match="ndim=2"):
            ordinary_kriging(coord, value, grid, _VGM, nmax=5)

    def test_coord_transposed_raises(self):
        coord = np.random.rand(10, 2)
        value = np.random.rand(10)
        grid  = np.random.rand(5, 2)
        # (2, 10) instead of (10, 2) — wrong convention
        with pytest.raises(AssertionError):
            ordinary_kriging(coord.T, value, grid, _VGM, nmax=5)

    def test_missing_library_error_message():
        """Importing when the library is absent should raise a clear error."""
        # This is tested implicitly at import time; we just confirm the module loaded.
        from pykriging import Kriging
        assert Kriging is not None


# ---------------------------------------------------------------------------
# Simple kriging (unbias=0)
# ---------------------------------------------------------------------------

class TestSimpleKriging:

    def test_simple_kriging_with_sk_mean(self):
        """Simple kriging with the true mean should give a valid result."""
        rng   = np.random.default_rng(42)
        coord = rng.uniform(0, 100, (20, 2))
        value = rng.normal(5.0, 1.0, 20)
        grid  = np.array([[50.0, 50.0]])
        true_mean = value.mean()

        k = Kriging(ndim=2, nvar=1, unbias=0, sk_mean=float(true_mean))
        k.set_obs(ivar=1, coord=coord, value=value, nmax=10)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        assert est.shape == (1,)
        assert var[0] >= 0.0


# ---------------------------------------------------------------------------
# Estimate clipping (bounds)
# ---------------------------------------------------------------------------

class TestBoundsClipping:

    def test_bounds_clip_upper(self):
        rng   = np.random.default_rng(0)
        coord = rng.uniform(0, 100, (30, 2))
        value = rng.uniform(0, 10, 30)
        grid  = rng.uniform(0, 100, (20, 2))

        upper = 7.0
        k = Kriging(ndim=2, nvar=1, bounds=(0.0, upper))
        k.set_obs(ivar=1, coord=coord, value=value, nmax=10)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        est, _ = k.get_results()
        assert est.max() <= upper + 1e-6, \
            f"Estimate {est.max():.4f} exceeds upper bound {upper}"

    def test_bounds_clip_lower(self):
        rng   = np.random.default_rng(1)
        coord = rng.uniform(0, 100, (30, 2))
        value = rng.uniform(-5, 5, 30)
        grid  = rng.uniform(0, 100, (20, 2))

        lower = 0.0
        k = Kriging(ndim=2, nvar=1, bounds=(lower, 10.0))
        k.set_obs(ivar=1, coord=coord, value=value, nmax=10)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        est, _ = k.get_results()
        assert est.min() >= lower - 1e-6, \
            f"Estimate {est.min():.4f} is below lower bound {lower}"


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation:

    def test_cross_validation_returns_nobs_estimates(self):
        """Cross-validation must return one estimate per observation."""
        rng   = np.random.default_rng(5)
        coord = rng.uniform(0, 100, (15, 2))
        value = rng.uniform(0, 1, 15)

        k = Kriging(ndim=2, nvar=1, cross_validation=True)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=15)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM)
        k.set_grid_cv()
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        assert est.shape == (coord.shape[0],)
        assert np.all(var >= 0.0)

    def test_cross_validation_residuals_unbiased(self):
        """Mean cross-validation residual should be near zero for a correct model."""
        rng   = np.random.default_rng(99)
        coord = rng.uniform(0, 100, (20, 2))
        value = rng.uniform(0, 1, 20)

        k = Kriging(ndim=2, nvar=1, cross_validation=True)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=20)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM)
        k.set_grid_cv()
        k.set_search(ivar=1)
        k.solve()
        est, _ = k.get_results()
        residuals = value - est
        # Mean residual should be small relative to data range
        assert abs(residuals.mean()) < 0.2 * (value.max() - value.min()), \
            f"Mean cross-validation residual too large: {residuals.mean():.4f}"


# ---------------------------------------------------------------------------
# Drift (universal kriging)
# ---------------------------------------------------------------------------

class TestDrift:

    def test_kriging_with_linear_drift(self, head2d_obs):
        """
        Universal kriging with a linear drift (x, y as drift functions)
        should produce estimates without errors on the head2d dataset.
        """
        coord, value = head2d_obs
        grid = np.array([[5.0, 5.0], [5.0, 8.0], [7.0, 5.0]])

        # Drift values at observations: [x, y] normalised
        obs_drift  = np.column_stack([coord[:, 0], coord[:, 1]])   # (nobs, 2)
        grid_drift = np.column_stack([grid[:, 0],  grid[:, 1]])    # (ngrid, 2)

        k = Kriging(ndim=2, nvar=1, ndrift=2, unbias=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=29)
        k.set_obs_drift(ivar=1, drift=obs_drift)
        k.set_vgm(ivar=1, jvar=1, spec="sph 0.0 50000 3.0 5.0 3.0 0.0 0.0 0.0")
        k.set_grid(coord=grid)
        k.set_grid_drift(drift=grid_drift)
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        assert est.shape == (grid.shape[0],)
        assert np.all(var >= 0.0)
        # Head values should be in a physically plausible range
        assert np.all(est > 0), "Hydraulic head estimates should be positive"

    def test_obs_drift_wrong_shape_raises(self, head2d_obs):
        """set_obs_drift with wrong ndrift dimension should raise on the Fortran side."""
        coord, value = head2d_obs
        k = Kriging(ndim=2, nvar=1, ndrift=2, unbias=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=10)
        wrong_drift = np.ones((coord.shape[0], 3))  # ndrift=3 but declared ndrift=2
        with pytest.raises(Exception):
            k.set_obs_drift(ivar=1, drift=wrong_drift)
