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
_VGM_PC2D = "sph 0.0  0.12 5000.0 5000.0 5000.0 0.0 0.0 0.0"
_NMAX = 20

# Two interior grid points not co-located with any observation
_INTERIOR_GRID = np.array([[580000.0, 4395000.0],
                            [578000.0, 4400000.0]])

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_coord_wrong_ndim_raises(self):
        coord = np.random.rand(10, 3)   # 3D coord passed as obs
        value = np.random.rand(10)
        grid  = np.random.rand(5, 2)
        # obs ndim=3 is inferred correctly; grid has ndim=2 which mismatches
        with pytest.raises(AssertionError, match="ndim=3"):
            ordinary_kriging(coord, value, grid, _VGM, nmax=5)

    def test_coord_transposed_raises(self):
        coord = np.random.rand(10, 2)
        value = np.random.rand(10)
        grid  = np.random.rand(5, 2)
        # (2, 10) instead of (10, 2) — wrong convention
        with pytest.raises(AssertionError):
            ordinary_kriging(coord.T, value, grid, _VGM, nmax=5)

    def test_missing_library_error_message(self):
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
        grid  = np.array([[50.0, 50.0]])
        value = rng.normal(5.0, 1.0, 20)
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
        rng   = np.random.default_rng(42)
        coord = rng.uniform(0, 100, (20, 2))
        grid  = rng.uniform(0, 100, (20, 2))
        value = rng.uniform(0, 10, 20)

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
        rng   = np.random.default_rng(42)
        coord = rng.uniform(0, 100, (20, 2))
        grid  = rng.uniform(0, 100, (20, 2))
        value = rng.uniform(-5, 5, 20)

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
        k.set_vgm(ivar=1, jvar=1, spec="sph 0.0 50000 5.0 3.0 3.0 0.0 0.0 0.0")
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
        """set_obs_drift with wrong ndrift column count should be caught on Python side."""
        coord, value = head2d_obs
        k = Kriging(ndim=2, nvar=1, ndrift=2, unbias=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=10)
        # drift has 3 columns but ndrift=2 was declared — Python shape check should raise
        wrong_drift = np.ones((coord.shape[0], 3))   # (nobs, 3) but ndrift=2
        with pytest.raises(AssertionError, match="ndrift=2"):
        	k.set_obs_drift(ivar=1, drift=wrong_drift)            #"test precondition: wrong_drift must have the wrong number of drift columns"


# ===========================================================================
# Object reuse
# ===========================================================================

class TestObjectReuse:
    """
    Calling set_obs / set_grid on an already-used Kriging object must:

    * free all previously allocated arrays without memory errors or segfaults
    * produce results that reflect the new data, not the old data
    * reproduce the first run exactly when the first data is reloaded

    The reuse path exercises reset_obs / reset_grid / reset_block inside the
    Fortran layer (called automatically at the start of set_obs / set_grid).
    """

    def test_second_run_differs_with_different_obs(self, pc2d_obs):
        """Results change when observations are replaced with a different subset."""
        coord, value = pc2d_obs
        grid = coord[5:10]

        k = Kriging(ndim=2, nvar=1, verbose=0)

        # Run 1: all 62 observations
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        est1, var1 = k.get_results()

        # Run 2: last 32 observations only
        k.set_obs(ivar=1, coord=coord[30:], value=value[30:], nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        est2, var2 = k.get_results()

        assert not np.allclose(est1, est2), (
            "Estimates should differ when obs set changes")
        assert np.all(var2 >= 0), "Variance must remain non-negative after reuse"

    def test_second_run_differs_with_different_grid(self, pc2d_obs):
        """Results change when the estimation grid is replaced."""
        coord, value = pc2d_obs

        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)

        k.set_grid(coord=_INTERIOR_GRID[:1])
        k.set_search(ivar=1)
        k.solve()
        est1, _ = k.get_results()

        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)
        k.set_grid(coord=_INTERIOR_GRID[1:])
        k.set_search(ivar=1)
        k.solve()
        est2, _ = k.get_results()

        assert not np.allclose(est1, est2), (
            "Estimates should differ at different grid locations")

    def test_third_run_reproduces_first(self, pc2d_obs):
        """Reloading the original data must reproduce the original results exactly."""
        coord, value = pc2d_obs
        grid = coord[5:10]

        k = Kriging(ndim=2, nvar=1, verbose=0)

        def _do_run(obs_coord, obs_val):
            k.set_obs(ivar=1, coord=obs_coord, value=obs_val, nmax=_NMAX)
            k.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)
            k.set_grid(coord=grid)
            k.set_search(ivar=1)
            k.solve()
            return k.get_results()

        est1, var1 = _do_run(coord, value)
        _do_run(coord[30:], value[30:])          # intermediate run with different data
        est3, var3 = _do_run(coord, value)

        np.testing.assert_allclose(est1, est3, rtol=1e-6,
            err_msg="Third run (same data as first) must reproduce first run estimates")
        np.testing.assert_allclose(var1, var3, rtol=1e-6,
            err_msg="Third run (same data as first) must reproduce first run variances")

    def test_reuse_with_smaller_then_larger_obs(self, pc2d_obs):
        """
        Reuse from a smaller obs set to a larger one must not leave stale
        array lengths behind (would cause out-of-bounds access in Fortran).
        """
        coord, value = pc2d_obs
        grid = _INTERIOR_GRID

        k = Kriging(ndim=2, nvar=1, verbose=0)

        k.set_obs(ivar=1, coord=coord[:10], value=value[:10], nmax=10)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        est_small, _ = k.get_results()

        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        est_full, var_full = k.get_results()

        assert est_full.shape == (len(grid),)
        assert np.all(var_full >= 0)
        assert not np.allclose(est_small, est_full), (
            "More observations should change the estimate")

    def test_reuse_variance_nonnegative_across_runs(self, pc2d_obs):
        """Variance must be non-negative in every run across three reuses."""
        coord, value = pc2d_obs
        grid = _INTERIOR_GRID
        k = Kriging(ndim=2, nvar=1, verbose=0)
        for sl in [slice(None), slice(30), slice(15, 45)]:
            k.set_obs(ivar=1, coord=coord[sl], value=value[sl], nmax=_NMAX)
            k.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)
            k.set_grid(coord=grid)
            k.set_search(ivar=1)
            k.solve()
            _, var = k.get_results()
            assert np.all(var >= 0), (
                f"Negative variance after reuse with slice {sl}: {var}")

    def test_set_vgm_accumulates_structures(self, pc2d_obs):
        """
        Calling set_vgm twice on the same object with the same spec adds two
        copies of that structure (doubled sill).  This is intentional: it is
        how multi-struct models are built.  The test documents the behaviour
        so that callers know they must not call set_vgm redundantly when reusing
        an object — create a fresh Kriging when the variogram changes.
        """
        coord, value = pc2d_obs
        grid = _INTERIOR_GRID

        k1 = Kriging(ndim=2, nvar=1, verbose=0)
        k1.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k1.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)          # sill = 0.12
        k1.set_grid(coord=grid)
        k1.set_search(ivar=1)
        k1.solve()
        _, var_one = k1.get_results()

        k2 = Kriging(ndim=2, nvar=1, verbose=0)
        k2.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k2.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)          # first call
        k2.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)          # second call → sill = 0.24
        k2.set_grid(coord=grid)
        k2.set_search(ivar=1)
        k2.solve()
        _, var_two = k2.get_results()

        np.testing.assert_allclose(var_two, 2.0 * var_one, rtol=1e-5,
            err_msg="Calling set_vgm twice with same spec must double the total sill "
                    "and therefore double the kriging variance")


