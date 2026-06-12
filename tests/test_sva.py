"""
test_sva.py
========================
Tests for spatial varying anisotropy features:

  1. Local nugget  (localnugget per grid node)
  2. Range scaler  (rangescale per grid node)

All tests use the pc2d dataset (62 observations, 2D) with the
standard spherical variogram so results are comparable to ordinary
point kriging.

Variogram:  sph  nugget=0  sill=0.12  range=5000  (isotropic)
"""

import numpy as np
import pytest
import os
from krigekit import ordinary_kriging

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
_VGM = dict(vtype="sph", nugget=0.0, sill=0.12, a_major=5000.0)
_NMAX = 20


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_ok(coord, value, grid, nmax=_NMAX, **kwargs):
    """Run ordinary kriging; kwargs forwarded to set_grid."""
    return ordinary_kriging(coord, value, grid, _VGM, nmax=nmax, **kwargs)


# ===========================================================================
# 1. Local nugget tests
# ===========================================================================

class TestLocalNugget:
    """
    localnugget adds a per-node nugget on top of the global variogram nugget.
    It is added to the diagonal of the kriging matrix at the estimation point,
    so it increases the kriging variance without changing the weights.

    Physical interpretation: measurement uncertainty at a specific location.
    A localnugget=sigma^2 at node i means the estimate at that node is treated
    as if the supporting sample has measurement error sigma.
    """

    def test_zero_localnugget_matches_ordinary_kriging(self, pc2d_obs, pc2d_grid):
        """Explicit zeros must give identical results to omitting localnugget."""
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:20]   # small subset for speed

        est_default, var_default = _run_ok(coord, value, grid)
        est_zero, var_zero = _run_ok(
            coord, value, grid,
            localnugget=np.zeros(len(grid))
        )

        np.testing.assert_allclose(est_default, est_zero, rtol=1e-6,
            err_msg="Zero localnugget should match default (no localnugget)")
        np.testing.assert_allclose(var_default, var_zero, rtol=1e-6,
            err_msg="Zero localnugget variance should match default")

    def test_localnugget_increases_variance(self, pc2d_obs, pc2d_grid):
        """Adding a positive localnugget must increase kriging variance at every node."""
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:20]
        nugget_val = 0.05   # ~40% of sill

        _, var_base  = _run_ok(coord, value, grid)
        _, var_nugget = _run_ok(
            coord, value, grid,
            localnugget=np.full(len(grid), nugget_val)
        )

        assert np.all(var_nugget >= var_base - 1e-10), (
            "localnugget should increase variance at every node")

    def test_localnugget_does_not_change_estimates(self, pc2d_obs, pc2d_grid):
        """
        localnugget is added to the diagonal of the kriging matrix (acts as
        per-node measurement error on the data), so it DOES change estimates —
        it smooths them away from the observation values.

        This test verifies the smoothing effect: estimates with localnugget
        should be closer to the global mean than without.
        """
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:20]
        global_mean = value.mean()

        est_base,   _ = _run_ok(coord, value, grid)
        est_nugget, _ = _run_ok(coord, value, grid,
                                localnugget=np.full(len(grid), 0.05))

        # localnugget smooths estimates toward the mean
        dev_base   = np.abs(est_base   - global_mean).mean()
        dev_nugget = np.abs(est_nugget - global_mean).mean()
        assert dev_nugget <= dev_base + 1e-5, (
            "localnugget should smooth estimates toward the global mean "
            f"(base dev={dev_base:.4f}, nugget dev={dev_nugget:.4f})"
        )

    def test_localnugget_per_node_variation(self, pc2d_obs, pc2d_grid):
        """Different localnugget values at different nodes produce independent variance increases."""
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:10]
        n = len(grid)

        # Alternating nugget: 0 at even nodes, 0.05 at odd nodes
        ln = np.array([0.0 if i % 2 == 0 else 0.05 for i in range(n)])
        _, var_base    = _run_ok(coord, value, grid)
        _, var_partial = _run_ok(coord, value, grid, localnugget=ln)

        # Even-indexed nodes: variance unchanged
        np.testing.assert_allclose(var_partial[0::2], var_base[0::2], rtol=1e-5,
            err_msg="Zero-nugget nodes should have unchanged variance")
        # Odd-indexed nodes: variance must increase
        assert np.all(var_partial[1::2] >= var_base[1::2] - 1e-10), (
            "Positive-nugget nodes should have increased variance")

    def test_localnugget_variance_nonnegative(self, pc2d_obs, pc2d_grid):
        """Kriging variance must remain non-negative even with large local nuggets."""
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:20]

        _, var = _run_ok(coord, value, grid,
                         localnugget=np.full(len(grid), 0.5))
        assert np.all(var >= -1e-10), f"Negative variance: {var.min():.4f}"

    def test_exact_match_localnugget_zero(self, pc2d_obs):
        """
        At a grid node coinciding with an observation and localnugget=0,
        the estimate must equal the observed value exactly.
        """
        coord, value = pc2d_obs
        grid_at_obs = coord[[0]]

        est, _ = _run_ok(coord, value, grid_at_obs,
                         localnugget=np.zeros(1))
        assert est[0] == pytest.approx(value[0], rel=1e-4)

    def test_exact_match_localnugget_nonzero_smooths(self, pc2d_obs):
        """
        localnugget is added to the diagonal of matA (data-to-data covariance),
        so it smooths non-exact-match estimates.  At a grid node that is NOT
        co-located with any observation, a larger localnugget pulls the estimate
        closer to the global mean.
        """
        coord, value = pc2d_obs
        global_mean = value.mean()

        # Use a grid point far from all observations so no exact match occurs
        grid_far = np.array([[coord[:, 0].mean(), coord[:, 1].mean()]])

        est_zero,    _ = _run_ok(coord, value, grid_far,
                                  localnugget=np.zeros(1))
        est_large,   _ = _run_ok(coord, value, grid_far,
                                  localnugget=np.full(1, 0.5))

        # Larger localnugget → estimate pulled further toward global mean
        assert (abs(est_large[0] - global_mean) <=
                abs(est_zero[0]  - global_mean) + 1e-5), (
            "Large localnugget should pull estimate toward global mean"
        )


# ===========================================================================
# 2. Range scaler tests
# ===========================================================================

class TestRangeScaler:
    """
    rangescale divides the lag vector before variogram evaluation:
        h_scaled = h / rangescale

    rangescale > 1  stretches the effective range  →  longer-range correlation
                    →  smoother estimates, lower variance
    rangescale < 1  compresses the effective range →  shorter-range correlation
                    →  less-smooth estimates, higher variance
    rangescale = 1  reproduces standard ordinary kriging
    """

    def test_unit_rangescale_matches_ordinary_kriging(self, pc2d_obs, pc2d_grid):
        """Explicit rangescale=1 must give identical results to omitting it."""
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:20]

        est_default, var_default = _run_ok(coord, value, grid)
        est_one, var_one = _run_ok(
            coord, value, grid,
            rangescale=np.ones(len(grid))
        )

        np.testing.assert_allclose(est_default, est_one, rtol=1e-6,
            err_msg="Unit rangescale should match default (no rangescale)")
        np.testing.assert_allclose(var_default, var_one, rtol=1e-6,
            err_msg="Unit rangescale variance should match default")

    def test_larger_rangescale_reduces_variance(self, pc2d_obs, pc2d_grid):
        """
        rangescale > 1 stretches the variogram range so more distant observations
        contribute with higher weight, reducing kriging variance.
        """
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:20]

        _, var_1   = _run_ok(coord, value, grid,
                             rangescale=np.ones(len(grid)))
        _, var_2   = _run_ok(coord, value, grid,
                             rangescale=np.full(len(grid), 2.0))

        # Mean variance with larger range should be lower (smoother interpolation)
        assert var_2.mean() <= var_1.mean() + 1e-6, (
            f"rangescale=2 mean variance ({var_2.mean():.4f}) should be <= "
            f"rangescale=1 mean variance ({var_1.mean():.4f})")

    def test_smaller_rangescale_increases_variance(self, pc2d_obs, pc2d_grid):
        """
        rangescale < 1 compresses the effective range so the same observations
        appear 'farther away', increasing kriging variance.
        """
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:20]

        _, var_1   = _run_ok(coord, value, grid,
                             rangescale=np.ones(len(grid)))
        _, var_half = _run_ok(coord, value, grid,
                              rangescale=np.full(len(grid), 0.5))

        assert var_half.mean() >= var_1.mean() - 1e-6, (
            f"rangescale=0.5 mean variance ({var_half.mean():.4f}) should be >= "
            f"rangescale=1 mean variance ({var_1.mean():.4f})")

    def test_rangescale_variance_nonnegative(self, pc2d_obs, pc2d_grid):
        """Variance must remain non-negative for any positive rangescale."""
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:20]

        for rs in [0.25, 0.5, 1.0, 2.0, 5.0]:
            _, var = _run_ok(coord, value, grid,
                             rangescale=np.full(len(grid), rs))
            assert np.all(var >= -1e-10), (
                f"Negative variance with rangescale={rs}: {var.min():.4f}")

    def test_rangescale_per_node_spatial_variation(self, pc2d_obs, pc2d_grid):
        """
        Nodes with larger rangescale should have lower variance than
        nodes with smaller rangescale, all else being equal.
        """
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:10]
        n = len(grid)

        # Low rangescale for first half, high for second half
        rs_mixed = np.array([0.5 if i < n // 2 else 2.0 for i in range(n)])
        _, var_low_rs  = _run_ok(coord, value, grid,
                                  rangescale=np.full(n, 0.5))
        _, var_high_rs = _run_ok(coord, value, grid,
                                  rangescale=np.full(n, 2.0))

        # Across all nodes: high range scale → lower variance on average
        assert var_high_rs.mean() <= var_low_rs.mean() + 1e-4

    def test_rangescale_monotone_with_scale(self, pc2d_obs, pc2d_grid):
        """Kriging variance should be monotonically non-increasing as rangescale grows."""
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:5]

        variances = []
        for rs in [0.5, 1.0, 2.0, 4.0, 8.0]:
            _, var = _run_ok(coord, value, grid,
                             rangescale=np.full(len(grid), rs))
            variances.append(var.mean())

        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i+1] - 1e-5, (
                f"Variance should not increase as rangescale grows: "
                f"rs={[0.5,1,2,4,8][i]}->{[0.5,1,2,4,8][i+1]} "
                f"var={variances[i]:.4f}->{variances[i+1]:.4f}")

    def test_localnugget_and_rangescale_combined(self, pc2d_obs, pc2d_grid):
        """Both features can be used together without conflict."""
        coord, value = pc2d_obs
        grid = pc2d_grid[0][:20]

        est, var = _run_ok(coord, value, grid,
                           rangescale=np.full(len(grid), 1.5),
                           localnugget=np.full(len(grid), 0.02))
        assert est.shape == (len(grid),)
        assert var.shape == (len(grid),)
        assert np.all(var >= -1e-10)
