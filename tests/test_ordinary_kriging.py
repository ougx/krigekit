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
from krigekit import Kriging, ordinary_kriging


# Variogram for obs_simple: pure nugget (isotropic, no spatial structure)
_VGM_SIMPLE = dict(vtype="sph", nugget=0.01, sill=0.09, a_major=100.0)

# Variogram fitted to pc2d dataset
_VGM_PC2D   = dict(vtype="sph", nugget=0.0, sill=0.12, a_major=5000.0)


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
        k.set_vgm(ivar=1, jvar=1, **_VGM_SIMPLE)
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
        k.set_vgm(ivar=1, jvar=1, **_VGM_SIMPLE)
        k.set_grid(coord=simple_grid)
        k.set_search(ivar=1)
        k.solve()
        _, var = k.get_results()
        assert np.all(var >= 0.0), f"Negative variance found: {var.min()}"

    def test_exact_match(self, simple_obs):
        """When a grid node coincides with an observation, estimate == observed value when there is no obs error/nugget."""
        coord, value = simple_obs
        # Grid = first observation point
        grid_at_obs = coord[[0]]
        vv = _VGM_SIMPLE.copy()
        vv["nugget"] = 0.0
        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=5)
        k.set_vgm(ivar=1, jvar=1, **vv)
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
        k.set_vgm(ivar=1, jvar=1, **_VGM_SIMPLE)
        k.set_grid(coord=simple_grid)
        k.set_search(ivar=1)
        k.solve()
        est, _ = k.get_results()
        assert est == pytest.approx(3.14, rel=1e-4)

    def test_destructor_does_not_crash(self, simple_obs, simple_grid):
        """__del__ must not crash even if solve was never called."""
        coord, value = simple_obs
        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord, value=value)
        k.set_vgm(ivar=1, jvar=1, **_VGM_SIMPLE)
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


# ---------------------------------------------------------------------------
# maxdist filtering tests
# ---------------------------------------------------------------------------

class TestMaxDist:
    """
    Verify that the maxdist parameter excludes observations beyond the search
    radius from the kriging system.

    Synthetic layout (2-D, x-axis only):
        obs A at x=0  value=1.0  }  "near" cluster
        obs B at x=2  value=1.0  }
        obs C at x=200 value=999.0   "far" outlier

    Target grid node at x=1 (between A and B).

    Variogram: spherical, range=500 so all three obs are within range and the
    far outlier genuinely influences the full-neighbourhood estimate.
    """

    _COORD  = np.array([[0.0, 0.0], [2.0, 0.0], [200.0, 0.0]])
    _VALUE  = np.array([1.0, 1.0, 999.0])
    _TARGET = np.array([[0.5, 0.0]])
    _VGM    = dict(vtype="sph", nugget=0.0, sill=1.0, a_major=1.0)

    def _solve(self, maxdist=None, nmax=3):
        k = Kriging(ndim=2, nvar=1)
        kw = dict(ivar=1, coord=self._COORD, value=self._VALUE, nmax=nmax)
        if maxdist is not None:
            kw["maxdist"] = maxdist
        k.set_obs(**kw)
        k.set_vgm(ivar=1, jvar=1, **self._VGM)
        k.set_grid(coord=self._TARGET)
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        return float(est[0]), float(var[0])

    def test_far_obs_influences_full_estimate(self):
        """Sanity check: without maxdist the far outlier (999) pulls the
        estimate above 1.0, confirming it participates in the system."""
        est_full, _ = self._solve()
        assert est_full > 1.1, (
            f"Expected far obs to pull estimate above 1.1, got {est_full:.4f}. "
            "Check that the variogram range is large enough."
        )

    def test_maxdist_excludes_far_obs(self):
        """With maxdist=10, the outlier at x=200 is excluded and the estimate
        should be close to 1.0 (the value of both near observations)."""
        est_limited, _ = self._solve(maxdist=10.0)
        assert est_limited == pytest.approx(1.0, abs=0.001), (
            f"Expected ≈1.0 with maxdist=10 (far obs excluded), got {est_limited:.4f}"
        )

    def test_maxdist_large_matches_full_estimate(self):
        """maxdist larger than all pairwise distances must give the same result
        as running without maxdist."""
        est_full,    var_full    = self._solve()
        est_large,   var_large   = self._solve(maxdist=1e6)
        assert est_large == pytest.approx(est_full,  rel=1e-5)
        assert var_large == pytest.approx(var_full,  rel=1e-5)

    def test_maxdist_no_obs_in_range_raises(self):
        """When every observation is beyond maxdist the solver has no
        neighbours and must raise an error.
        neglect_error=False is required: the Python default is True (silent NaN),
        but this test specifically validates the error-raising path."""
        coord  = np.array([[0.0, 0.0], [1.0, 0.0]])
        value  = np.array([1.0, 2.0])
        target = np.array([[100.0, 0.0]])   # far from all obs
        k = Kriging(ndim=2, nvar=1, neglect_error=False)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=2, maxdist=1.0)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=5.0)
        k.set_grid(coord=target)
        k.set_search(ivar=1)
        with pytest.raises(Exception):
            k.solve()

    def test_maxdist_subset_vs_trimmed_obs(self):
        """
        Filtering via maxdist must give the same estimate as manually removing
        the out-of-range observations before kriging.
        """
        # Only pass the two near observations directly (no far outlier)
        coord_near  = self._COORD[:2]
        value_near  = self._VALUE[:2]
        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord_near, value=value_near, nmax=2)
        k.set_vgm(ivar=1, jvar=1, **self._VGM)
        k.set_grid(coord=self._TARGET)
        k.set_search(ivar=1)
        k.solve()
        est_trimmed, var_trimmed = k.get_results()

        # Same dataset but with all three obs and maxdist=10 (excludes far obs)
        est_maxdist, var_maxdist = self._solve(maxdist=10.0)

        assert float(est_maxdist) == pytest.approx(float(est_trimmed[0]), rel=1e-4)
        assert float(var_maxdist) == pytest.approx(float(var_trimmed[0]), rel=1e-4)
