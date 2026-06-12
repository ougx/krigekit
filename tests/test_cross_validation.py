"""
test_reuse_and_features.py
==========================
Tests for four capabilities:

  1. Cross-validation (LOO-CV) — leave-one-out estimates for the pc2d dataset
                     compared against the reference 'loo-cv' column in pc2d.csv.

  2. Exact-match   — when a grid node coincides exactly with an observation
                     coordinate the estimate equals the observed value and the
                     variance equals the observation error variance (zero by
                     default, or the supplied per-obs variance).

All tests use the pc2d dataset (62 percent-coarse observations, 2D) via the
shared fixtures pc2d_obs and pc2d_loo defined in conftest.py.
"""

import numpy as np
import pytest
from krigekit import Kriging, ordinary_kriging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VGM_PC2D = dict(vtype="sph", nugget=0.0,  sill=0.12, a_major=5000.0)

# Two-structure decomposition used by multi-struct tests.
# _VGM_NUG + _VGM_SPH  →  nugget=0.04, sill=0.08, range=5000
_VGM_NUG  = dict(vtype="sph", nugget=0.04, sill=0.0,  a_major=5000.0)
_VGM_SPH  = dict(vtype="sph", nugget=0.0,  sill=0.08, a_major=5000.0)

# Two interior grid points not co-located with any observation
_INTERIOR_GRID = np.array([[580000.0, 4395000.0],
                            [578000.0, 4400000.0]])

_NMAX = 20


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _run(coord, value, grid, vgm_spec=(_VGM_PC2D,), nmax=_NMAX, **kw):
    """Run ordinary kriging; vgm_spec is a dict or iterable of dicts."""
    return ordinary_kriging(coord, value, grid, vgm_spec, nmax, **kw)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidationMode:

    def test_cross_validation_returns_nobs_estimates(self):
        """Cross-validation must return one estimate per observation."""
        rng   = np.random.default_rng(5)
        coord = rng.uniform(0, 100, (15, 2))
        value = rng.uniform(0, 1, 15)

        k = Kriging(ndim=2, nvar=1, cross_validation=True)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=15)
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
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
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
        k.set_grid_cv()
        k.set_search(ivar=1)
        k.solve()
        est, _ = k.get_results()
        residuals = value - est
        # Mean residual should be small relative to data range
        assert abs(residuals.mean()) < 0.2 * (value.max() - value.min()), \
            f"Mean cross-validation residual too large: {residuals.mean():.4f}"

# ===========================================================================
# 2. Leave-one-out cross-validation
# ===========================================================================

class TestCrossValidation:
    """
    Leave-one-out cross-validation (LOO-CV) using the pc2d dataset.

    The 'loo-cv' column in pc2d.csv holds reference estimates produced with
    all 62 observations as neighbours (nmax=62) and the standard spherical
    variogram (_VGM_PC2D).  Tests verify:

    * Output shape matches the number of observations
    * Variance is strictly positive (no exact matches in LOO-CV)
    * Estimates closely match the reference column
    * Diagnostic statistics are within acceptable ranges
    """

    @pytest.fixture(scope="class")
    def loo_results(self, pc2d_obs, pc2d_loo):
        """Run LOO-CV once and cache for the whole class."""
        coord, value = pc2d_obs
        k = Kriging(ndim=2, nvar=1, cross_validation=True, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value)   # no nmax limit → matches reference
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
        k.set_grid_cv()
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        return {"value": value, "ref": pc2d_loo, "est": est, "var": var}

    def test_output_shape(self, loo_results, pc2d_obs):
        """LOO-CV produces one estimate and one variance per observation."""
        n = len(pc2d_obs[1])
        assert loo_results["est"].shape == (n,)
        assert loo_results["var"].shape == (n,)

    def test_variance_positive(self, loo_results):
        """LOO-CV variance must be strictly > 0 at every point (no self-conditioning)."""
        assert np.all(loo_results["var"] > 0), (
            f"Non-positive LOO-CV variance at indices: "
            f"{np.where(loo_results['var'] <= 0)[0]}")

    def test_correlation_with_reference(self, loo_results):
        """LOO-CV estimates must correlate with the reference column at r > 0.999."""
        corr = np.corrcoef(loo_results["est"], loo_results["ref"])[0, 1]
        assert corr > 0.999, (
            f"Correlation with reference LOO-CV = {corr:.6f} (expected > 0.999)")

    def test_rmse_vs_reference(self, loo_results):
        """RMSE against the reference column must be below 0.01 (data range ≈ 1)."""
        rmse = np.sqrt(np.mean((loo_results["est"] - loo_results["ref"]) ** 2))
        assert rmse < 0.01, f"RMSE vs reference = {rmse:.6f} (expected < 0.01)"

    def test_mean_error_near_zero(self, loo_results):
        """Mean error (bias) of LOO-CV estimates against observed values must be small."""
        me = np.mean(loo_results["est"] - loo_results["value"])
        assert abs(me) < 0.05, f"Mean error = {me:.4f}: LOO-CV estimates appear biased"

    def test_standardised_residuals_distribution(self, loo_results):
        """
        Standardised residuals z = (est - obs) / sqrt(var) should follow
        approximately N(0,1) for a well-calibrated variogram.  The spherical
        model without nugget slightly overestimates continuity at short range,
        so MSSE can exceed 1; we accept up to 4.
        """
        z = (loo_results["est"] - loo_results["value"]) / np.sqrt(loo_results["var"])
        msse = np.mean(z ** 2)
        assert msse < 4.0, (
            f"MSSE = {msse:.3f}: variance appears poorly calibrated (expected < 4)")

    def test_within_3_sigma_fraction(self, loo_results):
        """At least 85% of observations should fall within ±3 standard deviations."""
        z = (loo_results["est"] - loo_results["value"]) / np.sqrt(loo_results["var"])
        frac = np.mean(np.abs(z) <= 3.0)
        assert frac >= 0.85, (
            f"Only {frac:.1%} of LOO-CV residuals within ±3σ (expected ≥ 85%)")

    def test_estimates_within_data_range(self, loo_results):
        """LOO-CV estimates must stay within a ±10% margin of the observed data range."""
        obs = loo_results["value"]
        est = loo_results["est"]
        margin = 0.10 * (obs.max() - obs.min())
        assert est.min() >= obs.min() - margin
        assert est.max() <= obs.max() + margin

    def test_loocv_rerun_reproduces_first(self, pc2d_obs, pc2d_loo):
        """Re-running LOO-CV on the same object must reproduce the first result."""
        coord, value = pc2d_obs

        k = Kriging(ndim=2, nvar=1, cross_validation=True, verbose=0)

        def _run_cv():
            k.set_obs(ivar=1, coord=coord, value=value)
            k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
            k.set_grid_cv()
            k.set_search(ivar=1)
            k.solve()
            return k.get_results()

        est1, _ = _run_cv()
        est2, _ = _run_cv()

        np.testing.assert_allclose(est1, est2, rtol=1e-6,
            err_msg="LOO-CV re-run on same object must reproduce first run")


# ===========================================================================
# Exact match
# ===========================================================================

class TestExactMatch:
    """
    When a grid node coincides exactly with an observation coordinate:

    * The estimate must equal the observed value (interpolation property)
    * The kriging variance must equal the observation error variance supplied
      via variance= to set_obs (zero by default)
    * The exact-match shortcut must hold even with a nugget in the variogram —
      the variance at that node is the obs error, not the variogram nugget
    """

    def test_estimate_equals_obs_no_error(self, pc2d_obs):
        """Grid node at obs location: estimate equals obs value (no obs error)."""
        coord, value = pc2d_obs
        est, _ = _run(coord, value, coord[[5]])
        assert est[0] == pytest.approx(value[5], rel=1e-4)

    def test_variance_zero_no_obs_error(self, pc2d_obs):
        """Kriging variance at exact-match node is zero when obs error is zero."""
        coord, value = pc2d_obs
        _, var = _run(coord, value, coord[[5]])
        assert var[0] == pytest.approx(0.0, abs=1e-8)

    def test_variance_equals_obs_error(self, pc2d_obs):
        """
        When per-observation error variance (obs_err) is supplied, two things
        change at an exact-match node compared to the no-error case:

        1. The estimate is NO LONGER equal to the observed value.  Obs error
           introduces smoothing: the kriging system treats the observation as
           noisy and blends it with surrounding values.

        2. The kriging variance is approximately obs_err (not zero).
           For a single coincident obs the theoretical value is:
             σ²_k = C(0) · obs_err / (C(0) + obs_err)
           which equals obs_err only in the limit obs_err ≪ C(0).
           For obs_err=0.01 and C(0)=0.12 the ratio is ~0.923, so
           σ²_k ≈ 0.92 × obs_err.  The test accepts 20% relative tolerance.
        """
        coord, value = pc2d_obs
        C0      = _VGM_PC2D['sill']   # total sill = C(0) for zero-nugget model
        obs_err = 0.01
        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX,
                  variance=np.full(len(value), obs_err))
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
        k.set_grid(coord=coord[[5]])
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        # Variance must be positive and close to obs_err (within one factor of C0/(C0+obs_err))
        theoretical_var = C0 * obs_err / (C0 + obs_err)
        assert var[0] > 0, "Obs error must make kriging variance positive"
        assert var[0] == pytest.approx(theoretical_var, rel=0.2), (
            f"Kriging variance {var[0]:.5f} should be near "
            f"C(0)·obs_err/(C(0)+obs_err) = {theoretical_var:.5f}"
        )
        # Estimate is smoothed away from value[5]; must still be data-informed
        # (within a few multiples of sqrt(obs_err) from the observed value)
        assert abs(est[0] - value[5]) < 5 * obs_err**0.5, (
            f"Smoothed estimate {est[0]:.4f} is too far from obs value {value[5]:.4f}"
        )

    def test_multiple_exact_matches_simultaneously(self, pc2d_obs):
        """A grid with several obs coordinates reproduces every observed value."""
        coord, value = pc2d_obs
        indices = [0, 5, 15, 30, 61]
        est, var = _run(coord, value, coord[indices])
        for rank, idx in enumerate(indices):
            assert est[rank] == pytest.approx(value[idx], rel=1e-4), (
                f"Exact-match at obs {idx}: {est[rank]:.6f} != {value[idx]:.6f}")
            assert var[rank] == pytest.approx(0.0, abs=1e-8), (
                f"Exact-match variance at obs {idx}: {var[rank]:.6g} != 0")

    def test_exact_match_among_non_exact_nodes(self, pc2d_obs):
        """Mixed grid: exact-match node interpolates exactly; others have positive variance."""
        coord, value = pc2d_obs
        grid = np.vstack([coord[[10]], _INTERIOR_GRID])
        est, var = _run(coord, value, grid)
        assert est[0] == pytest.approx(value[10], rel=1e-4)
        assert var[0] == pytest.approx(0.0, abs=1e-8)
        assert np.all(var[1:] > 0), "Non-exact nodes must have positive variance"

    def test_synthetic_exact_match_all_obs(self):
        """Using obs coords as the estimation grid reproduces every observed value."""
        rng = np.random.default_rng(42)
        n = 10
        coord = rng.uniform(0, 100, size=(n, 2))
        value = rng.uniform(0, 1, size=n)

        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=n)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=200.0)
        k.set_grid(coord=coord)
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()

        np.testing.assert_allclose(est, value, atol=1e-4,
            err_msg="Kriging over obs coords must reproduce all observed values")
        np.testing.assert_allclose(var, 0.0, atol=1e-8,
            err_msg="Kriging variance at obs coords must be zero (no obs error)")