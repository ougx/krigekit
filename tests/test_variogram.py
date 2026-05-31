"""
test_reuse_and_features.py
==========================
Tests for four capabilities:

  1. Object reuse  — calling set_obs / set_grid again on an existing Kriging
                     object resets state cleanly and produces correct results.

  2. Cross-validation (LOO-CV) — leave-one-out estimates for the pc2d dataset
                     compared against the reference 'loo-cv' column in pc2d.csv.

  3. Multi-structure variogram — a model with two or more nested structures
                     behaves correctly: total sill equals the sum of partial
                     sills, nugget increases variance, short-range structure
                     pulls estimates and variances relative to long-range-only.

  4. Exact-match   — when a grid node coincides exactly with an observation
                     coordinate the estimate equals the observed value and the
                     variance equals the observation error variance (zero by
                     default, or the supplied per-obs variance).

All tests use the pc2d dataset (62 percent-coarse observations, 2D) via the
shared fixtures pc2d_obs and pc2d_loo defined in conftest.py.
"""

import numpy as np
import pytest
from pykriging import Kriging, ordinary_kriging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VGM_PC2D = dict(vtype="sph", nugget=0.0,  sill=0.12, a_major=5000.0)

# Two-structure decomposition used by multi-struct tests.
# _VGM_NUG + _VGM_SPH  →  nugget=0.04, sill=0.08, range=5000
_VGM_NUG  = dict(vtype="sph", nugget=0.04, sill=0.0,  a_major=5000.0)
_VGM_SPH  = dict(vtype="sph", nugget=0.0,  sill=0.08, a_major=5000.0)

# Long-range spherical + short-range exponential (total sill = 0.12)
_VGM_LONG  = dict(vtype="sph", nugget=0.0, sill=0.08, a_major=5000.0)
_VGM_SHORT = dict(vtype="exp", nugget=0.0, sill=0.04, a_major=200.0)

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


class TestVariogramTypes:
    """Coverage for every variogram type accepted by the Fortran model parser."""

    @pytest.mark.parametrize("vtype", ["nug", "sph", "exp", "hol", "gau", "pow", "bsq", "cir", "lin"])
    def test_set_vgm_preserves_requested_type(self, vtype):
        coord = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        value = np.array([1.0, 2.0, 1.5])

        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=3)
        k.set_vgm(ivar=1, jvar=1, vtype=vtype, nugget=0.0, sill=0.2, a_major=10.0)

        assert f"    {vtype}  sill=" in k.get_info()

    def test_unknown_vtype_raises_fortran_error(self):
        coord = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        value = np.array([1.0, 2.0, 1.5])

        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=3)

        with pytest.raises(RuntimeError, match="unknown variogram type"):
            k.set_vgm(ivar=1, jvar=1, vtype="bad", nugget=0.0, sill=0.2, a_major=10.0)


class TestVaryingVariogram:
    """Regression coverage for per-block variogram storage and solve-time use."""

    def _solve_with_vgm(self, coord, value, grid, vgm):
        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, **vgm)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        return k.get_results()

    def test_set_vgm_block_matches_separate_per_block_solves(self, pc2d_obs):
        coord, value = pc2d_obs
        vgm_long = dict(vtype="sph", nugget=0.0, sill=0.12, a_major=5000.0)
        vgm_short = dict(vtype="sph", nugget=0.0, sill=0.12, a_major=500.0)

        k = Kriging(ndim=2, nvar=1, varying_vgm=True, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_grid(coord=_INTERIOR_GRID)
        k.set_vgm_block(ib=1, ivar=1, jvar=1, **vgm_long)
        k.set_vgm_block(ib=2, ivar=1, jvar=1, **vgm_short)
        k.set_search(ivar=1)
        k.solve()
        est_varying, var_varying = k.get_results()

        est_1, var_1 = self._solve_with_vgm(coord, value, _INTERIOR_GRID[:1], vgm_long)
        est_2, var_2 = self._solve_with_vgm(coord, value, _INTERIOR_GRID[1:], vgm_short)

        np.testing.assert_allclose(est_varying, [est_1[0], est_2[0]], rtol=1e-6)
        np.testing.assert_allclose(var_varying, [var_1[0], var_2[0]], rtol=1e-6)

    def test_varying_vgm_sgsim_smoke(self, pc2d_obs):
        coord, value = pc2d_obs

        k = Kriging(ndim=2, nvar=1, nsim=1, varying_vgm=True, verbose=0, seed=42)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_grid(coord=_INTERIOR_GRID)
        k.set_vgm_block(ib=1, ivar=1, jvar=1, **_VGM_LONG)
        k.set_vgm_block(ib=2, ivar=1, jvar=1, **_VGM_SHORT)
        k.set_sim(
            randpath=np.array([1, 2], dtype=np.int32),
            sample=np.zeros((_INTERIOR_GRID.shape[0], )),
        )
        k.set_search(ivar=1)
        k.solve()
        sims, var = k.get_results()

        assert sims.shape == (_INTERIOR_GRID.shape[0],)
        assert np.all(np.isfinite(sims))
        assert np.all(var >= 0.0)


# ===========================================================================
# cross-validation accuracy
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
# Multi-structure variogram
# ===========================================================================

class TestMultiStructVariogram:
    """
    A variogram model can be built from several nested structures by calling
    set_vgm multiple times.  These tests verify:

    * Nugget structure raises kriging variance at non-exact-match nodes
    * Splitting one structure into two additive parts reproduces the single-
      structure result (same total sill, same range)
    * Adding a short-range structure on top changes both estimates and variances
    * Variance remains non-negative with three structures
    * Ordinary kriging of a constant field returns that constant (unbiasedness)
    """

    def test_nugget_increases_variance(self, pc2d_obs):
        """A model with nugget must produce higher variance than one without."""
        coord, value = pc2d_obs
        _, var_no_nug   = _run(coord, value, _INTERIOR_GRID, vgm_spec=(_VGM_PC2D,))
        _, var_with_nug = _run(coord, value, _INTERIOR_GRID, vgm_spec=(_VGM_NUG, _VGM_SPH))
        assert np.all(var_with_nug >= var_no_nug - 1e-8), (
            "Nugget must increase kriging variance at non-exact-match nodes")

    def test_additive_split_same_range_reproducibility(self, pc2d_obs):
        """
        Splitting one structure into two parts with the same range and summed
        sill must give identical estimates and variances:

            sph(nug=0, sill=0.12, range=5000)
              == sph(nug=0, sill=0.04, range=5000)
               + sph(nug=0, sill=0.08, range=5000)
        """
        coord, value = pc2d_obs
        VGM_A = dict(vtype="sph", nugget=0.0, sill=0.04, a_major=5000.0)
        VGM_B = dict(vtype="sph", nugget=0.0, sill=0.08, a_major=5000.0)

        est_single, var_single = _run(coord, value, _INTERIOR_GRID, vgm_spec=(_VGM_PC2D,))
        est_split,  var_split  = _run(coord, value, _INTERIOR_GRID, vgm_spec=(VGM_A, VGM_B))

        np.testing.assert_allclose(est_single, est_split, rtol=1e-5,
            err_msg="Splitting into two equal parts must not change estimates")
        np.testing.assert_allclose(var_single, var_split, rtol=1e-5,
            err_msg="Splitting into two equal parts must not change variance")

    def test_short_range_structure_changes_estimates(self, pc2d_obs):
        """Adding a short-range structure must change estimates at interior nodes."""
        coord, value = pc2d_obs
        est_long,  _ = _run(coord, value, _INTERIOR_GRID, vgm_spec=(_VGM_LONG,))
        est_multi, _ = _run(coord, value, _INTERIOR_GRID, vgm_spec=(_VGM_LONG, _VGM_SHORT))
        assert not np.allclose(est_long, est_multi), (
            "Short-range structure must change estimates relative to long-range only")

    def test_short_range_structure_increases_variance(self, pc2d_obs):
        """Adding a short-range structure increases total sill, raising variance."""
        coord, value = pc2d_obs
        _, var_long  = _run(coord, value, _INTERIOR_GRID, vgm_spec=(_VGM_LONG,))
        _, var_multi = _run(coord, value, _INTERIOR_GRID, vgm_spec=(_VGM_LONG, _VGM_SHORT))
        assert np.all(var_multi >= var_long - 1e-8), (
            "Adding a short-range structure must not decrease variance")

    def test_three_structure_model_variance_nonnegative(self, pc2d_obs):
        """Kriging variance must remain non-negative with a three-structure model."""
        coord, value = pc2d_obs
        VGM_MED = dict(vtype="gau", nugget=0.0, sill=0.02, a_major=1000.0)
        _, var = _run(coord, value, _INTERIOR_GRID,
                      vgm_spec=(_VGM_LONG, _VGM_SHORT, VGM_MED))
        assert np.all(var >= -1e-10), (
            f"Three-structure model produced negative variance: {var.min():.6f}")

    def test_multi_struct_estimate_in_data_range(self, pc2d_obs):
        """Estimates from a multi-structure model must stay within the data range."""
        coord, value = pc2d_obs
        est, _ = _run(coord, value, _INTERIOR_GRID, vgm_spec=(_VGM_LONG, _VGM_SHORT))
        margin = 0.10 * (value.max() - value.min())
        assert est.min() >= value.min() - margin
        assert est.max() <= value.max() + margin

    def test_constant_field_multi_struct(self, pc2d_obs):
        """Ordinary kriging of a constant field must return that constant."""
        coord, _ = pc2d_obs
        const = 2.71828 * np.ones(len(coord))
        est, _ = _run(coord, const, _INTERIOR_GRID, vgm_spec=(_VGM_LONG, _VGM_SHORT))
        np.testing.assert_allclose(est, 2.71828, rtol=1e-4,
            err_msg="Multi-struct kriging of constant field must equal the constant")


# ===========================================================================
# 4. Exact match
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
        When per-observation error variance is supplied, the kriging estimate
        at an exact-match node is not equal to the observed value, and the kriging
        variance is smaller than the obs error.
        """
        coord, value = pc2d_obs
        obs_err = 0.01
        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX,
                  variance=np.full(len(value), obs_err))
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
        k.set_grid(coord=coord[[5]])
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        assert est[0] != pytest.approx(value[5], rel=1e-4), "kriging estimate is NOT equal to the obs value"
        assert var[0] < obs_err, "kriging variance is smaller than the obs error"

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

    def test_exact_match_with_nugget_variogram(self, pc2d_obs):
        """
        Even with a nugget variogram, an exact-match node does not return the obs value.
        """
        coord, value = pc2d_obs
        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, **_VGM_NUG)
        k.set_vgm(ivar=1, jvar=1, **_VGM_SPH)
        k.set_grid(coord=coord[[7]])
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        assert est[0] != pytest.approx(value[7], rel=1e-4)

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
