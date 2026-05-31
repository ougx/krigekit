"""
test_update_obs_value.py
========================
Tests for Kriging.update_obs_value(), which replaces observation values
in-place (coordinates and kd-tree unchanged) — primarily for weight reuse.

Coverage
--------
Auto-switch (primary use case)
    After the first solve() with store_weight=True, update_obs_value() automatically
    enables use_old_weight mode so the same object can be reused directly:
        k.solve()              # computes and stores weights
        k.update_obs_value()   # switches to weight-reuse mode
        k.solve()              # reuses weights with new values

Basic correctness
    Same values → same estimates.
    Doubled values → doubled estimates (ordinary kriging is linear, weights sum to 1).
    Shifted values → shifted estimates by the same constant.

Weight reuse
    store_weight + update_obs_value + re-solve matches a fresh solve with new values.
    Updating to new values then restoring originals reproduces original estimates.

Multi-variable
    update_obs_value works for each ivar in a co-kriging setup.

Error handling
    Passing a value array of wrong length raises ValueError before calling Fortran.
"""

import numpy as np
import pytest
from pykriging import Kriging


_VGM  = dict(vtype="sph", nugget=0.01, sill=0.09, a_major=100.0)
_NMAX = 5


def _build_and_solve(coord, value, grid, **kwargs):
    k = Kriging(ndim=2, nvar=1, **kwargs)
    k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
    k.set_grid(coord=grid)
    if not k.use_old_weight:
        k.set_vgm(ivar=1, jvar=1, **_VGM)
        k.set_search(ivar=1)
    k.solve()
    return k


# ===========================================================================
# Auto-switch (primary use case)
# ===========================================================================

class TestUpdateObsValueAutoSwitch:
    """
    After the first solve() with store_weight=True, update_obs_value() flips
    the object to use_old_weight mode automatically.  The caller can then just
    do k.update_obs_value(...); k.solve() without any extra object setup.
    """

    def test_auto_switch_result_matches_fresh_solve(self, simple_obs, simple_grid):
        """
        Solve once with store_weight=True, then update_obs_value + solve on the
        same object.  Result must match a fully independent fresh solve with the
        new values.
        """
        coord, value = simple_obs
        value2 = value * 1.5 + 0.1

        est_ref, var_ref = _build_and_solve(coord, value2, simple_grid).get_results()

        k = _build_and_solve(coord, value, simple_grid, store_weight=True)
        k.update_obs_value(ivar=1, value=value2)
        k.solve()
        est, var = k.get_results()

        np.testing.assert_allclose(est, est_ref, rtol=1e-5,
            err_msg="Auto-switch estimate differs from fresh solve with new values")
        np.testing.assert_allclose(var, var_ref, rtol=1e-5,
            err_msg="Auto-switch variance differs from fresh solve with new values")

    def test_auto_switch_linearity(self, simple_obs, simple_grid):
        """After auto-switch, ordinary kriging linearity still holds."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid, store_weight=True)
        est_orig, _ = k.get_results()

        k.update_obs_value(ivar=1, value=value * 2.0)
        k.solve()
        est_double, _ = k.get_results()

        np.testing.assert_allclose(est_double, est_orig * 2.0, rtol=1e-6,
            err_msg="Auto-switch: doubling obs values should double the OK estimate")

    def test_auto_switch_multiple_rounds(self, simple_obs, simple_grid):
        """Multiple sequential update+solve rounds each give the correct estimate."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid, store_weight=True)

        for scale in [2.0, 0.5, 3.0, 1.0]:
            k.update_obs_value(ivar=1, value=value * scale)
            k.solve()
            est, _ = k.get_results()

            est_ref, _ = _build_and_solve(coord, value * scale, simple_grid).get_results()
            np.testing.assert_allclose(est, est_ref, rtol=1e-5,
                err_msg=f"Auto-switch round scale={scale}: estimate differs from fresh solve")

    def test_auto_switch_no_store_weight_no_switch(self, simple_obs, simple_grid):
        """
        Without store_weight=True, update_obs_value does not switch to
        use_old_weight — the next solve recomputes normally.
        """
        coord, value = simple_obs
        value2 = value + 1.0

        # plain solve (no store_weight) → solve is a full recompute
        k = _build_and_solve(coord, value, simple_grid)
        k.update_obs_value(ivar=1, value=value2)
        k.solve()
        est, _ = k.get_results()

        est_ref, _ = _build_and_solve(coord, value2, simple_grid).get_results()
        np.testing.assert_allclose(est, est_ref, rtol=1e-5,
            err_msg="Without store_weight, update+solve should still give correct result")


# ===========================================================================
# Basic correctness
# ===========================================================================

class TestUpdateObsValueBasic:

    def test_same_values_same_result(self, simple_obs, simple_grid):
        """Updating with identical values and re-solving reproduces the original estimate."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid)
        est_orig, var_orig = k.get_results()

        k.update_obs_value(ivar=1, value=value)
        k.solve()
        est_upd, var_upd = k.get_results()

        np.testing.assert_allclose(est_upd, est_orig, rtol=1e-10,
            err_msg="update_obs_value with same values changed the estimate")
        np.testing.assert_allclose(var_upd, var_orig, rtol=1e-10,
            err_msg="update_obs_value with same values changed the variance")

    def test_doubled_values_double_estimate(self, simple_obs, simple_grid):
        """OK is linear: doubling all obs values doubles every block estimate."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid)
        est_orig, _ = k.get_results()

        k.update_obs_value(ivar=1, value=value * 2.0)
        k.solve()
        est_upd, _ = k.get_results()

        np.testing.assert_allclose(est_upd, est_orig * 2.0, rtol=1e-6,
            err_msg="update_obs_value: doubling obs values should double the OK estimate")

    def test_shifted_values_shift_estimate(self, simple_obs, simple_grid):
        """OK is linear: adding a constant to all obs shifts every block estimate by the same constant."""
        coord, value = simple_obs
        shift = 5.0
        k = _build_and_solve(coord, value, simple_grid)
        est_orig, _ = k.get_results()

        k.update_obs_value(ivar=1, value=value + shift)
        k.solve()
        est_upd, _ = k.get_results()

        np.testing.assert_allclose(est_upd, est_orig + shift, rtol=1e-6, atol=1e-8,
            err_msg="update_obs_value: constant shift to obs should shift OK estimate by same amount")

    def test_variance_unchanged_after_update(self, simple_obs, simple_grid):
        """Kriging variance depends only on geometry and variogram, not on obs values."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid)
        _, var_orig = k.get_results()

        k.update_obs_value(ivar=1, value=value * 99.0 + 7.0)
        k.solve()
        _, var_upd = k.get_results()

        np.testing.assert_allclose(var_upd, var_orig, rtol=1e-10,
            err_msg="Kriging variance should be independent of obs values")


# ===========================================================================
# Weight reuse
# ===========================================================================

class TestUpdateObsValueWeightReuse:

    def test_weight_reuse_matches_fresh_solve(self, simple_obs, simple_grid):
        """
        Solve with store_weight, update values, re-solve.
        Result must match a fully independent fresh solve with the new values.
        """
        coord, value = simple_obs
        value2 = value * 1.5 + 0.1

        # Reference: full independent solve with new values
        est_ref, var_ref = _build_and_solve(coord, value2, simple_grid).get_results()

        # Weight-reuse path: solve once, store weights, update values, re-solve
        k = _build_and_solve(coord, value, simple_grid, store_weight=True)
        k.update_obs_value(ivar=1, value=value2)
        k.solve()
        est_reuse, var_reuse = k.get_results()

        np.testing.assert_allclose(est_reuse, est_ref, rtol=1e-5,
            err_msg="Weight-reuse + update_obs_value estimate differs from fresh solve")
        np.testing.assert_allclose(var_reuse, var_ref, rtol=1e-5,
            err_msg="Weight-reuse + update_obs_value variance differs from fresh solve")

    def test_set_weights_then_update_obs_value(self, simple_obs, simple_grid):
        """
        get_weights → set_weights → update_obs_value → solve.
        Result must match a fresh solve with the new values.
        """
        coord, value = simple_obs
        value2 = value * 2.0

        est_ref, _ = _build_and_solve(coord, value2, simple_grid).get_results()

        k_store = _build_and_solve(coord, value, simple_grid, store_weight=True)
        w = k_store.get_weights()

        k_reuse = Kriging(ndim=2, nvar=1, use_old_weight=True)
        k_reuse.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k_reuse.set_grid(coord=simple_grid)
        k_reuse.set_weights(w)
        k_reuse.update_obs_value(ivar=1, value=value2)
        k_reuse.solve()
        est_reuse, _ = k_reuse.get_results()

        np.testing.assert_allclose(est_reuse, est_ref, rtol=1e-5,
            err_msg="set_weights + update_obs_value estimate differs from fresh solve")

    def test_update_then_restore(self, simple_obs, simple_grid):
        """Updating values then restoring the originals reproduces the original estimate."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid)
        est_orig, _ = k.get_results()

        k.update_obs_value(ivar=1, value=value * 99.0)
        k.solve()

        k.update_obs_value(ivar=1, value=value)
        k.solve()
        est_restored, _ = k.get_results()

        np.testing.assert_allclose(est_restored, est_orig, rtol=1e-10,
            err_msg="Restoring original values should reproduce original estimate")

    def test_multiple_sequential_updates(self, simple_obs, simple_grid):
        """Multiple sequential update+solve calls each produce consistent results."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid)

        for scale in [1.0, 2.0, 0.5, 3.0, 1.0]:
            k.update_obs_value(ivar=1, value=value * scale)
            k.solve()
            est, _ = k.get_results()

            est_ref, _ = _build_and_solve(coord, value * scale, simple_grid).get_results()
            np.testing.assert_allclose(est, est_ref, rtol=1e-5,
                err_msg=f"Sequential update with scale={scale} differs from fresh solve")


# ===========================================================================
# Multi-variable (co-kriging)
# ===========================================================================

class TestUpdateObsValueCokriging:

    _VGM_CK = [
        dict(ivar=1, jvar=1, vtype="sph", nugget=0.01, sill=0.09, a_major=100.0),
        dict(ivar=1, jvar=2, vtype="sph", nugget=0.00, sill=0.03, a_major=100.0),
        dict(ivar=2, jvar=2, vtype="sph", nugget=0.02, sill=0.16, a_major=100.0),
    ]

    def _build(self, coord, v1, v2, grid, **kwargs):
        k = Kriging(ndim=2, nvar=2, **kwargs)
        k.set_obs(ivar=1, coord=coord, value=v1, nmax=_NMAX)
        k.set_obs(ivar=2, coord=coord, value=v2, nmax=_NMAX)
        k.set_grid(coord=grid)
        if not k.use_old_weight:
            for vgm in self._VGM_CK:
                k.set_vgm(**vgm)
            k.set_search(ivar=1)
            k.set_search(ivar=2)
        k.solve()
        return k

    def test_update_all_vars_doubled(self, simple_obs, simple_grid):
        """Doubling all co-kriging obs values doubles all estimates (system is linear)."""
        coord, v1 = simple_obs
        v2 = v1 * 2.0 + 0.5
        k = self._build(coord, v1, v2, simple_grid)
        est_orig, _ = k.get_results()

        k.update_obs_value(ivar=1, value=v1 * 2.0)
        k.update_obs_value(ivar=2, value=v2 * 2.0)
        k.solve()
        est_upd, _ = k.get_results()

        np.testing.assert_allclose(est_upd, est_orig * 2.0, rtol=1e-5,
            err_msg="Co-kriging: doubling all obs values should double all estimates")

    def test_update_matches_fresh_cokriging_solve(self, simple_obs, simple_grid):
        """update_obs_value on co-kriging matches a fresh co-kriging solve with new values."""
        coord, v1 = simple_obs
        v2        = v1 * 1.8 + 0.3
        v1_new    = v1 * 0.7 + 1.2
        v2_new    = v2 * 0.9 - 0.1

        est_ref, _ = self._build(coord, v1_new, v2_new, simple_grid).get_results()

        k = self._build(coord, v1, v2, simple_grid)
        k.update_obs_value(ivar=1, value=v1_new)
        k.update_obs_value(ivar=2, value=v2_new)
        k.solve()
        est_upd, _ = k.get_results()

        np.testing.assert_allclose(est_upd, est_ref, rtol=1e-5,
            err_msg="Co-kriging update_obs_value result differs from fresh co-kriging solve")


# ===========================================================================
# Error handling
# ===========================================================================

class TestUpdateObsValueErrors:

    def test_too_long_raises(self, simple_obs, simple_grid):
        """Passing a value array that is one element too long raises ValueError."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid)
        with pytest.raises(ValueError, match="must match nobs"):
            k.update_obs_value(ivar=1, value=np.append(value, [99.0]))

    def test_too_short_raises(self, simple_obs, simple_grid):
        """Passing a value array that is one element too short raises ValueError."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid)
        with pytest.raises(ValueError, match="must match nobs"):
            k.update_obs_value(ivar=1, value=value[:-1])

    def test_error_message_includes_nobs(self, simple_obs, simple_grid):
        """The ValueError message contains the expected nobs for the variable."""
        coord, value = simple_obs
        k = _build_and_solve(coord, value, simple_grid)
        nobs = value.shape[0]
        with pytest.raises(ValueError, match=str(nobs)):
            k.update_obs_value(ivar=1, value=value[:-1])
