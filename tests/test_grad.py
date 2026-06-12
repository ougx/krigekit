"""
test_grad.py
============
Tests for gradient observation pairs (Delhomme 1979: "Kriging in hydrology").

A gradient pair (xs1, xs2, d) encodes the finite-difference constraint

    Z(xs1) - Z(xs2) = d

as an additional row/column in the kriging matrix.  For no-flow (zero-flux)
boundaries, d = 0.  The constraint is *exact* when variance = 0 (default) and
relaxed when a positive gradient variance is supplied.

Coverage
--------
Dimensions / API
    ngroups equals ngroups_base (= nvar, no SGSIM) when set_grad is not called.
    ngroups expands to ngroups_base + nvar after set_grad.
    Weight store reflects the expanded ngroups.
    Grad group inear indices are sequential (1-based).

Null properties — pure-nugget model
    In a pure-nugget model all cross-covariances between distinct locations
    are zero.  Adding any gradient pair (regardless of grad_value) leaves the
    kriging system block-diagonal: the grad block solves to theta = 0 and the
    obs block is unchanged.  Estimate and variance are identical to the no-grad
    case.  Derived analytically in the class docstring.

Directional correctness
    Setting grad_val > 0 means Z(xs1) > Z(xs2), i.e. the field is higher near
    xs1.  A target at xs1 therefore has a *higher* estimate than without the
    constraint.  Tested with zero obs values so the estimate is purely
    theta * grad_val.

Linearity
    The kriging system is linear: halving grad_val halves the estimate (with
    fixed-zero obs values).

Unbiasedness — constant-shift invariance
    Adding a constant C to all obs values shifts the estimate by C.  The
    grad_val = Z(xs1) - Z(xs2) is a *difference*, so it is unchanged when all
    values shift by C.

Soft constraints
    Setting a large gradient variance relaxes the constraint, pulling the
    estimate toward the no-grad solution.

Weight-store with gradient data
    get_weights() returns ngroups = ngroups_base + nvar groups when grad is set.
    Estimate reconstructed manually from stored obs weights and grad weights
    matches solve() output exactly:
        est[b] = sum_i  w[b,0,i] * z_obs[inear[b,0,i]-1]
               + sum_j  w[b,1,j] * d_grad[inear[b,1,j]-1]
    where group-0 = obs, group-1 = grad (nvar=1, no SGSIM case).

use_old_weight round-trip with gradient data
    Storing weights to a factor file and reloading with use_old_weight=True
    reproduces the same estimates and variances, *provided* set_grad is called
    before the second solve (to supply grad values to estimate_block).
"""

import numpy as np
import pytest
from krigekit import Kriging


# ============================================================================
# Module-level variogram specs
# ============================================================================

# Spherical: no nugget, sill = 4, range = 10
_SPH = dict(vtype="sph", nugget=0.0, sill=4.0,
            a_major=10.0, a_minor1=10.0, a_minor2=10.0, azimuth=0.0)

# Pure-nugget: C(0) = 2, C(h > 0) = 0 (sill = 0 → no spatial structure)
_NUG = dict(vtype="sph", nugget=2.0, sill=0.0,
            a_major=1.0, a_minor1=1.0, a_minor2=1.0, azimuth=0.0)

_NMAX = 50


# ============================================================================
# Helper builders
# ============================================================================

def _ok1d(obs_x, obs_z, target_x, vgm_spec, *,
          grad_c1=None, grad_c2=None, grad_val=None,
          grad_var=None, store_weight=False, **kw):
    """
    Build and solve 1-D ordinary-kriging (nvar=1) at a single target location.

    Parameters
    ----------
    obs_x : array_like, shape (n,)
        Observation 1-D x-coordinates.
    obs_z : array_like, shape (n,)
        Observed values.
    target_x : float
        Target estimation coordinate.
    vgm_spec : dict
        Keyword arguments for k.set_vgm.
    grad_c1, grad_c2 : array_like, shape (ng,), optional
        Positive-side and negative-side coordinates of each gradient pair.
        If provided, all three must be given together.
    grad_val : array_like, shape (ng,), optional
        Known differences Z(xs1) - Z(xs2) for each pair.
    grad_var : array_like, shape (ng,), optional
        Gradient observation variances; 0 = exact constraint (default).
    store_weight : bool
        Forward to the Kriging constructor.
    **kw
        Additional keyword arguments for the Kriging constructor.

    Returns
    -------
    k : Kriging
        Solved Kriging object.  Call ``get_results()`` or ``get_weights()``.
    """
    coord = np.asarray(obs_x, dtype=float).reshape(-1, 1)
    tgt   = np.array([[float(target_x)]])

    k = Kriging(ndim=1, nvar=1, store_weight=store_weight, **kw)
    k.set_obs(1, coord=coord, value=np.asarray(obs_z, dtype=float), nmax=_NMAX)
    k.set_vgm(1, 1, **vgm_spec)
    k.set_grid(tgt)
    k.set_search(1)

    if grad_c1 is not None:
        c1   = np.asarray(grad_c1, dtype=float).reshape(-1, 1)
        c2   = np.asarray(grad_c2, dtype=float).reshape(-1, 1)
        gv   = np.asarray(grad_val, dtype=float).ravel()
        gvar = (np.asarray(grad_var, dtype=float).ravel()
                if grad_var is not None else None)
        k.set_grad(coord1=c1, coord2=c2, grad_value=gv, ivar=1, variance=gvar)

    k.solve()
    return k


def _scalar(k):
    """Return (est, var) as Python floats for a single-block result."""
    est, var = k.get_results()
    return float(est[0]), float(var[0])


# ============================================================================
# 1. Dimensions and API
# ============================================================================

class TestGradDimensions:
    """
    Verify that ngroups is nvar (= 1) without grad and expands to
    ngroups_base + nvar (= 2) after set_grad.  Checked via the weight store.
    """

    def test_weight_store_ngroups_base_no_grad(self):
        """
        Without set_grad: ngroups = nvar = 1.
        Weight store shape[1] (ngroups axis) must equal 1.
        """
        k = _ok1d([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], 5.0,
                  _SPH, store_weight=True)
        w = k.get_weights()
        assert w["nnear"].shape[1] == 1, (
            f"Expected ngroups=1 without grad, got {w['nnear'].shape[1]}"
        )

    def test_weight_store_ngroups_expanded_with_grad(self):
        """
        With set_grad (ivar=1): ngroups = ngroups_base + nvar = 1 + 1 = 2.
        Weight store shape[1] must equal 2.
        """
        k = _ok1d(
            [0.0, 2.0], [0.0, 0.0], 5.0, _SPH,
            grad_c1=[3.0], grad_c2=[4.0], grad_val=[1.0],
            store_weight=True,
        )
        w = k.get_weights()
        assert w["nnear"].shape[1] == 2, (
            f"Expected ngroups=2 with grad (ngroups_base=1 + nvar=1), "
            f"got {w['nnear'].shape[1]}"
        )

    def test_grad_group_nnear_equals_ngrad(self):
        """
        The grad group (index 1) nnear must equal the number of grad pairs.
        """
        n_pairs = 3
        gc1 = [3.0, 4.0, 5.0]
        gc2 = [3.5, 4.5, 5.5]
        gv  = [0.0, 0.0, 0.0]

        k = _ok1d(
            [0.0, 10.0], [0.0, 0.0], 7.0, _SPH,
            grad_c1=gc1, grad_c2=gc2, grad_val=gv,
            store_weight=True,
        )
        w = k.get_weights()
        nn_grad = int(w["nnear"][0, 1])
        assert nn_grad == n_pairs, (
            f"Expected nnear[block=0, grad-group] = {n_pairs}, got {nn_grad}"
        )

    def test_grad_group_inear_sequential(self):
        """
        Grad-group inear indices must be sequential 1-based integers
        [1, 2, ..., ngrad], because every grad pair participates in every block.
        """
        n_pairs = 2
        gc1 = [3.0, 4.0]
        gc2 = [3.5, 4.5]
        gv  = [1.0, 2.0]

        k = _ok1d(
            [0.0, 10.0], [0.0, 0.0], 5.0, _SPH,
            grad_c1=gc1, grad_c2=gc2, grad_val=gv,
            store_weight=True,
        )
        w = k.get_weights()
        nn_grad = int(w["nnear"][0, 1])
        inear_grad = w["inear"][0, 1, :nn_grad]   # 1-based

        expected = np.arange(1, n_pairs + 1, dtype=np.int32)
        np.testing.assert_array_equal(inear_grad, expected,
            err_msg="Grad-group inear must be sequential 1..ngrad")

    def test_obs_group_dimensions_unchanged_by_grad(self):
        """
        Adding grad pairs does not alter the obs-group (index 0) dimensions.
        nnear[block=0, group=0] equals the number of obs within the search
        neighbourhood — unaffected by the gradient constraint.
        """
        obs_x = [0.0, 2.0, 4.0]
        obs_z = [1.0, 2.0, 3.0]
        target = 1.0

        k_no_grad = _ok1d(obs_x, obs_z, target, _SPH, store_weight=True)
        k_with_grad = _ok1d(
            obs_x, obs_z, target, _SPH,
            grad_c1=[6.0], grad_c2=[7.0], grad_val=[0.0],
            store_weight=True,
        )

        w0 = k_no_grad.get_weights()
        w1 = k_with_grad.get_weights()

        assert w0["nnear"][0, 0] == w1["nnear"][0, 0], (
            "Obs-group nnear must not change when a grad pair is added"
        )


# ============================================================================
# 2. Null properties — pure-nugget model
# ============================================================================

class TestGradPureNugget:
    """
    Analytical derivation
    ---------------------
    Pure-nugget variogram: C(0) = C0, C(h) = 0 for h > 0.

    Kriging system with n obs and 1 grad pair:

        obs row i:   C0·w_i + 0·θ + μ = 0          (rhs = C(xi, x0) = 0, x0 ≠ xi)
        grad row:    0·w_i + 2C0·θ + 0 = 0          (rhs = C(xs1,x0) - C(xs2,x0) = 0)
        unbiasedness: Σw_i = 1

    Solution: θ = 0, w_i = 1/n, μ = -C0/n.
    Estimate: Z* = mean(z).
    Variance: σ² = C0 + C0/n = C0·(1 + 1/n).

    The grad pair is completely invisible: est and var match the no-grad case.
    This holds regardless of grad_value, because the grad weight θ = 0.
    """

    _OBS_X   = [0.0, 1.0, 2.0]          # n = 3
    _OBS_Z   = [1.0, 3.0, 5.0]          # mean = 3.0
    _TARGET  = 10.0                      # new location
    _GRAD_C1 = [5.0]
    _GRAD_C2 = [6.0]
    _C0      = 2.0
    _N       = 3
    _MEAN    = 3.0
    _VAR_EXPECTED = _C0 * (1.0 + 1.0 / _N)   # = 8/3

    def _solve(self, **gkw):
        return _scalar(_ok1d(
            self._OBS_X, self._OBS_Z, self._TARGET, _NUG, **gkw
        ))

    def test_no_grad_baseline_estimate(self):
        """Baseline: no grad → est = mean = 3.0."""
        est, _ = self._solve()
        np.testing.assert_allclose(est, self._MEAN, rtol=1e-6)

    def test_no_grad_baseline_variance(self):
        """Baseline: no grad → var = C0·(1+1/n) = 8/3."""
        _, var = self._solve()
        np.testing.assert_allclose(var, self._VAR_EXPECTED, rtol=1e-5)

    def test_zero_grad_val_estimate_unchanged(self):
        """grad_val = 0: est equals no-grad baseline (θ = 0 in nugget model)."""
        est, _ = self._solve(
            grad_c1=self._GRAD_C1, grad_c2=self._GRAD_C2, grad_val=[0.0]
        )
        np.testing.assert_allclose(est, self._MEAN, rtol=1e-6,
            err_msg="Pure-nugget: est must equal mean even with grad pair")

    def test_zero_grad_val_variance_unchanged(self):
        """grad_val = 0: var equals no-grad baseline (θ = 0 in nugget model)."""
        _, var = self._solve(
            grad_c1=self._GRAD_C1, grad_c2=self._GRAD_C2, grad_val=[0.0]
        )
        np.testing.assert_allclose(var, self._VAR_EXPECTED, rtol=1e-5,
            err_msg="Pure-nugget: var must equal C0(1+1/n) even with grad pair")

    def test_nonzero_grad_val_estimate_unchanged(self):
        """
        grad_val = 10 (large non-zero value): est still equals mean.
        The grad weight is zero in the nugget model, so grad_val has no effect.
        """
        est, _ = self._solve(
            grad_c1=self._GRAD_C1, grad_c2=self._GRAD_C2, grad_val=[10.0]
        )
        np.testing.assert_allclose(est, self._MEAN, rtol=1e-6,
            err_msg="Pure-nugget: est must be invariant to non-zero grad_val")

    def test_nonzero_grad_val_variance_unchanged(self):
        """
        grad_val = 10: var still equals C0·(1+1/n).
        The nugget model blocks all spatial cross-covariances; grad pair is inert.
        """
        _, var = self._solve(
            grad_c1=self._GRAD_C1, grad_c2=self._GRAD_C2, grad_val=[10.0]
        )
        np.testing.assert_allclose(var, self._VAR_EXPECTED, rtol=1e-5,
            err_msg="Pure-nugget: var must be invariant to non-zero grad_val")


# ============================================================================
# 3. Directional correctness
# ============================================================================

class TestGradDirectional:
    """
    With zero obs values (z = 0 everywhere), the kriging estimate at target x0
    reduces to:

        Z*(x0) = θ(x0) · grad_val

    where θ(x0) is the grad weight for target x0 (sign depends on geometry).

    For spherical variogram with no nugget and a grad pair xs1=3, xs2=7:
    - Target at xs1=3: rhs_grad = C(xs1,xs1) - C(xs2,xs1) = C(0) - C(4) > 0
      → θ(3) > 0  → est(3) > 0 when grad_val > 0.
    - Target at xs2=7: rhs_grad = C(xs1,xs2) - C(xs2,xs2) = C(4) - C(0) < 0
      → θ(7) < 0  → est(7) < 0 when grad_val > 0.
    - Target at midpoint x=5: rhs_grad = C(3,5) - C(7,5) = C(2) - C(2) = 0
      → θ(5) = 0  → est(5) = 0 regardless of grad_val.

    All three behaviours are tested below.
    """

    # Obs at x=0 and x=10 with z=0: the "background" is flat-zero.
    # Grad pair: xs1=3, xs2=7 (so grad constraint is between x=3 and x=7).
    _OBS_X = [0.0, 10.0]
    _OBS_Z = [0.0, 0.0]
    _GC1   = [3.0]
    _GC2   = [7.0]

    def _est(self, target, gval):
        """Solve and return (est, var) at `target` with grad_val = `gval`."""
        return _scalar(_ok1d(
            self._OBS_X, self._OBS_Z, target, _SPH,
            grad_c1=self._GC1, grad_c2=self._GC2, grad_val=[gval],
        ))

    # ------------------------------------------------------------------
    # Target at xs1 = 3 (near positive side)
    # ------------------------------------------------------------------

    def test_positive_grad_increases_estimate_at_xs1(self):
        """
        grad_val > 0 (Z(xs1) > Z(xs2)) → est at xs1 is positive.
        Without grad: est = 0 (z=0 everywhere).
        """
        est_no_grad, _ = _scalar(_ok1d(self._OBS_X, self._OBS_Z, 3.0, _SPH))
        est_with_grad, _ = self._est(3.0, gval=4.0)
        assert est_with_grad > est_no_grad, (
            f"est at xs1=3 with grad_val=4 ({est_with_grad:.4f}) should be > "
            f"no-grad est ({est_no_grad:.4f})"
        )

    def test_negative_grad_decreases_estimate_at_xs1(self):
        """
        grad_val < 0 (Z(xs1) < Z(xs2)) → est at xs1 is negative.
        """
        est_no_grad, _ = _scalar(_ok1d(self._OBS_X, self._OBS_Z, 3.0, _SPH))
        est_with_grad, _ = self._est(3.0, gval=-4.0)
        assert est_with_grad < est_no_grad, (
            f"est at xs1=3 with grad_val=-4 ({est_with_grad:.4f}) should be < "
            f"no-grad est ({est_no_grad:.4f})"
        )

    def test_antisymmetry_at_xs1(self):
        """
        est(xs1, +d) = -est(xs1, -d):
        sign of grad_val exactly reverses the estimate at xs1 (z=0 obs baseline).
        """
        est_pos, _ = self._est(3.0, gval= 4.0)
        est_neg, _ = self._est(3.0, gval=-4.0)
        np.testing.assert_allclose(est_pos, -est_neg, rtol=1e-5,
            err_msg="Antisymmetry: est(+d) must equal -est(-d) for zero-obs baseline")

    # ------------------------------------------------------------------
    # Target at xs2 = 7 (near negative side)
    # ------------------------------------------------------------------

    def test_positive_grad_decreases_estimate_at_xs2(self):
        """
        grad_val > 0 → Z(xs2) < Z(xs1), so est at xs2=7 is pulled negative.
        """
        est_no_grad, _ = _scalar(_ok1d(self._OBS_X, self._OBS_Z, 7.0, _SPH))
        est_with_grad, _ = self._est(7.0, gval=4.0)
        assert est_with_grad < est_no_grad, (
            f"est at xs2=7 with grad_val=4 ({est_with_grad:.4f}) should be < "
            f"no-grad est ({est_no_grad:.4f})"
        )

    # ------------------------------------------------------------------
    # Target at midpoint = 5 (symmetric → grad rhs = 0)
    # ------------------------------------------------------------------

    def test_midpoint_estimate_zero_regardless_of_grad_val(self):
        """
        The grad pair xs1=3, xs2=7 is symmetric about x=5.
        At x=5: rhs_grad = C(xs1,5) - C(xs2,5) = C(2) - C(2) = 0 → θ = 0.
        Estimate = 0 regardless of grad_val.
        """
        est_d0, _  = self._est(5.0, gval= 0.0)
        est_d4, _  = self._est(5.0, gval= 4.0)
        est_dm4, _ = self._est(5.0, gval=-4.0)
        np.testing.assert_allclose(est_d0,  0.0, atol=1e-5)
        np.testing.assert_allclose(est_d4,  0.0, atol=1e-5,
            err_msg="Midpoint est must be 0 when grad pair is symmetric about target")
        np.testing.assert_allclose(est_dm4, 0.0, atol=1e-5,
            err_msg="Midpoint est must be 0 when grad pair is symmetric about target")

    # ------------------------------------------------------------------
    # Linearity: est(d) proportional to d for zero-obs baseline
    # ------------------------------------------------------------------

    def test_estimate_linear_in_grad_val(self):
        """
        With z=0 obs everywhere: est(d) = θ·d → est(2d) = 2·est(d).
        Kriging is a linear estimator; the grad weight θ does not depend on d.
        """
        est1, _ = self._est(3.0, gval=2.0)
        est2, _ = self._est(3.0, gval=4.0)
        np.testing.assert_allclose(est2, 2.0 * est1, rtol=1e-5,
            err_msg="est(2·grad_val) must equal 2·est(grad_val) (linearity)")

    def test_estimate_linear_in_grad_val_negative(self):
        """
        Linearity holds for negative grad_val: est(-3d) = -3·est(d).
        """
        est_ref, _ = self._est(3.0, gval=1.0)
        est_neg, _ = self._est(3.0, gval=-3.0)
        np.testing.assert_allclose(est_neg, -3.0 * est_ref, rtol=1e-5,
            err_msg="est(-3·grad_val) must equal -3·est(grad_val) (linearity)")


# ============================================================================
# 4. Unbiasedness — constant-shift invariance
# ============================================================================

class TestGradUnbiasedness:
    """
    For ordinary kriging (weights sum to 1), adding a constant C to all obs
    values shifts the estimate by C.

    The gradient value is a *difference* Z(xs1) - Z(xs2), so shifting every
    Z by C leaves grad_val unchanged.  Therefore:

        est(z + C, grad_val) = est(z, grad_val) + C

    The variance is unaffected (it depends only on the kriging weights and
    variogram, not on the obs values).
    """

    _OBS_X  = [0.0, 1.5, 4.0]
    _OBS_Z  = [1.0, 3.0, 2.0]
    _GC1    = [2.5]
    _GC2    = [3.0]
    _GV     = [1.0]
    _TARGET = 2.0
    _SHIFT  = 100.0

    def _solve(self, obs_z):
        return _scalar(_ok1d(
            self._OBS_X, obs_z, self._TARGET, _SPH,
            grad_c1=self._GC1, grad_c2=self._GC2, grad_val=self._GV,
        ))

    def test_shift_changes_estimate_by_same_constant(self):
        """Shifting all obs by C shifts the estimate by C."""
        est_base, _  = self._solve(self._OBS_Z)
        est_shift, _ = self._solve(np.array(self._OBS_Z) + self._SHIFT)
        np.testing.assert_allclose(est_shift, est_base + self._SHIFT, rtol=1e-5,
            err_msg="Constant shift of obs must shift estimate by the same constant")

    def test_shift_does_not_change_variance(self):
        """Shifting all obs by C does not change the kriging variance."""
        _, var_base  = self._solve(self._OBS_Z)
        _, var_shift = self._solve(np.array(self._OBS_Z) + self._SHIFT)
        np.testing.assert_allclose(var_shift, var_base, rtol=1e-5,
            err_msg="Constant shift of obs must not change kriging variance")

    def test_negative_shift(self):
        """Shifting all obs by -C shifts the estimate by -C."""
        neg = -self._SHIFT
        est_base, _  = self._solve(self._OBS_Z)
        est_neg, _   = self._solve(np.array(self._OBS_Z) + neg)
        np.testing.assert_allclose(est_neg, est_base + neg, rtol=1e-5)


# ============================================================================
# 5. Variance properties
# ============================================================================

class TestGradVariance:
    """Physical sanity checks on the kriging variance with grad data."""

    _OBS_X  = [0.0, 3.0, 7.0]
    _OBS_Z  = [1.0, 2.0, 1.5]
    _GC1    = [4.0]
    _GC2    = [5.0]
    _GV     = [1.0]
    _TARGET = 2.5

    def test_variance_nonnegative_with_grad(self):
        """Kriging variance must be >= 0 when a grad pair is present."""
        _, var = _scalar(_ok1d(
            self._OBS_X, self._OBS_Z, self._TARGET, _SPH,
            grad_c1=self._GC1, grad_c2=self._GC2, grad_val=self._GV,
        ))
        assert var >= -1e-8, f"Kriging variance is negative with grad: var = {var:.2e}"

    def test_variance_bounded_by_sill_with_grad(self):
        """Kriging variance must not exceed the variogram sill (C(0) = 4)."""
        _, var = _scalar(_ok1d(
            self._OBS_X, self._OBS_Z, self._TARGET, _SPH,
            grad_c1=self._GC1, grad_c2=self._GC2, grad_val=self._GV,
        ))
        sill = _SPH["sill"]
        assert var <= sill * 1.01, (
            f"Variance {var:.4f} exceeds sill {sill} — kriging system may be unstable"
        )

    def test_soft_constraint_has_larger_variance_than_exact(self):
        """
        A large grad_variance (soft constraint) relaxes the constraint and
        therefore provides *less* conditioning than the exact (variance=0) case.
        Kriging variance with the soft constraint >= variance with exact constraint.

        A large variance effectively removes the grad pair from the system;
        the variance approaches that of the no-grad case.
        """
        _, var_exact = _scalar(_ok1d(
            self._OBS_X, self._OBS_Z, self._TARGET, _SPH,
            grad_c1=self._GC1, grad_c2=self._GC2, grad_val=self._GV,
            grad_var=[0.0],   # exact
        ))
        _, var_soft = _scalar(_ok1d(
            self._OBS_X, self._OBS_Z, self._TARGET, _SPH,
            grad_c1=self._GC1, grad_c2=self._GC2, grad_val=self._GV,
            grad_var=[1e6],   # very soft: nearly unconstrained
        ))
        _, var_no_grad = _scalar(_ok1d(
            self._OBS_X, self._OBS_Z, self._TARGET, _SPH,
        ))
        # Soft grad var >= exact grad var
        assert var_soft >= var_exact - 1e-6, (
            f"Soft constraint variance ({var_soft:.4f}) should be >= "
            f"exact constraint variance ({var_exact:.4f})"
        )
        # Soft grad var approaches no-grad var as grad_variance → ∞
        np.testing.assert_allclose(var_soft, var_no_grad, rtol=1e-3,
            err_msg="Very large grad_variance should approach the no-grad variance")

    def test_multiple_grad_pairs_variance_nonneg(self):
        """Variance is non-negative with multiple gradient pairs."""
        gc1 = [2.0, 4.0, 6.0]
        gc2 = [2.5, 4.5, 6.5]
        gv  = [1.0, 0.0, -1.0]
        for tgt in [1.0, 3.0, 5.0, 8.0]:
            _, var = _scalar(_ok1d(
                [0.0, 10.0], [0.0, 0.0], tgt, _SPH,
                grad_c1=gc1, grad_c2=gc2, grad_val=gv,
            ))
            assert var >= -1e-8, (
                f"Negative variance at target={tgt} with multiple grad pairs: {var:.2e}"
            )


# ============================================================================
# 6. Weight-store with gradient data
# ============================================================================

class TestGradWeightStore:
    """
    Verify the weight store contents when a gradient pair is present.

    Layout for nvar=1, nsim=0, 1 grad pair:
        weight shape: (nblock, ngroups=2, nmax)
        group 0: obs kriging weights  (applied to obs values)
        group 1: grad kriging weights (applied to grad_val differences)

    Estimate reconstruction:
        est[b] = sum_i  weight[b,0,i] * obs_z[inear[b,0,i]-1]
               + sum_j  weight[b,1,j] * grad_val[inear[b,1,j]-1]

    Note: inear[b,1,j] for the grad group is always 1-based sequential
    (1, 2, ..., ngrad), so inear[b,1,j]-1 gives 0-based grad_val indices.
    """

    _OBS_X  = [0.0, 2.0, 5.0]
    _OBS_Z  = [1.0, 3.0, 2.0]
    _GC1    = [3.5]
    _GC2    = [4.5]
    _GV     = [2.0]
    _TARGET = 1.5

    def _build(self, **kw):
        return _ok1d(
            self._OBS_X, self._OBS_Z, self._TARGET, _SPH,
            grad_c1=self._GC1, grad_c2=self._GC2, grad_val=self._GV,
            store_weight=True, **kw,
        )

    def test_weight_store_shape_expanded(self):
        """Weight store ngroups = 2 (ngroups_base=1 + nvar=1) with grad."""
        k = self._build()
        w = k.get_weights()
        assert w["weight"].shape[1] == 2, (
            f"Expected ngroups=2, got {w['weight'].shape[1]}"
        )

    def test_obs_weights_sum_to_one(self):
        """
        Ordinary kriging: obs-group weights (group 0) must sum to 1.
        The grad weights (group 1) have no unbiasedness constraint.
        """
        k = self._build()
        w = k.get_weights()
        nn_obs  = int(w["nnear"][0, 0])
        wsum    = float(w["weight"][0, 0, :nn_obs].sum())
        assert wsum == pytest.approx(1.0, abs=1e-5), (
            f"Obs weights sum to {wsum:.8f}, expected 1.0"
        )

    def test_estimate_reconstruction_obs_plus_grad(self):
        """
        Manually reconstruct the estimate from stored weights.

        est = (obs weights · obs values) + (grad weights · grad values)

        This verifies that the weight store captures both contributions
        and that inear for the grad group is indeed sequential.
        """
        k   = self._build()
        w   = k.get_weights()
        obs_z  = np.asarray(self._OBS_Z, dtype=float)
        grad_v = np.asarray(self._GV, dtype=float)

        ib = 0
        # Obs contribution
        nn_obs   = int(w["nnear"][ib, 0])
        obs_idx  = w["inear"][ib, 0, :nn_obs] - 1   # 0-based
        wt_obs   = w["weight"][ib, 0, :nn_obs]
        est_obs  = float(np.dot(wt_obs, obs_z[obs_idx]))

        # Grad contribution
        nn_grad  = int(w["nnear"][ib, 1])
        grad_idx = w["inear"][ib, 1, :nn_grad] - 1  # sequential 0,1,...,ngrad-1
        wt_grad  = w["weight"][ib, 1, :nn_grad]
        est_grad = float(np.dot(wt_grad, grad_v[grad_idx]))

        est_manual = est_obs + est_grad

        # Reference from solve()
        est_solve, _ = k.get_results()
        est_ref = float(est_solve[0])

        np.testing.assert_allclose(
            est_manual, est_ref, rtol=1e-5, atol=1e-7,
            err_msg=(
                f"Manual reconstruction (obs={est_obs:.6f} + grad={est_grad:.6f} "
                f"= {est_manual:.6f}) differs from solve() ({est_ref:.6f})"
            ),
        )

    def test_unused_weight_slots_are_zero(self):
        """Slots beyond nnear are zero-padded in weight and inear arrays."""
        k = self._build()
        w = k.get_weights()
        nb, ng, nm = w["weight"].shape
        for ib in range(nb):
            for ig in range(ng):
                nn = w["nnear"][ib, ig]
                assert np.all(w["weight"][ib, ig, nn:] == 0.0), (
                    f"Non-zero weight in padding: block={ib} group={ig}"
                )
                assert np.all(w["inear"][ib, ig, nn:] == 0), (
                    f"Non-zero inear in padding: block={ib} group={ig}"
                )

    def test_multiple_grid_points_reconstruction(self):
        """
        Reconstruction accuracy holds across a multi-block grid (3 targets).
        Tests that the weight store saves correctly for all blocks, not just ib=0.
        """
        obs_x = [0.0, 3.0, 7.0]
        obs_z = [1.0, 2.0, 3.0]
        gc1   = [1.5]
        gc2   = [2.0]
        gv    = [1.5]
        targets = [0.5, 2.5, 5.0]

        coords  = np.array(obs_x, dtype=float).reshape(-1, 1)
        tgt_arr = np.array(targets, dtype=float).reshape(-1, 1)
        grad_v  = np.asarray(gv, dtype=float)

        k = Kriging(ndim=1, nvar=1, store_weight=True)
        k.set_obs(1, coord=coords, value=np.array(obs_z, dtype=float), nmax=_NMAX)
        k.set_vgm(1, 1, **_SPH)
        k.set_grid(tgt_arr)
        k.set_search(1)
        k.set_grad(
            coord1=np.array(gc1, dtype=float).reshape(-1, 1),
            coord2=np.array(gc2, dtype=float).reshape(-1, 1),
            grad_value=grad_v, ivar=1,
        )
        k.solve()

        est_ref, _ = k.get_results()
        w = k.get_weights()
        obs_z_arr = np.array(obs_z, dtype=float)

        for ib in range(len(targets)):
            nn_obs   = int(w["nnear"][ib, 0])
            obs_idx  = w["inear"][ib, 0, :nn_obs] - 1
            est_obs  = float(np.dot(w["weight"][ib, 0, :nn_obs], obs_z_arr[obs_idx]))

            nn_grad  = int(w["nnear"][ib, 1])
            grad_idx = w["inear"][ib, 1, :nn_grad] - 1
            est_grd  = float(np.dot(w["weight"][ib, 1, :nn_grad], grad_v[grad_idx]))

            est_manual = est_obs + est_grd
            np.testing.assert_allclose(
                est_manual, float(est_ref[ib]), rtol=1e-5, atol=1e-7,
                err_msg=f"Weight reconstruction mismatch at target index {ib}",
            )


# ============================================================================
# 7. use_old_weight round-trip with gradient data
# ============================================================================

class TestGradUseOldWeight:
    """
    Factor-file round-trip when gradient pairs are present.

    When use_old_weight=True, estimate_block uses the stored weights but
    still needs the grad values (stored in self%grad) to compute the
    estimate contribution.  Therefore set_grad MUST be called before the
    second solve.

    The test verifies that reloading the weights reproduces the same
    estimates and variances as the original solve.
    """

    _OBS_X  = [0.0, 2.0, 5.0, 8.0]
    _OBS_Z  = [1.0, 4.0, 3.0, 2.0]
    _GC1    = [3.0, 6.0]
    _GC2    = [3.5, 6.5]
    _GV     = [1.5, -1.0]
    _TARGETS = [1.0, 4.0, 7.0]

    def _build_first(self, weight_file):
        """First run: solve + write weights to factor file."""
        coords  = np.array(self._OBS_X, dtype=float).reshape(-1, 1)
        tgt     = np.array(self._TARGETS, dtype=float).reshape(-1, 1)
        k = Kriging(ndim=1, nvar=1, store_weight=True, weight_file=weight_file)
        k.set_obs(1, coord=coords, value=np.array(self._OBS_Z, dtype=float),
                  nmax=_NMAX)
        k.set_vgm(1, 1, **_SPH)
        k.set_grid(tgt)
        k.set_search(1)
        k.set_grad(
            coord1=np.array(self._GC1, dtype=float).reshape(-1, 1),
            coord2=np.array(self._GC2, dtype=float).reshape(-1, 1),
            grad_value=np.array(self._GV, dtype=float),
            ivar=1,
        )
        k.solve()
        return k

    def _build_second(self, weight_file):
        """
        Second run: reload weights from factor file.
        set_grad is required to supply grad values to estimate_block.
        """
        coords  = np.array(self._OBS_X, dtype=float).reshape(-1, 1)
        tgt     = np.array(self._TARGETS, dtype=float).reshape(-1, 1)
        k = Kriging(ndim=1, nvar=1, use_old_weight=True, weight_file=weight_file)
        k.set_obs(1, coord=coords, value=np.array(self._OBS_Z, dtype=float),
                  nmax=_NMAX)
        k.set_vgm(1, 1, **_SPH)
        k.set_grid(tgt)
        k.set_search(1)
        # set_grad required: estimate_block uses grad%value from the Fortran struct
        k.set_grad(
            coord1=np.array(self._GC1, dtype=float).reshape(-1, 1),
            coord2=np.array(self._GC2, dtype=float).reshape(-1, 1),
            grad_value=np.array(self._GV, dtype=float),
            ivar=1,
        )
        k.solve()
        return k

    def test_use_old_weight_reproduces_estimates(self, tmp_path):
        """use_old_weight + grad reproduces the same estimates."""
        wf = str(tmp_path / "grad_weights.fac")
        k1 = self._build_first(wf)
        k2 = self._build_second(wf)

        est1, _ = k1.get_results()
        est2, _ = k2.get_results()
        np.testing.assert_allclose(est2, est1, rtol=1e-5, atol=1e-8,
            err_msg="use_old_weight estimates differ from original solve with grad")

    def test_use_old_weight_reproduces_variance(self, tmp_path):
        """use_old_weight + grad reproduces the same kriging variances."""
        wf = str(tmp_path / "grad_weights_var.fac")
        k1 = self._build_first(wf)
        k2 = self._build_second(wf)

        _, var1 = k1.get_results()
        _, var2 = k2.get_results()
        np.testing.assert_allclose(var2, var1, rtol=1e-5, atol=1e-8,
            err_msg="use_old_weight variances differ from original solve with grad")

    def test_changed_obs_values_with_old_weights(self, tmp_path):
        """
        Changing obs values while reusing weights gives a linearly-shifted estimate.

        For OK with stored weights and constant C added to all obs values,
        the estimate shifts by C (Σw = 1) and the grad contribution is unchanged
        (grad_val = difference of field values, unchanged by adding C everywhere).
        """
        wf = str(tmp_path / "grad_shift.fac")
        k1 = self._build_first(wf)
        est1, _ = k1.get_results()
        w1 = k1.get_weights()

        C = 5.0
        coords  = np.array(self._OBS_X, dtype=float).reshape(-1, 1)
        tgt     = np.array(self._TARGETS, dtype=float).reshape(-1, 1)
        k2 = Kriging(ndim=1, nvar=1, use_old_weight=True, weight_file=wf)
        k2.set_obs(1, coord=coords,
                   value=np.array(self._OBS_Z, dtype=float) + C,  # shifted
                   nmax=_NMAX)
        k2.set_vgm(1, 1, **_SPH)
        k2.set_grid(tgt)
        k2.set_search(1)
        k2.set_grad(
            coord1=np.array(self._GC1, dtype=float).reshape(-1, 1),
            coord2=np.array(self._GC2, dtype=float).reshape(-1, 1),
            grad_value=np.array(self._GV, dtype=float),  # grad_val unchanged
            ivar=1,
        )
        k2.solve()
        est2, _ = k2.get_results()

        np.testing.assert_allclose(est2, est1 + C, rtol=1e-5,
            err_msg="Shifting obs values by C must shift estimate by C with stored weights")
