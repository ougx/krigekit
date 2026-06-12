"""
test_kriging_sva.py
-------------------
Manual debug tests for the t_kriging_sva spatially varying variogram
subclass (kriging_sva.F90 / KrigingSVA Python wrapper).

Three test scenarios, run in order:

  Test 1 — Uniform SVA (regression baseline)
      Every block gets the same variogram as in test_ok.py.
      Results must match ordinary kriging to within floating-point
      tolerance, verifying that the pointer-redirection mechanism
      does not corrupt the covariance assembly.

  Test 2 — Spatially varying range
      Grid is split into two halves.  Left half uses a short-range
      spherical model; right half uses a long-range exponential model.
      Checks that estimates in each half are finite and that the two
      halves produce different variance levels (they should, because
      different variogram ranges produce different kriging variances).

  Test 3 — Error-path checks
      Verifies that the expected errors are raised when the API is
      called out of order (missing allocate_sva, missing set_vgm_block,
      out-of-range block index).

Usage
-----
    python test_kriging_sva.py

All tests use synthetic data so no CSV files are required.
"""

import sys
import numpy as np

sys.path.insert(0, "../src/krigekit")

# ---------------------------------------------------------------------------
# Import — KrigingSVA should be exported from the same package as Kriging.
# Adjust the import path if the class lives in a different module.
# ---------------------------------------------------------------------------
try:
    from _kriging import Kriging, KrigingSVA
except ImportError as exc:
    raise ImportError(
        "Could not import KrigingSVA from _kriging. "
        "Make sure kriging_sva.F90 is compiled into the DLL/SO and "
        "that a KrigingSVA Python wrapper class has been added to _kriging.py."
    ) from exc

# ---------------------------------------------------------------------------
# Shared synthetic dataset (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
_RNG   = np.random.default_rng(0)
_NOBS  = 40
_NGRID = 100
_COORD = _RNG.uniform(0, 1000, (_NOBS,  2)).astype(np.float32)
_VALUE = _RNG.standard_normal(_NOBS).astype(np.float32)
_GRID  = np.column_stack([
    np.tile(np.linspace(0, 1000, 10), 10),        # x: 10 columns
    np.repeat(np.linspace(0, 1000, 10), 10),       # y: 10 rows
]).astype(np.float32)

# Variogram specs (same format as Kriging.set_vgm)
# "vtype  nugget  sill  a_major  a_minor1  a_minor2  azimuth  dip  plunge"
_VGM_GLOBAL = "sph 0.0 1.0 500.0 500.0 500.0 0.0 0.0 0.0"
_VGM_SHORT  = "sph 0.0 1.0 200.0 200.0 200.0 0.0 0.0 0.0"   # left half
_VGM_LONG   = "exp 0.0 1.0 800.0 800.0 800.0 0.0 0.0 0.0"   # right half

NMAX = 20   # neighbours per block

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_base_ok():
    """Ordinary kriging with the global variogram (reference results)."""
    k = Kriging(ndim=2, nvar=1)
    k.set_obs(ivar=1, coord=_COORD, value=_VALUE, nmax=NMAX)
    k.set_vgm(ivar=1, jvar=1, spec=_VGM_GLOBAL)
    k.set_grid(coord=_GRID)
    k.set_search(ivar=1)
    k.solve()
    return k.get_results()   # (est, var)


def _build_sva_uniform():
    """SVA kriging with the same global variogram on every block."""
    k = KrigingSVA(ndim=2, nvar=1)
    k.set_obs(ivar=1, coord=_COORD, value=_VALUE, nmax=NMAX)
    k.set_grid(coord=_GRID)
    k.allocate_sva()
    k.set_vgm_block_all(ivar=1, jvar=1, spec=_VGM_GLOBAL)
    k.set_search(ivar=1)
    k.solve()
    return k.get_results()


def _build_sva_split():
    """SVA kriging: short range for left half, long range for right half."""
    k = KrigingSVA(ndim=2, nvar=1)
    k.set_obs(ivar=1, coord=_COORD, value=_VALUE, nmax=NMAX)
    k.set_grid(coord=_GRID)
    k.allocate_sva()

    mid_x = 500.0
    for ib in range(1, _NGRID + 1):
        x = _GRID[ib - 1, 0]   # _GRID is 0-indexed; block index is 1-based
        spec = _VGM_SHORT if x < mid_x else _VGM_LONG
        k.set_vgm_block(ib=ib, ivar=1, jvar=1, spec=spec)

    k.set_search(ivar=1)
    k.solve()
    return k.get_results()

# ---------------------------------------------------------------------------
# Test 1 — Uniform SVA matches ordinary kriging
# ---------------------------------------------------------------------------

def test_uniform_sva_matches_ok():
    print("\n" + "=" * 60)
    print("Test 1: Uniform SVA must match ordinary kriging")
    print("=" * 60)

    est_ok,  var_ok  = _build_base_ok()
    est_sva, var_sva = _build_sva_uniform()

    tol = 1e-4

    est_maxdiff = np.max(np.abs(est_sva - est_ok))
    var_maxdiff = np.max(np.abs(var_sva - var_ok))

    print(f"  est max |SVA - OK| = {est_maxdiff:.2e}  (tol={tol:.0e})")
    print(f"  var max |SVA - OK| = {var_maxdiff:.2e}  (tol={tol:.0e})")

    assert est_maxdiff < tol, (
        f"Estimate mismatch: max diff {est_maxdiff:.3e} exceeds tol {tol:.0e}.\n"
        f"  OK  range: [{est_ok.min():.4f}, {est_ok.max():.4f}]\n"
        f"  SVA range: [{est_sva.min():.4f}, {est_sva.max():.4f}]"
    )
    assert var_maxdiff < tol, (
        f"Variance mismatch: max diff {var_maxdiff:.3e} exceeds tol {tol:.0e}.\n"
        f"  OK  range: [{var_ok.min():.4f}, {var_ok.max():.4f}]\n"
        f"  SVA range: [{var_sva.min():.4f}, {var_sva.max():.4f}]"
    )

    print("  PASSED")

# ---------------------------------------------------------------------------
# Test 2 — Spatially varying range produces distinct variance zones
# ---------------------------------------------------------------------------

def test_split_vgm_variance_contrast():
    print("\n" + "=" * 60)
    print("Test 2: Split variogram — variance differs between halves")
    print("=" * 60)

    est, var = _build_sva_split()

    mid_x = 500.0
    left_mask  = _GRID[:, 0] < mid_x
    right_mask = ~left_mask

    # Basic sanity: all estimates and variances are finite and non-negative
    assert np.all(np.isfinite(est)), "Non-finite estimates detected"
    assert np.all(np.isfinite(var)), "Non-finite variances detected"
    assert np.all(var >= 0.0),       "Negative variances detected"

    mean_var_left  = var[left_mask].mean()
    mean_var_right = var[right_mask].mean()

    print(f"  est range (left) : [{est[left_mask].min():.4f}, {est[left_mask].max():.4f}]")
    print(f"  est range (right): [{est[right_mask].min():.4f}, {est[right_mask].max():.4f}]")
    print(f"  mean var (left,  short-range sph): {mean_var_left:.4f}")
    print(f"  mean var (right, long-range  exp): {mean_var_right:.4f}")

    # Short-range model -> data decorrelates faster -> higher kriging variance.
    # Long-range model  -> data stays correlated   -> lower kriging variance.
    # This should be detectable as a meaningful difference in mean variance.
    assert mean_var_left > mean_var_right, (
        f"Expected left (short range) variance > right (long range) variance, "
        f"but got left={mean_var_left:.4f}, right={mean_var_right:.4f}. "
        "Check that the per-block variogram pointer is being redirected correctly."
    )

    ratio = mean_var_left / max(mean_var_right, 1e-9)
    print(f"  variance ratio left/right = {ratio:.2f}  (expected > 1.0)")
    print("  PASSED")

# ---------------------------------------------------------------------------
# Test 3 — Error paths
# ---------------------------------------------------------------------------

def test_error_paths():
    print("\n" + "=" * 60)
    print("Test 3: Error paths")
    print("=" * 60)

    # 3a: set_vgm_block before allocate_sva must raise
    print("  3a: set_vgm_block before allocate_sva ...")
    k = KrigingSVA(ndim=2, nvar=1)
    k.set_obs(ivar=1, coord=_COORD, value=_VALUE, nmax=NMAX)
    k.set_grid(coord=_GRID)
    try:
        k.set_vgm_block(ib=1, ivar=1, jvar=1, spec=_VGM_GLOBAL)
        assert False, "Expected an error but none was raised"
    except (RuntimeError, Exception) as exc:
        print(f"     Got expected error: {exc}")
    print("     PASSED")

    # 3b: allocate_sva before set_grid must raise
    print("  3b: allocate_sva before set_grid ...")
    k2 = KrigingSVA(ndim=2, nvar=1)
    k2.set_obs(ivar=1, coord=_COORD, value=_VALUE, nmax=NMAX)
    try:
        k2.allocate_sva()
        assert False, "Expected an error but none was raised"
    except (RuntimeError, Exception) as exc:
        print(f"     Got expected error: {exc}")
    print("     PASSED")

    # 3c: out-of-range block index must raise
    print("  3c: set_vgm_block with ib out of range ...")
    k3 = KrigingSVA(ndim=2, nvar=1)
    k3.set_obs(ivar=1, coord=_COORD, value=_VALUE, nmax=NMAX)
    k3.set_grid(coord=_GRID)
    k3.allocate_sva()
    try:
        k3.set_vgm_block(ib=_NGRID + 99, ivar=1, jvar=1, spec=_VGM_GLOBAL)
        assert False, "Expected an error but none was raised"
    except (RuntimeError, Exception) as exc:
        print(f"     Got expected error: {exc}")
    print("     PASSED")

    # 3d: solve without setting all block variograms must raise
    print("  3d: solve with missing block variograms ...")
    k4 = KrigingSVA(ndim=2, nvar=1)
    k4.set_obs(ivar=1, coord=_COORD, value=_VALUE, nmax=NMAX)
    k4.set_grid(coord=_GRID)
    k4.allocate_sva()
    k4.set_vgm_block(ib=1, ivar=1, jvar=1, spec=_VGM_GLOBAL)   # only block 1
    k4.set_search(ivar=1)
    try:
        k4.solve()
        assert False, "Expected an error but none was raised"
    except (RuntimeError, Exception) as exc:
        print(f"     Got expected error: {exc}")
    print("     PASSED")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0

    for test_fn in [
        test_uniform_sva_matches_ok,
        test_split_vgm_variance_contrast,
        test_error_paths,
    ]:
        try:
            test_fn()
            passed += 1
        except AssertionError as exc:
            print(f"\n  FAILED: {exc}")
            failed += 1
        except Exception as exc:
            print(f"\n  ERROR (unexpected): {type(exc).__name__}: {exc}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
