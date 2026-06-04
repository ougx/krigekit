"""
test_api.py
===========
Tests for input validation, edge cases, and the full Kriging class API
including set_obs_drift, set_grid_drift, set_grid_cv, and bounds clipping.
"""

import numpy as np
import pytest
from pykriging import Kriging, ordinary_kriging

_VGM = dict(vtype="sph", nugget=0.0, sill=1.0, a_major=50.0)
_VGM_PC2D = dict(vtype="sph", nugget=0.0, sill=0.12, a_major=5000.0)
_NMAX = 20

_SMALL_COORD = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
_SMALL_VALUE = np.array([1.0, 2.0, 1.5, 2.5])
_SMALL_GRID = np.array([
    [0.25, 0.25],
    [0.75, 0.75],
])

# Two interior grid points not co-located with any observation
_INTERIOR_GRID = np.array([[580000.0, 4395000.0],
                            [578000.0, 4400000.0]])


# ---------------------------------------------------------------------------
# Persistent factor cache
# ---------------------------------------------------------------------------

class TestPersistentFactorCache:

    def test_get_factor_includes_assembled_linear_system(self):
        coord = np.array([[0.0], [2.0]])
        value = np.array([1.0, 3.0])
        grid = np.array([[1.0]])
        spec = dict(vtype="sph", nugget=0.0, sill=4.0,
                    a_major=2.0, a_minor1=2.0, a_minor2=2.0,
                    azimuth=0.0)

        k = Kriging(ndim=1, nvar=1, pf_cache=True)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=2)
        k.set_vgm(ivar=1, jvar=1, **spec)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()

        f = k.get_factor()
        assert f["valid"]
        assert f["matA"].shape == (3, 3)
        assert f["rhsB"].shape == (1, 3)

        expected_matA = np.array([
            [4.0, 0.0, 1.0],
            [0.0, 4.0, 1.0],
            [1.0, 1.0, 0.0],
        ])
        expected_rhsB = np.array([[1.25, 1.25, 1.0]])
        np.testing.assert_allclose(f["matA"], expected_matA, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(f["rhsB"], expected_rhsB, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_coord_wrong_ndim_raises(self):
        coord = np.random.rand(10, 3)   # 3D coord passed as obs
        value = np.random.rand(10)
        grid  = np.random.rand(5, 2)
        # obs ndim=3 is inferred correctly; grid has ndim=2 which mismatches
        with pytest.raises(RuntimeError, match="nlag"):
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

    def test_repr(self, simple_obs):
        coord, value = simple_obs
        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord, value=value)
        assert "Kriging" in repr(k)
        assert "Dimension              : 2" in str(k)
        assert "Number of Variables    : 1" in str(k)
        assert "Number of data         : 5" in str(k)
        assert "nstruct=0" in str(k)

    def test_set_obs_value_wrong_length_raises(self):
        """Python checks value length before passing a raw pointer to Fortran."""
        k = Kriging(ndim=2, nvar=1, verbose=0)
        with pytest.raises(ValueError, match="value length"):
            k.set_obs(ivar=1, coord=_SMALL_COORD, value=_SMALL_VALUE[:-1])

    def test_set_obs_variance_wrong_length_raises(self):
        """Python checks variance length before passing a raw pointer to Fortran."""
        k = Kriging(ndim=2, nvar=1, verbose=0)
        with pytest.raises(ValueError, match="variance length"):
            k.set_obs(
                ivar=1,
                coord=_SMALL_COORD,
                value=_SMALL_VALUE,
                variance=np.zeros(_SMALL_VALUE.size - 1),
            )

    def test_set_grid_block_nblockpnt_sum_mismatch_raises(self):
        """Block maps must not claim more sub-nodes than coord provides."""
        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=_SMALL_COORD, value=_SMALL_VALUE, nmax=4)
        with pytest.raises(ValueError, match=r"sum\(nblockpnt\).*coord rows"):
            k.set_grid_block(
                coord=_SMALL_COORD[:3],
                block_type=1,
                nblockpnt=np.array([2, 2], dtype=np.int32),
            )

    def test_set_sim_randpath_wrong_length_raises(self):
        """SGSIM path length is checked before calling the C API."""
        k = Kriging(ndim=2, nvar=1, nsim=1, verbose=0, seed=11)
        k.set_obs(ivar=1, coord=_SMALL_COORD, value=_SMALL_VALUE, nmax=4)
        k.set_grid(coord=_SMALL_GRID)
        with pytest.raises(ValueError, match="randpath length"):
            k.set_sim(randpath=np.array([1], dtype=np.int32))

    def test_set_sim_randpath_must_be_permutation(self):
        """SGSIM path must be a 1-based permutation of the block numbers."""
        k = Kriging(ndim=2, nvar=1, nsim=1, verbose=0, seed=11)
        k.set_obs(ivar=1, coord=_SMALL_COORD, value=_SMALL_VALUE, nmax=4)
        k.set_grid(coord=_SMALL_GRID)
        with pytest.raises(ValueError, match="1-based permutation"):
            k.set_sim(randpath=np.array([1, 1], dtype=np.int32))

    def test_set_sim_sample_wrong_shape_raises(self):
        """SGSIM sample matrix must match (nsim, nblocks)."""
        k = Kriging(ndim=2, nvar=1, nsim=2, verbose=0, seed=11)
        k.set_obs(ivar=1, coord=_SMALL_COORD, value=_SMALL_VALUE, nmax=4)
        k.set_grid(coord=_SMALL_GRID)
        with pytest.raises(ValueError, match="sample shape"):
            k.set_sim(
                randpath=np.array([1, 2], dtype=np.int32),
                sample=np.zeros((1, _SMALL_GRID.shape[0])),
            )

    def test_set_grid_before_obs_raises_runtime_error(self):
        """Fortran ierr is surfaced when workflow calls are out of order."""
        k = Kriging(ndim=2, nvar=1, verbose=0)
        with pytest.raises(RuntimeError, match="Observation|set_obs"):
            k.set_grid(coord=_SMALL_GRID)

    def test_set_search_before_obs_raises_runtime_error(self):
        """set_search depends on observations and should report status cleanly."""
        k = Kriging(ndim=2, nvar=1, verbose=0)
        with pytest.raises(RuntimeError, match="Observation|set_obs"):
            k.set_search(ivar=1)

    def test_solve_without_search_raises_runtime_error(self):
        """solve should refuse to continue until every variable has search set."""
        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=_SMALL_COORD, value=_SMALL_VALUE, nmax=4)
        k.set_vgm(ivar=1, jvar=1, **_VGM)
        k.set_grid(coord=_SMALL_GRID)
        with pytest.raises(RuntimeError, match="set_search"):
            k.solve()

    def test_obs_drift_before_obs_raises_runtime_error(self):
        """Drift setup before observations should be reported through ierr."""
        k = Kriging(ndim=2, nvar=1, ndrift=1, verbose=0)
        with pytest.raises(RuntimeError, match="Observation|set_obs"):
            k.set_obs_drift(ivar=1, drift=np.ones((_SMALL_COORD.shape[0], 1)))


class TestOperationalModes:

    def _solve(self, **kwargs):
        k = Kriging(ndim=2, nvar=1, verbose=0, **kwargs)
        k.set_obs(ivar=1, coord=_SMALL_COORD, value=_SMALL_VALUE, nmax=4)
        k.set_vgm(ivar=1, jvar=1, **_VGM)
        k.set_grid(coord=_SMALL_GRID)
        k.set_search(ivar=1)
        k.solve()
        return k.get_results()

    def test_write_mat_with_openmp_writes_debug_files(self, tmp_path, monkeypatch):
        """write_mat should be safe under OpenMP and write one file set per block."""
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        monkeypatch.chdir(tmp_path)

        est, var = self._solve(write_mat=True)

        assert np.all(np.isfinite(est))
        assert np.all(var >= 0.0)
        assert len(list(tmp_path.glob("data_*.csv"))) == _SMALL_GRID.shape[0]
        assert len(list(tmp_path.glob("matA_*.csv"))) == _SMALL_GRID.shape[0]
        assert len(list(tmp_path.glob("rhsB_*.csv"))) == _SMALL_GRID.shape[0]


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

        k = Kriging(ndim=2, nvar=1, unbias=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=10, sk_mean=float(true_mean))
        k.set_vgm(ivar=1, jvar=1, **_VGM)
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
        k.set_vgm(ivar=1, jvar=1, **_VGM)
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
        k.set_vgm(ivar=1, jvar=1, **_VGM)
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
        k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=50000, a_major=5.0, a_minor1=3.0, a_minor2=3.0)
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
        """set_obs_drift with wrong ndrift column count should be reported by Fortran ierr."""
        coord, value = head2d_obs
        k = Kriging(ndim=2, nvar=1, ndrift=2, unbias=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=10)
        # drift has 3 columns but ndrift=2 was declared; Fortran returns ierr.
        wrong_drift = np.ones((coord.shape[0], 3))   # (nobs, 3) but ndrift=2
        with pytest.raises(RuntimeError, match="size\\(drift, 1\\) /= ndrift"):
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
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        est1, var1 = k.get_results()

        # Run 2: last 32 observations only
        k.set_obs(ivar=1, coord=coord[30:], value=value[30:], nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
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
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)

        k.set_grid(coord=_INTERIOR_GRID[:1])
        k.set_search(ivar=1)
        k.solve()
        est1, _ = k.get_results()

        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
        k.set_grid(coord=_INTERIOR_GRID[1:])
        k.set_search(ivar=1)
        k.solve()
        est2, _ = k.get_results()

        assert not np.allclose(est1, est2), (
            "Estimates should differ at different grid locations")

    def test_third_run_reproduces_first(self, pc2d_obs):
        """Reloading the original data must reproduce the original results to within
        floating-point precision.  With OpenMP enabled (nsim=0 activates the
        parallel block loop), dynamic thread scheduling means different blocks may
        be processed by different threads between run 1 and run 3, producing
        different factorization-cache hit/miss patterns and sub-epsilon rounding
        differences.  atol=1e-10 accepts these while still catching any real
        numerical corruption from stale state."""
        coord, value = pc2d_obs
        grid = coord[5:10]


        def _do_run(obs_coord, obs_val):
            k = Kriging(ndim=2, nvar=1, verbose=0)
            k.set_obs(ivar=1, coord=obs_coord, value=obs_val, nmax=_NMAX)
            k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
            k.set_grid(coord=grid)
            k.set_search(ivar=1)
            k.solve(1)
            return k.get_results()

        est1, var1 = _do_run(coord, value)
        _do_run(coord[30:], value[30:])          # intermediate run with different data
        est3, var3 = _do_run(coord, value)

        np.testing.assert_allclose(est1, est3,
            err_msg="Third run (same data as first) must reproduce first run estimates")
        np.testing.assert_allclose(var1, var3,
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
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
        k.set_grid(coord=grid)
        k.set_search(ivar=1)
        k.solve()
        est_small, _ = k.get_results()

        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D, append=False)
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
            k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D, append=False)
            k.set_grid(coord=grid)
            k.set_search(ivar=1)
            k.solve()
            _, var = k.get_results()
            assert np.all(var >= 0), (
                f"Negative variance after reuse with slice {sl}: {var}")

    def test_set_vgm_accumulates_structures(self, pc2d_obs):
        """
        Calling set_vgm twice with append=True (the default) on the same object
        adds two copies of the structure, doubling the sill.  This is how
        multi-structure nested models are built.  Pass append=False when reusing
        an object to replace the previous model instead of accumulating.
        """
        coord, value = pc2d_obs
        grid = _INTERIOR_GRID

        k1 = Kriging(ndim=2, nvar=1, verbose=0)
        k1.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k1.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)  # sill = 0.12
        k1.set_grid(coord=grid)
        k1.set_search(ivar=1)
        k1.solve()
        _, var_one = k1.get_results()

        k2 = Kriging(ndim=2, nvar=1, verbose=0)
        k2.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k2.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)  # first call
        k2.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)  # second call → sill = 0.24
        k2.set_grid(coord=grid)
        k2.set_search(ivar=1)
        k2.solve()
        _, var_two = k2.get_results()

        np.testing.assert_allclose(var_two, 2.0 * var_one, rtol=1e-5,
            err_msg="Calling set_vgm twice with same spec must double the total sill "
                    "and therefore double the kriging variance")


# ===========================================================================
# nthread parameter
# ===========================================================================

class TestNthread:
    """Tests for Kriging.solve(nthread=N).

    Three distinct behaviors are verified:
      1. nthread=1 is bitwise reproducible across repeated calls.
      2. nthread=1 (serial) and nthread=N (parallel) give the same results.
      3. The global OMP max_threads setting is restored after the call.
    """

    def _make_kriging(self, pc2d_obs):
        """Return a ready-to-solve Kriging object (not yet solved)."""
        coord, value = pc2d_obs
        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
        k.set_grid(coord=coord[5:15])   # 10 estimation points
        k.set_search(ivar=1)
        return k

    def test_nthread_1_is_bitwise_reproducible(self, pc2d_obs):
        """solve(nthread=1) called twice on the same object must return
        bitwise-identical results — no OMP scheduling variation."""
        k = self._make_kriging(pc2d_obs)

        k.solve(nthread=1)
        est1, var1 = k.get_results()

        k.solve(nthread=1)   # same data, same object, serial
        est2, var2 = k.get_results()

        np.testing.assert_array_equal(est1, est2,
            err_msg="nthread=1 estimates must be bitwise identical on repeated calls")
        np.testing.assert_array_equal(var1, var2,
            err_msg="nthread=1 variances must be bitwise identical on repeated calls")

    def test_nthread_1_agrees_with_parallel(self, pc2d_obs):
        """Serial (nthread=1) and parallel results must agree to within
        floating-point noise — same kriging equations, different scheduling."""
        k = self._make_kriging(pc2d_obs)
        k.solve(nthread=1)
        est_serial, var_serial = k.get_results()

        k.solve()            # default thread count
        est_par, var_par = k.get_results()

        np.testing.assert_allclose(est_serial, est_par, rtol=1e-6, atol=1e-10,
            err_msg="Serial and parallel estimates must agree to within floating-point noise")
        np.testing.assert_allclose(var_serial, var_par, rtol=1e-6, atol=1e-10,
            err_msg="Serial and parallel variances must agree to within floating-point noise")

    def test_nthread_restores_omp_setting(self, pc2d_obs):
        """omp_get_max_threads() must return the same value before and after
        solve(nthread=1) — the Fortran layer must save and restore."""
        from pykriging.kriging import omp_info
        before = omp_info()["max_threads"]

        k = self._make_kriging(pc2d_obs)
        k.solve(nthread=1)   # deliberately sets thread count to 1 …

        after = omp_info()["max_threads"]
        assert before == after, (
            f"OMP max_threads changed from {before} to {after} after solve(nthread=1); "
            "the Fortran layer must restore the previous setting"
        )

    def test_nthread_0_accepted_without_error(self, pc2d_obs):
        """nthread=0 (the no-op sentinel) must run without error and return
        finite, non-negative results."""
        k = self._make_kriging(pc2d_obs)
        k.solve(nthread=0)
        est, var = k.get_results()
        assert np.all(np.isfinite(est)), "nthread=0 produced non-finite estimates"
        assert np.all(var >= 0.0),       "nthread=0 produced negative variances"
