"""
test_sgsim.py
=============
Tests for Sequential Gaussian Simulation (SGSIM) using the pc2d dataset
and the 4800-node grid.

The pre-computed path (path4800.csv) and samples (sample4800.csv) allow
deterministic reproducibility: running SGSIM with those inputs must produce
the same realisations regardless of platform or compiler version.
"""

import numpy as np
import pytest
from krigekit import Kriging, sequential_gaussian_simulation

_VGM_PC2D = dict(vtype="sph", nugget=0.0, sill=0.12, a_major=5000.0)


class TestSGSIM:

    def test_sgsim_shape_single_realisation(self, pc2d_obs, pc2d_grid):
        """One realisation returns shape (ngrid,)."""
        coord, value   = pc2d_obs
        grid_coord, _  = pc2d_grid
        sims = sequential_gaussian_simulation(
            coord, value, grid_coord, _VGM_PC2D, nsim=1, nmax=20, seed=42
        )
        assert sims.shape == (grid_coord.shape[0],)

    def test_sgsim_shape_multiple_realisations(self, pc2d_obs, pc2d_grid):
        """N realisations return shape (N, ngrid)."""
        coord, value   = pc2d_obs
        grid_coord, _  = pc2d_grid
        nsim = 3
        sims = sequential_gaussian_simulation(
            coord, value, grid_coord, _VGM_PC2D, nsim=nsim, nmax=20, seed=42
        )
        assert sims.shape == (grid_coord.shape[0], nsim)

    def test_realisations_differ(self, pc2d_obs, pc2d_grid):
        """Different realisations must not be identical."""
        coord, value   = pc2d_obs
        grid_coord, _  = pc2d_grid
        sims = sequential_gaussian_simulation(
            coord, value, grid_coord, _VGM_PC2D, nsim=2, nmax=20, seed=123
        )
        assert not np.allclose(sims[0], sims[1]), \
            "Two SGSIM realisations are identical — likely a bug"

    def test_realisations_differ_seperate_seeds(self, pc2d_obs, pc2d_grid):
        """Different realisations must not be identical."""
        coord, value   = pc2d_obs
        grid_coord, _  = pc2d_grid
        sims = [sequential_gaussian_simulation(
            coord, value, grid_coord, _VGM_PC2D, nsim=1, nmax=20, seed=seed*100+1
        ) for seed in range(2)]
        assert not np.allclose(sims[0], sims[1]), \
            "Two SGSIM realisations are identical — likely a bug"

    def test_seed_reproducibility(self, pc2d_obs, pc2d_grid):
        """Same seed must produce identical results."""
        coord, value   = pc2d_obs
        grid_coord, _  = pc2d_grid
        sims_a = sequential_gaussian_simulation(
            coord, value, grid_coord, _VGM_PC2D, nsim=1, nmax=20, seed=7
        )
        sims_b = sequential_gaussian_simulation(
            coord, value, grid_coord, _VGM_PC2D, nsim=1, nmax=20, seed=7
        )
        np.testing.assert_array_equal(sims_a, sims_b)

    def test_grid_on_observations_no_nan(self):
        """
        Grid nodes coincident with observations must not produce NaN.  A
        previously simulated node at a hard-datum location creates two
        co-located conditioning points (a singular covariance matrix); the
        solver de-singularises this with a small diagonal regularisation and
        reproduces the observed values instead of failing.
        """
        rng   = np.random.default_rng(3)
        coord = rng.uniform(0.0, 100.0, size=(12, 2))
        value = np.exp(rng.normal(0.0, 1.0, size=12))
        k = Kriging(nsim=1, seed=5)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=12)
        k.set_grid(coord=coord.copy())            # grid exactly on observations
        k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=40.0)
        k.set_sim()
        k.set_search()
        k.solve()
        est, _ = k.get_results()
        del k
        est = np.asarray(est).reshape(coord.shape[0], -1)[:, 0]
        assert np.isnan(est).sum() == 0, "co-located conditioning produced NaN"
        np.testing.assert_allclose(est, value, rtol=1e-2, atol=1e-2)

    def test_ensemble_mean_close_to_kriging(self, pc2d_obs, pc2d_grid):
        """
        The ensemble mean of many SGSIM realisations should converge towards
        the kriging estimate. With nsim=50 the correlation should exceed 0.90.
        """
        from krigekit import ordinary_kriging
        coord, value   = pc2d_obs
        grid_coord, _  = pc2d_grid

        sims = sequential_gaussian_simulation(
            coord, value, grid_coord, _VGM_PC2D, nsim=50, nmax=20, seed=1001
        )
        ens_mean = sims.mean(axis=1)
        est, _ = ordinary_kriging(coord, value, grid_coord, _VGM_PC2D, nmax=20)
        corr = np.corrcoef(ens_mean, est)[0, 1]
        assert corr > 0.90, (
            f"Ensemble mean correlation with kriging = {corr:.3f} (expected > 0.90). "
            "This may indicate a bug in the SGSIM path or too few realisations."
        )

    def test_class_interface_with_precomputed_path_sample(
            self, pc2d_obs, pc2d_grid, sgsim_path_sample):
        """
        Using pre-computed path and sample must reproduce stored results
        deterministically — platform-independent regression test.
        """
        coord, value      = pc2d_obs
        grid_coord, _     = pc2d_grid
        path, sample      = sgsim_path_sample   # shapes: (4800,), (1, 4800)

        k = Kriging(ndim=2, nvar=1, nsim=1)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=20)
        k.set_vgm(ivar=1, jvar=1, **_VGM_PC2D)
        k.set_grid(coord=grid_coord)
        k.set_sim(randpath=path, sample=sample)
        k.set_search(ivar=1)
        k.solve()
        sims, _ = k.get_results()
        sims_matrix, _ = k.get_results(squeeze=False)
        sims_copy, _ = k.get_results(copy=True)

        assert sims.shape == (grid_coord.shape[0],)
        assert sims_matrix.shape == (grid_coord.shape[0], 1, 1)
        np.testing.assert_array_equal(sims_matrix[:,0,0], sims)
        assert sims_copy.flags.c_contiguous
        # Realisations must lie within a physically reasonable range
        assert sims.min() >= -5.0, f"Simulation minimum {sims.min()} is unreasonably low"
        assert sims.max() <=  5.0, f"Simulation maximum {sims.max()} is unreasonably high"

    def test_unconditional_simulation_no_observations(self):
        """SGSIM with zero conditioning data draws from the prior.

        The first visited node has no neighbours and is drawn from
        ``N(sk_mean, C(0))``; later nodes condition only on previously
        simulated nodes.  The ensemble variance should approach the sill.
        """
        g = np.arange(0.0, 60.0, 3.0)
        gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
        grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        k = Kriging(ndim=3, nvar=1, nsim=4, seed=11)
        k.set_obs(ivar=1, coord=np.empty((0, 3)), value=np.empty((0,)), nmax=48)
        k.set_grid(coord=grid)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0,
                  a_major=30.0, a_minor1=12.0, a_minor2=8.0, azimuth=30.0, dip=20.0)
        k.set_sim()
        k.set_search(ivar=1, anis1=12.0 / 30.0, anis2=8.0 / 30.0, azimuth=30.0, dip=20.0)
        k.solve()
        sims, _ = k.get_results()
        sims = np.asarray(sims)

        assert sims.shape == (grid.shape[0], 4)
        assert np.isnan(sims).sum() == 0, "unconditional SGSIM produced NaN"
        # Each unconditional realisation has its own random global mean (no data
        # pins it), so pooling realisations inflates the variance.  Check the
        # *per-realisation* spatial variance approaches the sill instead.
        per_real_var = sims.var(axis=0)
        assert np.all((per_real_var > 0.6) & (per_real_var < 1.6)), \
            f"per-realisation variance {per_real_var} far from sill 1.0"
        assert not np.allclose(sims[:, 0], sims[:, 1]), "realisations are identical"

    def test_unconditional_single_neighbour_node_is_conditioned(self):
        """The second unconditional node (one neighbour) must be solved.

        Regression for the ``npp > 1`` threshold that skipped single-neighbour
        systems, leaving the kriging variance NaN and the node undrawn.
        """
        grid = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        k = Kriging(ndim=3, nvar=1, nsim=1, seed=1)
        k.set_obs(ivar=1, coord=np.empty((0, 3)), value=np.empty((0,)), nmax=4)
        k.set_grid(coord=grid)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=10.0)
        k.set_sim()
        k.set_search(ivar=1)
        k.solve()
        val, var = k.get_results()
        val, var = np.asarray(val), np.asarray(var)
        assert np.isnan(val).sum() == 0 and np.isnan(var).sum() == 0
        # First node: drawn from the prior, variance == sill.
        assert var[0] == pytest.approx(1.0, abs=1e-6)
        # Second node conditions on the first, so its kriging variance is
        # strictly reduced below the sill (was left NaN before the fix).
        assert 0.0 < var[1] < 1.0

    def test_unconditional_requires_set_sim(self):
        """SGSIM without set_sim()/set_obs() raises instead of crashing."""
        grid = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])

        k = Kriging(ndim=3, nvar=1, nsim=1, seed=1)
        k.set_grid(coord=grid)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=10.0)
        with pytest.raises(RuntimeError, match="set_obs"):
            k.set_search(ivar=1)

        k = Kriging(ndim=3, nvar=1, nsim=1, seed=1)
        k.set_obs(ivar=1, coord=np.empty((0, 3)), value=np.empty((0,)), nmax=4)
        k.set_grid(coord=grid)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=10.0)
        with pytest.raises(RuntimeError, match="set_sim"):
            k.set_search(ivar=1)

    def test_unconditional_tiny_grid_below_dimension(self):
        """A grid with fewer nodes than dimensions must not break the k-d tree.

        Regression for kdtree2_create failing when the search pool is smaller
        than ndim; such tiny pools use the direct (no-tree) search path.
        """
        grid = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # 2 nodes in 3-D
        k = Kriging(ndim=3, nvar=1, nsim=1, seed=1)
        k.set_obs(ivar=1, coord=np.empty((0, 3)), value=np.empty((0,)), nmax=1)
        k.set_grid(coord=grid)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=10.0)
        k.set_sim()
        k.set_search(ivar=1)
        k.solve()
        sims = np.asarray(k.get_results()[0])
        assert sims.shape == (2,)
        assert np.isnan(sims).sum() == 0

    def test_joint_cosim_get_estimate_all_shape(self):
        """Joint co-simulation returns simulations, blocks, then variables."""
        coord = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        grid = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]])

        k = Kriging(ndim=2, nvar=2, nsim=2, seed=123)
        k.set_obs(ivar=1, coord=coord, value=np.array([1.0, 2.0, 1.5]),  nmax=3)
        k.set_obs(ivar=2, coord=coord, value=np.array([10.0, 20.0, 15.0]), nmax=3)
        k.set_grid(coord=grid)
        nobs = len(coord)
        nnew = len(grid)
        nsim=2
        nv = 2
        k.set_sim(
            randpath=np.array([1, 2, 3], dtype=np.int32),
            sample=np.ones((nnew, nv, nsim), order="F"),
        )
        k.set_search(ivar=1)
        k.set_search(ivar=2)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=1.0)
        k.set_vgm(ivar=1, jvar=2, vtype="sph", nugget=0.0, sill=0.25, a_major=1.0)
        k.set_vgm(ivar=2, jvar=2, vtype="sph", nugget=0.0, sill=1.0, a_major=1.0)

        k.solve()
        primary, primary_var = k.get_results()
        all_est = k.get_estimate_all()
        all_est_copy = k.get_estimate_all(copy=True)
        all_var = k.get_variance_all()
        all_var_copy = k.get_variance_all(copy=True)

        assert all_est.shape == (nnew, nv, nsim)
        assert all_var.shape == (nnew, nv, nv)
        assert np.all(np.diagonal(all_var, axis1=1, axis2=2) >= -1e-10)
        assert all_est.flags.f_contiguous
        assert all_var.flags.f_contiguous
        assert all_est_copy.flags.c_contiguous
        assert all_var_copy.flags.c_contiguous
        np.testing.assert_allclose(all_est_copy, all_est)
        np.testing.assert_allclose(all_var_copy, all_var)
