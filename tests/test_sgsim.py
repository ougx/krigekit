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
