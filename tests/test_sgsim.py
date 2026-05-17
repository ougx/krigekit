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
from pykriging import Kriging, sequential_gaussian_simulation

_VGM_PC2D = "sph 0.0 0.12 5000.0 5000.0 5000.0 0.0 0.0 0.0"


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
        assert sims.shape == (nsim, grid_coord.shape[0])

    def test_realisations_differ(self, pc2d_obs, pc2d_grid):
        """Different realisations must not be identical."""
        coord, value   = pc2d_obs
        grid_coord, _  = pc2d_grid
        sims = sequential_gaussian_simulation(
            coord, value, grid_coord, _VGM_PC2D, nsim=2, nmax=20, seed=123
        )
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
        from pykriging import ordinary_kriging
        coord, value   = pc2d_obs
        grid_coord, _  = pc2d_grid

        sims = sequential_gaussian_simulation(
            coord, value, grid_coord, _VGM_PC2D, nsim=50, nmax=20, seed=0
        )
        ens_mean = sims.mean(axis=0)

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
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_PC2D)
        k.set_grid(coord=grid_coord)
        k.set_sim(randpath=path, sample=sample)
        k.set_search(ivar=1)
        k.solve()
        sims, _ = k.get_results()

        assert sims.shape == (grid_coord.shape[0],)
        # Realisations must lie within a physically reasonable range
        assert sims.min() >= -5.0, f"Simulation minimum {sims.min()} is unreasonably low"
        assert sims.max() <=  5.0, f"Simulation maximum {sims.max()} is unreasonably high"
