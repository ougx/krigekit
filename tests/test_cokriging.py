"""
test_cokriging.py
=================
Tests for ordinary co-kriging using the Walker Lake dataset.

Dataset
-------
walker.csv contains 470 observations of V (primary) and U (secondary) in
a 260 x 300 unit domain. We use only rows where both V and U are observed
(U != -999). The variogram models are simplified sphericals consistent with
the published Walker Lake study parameters.
"""

import numpy as np
import pytest
from pykriging import Kriging, cokriging

# Walker Lake variogram parameters (simplified)
# V auto-variogram
_VGM_VV = "sph 0.0 1.0 50.0 80.0 50.0 0.0 0.0 0.0"
# U auto-variogram
_VGM_UU = "sph 0.0 1.0 50.0 80.0 50.0 0.0 0.0 0.0"
# Cross-variogram (b12^2 <= b11*b22 = 1.0 satisfied)
_VGM_VU = "sph 0.0 0.8 50.0 80.0 50.0 0.0 0.0 0.0"

# Small regular test grid (5x5)
_GRID_X = np.linspace(10, 250, 5)
_GRID_Y = np.linspace(10, 290, 5)
_GRID   = np.array([[x, y] for x in _GRID_X for y in _GRID_Y])


class TestCoKriging:

    def test_cokriging_result_shapes(self, walker_obs):
        (coord_v, val_v), (coord_u, val_u) = walker_obs
        # Standardise both variables to unit variance for numerical stability
        val_v = (val_v - val_v.mean()) / val_v.std()
        val_u = (val_u - val_u.mean()) / val_u.std()

        est, var = cokriging(
            obs_coords=[coord_v, coord_u],
            obs_values=[val_v, val_u],
            grid_coord=_GRID,
            variogram_specs={
                (1, 1): _VGM_VV,
                (2, 2): _VGM_UU,
                (1, 2): _VGM_VU,
            },
            nmax=20,
        )
        assert est.shape == (_GRID.shape[0],)
        assert var.shape == (_GRID.shape[0],)

    def test_cokriging_variance_nonnegative(self, walker_obs):
        (coord_v, val_v), (coord_u, val_u) = walker_obs
        val_v = (val_v - val_v.mean()) / val_v.std()
        val_u = (val_u - val_u.mean()) / val_u.std()

        _, var = cokriging(
            obs_coords=[coord_v, coord_u],
            obs_values=[val_v, val_u],
            grid_coord=_GRID,
            variogram_specs={(1,1):_VGM_VV, (2,2):_VGM_UU, (1,2):_VGM_VU},
            nmax=20,
        )
        assert np.all(var >= 0.0), f"Negative variance: {var.min()}"

    def test_cokriging_class_interface(self, walker_obs):
        """Same run via the Kriging class to verify the two-variable workflow."""
        (coord_v, val_v), (coord_u, val_u) = walker_obs
        val_v = (val_v - val_v.mean()) / val_v.std()
        val_u = (val_u - val_u.mean()) / val_u.std()

        k = Kriging(ndim=2, nvar=2)
        k.set_obs(ivar=1, coord=coord_v, value=val_v, nmax=20)
        k.set_obs(ivar=2, coord=coord_u, value=val_u, nmax=20)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM_VV)
        k.set_vgm(ivar=2, jvar=2, spec=_VGM_UU)
        k.set_vgm(ivar=1, jvar=2, spec=_VGM_VU)
        k.set_grid(coord=_GRID)
        k.set_search(ivar=1)
        k.set_search(ivar=2)
        k.solve()
        est, var = k.get_results()
        assert est.shape == (_GRID.shape[0],)
        assert np.all(var >= 0.0)

    def test_cokriging_better_than_kriging_on_sparse_primary(self, walker_obs):
        """
        Co-kriging should produce smaller mean variance than ordinary kriging
        when secondary data is abundant: a basic sanity check that the secondary
        variable is contributing information.
        """
        (coord_v, val_v), (coord_u, val_u) = walker_obs
        val_v = (val_v - val_v.mean()) / val_v.std()
        val_u = (val_u - val_u.mean()) / val_u.std()

        # Use only every 5th primary observation to simulate sparsity
        coord_v_sparse = coord_v[::5]
        val_v_sparse   = val_v[::5]

        # Ordinary kriging with sparse primary only
        k_ok = Kriging(ndim=2, nvar=1)
        k_ok.set_obs(ivar=1, coord=coord_v_sparse, value=val_v_sparse, nmax=20)
        k_ok.set_vgm(ivar=1, jvar=1, spec=_VGM_VV)
        k_ok.set_grid(coord=_GRID)
        k_ok.set_search(ivar=1)
        k_ok.solve()
        _, var_ok = k_ok.get_results()

        # Co-kriging with sparse primary + full secondary
        _, var_cok = cokriging(
            obs_coords=[coord_v_sparse, coord_u],
            obs_values=[val_v_sparse, val_u],
            grid_coord=_GRID,
            variogram_specs={(1,1):_VGM_VV, (2,2):_VGM_UU, (1,2):_VGM_VU},
            nmax=20,
        )
        assert var_cok.mean() <= var_ok.mean(), (
            f"Co-kriging mean variance ({var_cok.mean():.4f}) should be <= "
            f"ordinary kriging mean variance ({var_ok.mean():.4f})"
        )
