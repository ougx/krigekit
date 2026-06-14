"""
test_uscore.py
==============
Tests for the uniform quantile transform used by SGSIM.
"""

import numpy as np
import pytest
from krigekit import Kriging

_VGM = dict(vtype="sph", nugget=0.0, sill=1.0 / 12.0, a_major=40.0)


def _skewed_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, 100.0, size=(n, 2))
    value = np.exp(rng.normal(0.0, 1.0, size=n))
    return coord, value


def _grid(nx=12, ny=12):
    gx, gy = np.meshgrid(np.linspace(0, 100, nx), np.linspace(0, 100, ny))
    return np.column_stack([gx.ravel(), gy.ravel()])


def _run(coord, value, grid, nsim=1, seed=42, **us_kw):
    k = Kriging(nsim=nsim, seed=seed)
    k.set_obs(ivar=1, coord=coord, value=value, nmax=len(value))
    k.set_uscore(ivar=1, **us_kw)
    k.set_grid(coord=grid)
    k.set_vgm(ivar=1, jvar=1, **_VGM)
    k.set_sim()
    k.set_search()
    k.solve()
    est, _ = k.get_results()
    del k
    return np.asarray(est).reshape(grid.shape[0], -1)


class TestUscoreApi:

    def test_requires_nsim(self):
        coord, value = _skewed_data()
        k = Kriging(nsim=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=len(value))
        with pytest.raises(ValueError):
            k.set_uscore(ivar=1)
        del k

    def test_requires_obs_or_explicit_bounds(self):
        k = Kriging(nsim=2)
        with pytest.raises(Exception):
            k.set_uscore(ivar=1)
        del k

    def test_quantile_alias(self):
        coord, value = _skewed_data()
        grid = _grid(4, 4)
        k = Kriging(nsim=1, seed=3)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=len(value))
        k.set_quantile(ivar=1)
        k.set_grid(coord=grid)
        k.set_vgm(ivar=1, jvar=1, **_VGM)
        k.set_sim()
        k.set_search()
        k.solve()
        est, _ = k.get_results()
        assert np.all(np.isfinite(est))
        del k

    def test_rejects_stacked_transforms(self):
        coord, value = _skewed_data()
        k = Kriging(nsim=1)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=len(value))
        k.set_nscore(ivar=1)
        with pytest.raises(RuntimeError):
            k.set_uscore(ivar=1)
        del k


class TestUscoreSimulation:

    def test_realisations_finite_and_bounded(self):
        coord, value = _skewed_data()
        grid = _grid()
        sims = _run(coord, value, grid, nsim=4, seed=1)
        assert np.all(np.isfinite(sims))
        assert sims.min() >= value.min() - 1e-6
        assert sims.max() <= value.max() + 1e-6

    def test_exact_conditioning_at_observations(self):
        coord, value = _skewed_data(n=40, seed=3)
        sims = _run(coord, value, coord.copy(), nsim=1, seed=5)
        assert np.isnan(sims).sum() == 0
        np.testing.assert_allclose(sims[:, 0], value, rtol=1e-2, atol=1e-2)

    def test_wider_bounds_allow_extrapolation(self):
        coord, value = _skewed_data(seed=4)
        grid = _grid()
        zmin = float(value.min()) - 5.0
        zmax = float(value.max()) + 5.0
        sims = _run(coord, value, grid, nsim=4, seed=8,
                    zmin=zmin, zmax=zmax, ltail="power", utail="power",
                    ltpar=1.5, utpar=1.5)
        assert sims.min() >= zmin - 1e-6
        assert sims.max() <= zmax + 1e-6
