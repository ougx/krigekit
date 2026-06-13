"""
test_nscore.py
==============
Tests for the normal-score (Gaussian anamorphosis) transform used by SGSIM.

The transform lives in the Fortran engine (behind the C API), so it is shared
by every client.  These tests exercise it through the Python wrapper:

  * set_nscore requires sequential simulation (nsim > 0);
  * realisations are back-transformed to data units and, with the default
    zmin/zmax = data range, stay within that range;
  * a grid node coincident with an observation reproduces the observed value
    after back-transform (exact conditioning, zero kriging variance);
  * seeded runs are reproducible;
  * enabling the transform changes the result versus simulating in native space.

The variogram passed to set_vgm is the variogram of the normal scores
(unit sill), as the workflow requires.
"""

import numpy as np
import pytest
from krigekit import Kriging

# Normal-score variogram (unit sill); domain is [0, 100]^2.
_VGM = dict(vtype="sph", nugget=0.0, sill=1.0, a_major=40.0)


def _skewed_data(n=60, seed=0):
    """Strongly right-skewed (lognormal) data on a 2-D domain."""
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, 100.0, size=(n, 2))
    value = np.exp(rng.normal(0.0, 1.0, size=n))
    return coord, value


def _grid(nx=12, ny=12):
    gx, gy = np.meshgrid(np.linspace(0, 100, nx), np.linspace(0, 100, ny))
    return np.column_stack([gx.ravel(), gy.ravel()])


def _run(coord, value, grid, nsim=1, seed=42, nscore=True, **ns_kw):
    k = Kriging(nsim=nsim, seed=seed)
    k.set_obs(ivar=1, coord=coord, value=value, nmax=len(value))
    if nscore:
        k.set_nscore(ivar=1, **ns_kw)
    k.set_grid(coord=grid)
    k.set_vgm(ivar=1, jvar=1, **_VGM)
    k.set_sim()
    k.set_search()
    k.solve()
    est, _ = k.get_results()
    del k
    return np.asarray(est).reshape(grid.shape[0], -1)   # (ngrid, nsim)


# ---------------------------------------------------------------------------
class TestNscoreApi:

    def test_requires_nsim(self):
        coord, value = _skewed_data()
        k = Kriging(nsim=0)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=len(value))
        with pytest.raises(ValueError):
            k.set_nscore(ivar=1)
        del k

    def test_requires_obs_or_explicit_bounds(self):
        k = Kriging(nsim=2)
        with pytest.raises(Exception):
            k.set_nscore(ivar=1)        # no obs cached and no zmin/zmax given
        del k


# ---------------------------------------------------------------------------
class TestNscoreSimulation:

    def test_realisations_finite_and_bounded(self):
        coord, value = _skewed_data()
        grid = _grid()
        sims = _run(coord, value, grid, nsim=4, seed=1)
        assert np.all(np.isfinite(sims))
        # default zmin/zmax = data range -> realisations stay within it
        assert sims.min() >= value.min() - 1e-6
        assert sims.max() <= value.max() + 1e-6

    def test_reproduces_data_distribution(self):
        """
        The pooled marginal of many realisations reproduces the data
        distribution (the defining property of the normal-score transform):
        the simulated median and mean track the data, and no NaNs appear when
        the grid does not coincide with observations.  Deterministic (seeded).
        """
        coord, value = _skewed_data(n=80, seed=11)
        grid = _grid(20, 20)
        sims = _run(coord, value, grid, nsim=50, seed=2)
        pooled = sims.ravel()
        assert np.isnan(pooled).sum() == 0
        # bounded by the data range (default zmin/zmax)
        assert pooled.min() >= value.min() - 1e-6
        assert pooled.max() <= value.max() + 1e-6
        iqr = np.percentile(value, 75) - np.percentile(value, 25)
        assert abs(np.median(pooled) - np.median(value)) <= 0.5 * iqr
        assert abs(pooled.mean() - value.mean()) <= 0.4 * value.mean()

    def test_exact_conditioning_at_observations(self):
        """
        With the grid placed on observation locations, the back-transformed
        realisation reproduces the observed values and contains no NaN — the
        co-located conditioning point (hard datum + simulated node) is
        de-singularised by the solver.
        """
        coord, value = _skewed_data(n=40, seed=3)
        grid = coord.copy()
        sims = _run(coord, value, grid, nsim=1, seed=5)
        assert np.isnan(sims).sum() == 0
        np.testing.assert_allclose(sims[:, 0], value, rtol=1e-2, atol=1e-2)

    def test_seed_reproducibility(self):
        coord, value = _skewed_data()
        grid = _grid()
        a = _run(coord, value, grid, nsim=2, seed=99)
        b = _run(coord, value, grid, nsim=2, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_differs_from_native_space(self):
        coord, value = _skewed_data()
        grid = _grid()
        with_ns = _run(coord, value, grid, nsim=1, seed=7, nscore=True)
        without = _run(coord, value, grid, nsim=1, seed=7, nscore=False)
        assert not np.allclose(with_ns, without)

    def test_wider_bounds_allow_extrapolation(self):
        """With zmin/zmax wider than the data and power tails, realisations may
        extend beyond the data range but stay within [zmin, zmax]."""
        coord, value = _skewed_data(seed=4)
        grid = _grid()
        zmin = float(value.min()) - 5.0
        zmax = float(value.max()) + 5.0
        sims = _run(coord, value, grid, nsim=4, seed=8,
                    zmin=zmin, zmax=zmax, ltail="power", utail="power",
                    ltpar=1.5, utpar=1.5)
        assert sims.min() >= zmin - 1e-6
        assert sims.max() <= zmax + 1e-6
