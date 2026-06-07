"""
Tests for IndicatorKriging (MIK estimation and SIS).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from pykriging import IndicatorKriging


# ---------------------------------------------------------------------------
# Shared fixture: 3-category 2-D dataset
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

NOBS   = 40
NCAT   = 3
NDIM   = 2
NGRID  = 20
NMAX   = NOBS  # use all obs

OBS_COORD = RNG.uniform(0, 100, (NOBS, NDIM))
OBS_CAT   = RNG.integers(1, NCAT + 1, NOBS)   # labels 1, 2, 3
GRID_COORD = RNG.uniform(10, 90, (NGRID, NDIM))
CAT_LABELS = [1, 2, 3]

VGM = dict(vtype="sph", nugget=0.05, sill=0.25, a_major=60.0, a_minor1=60.0, a_minor2=60.0)


def _build_ik(nsim=0, seed=1):
    """Build a fully ready IndicatorKriging object.

    Call order required by the Fortran layer:
        set_obs  →  set_vgm  →  set_grid  →  [set_sim]  →  set_search
    For nsim>0 set_sim is inserted between set_grid and set_search.
    """
    ik = IndicatorKriging(ncat=NCAT, ndim=NDIM, nsim=nsim, seed=seed,
                          neglect_error=True, std_ck=True)
    ik.set_categorical_obs(coord=OBS_COORD, categories=OBS_CAT,
                           category_labels=CAT_LABELS, nmax=NMAX)
    # All K² variogram pairs are required; off-diagonal get the same model
    # (independent-kriging approximation — the post_solve normalisation
    # corrects the probabilities regardless of cross-variogram choice).
    for iv in range(1, NCAT + 1):
        for jv in range(1, NCAT + 1):
            ik.set_vgm(ivar=iv, jvar=jv, **VGM)
    ik.set_grid(coord=GRID_COORD)
    if nsim > 0:
        ik.set_sim()   # must come before set_search when nsim>0
    for k in range(1, NCAT + 1):
        ik.set_search(ivar=k)
    return ik


# ---------------------------------------------------------------------------
# Estimation tests (nsim=0)
# ---------------------------------------------------------------------------

class TestMIKEstimation:
    def test_shape(self):
        ik = _build_ik()
        ik.solve()
        probs, var = ik.get_results()
        # (ngrid, ncat) because nsim=0 and nvar=3>1
        assert probs.shape == (NGRID, NCAT), f"Unexpected shape: {probs.shape}"

    def test_probabilities_in_unit_interval(self):
        ik = _build_ik()
        ik.solve()
        probs, _ = ik.get_results()
        assert np.all(probs >= -1e-6), "Probability below 0"
        assert np.all(probs <= 1 + 1e-6), "Probability above 1"

    def test_probabilities_sum_to_one(self):
        ik = _build_ik()
        ik.solve()
        probs, _ = ik.get_results()
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                   err_msg="Probability rows do not sum to 1")

    def test_set_categorical_obs_label_mismatch_raises(self):
        ik = IndicatorKriging(ncat=3, ndim=2, seed=1)
        with pytest.raises(ValueError, match="ncat"):
            ik.set_categorical_obs(OBS_COORD, OBS_CAT, category_labels=[1, 2])

    def test_destroy_without_error(self):
        ik = _build_ik()
        ik.solve()
        del ik   # should not raise


# ---------------------------------------------------------------------------
# SIS tests (nsim>0)
# ---------------------------------------------------------------------------

NSIM = 10

class TestSIS:
    def test_shape(self):
        ik = _build_ik(nsim=NSIM)
        ik.solve()
        sims, _ = ik.get_results()
        assert sims.shape == (NGRID, NCAT, NSIM), f"Unexpected shape: {sims.shape}"

    def test_binary_indicators(self):
        ik = _build_ik(nsim=NSIM)
        ik.solve()
        sims, _ = ik.get_results()
        # Each (iblock, :, isim) must be a one-hot binary vector
        vals = set(np.unique(np.round(sims, 6)))
        assert vals.issubset({0.0, 1.0}), f"Non-binary values in SIS output: {vals}"

    def test_one_hot_per_block(self):
        ik = _build_ik(nsim=NSIM)
        ik.solve()
        sims, _ = ik.get_results()   # (ngrid, ncat, nsim)
        # For each realisation and each block, exactly one category is 1
        for i in range(NSIM):
            col_sums = sims[:, :, i].sum(axis=1)   # (ngrid,)
            np.testing.assert_allclose(
                col_sums, 1.0, atol=1e-5,
                err_msg=f"Realisation {i} has non-unit row sums: {col_sums}"
            )

    def test_multiple_categories_drawn(self):
        ik = _build_ik(nsim=50, seed=99)
        ik.solve()
        sims, _ = ik.get_results()
        # Each category should appear at least once across all blocks/realisations
        for k in range(NCAT):
            assert sims[:, k, :].max() > 0.5, \
                f"Category {k+1} never drawn in 50 realisations"

    def test_destroy_without_error(self):
        ik = _build_ik(nsim=NSIM)
        ik.solve()
        del ik
