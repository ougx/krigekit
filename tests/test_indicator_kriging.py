"""
Tests for IndicatorKriging (MIK estimation and SIS).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from krigekit import IndicatorKriging, IndicatorVariogramSystem
from krigekit.variogram_system_indicator import encode_indicator_matrix


def _set_indicator_obs(ik, coord, categories, labels, nmax):
    """Encode raw categories and load them as indicator observations.

    Replaces the removed ``IndicatorKriging.set_categorical_obs`` convenience for
    cokriging-with-secondary cases, where the pure indicator system cannot own
    the extra continuous variable.
    """
    matrix = encode_indicator_matrix(categories, labels)
    for k in range(1, len(labels) + 1):
        ik.set_obs(ivar=k, coord=coord, value=matrix[:, k - 1], nmax=nmax)


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
    # Build the indicator block with the variogram system and transfer it.  The
    # off-diagonal pairs get the same (uniform) model — the post_solve
    # normalisation corrects probabilities regardless of cross-variogram choice.
    system = IndicatorVariogramSystem(categories=CAT_LABELS)
    system.set_categorical_obs(OBS_COORD, OBS_CAT, nmax=NMAX)
    system.set_indicator_vgm(sill_strategy="uniform", cross_strategy="uniform",
                             **VGM)
    system.apply(ik)
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
        system = IndicatorVariogramSystem(ncat=3)
        with pytest.raises(ValueError, match="ncat"):
            system.set_categorical_obs(OBS_COORD, OBS_CAT, category_labels=[1, 2])

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


def test_secondary_covariate_is_excluded_from_indicator_results():
    ik = IndicatorKriging(
        ncat=2, nvar=3, ndim=2, nsim=2, seed=7,
        neglect_error=True, std_ck=True,
    )
    categories = np.where(np.arange(NOBS) % 2 == 0, 1, 2)
    _set_indicator_obs(ik, OBS_COORD, categories, [1, 2], NMAX)
    ik.set_obs(
        ivar=3,
        coord=OBS_COORD,
        value=np.linspace(-1.0, 1.0, NOBS),
        nmax=NMAX,
    )
    for ivar in range(1, 4):
        for jvar in range(ivar, 4):
            ik.set_vgm(ivar, jvar, **VGM)
    ik.set_grid(GRID_COORD)
    ik.set_sim()
    for ivar in range(1, 4):
        ik.set_search(ivar)
    ik.solve()

    sims, variance = ik.get_results()
    assert sims.shape == (NGRID, 2, 2)
    assert variance.shape == (NGRID, 2, 2)
    np.testing.assert_allclose(sims.sum(axis=1), 1.0)


def _build_secondary_coik_1d(cross_sill, nsim=0, weight_file=""):
    """Build a small valid indicator/AEM cokriging system."""
    obs_coord = np.array([[0.0], [10.0], [20.0]])
    categories = np.array([1, 2, 1])
    secondary_coord = np.array([[0.0], [5.0], [10.0], [15.0], [20.0]])
    secondary_value = np.array([-1.0, 2.0, 1.0, -2.0, 0.5])
    grid_coord = np.array([[2.0], [8.0], [14.0]])

    ik = IndicatorKriging(
        ncat=2, nvar=3, ndim=1, nsim=nsim, seed=3,
        std_ck=True, store_weight=bool(weight_file),
        weight_file=str(weight_file),
    )
    _set_indicator_obs(ik, obs_coord, categories, [1, 2], 5)
    ik.set_obs(3, secondary_coord, secondary_value, nmax=5)

    # Positive-semidefinite coregionalization matrix.  The two indicators are
    # complementary; the secondary variable is negatively correlated with
    # category 1 and positively correlated with category 2.
    sill = np.array([
        [0.25, -0.25, -cross_sill],
        [-0.25, 0.25, cross_sill],
        [-cross_sill, cross_sill, 1.0],
    ])
    for ivar in range(3):
        for jvar in range(ivar, 3):
            ik.set_vgm(
                ivar + 1, jvar + 1, vtype="sph", nugget=0.0,
                sill=float(sill[ivar, jvar]),
                a_major=15.0, a_minor1=15.0, a_minor2=15.0,
            )

    ik.set_grid(grid_coord)
    if nsim:
        sample = np.full((len(grid_coord), 3, nsim), 0.6)
        ik.set_sim(
            randpath=np.arange(1, len(grid_coord) + 1, dtype=np.int32),
            sample=sample,
        )
    for ivar in range(1, 4):
        ik.set_search(ivar)
    return ik


def test_secondary_cross_covariance_changes_indicator_probabilities():
    independent = _build_secondary_coik_1d(cross_sill=0.0)
    correlated = _build_secondary_coik_1d(cross_sill=0.2)
    independent.solve()
    correlated.solve()

    prob_independent, _ = independent.get_results()
    prob_correlated, variance = correlated.get_results()

    assert np.max(np.abs(prob_correlated - prob_independent)) > 0.1
    np.testing.assert_allclose(variance, variance.swapaxes(1, 2), atol=1e-10)
    assert np.all(variance[:, 0, 1] < 0.0)


def test_secondary_covariate_has_no_simulated_neighbor_group(tmp_path):
    ik = _build_secondary_coik_1d(
        cross_sill=0.2,
        nsim=1,
        weight_file=tmp_path / "indicator_aem_weights.txt",
    )
    ik.solve()
    weights = ik.get_weights()

    # SGSIM group layout is obs(1:nvar), sim(1:nvar).  The indicators acquire
    # prior simulated nodes, while the AEM variable remains observation-only.
    assert np.any(weights["nnear"][1:, 3:5] > 0)
    np.testing.assert_array_equal(weights["nnear"][:, 5], 0)


def test_multivariate_sis_uses_variable_specific_simulation_constraints():
    """Prior simulated categories must not duplicate variable-1 constraints."""
    ngrid = 25
    grid = np.arange(ngrid, dtype=float)[:, None]
    empty_coord = np.empty((0, 1))
    empty_value = np.empty(0)

    ik = IndicatorKriging(
        ncat=2, ndim=1, nsim=4, seed=11,
        std_ck=True, neglect_error=False,
    )
    for ivar in (1, 2):
        ik.set_obs(
            ivar, empty_coord, empty_value,
            nmax=8, sk_mean=0.5,
        )
    ik.set_vgm(1, 1, "sph", nugget=0.0, sill=0.25, a_major=10.0)
    ik.set_vgm(1, 2, "sph", nugget=0.0, sill=0.0, a_major=10.0)
    ik.set_vgm(2, 2, "sph", nugget=0.0, sill=0.25, a_major=10.0)
    ik.set_grid(grid)
    ik.set_sim(randpath=np.arange(1, ngrid + 1, dtype=np.int32))
    ik.set_search(1)
    ik.set_search(2)

    # Before the regression fix, block 2 was singular because both simulated
    # groups used block%drift(:, 1, :) and therefore duplicated category 1's
    # unbiasedness row.
    ik.solve()
    simulations, variance = ik.get_results()
    assert np.all(np.isfinite(simulations))
    assert np.all(np.isfinite(variance))
    np.testing.assert_allclose(simulations.sum(axis=1), 1.0)
