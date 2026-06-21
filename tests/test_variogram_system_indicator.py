"""Tests for the indicator variogram system and its closure-aware fitting."""

import numpy as np
import pandas as pd
import pytest

from krigekit import IndicatorKriging, IndicatorVariogramSystem
from krigekit.variogram_kernels import calc_vgm
from krigekit.variogram_system_indicator import (
    contrast_basis,
    encode_indicator_matrix,
    indicator_covariance,
)


def test_helpers_encode_covariance_and_contrast_basis():
    values = np.array([1, 2, 3, 1, 2])
    matrix = encode_indicator_matrix(values, [1, 2, 3])
    assert matrix.shape == (5, 3)
    np.testing.assert_array_equal(matrix.sum(axis=1), np.ones(5))
    np.testing.assert_array_equal(matrix[:, 0], [1, 0, 0, 1, 0])

    p = np.array([0.2, 0.3, 0.5])
    b0 = indicator_covariance(p)
    np.testing.assert_allclose(b0, np.diag(p) - np.outer(p, p))
    np.testing.assert_allclose(b0.sum(axis=1), 0.0, atol=1e-12)        # closure
    assert np.linalg.eigvalsh(b0).min() >= -1e-12                      # PSD

    q = contrast_basis(3)
    assert q.shape == (3, 2)
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-12)         # orthonormal
    np.testing.assert_allclose(q.T @ np.ones(3), 0.0, atol=1e-12)      # ⟂ ones


def test_set_indicator_vgm_matches_legacy_strategies():
    props = np.array([0.18, 0.23, 0.21, 0.38])
    auto = props * (1.0 - props)

    # legacy "proportional": theoretical autos, cross = sqrt(s_i s_j)
    system = IndicatorVariogramSystem(categories=[1, 2, 3, 4])
    system.proportions = props
    system.set_indicator_vgm(vtype="sph", nugget=0.02, a_major=500.0,
                             a_minor1=80.0, azimuth=90.0,
                             sill_strategy="theoretical",
                             cross_strategy="proportional")
    for i in range(1, 5):
        for j in range(i, 5):
            comp = system.vgm[i, j].components[0]
            if i == j:
                np.testing.assert_allclose(comp.sill, auto[i - 1])
                assert comp.nugget == 0.02
            else:
                np.testing.assert_allclose(
                    comp.sill, np.sqrt(auto[i - 1] * auto[j - 1]))

    # legacy "same": uniform autos and crosses
    same = IndicatorVariogramSystem(categories=[1, 2, 3, 4])
    same.set_indicator_vgm(vtype="sph", nugget=0.02, sill=0.19, a_major=500.0,
                           sill_strategy="uniform", cross_strategy="uniform")
    for i in range(1, 5):
        for j in range(i, 5):
            np.testing.assert_allclose(same.vgm[i, j].components[0].sill, 0.19)


def test_closure_strategy_builds_indicator_covariance():
    p = np.array([0.2, 0.3, 0.5])
    system = IndicatorVariogramSystem(categories=["a", "b", "c"])
    system.proportions = p
    system.set_indicator_vgm(vtype="sph", a_major=4.0, sill_strategy="theoretical",
                             cross_strategy="closure")
    b0 = indicator_covariance(p)
    for i in range(1, 4):
        for j in range(i, 4):
            np.testing.assert_allclose(
                system.vgm[i, j].components[0].sill, b0[i - 1, j - 1])


def test_fit_indicator_lmc_recovers_closed_psd_matrix():
    p = np.array([0.2, 0.3, 0.5])
    b0 = indicator_covariance(p)
    rng = 4.0
    h = np.linspace(0.2, 8.0, 40)

    system = IndicatorVariogramSystem(categories=["a", "b", "c"])
    system.proportions = p
    system.set_indicator_vgm(vtype="sph", a_major=rng, sill_strategy="theoretical",
                             cross_strategy="closure")
    for i in range(1, 4):
        for j in range(i, 4):
            system.set_avg_vgm(i, j, pd.DataFrame({
                ("distance", "mean"): h,
                ("variogram", "mean"): calc_vgm("sph", h, psill=b0[i - 1, j - 1], rng=rng),
                ("variogram", "count"): np.ones_like(h),
            }))

    res = system.fit(method="closure", fit_nugget=False,
                     weight_col=("variogram", "count"))
    fitted = res.target
    assert res.success
    assert isinstance(fitted, IndicatorVariogramSystem)

    mats = fitted.get_lmc_matrices(include_nugget=False)
    np.testing.assert_allclose(mats[0], b0, atol=1e-4)
    np.testing.assert_allclose(mats[0].sum(axis=1), 0.0, atol=1e-8)   # closure
    assert np.linalg.eigvalsh(mats[0]).min() >= -1e-8                 # PSD
    fitted.validate_closure()


def test_set_categorical_obs_encodes_and_computes_proportions():
    rng = np.random.default_rng(0)
    coord = rng.uniform(0, 100, size=(60, 2))
    cats = rng.choice([10, 20, 30], size=60, p=[0.5, 0.3, 0.2])

    system = IndicatorVariogramSystem(ncat=3)
    system.set_categorical_obs(coord, cats, category_labels=[10, 20, 30], nmax=16)

    assert system.categories == (10, 20, 30)
    assert system.proportions.shape == (3,)
    np.testing.assert_allclose(system.proportions.sum(), 1.0)
    # each indicator mean equals its empirical proportion
    np.testing.assert_allclose(system.proportions[0], np.mean(cats == 10))


def test_apply_transfers_indicators_and_structures_to_kriging():
    rng = np.random.default_rng(1)
    coord = rng.uniform(0, 100, size=(40, 2))
    cats = rng.choice([1, 2, 3], size=40, p=[0.4, 0.35, 0.25])

    system = IndicatorVariogramSystem(categories=[1, 2, 3])
    system.set_categorical_obs(coord, cats, nmax=12)
    system.set_indicator_vgm(vtype="sph", a_major=30.0,
                             sill_strategy="theoretical", cross_strategy="closure")

    ik = IndicatorKriging(ncat=3, ndim=2)
    system.apply(ik)

    assert int(ik._nobs[0]) == 40
    # all 3 auto + 3 cross structures transferred
    assert ik._nvgm_struct[0, 0] == 1
    assert ik._nvgm_struct[0, 1] == 1

    mismatch = IndicatorKriging(ncat=4, ndim=2)
    with pytest.raises(ValueError):
        system.apply(mismatch)
