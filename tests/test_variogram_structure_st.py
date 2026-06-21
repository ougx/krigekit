"""Equivalence tests for the theoretical space-time structure."""

import numpy as np
import pytest

from krigekit import SpaceTimeVariogramModel, VariogramModel, VgmStructureST
from krigekit.variogram_kernels import calc_vgm


def _st_grid():
    return np.meshgrid(np.linspace(0.0, 6000.0, 7), np.linspace(0.0, 12.0, 5))


def test_product_sum_matches_model_evaluation_and_specs():
    params = (0.10, 0.06, -0.005, 5000.0, 9.0)
    aniso = {"anis1": 0.5, "azimuth": 0.0}

    st = VgmStructureST().set_product_sum(
        params, spatial_vtype="sph", temporal_vtype="gau", anisotropy=aniso)

    model = SpaceTimeVariogramModel()
    model.set_spacetime_anisotropy(anis1=0.5, azimuth=0.0)
    model.set_spacetime_params(params, spatial_vtype="sph", temporal_vtype="gau")

    hs, ht = _st_grid()
    np.testing.assert_allclose(
        st.calc_variogram(hs, ht), model.calc_spacetime_variogram(hs, ht))

    # cs/ct are rebuilt as single-component marginals carrying a+p and b+p
    assert st.ncomponent_spatial == 1 and st.ncomponent_temporal == 1
    assert st.cs is st.spatial and st.ct is st.temporal
    np.testing.assert_allclose(st.cs[0].sill, params[0] + params[2])
    np.testing.assert_allclose(st.ct[0].sill, params[1] + params[2])

    s_specs = st.to_kriging_specs(z_scale=5.0)
    m_specs = model.to_spacetime_kriging_specs(z_scale=5.0)
    assert s_specs["model"] == m_specs["model"] == "product_sum"
    np.testing.assert_allclose(s_specs["k_ps"], m_specs["k_ps"])
    np.testing.assert_allclose(s_specs["time_at"], m_specs["time_at"])
    for key in ("vtype", "sill", "a_major", "a_minor1", "a_minor2", "azimuth"):
        assert s_specs["spatial_spec"][key] == m_specs["spatial_spec"][key]
    for key in ("vtype", "sill", "at_k"):
        assert s_specs["temporal_spec"][key] == m_specs["temporal_spec"][key]


def test_sum_metric_matches_model_evaluation_and_specs():
    spatial = VariogramModel().set_vgm("sph", nugget=0.1, sill=2.0, a_major=10.0)
    temporal = VariogramModel()
    temporal.set_vgm("gau", nugget=0.05, sill=0.6, a_major=4.0)
    temporal.set_vgm("hol", sill=1.0, a_major=0.5, product=True)
    sm = (0.8, 1.2, 0.4, 3.0)

    st = VgmStructureST().set_sum_metric(
        spatial.structure, temporal.structure, sm,
        transform="lin", time_nugget=0.0, time_sill=1.0)

    model = SpaceTimeVariogramModel(spatial=spatial, temporal=temporal)
    model.sum_metric_spatial_model_ = spatial
    model.sum_metric_temporal_model_ = temporal
    model.sum_metric_transform_ = "lin"
    model.sum_metric_time_nugget_ = 0.0
    model.sum_metric_time_sill_ = 1.0
    model.sum_metric_params_ = np.asarray(sm, dtype=float)

    hs, ht = np.meshgrid(np.linspace(1.0, 15.0, 8), np.linspace(0.25, 6.0, 9))
    np.testing.assert_allclose(
        st.calc_variogram(hs, ht),
        model.calc_spacetime_sum_metric_variogram(hs, ht, sm))

    # cross-check against the explicit closed form
    dw = calc_vgm("lin", ht, rng=sm[3])
    hst = np.sqrt((hs / 10.0) ** 2 + dw ** 2)
    expected = (sm[0] * spatial.variogram(hs)
                + sm[1] * temporal.variogram(ht)
                + sm[2] * calc_vgm("sph", hst, rng=1.0))
    np.testing.assert_allclose(st.calc_variogram(hs, ht), expected)

    s_specs = st.to_kriging_specs()
    m_specs = model.to_sum_metric_kriging_specs()
    assert s_specs["model"] == m_specs["model"] == "sum_metric"
    np.testing.assert_allclose(s_specs["joint_sills"], m_specs["joint_sills"])
    np.testing.assert_allclose(s_specs["at"], m_specs["at"])
    np.testing.assert_allclose(
        s_specs["spatial_specs"][0]["sill"], m_specs["spatial_specs"][0]["sill"])
    np.testing.assert_allclose(
        s_specs["temporal_specs"][0]["sill"], m_specs["temporal_specs"][0]["sill"])


def test_validate_and_unconfigured_errors():
    st = VgmStructureST()
    with pytest.raises(RuntimeError):
        st.calc_variogram(1.0, 1.0)
    with pytest.raises(RuntimeError):
        st.validate()

    # product-sum covariance-conversion constraints
    with pytest.raises(ValueError):
        VgmStructureST().set_product_sum(
            (0.10, 0.004, -0.005, 5000.0, 9.0),
            spatial_vtype="sph", temporal_vtype="gau").validate()

    st.set_product_sum((0.10, 0.06, -0.005, 5000.0, 9.0),
                       spatial_vtype="sph", temporal_vtype="gau")
    assert st.validate() is st
    assert st.copy().model == "product_sum"
