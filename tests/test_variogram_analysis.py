import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from krigekit import Kriging
from krigekit.variogram import (
    _great_circle_dist,
    _fill_nan_nearest,
    VariogramModel,
    VariogramSystem,
    avg_vgm,
    calc_anisotropic_lag,
    calc_cov,
    calc_lag_vectors,
    calc_vgm,
    cross_vgm,
    directional_vgm,
    fit_vgm,
    filter_vgm,
    raw_vgm,
    raw_cross_vgm,
)


def test_theoretical_models_match_engine_core_shapes():
    h = np.array([0.0, 0.5, 1.0, 2.0])

    np.testing.assert_allclose(calc_cov("nug", h), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(calc_cov("sph", h), [1.0, 0.3125, 0.0, 0.0])
    np.testing.assert_allclose(calc_cov("exp", h), np.exp(-3.0 * h))
    np.testing.assert_allclose(calc_cov("gau", h), np.exp(-3.0625 * h * h))
    np.testing.assert_allclose(calc_cov("hol", h), np.cos(np.pi * h))
    np.testing.assert_allclose(calc_cov("pow", h), [1.0, 1.0 - 0.5 ** 1.5, 0.0, 0.0])
    np.testing.assert_allclose(calc_cov("bsq", h), [1.0, (1.0 - 0.5 ** 2) ** 2, 0.0, 0.0])

    rc = 0.5
    circular = 1.0 - (2.0 * rc * np.sqrt(1.0 - rc ** 2) + 2.0 * np.arcsin(rc)) / np.pi
    np.testing.assert_allclose(calc_cov("cir", h), [1.0, circular, 0.0, 0.0])
    np.testing.assert_allclose(calc_cov("lin", h), [1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(calc_cov("cyc", h), np.exp(-2.0 * np.sin(np.pi * h) ** 2))
    np.testing.assert_allclose(calc_cov("dco", h), np.exp(-3.0 * h) * np.cos(np.pi * h))


def test_variogram_model_additive_nested_structures():
    h = np.array([0.0, 0.5, 1.0])
    model = VariogramModel()
    model.set_vgm(vtype="sph", nugget=0.1, sill=0.9, a_major=1.0)
    model.set_vgm(vtype="exp", nugget=0.0, sill=0.2, a_major=2.0)

    expected_cov = (
        np.where(h <= 0.0, 1.0, calc_cov("sph", h, psill=0.9, rng=1.0))
        + np.where(h <= 0.0, 0.2, calc_cov("exp", h, psill=0.2, rng=2.0))
    )

    np.testing.assert_allclose(model.covariance(h), expected_cov)
    np.testing.assert_allclose(model.variogram(h), model.cov0 - expected_cov)


def test_variogram_model_product_structures_multiply_covariances():
    h = np.array([0.0, 0.25, 0.5])
    model = VariogramModel()
    model.set_vgm(vtype="exp", sill=1.0, a_major=5.0)
    model.set_vgm(vtype="hol", sill=1.0, a_major=1.0, product=True)

    expected_cov = calc_cov("exp", h, rng=5.0) * calc_cov("hol", h, rng=1.0)

    np.testing.assert_allclose(model.covariance(h), expected_cov)
    np.testing.assert_allclose(model.variogram(h), 1.0 - expected_cov)


def test_variogram_model_calc_variogram_applies_2d_anisotropy():
    model = VariogramModel()
    model.set_vgm(
        vtype="sph",
        sill=1.0,
        a_major=10.0,
        a_minor1=5.0,
        azimuth=0.0,
    )

    north = model.calc_variogram([0.0, 0.0], [0.0, 5.0])
    east = model.calc_variogram([0.0, 0.0], [5.0, 0.0])

    np.testing.assert_allclose(north, calc_vgm("sph", 5.0, psill=1.0, rng=10.0))
    np.testing.assert_allclose(east, calc_vgm("sph", 5.0, psill=1.0, rng=5.0))
    assert north < east


def test_shared_anisotropic_lag_matches_model_coordinate_evaluation():
    lag = calc_lag_vectors([0.0, 0.0], [[0.0, 5.0], [5.0, 0.0]])
    distance = calc_anisotropic_lag(
        lag,
        anis1=0.5,
        azimuth=0.0,
    )
    np.testing.assert_allclose(distance, [5.0, 10.0])

    model = VariogramModel().set_vgm(
        vtype="sph",
        sill=1.0,
        a_major=10.0,
        a_minor1=5.0,
        azimuth=0.0,
    )
    expected = calc_vgm("sph", distance, rng=10.0)
    actual = model.calc_variogram(
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 5.0], [5.0, 0.0]],
    )
    np.testing.assert_allclose(actual, expected)


def test_raw_vgm_preserves_physical_and_anisotropic_distances():
    cloud = raw_vgm(
        [[0.0, 0.0], [0.0, 4.0], [2.0, 0.0]],
        [0.0, 1.0, 2.0],
        cutoff=5.0,
        anisotropy={"anis1": 0.5, "azimuth": 0.0},
        verbose=False,
    )

    assert "anisotropic_distance" in cloud
    north = cloud.loc[(cloud["d0"] == 0.0) & (cloud["d1"] == 4.0)].iloc[0]
    east = cloud.loc[(cloud["d0"] == 2.0) & (cloud["d1"] == 0.0)].iloc[0]
    assert north["distance"] == 4.0
    assert north["anisotropic_distance"] == 4.0
    assert east["distance"] == 2.0
    assert east["anisotropic_distance"] == 4.0


def test_spacetime_variogram_between_applies_stored_anisotropy():
    model = VariogramModel()
    model.set_spacetime_anisotropy(anis1=0.5, azimuth=0.0)
    model.set_spacetime_params(
        (0.10, 0.06, -0.005, 10.0, 2.0),
        spatial_vtype="sph",
        temporal_vtype="gau",
    )

    north = model.calc_spacetime_variogram_between(
        [0.0, 0.0], [0.0, 5.0], 0.0, 0.0
    )
    east = model.calc_spacetime_variogram_between(
        [0.0, 0.0], [5.0, 0.0], 0.0, 0.0
    )
    assert north < east
    specs = model.to_spacetime_kriging_specs()
    assert specs["spatial_spec"]["a_major"] == 10.0
    assert specs["spatial_spec"]["a_minor1"] == 5.0


def test_variogram_model_calc_covariance_pairwise_matrix():
    model = VariogramModel()
    model.set_vgm(vtype="exp", sill=1.0, a_major=2.0)

    cov = model.calc_covariance(
        [[0.0], [1.0]],
        [[0.0], [2.0], [4.0]],
        pairwise=True,
    )

    assert cov.shape == (2, 3)
    np.testing.assert_allclose(cov[0, 0], model.cov0)
    np.testing.assert_allclose(cov[0, 1], calc_cov("exp", 2.0, rng=2.0))
    assert model.calc_covariance([[0.0]], [[0.0], [1.0]], pairwise=True).shape == (1, 2)


def test_variogram_model_append_false_replaces_structures():
    model = VariogramModel()
    model.set_vgm(vtype="sph", sill=0.5, a_major=1.0)
    model.set_vgm(vtype="exp", sill=0.2, a_major=2.0, append=False)

    specs = model.to_kriging_specs()
    assert len(specs) == 1
    assert specs[0]["vtype"] == "exp"
    assert specs[0]["sill"] == 0.2


def test_variogram_model_set_params_manually_updates_curve_and_state():
    h = np.array([2.5, 5.0])
    model = VariogramModel()
    model.set_vgm(vtype="sph", nugget=0.0, sill=1.0, a_major=10.0)

    model.set_params([2.0, 5.0, 0.2])

    np.testing.assert_allclose(model.params_, [2.0, 5.0, 0.2])
    assert model.pcov_ is None
    assert model.fitted_model_ is model
    np.testing.assert_allclose(
        model.variogram(h),
        calc_vgm("sph", h, psill=2.0, rng=5.0, nugget=0.2),
    )


def test_variogram_model_set_structure_params_updates_anisotropy():
    model = VariogramModel()
    model.set_vgm(vtype="sph", sill=1.0, a_major=10.0, a_minor1=10.0)

    model.set_structure_params(0, a_minor1=5.0, azimuth=0.0)

    north = model.calc_variogram([0.0, 0.0], [0.0, 5.0])
    east = model.calc_variogram([0.0, 0.0], [5.0, 0.0])
    assert north < east
    np.testing.assert_allclose(model.params_, [1.0, 10.0, 0.0])


def test_variogram_model_set_anisotropy_applies_ratio_to_nested_structures():
    model = VariogramModel()
    model.set_vgm(vtype="sph", sill=0.5, a_major=100.0)
    model.set_vgm(vtype="sph", sill=0.3, a_major=500.0)

    model.set_anisotropy(ratio_minor1=0.2, azimuth=90.0)

    specs = model.to_kriging_specs()
    np.testing.assert_allclose([spec["a_minor1"] for spec in specs], [20.0, 100.0])
    np.testing.assert_allclose([spec["azimuth"] for spec in specs], [90.0, 90.0])

    model.set_params([0.4, 120.0, 0.2, 600.0, 0.01])
    model.set_anisotropy(anis1=0.2)
    specs = model.to_kriging_specs()
    np.testing.assert_allclose([spec["a_minor1"] for spec in specs], [24.0, 120.0])


def test_variogram_model_calc_directional_average_uses_fixed_axes():
    model = VariogramModel()
    model.set_vgm(vtype="sph", sill=1.0, a_major=20.0, a_minor1=5.0, azimuth=90.0)
    raw = pd.DataFrame({
        "d0": [10.0, 20.0, 0.0, 0.0],
        "d1": [0.0, 0.0, 10.0, 20.0],
        "distance": [10.0, 20.0, 10.0, 20.0],
        "variogram": [0.1, 0.2, 0.8, 0.9],
    })

    avg = model.calc_directional_average(raw, h_width=10.0, angle_tol=5.0)

    major = avg.loc[avg["axis"] == "major", "variogram"].to_numpy()
    minor = avg.loc[avg["axis"] == "minor1", "variogram"].to_numpy()
    np.testing.assert_allclose(major, [0.1, 0.2])
    np.testing.assert_allclose(minor, [0.8, 0.9])


def test_variogram_model_fit_anisotropy_recovers_major_and_minor_ranges():
    truth = VariogramModel()
    truth.set_vgm(
        vtype="sph",
        sill=0.8,
        a_major=100.0,
        a_minor1=25.0,
        azimuth=30.0,
    )
    h = np.linspace(5.0, 95.0, 20)
    directions, _ = truth._principal_directions(dim=2)
    directional = pd.DataFrame({
        "direction": np.repeat([0, 1], len(h)),
        "lag": np.tile(h, 2),
        "count": 50.0,
    })
    coord0 = np.zeros((len(directional), 2))
    dir_index = directional["direction"].to_numpy()
    lag = directional["lag"].to_numpy()
    coord1 = directions[dir_index] * lag[:, None]
    directional["variogram"] = truth.calc_variogram(coord0, coord1)

    model = VariogramModel()
    model.set_vgm(
        vtype="sph",
        sill=0.5,
        a_major=80.0,
        a_minor1=40.0,
        azimuth=30.0,
    )

    fitted, _, params = model.fit_anisotropy(
        directional,
        p0=(0.5, 80.0, 40.0),
        fit_nugget=False,
        return_params=True,
    )

    np.testing.assert_allclose(params, [0.8, 100.0, 25.0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fitted.structures[0].a_major, 100.0, rtol=1e-5)
    np.testing.assert_allclose(fitted.structures[0].a_minor1, 25.0, rtol=1e-5)


def test_variogram_model_apply_to_kriging_replays_specs():
    model = VariogramModel()
    model.set_vgm(vtype="exp", sill=1.0, a_major=5.0)
    model.set_vgm(vtype="hol", sill=1.0, a_major=1.0, product=True)

    k = Kriging(ndim=2, nvar=1, verbose=0)
    k.set_obs(
        ivar=1,
        coord=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        value=np.array([1.0, 2.0, 1.5]),
        nmax=3,
    )
    model.apply_to(k, ivar=1, jvar=1)
    k.set_grid(coord=np.array([[0.5, 0.5]]))
    k.set_search(ivar=1)
    k.solve()
    est, var = k.get_results()

    assert k._nvgm_struct[0, 0] == 2
    assert np.isfinite(est[0])
    assert np.isfinite(var[0])


def test_variogram_model_apply_temporal_to_replays_product_specs():
    class TemporalRecorder:
        def __init__(self):
            self.calls = []

        def set_vgm_temporal(self, **kwargs):
            self.calls.append(kwargs)

    model = VariogramModel()
    model.set_vgm(vtype="gau", nugget=0.2, sill=3.0, a_major=25.0)
    model.set_vgm(vtype="hol", sill=1.0, a_major=0.5, product=True)

    recorder = TemporalRecorder()
    returned = model.apply_temporal_to(recorder, ivar=1, jvar=1)

    assert returned is recorder
    assert recorder.calls == [
        {
            "ivar": 1, "jvar": 1, "vtype": "gau",
            "nugget": 0.2, "sill": 3.0, "at_k": 25.0,
            "product": False,
        },
        {
            "ivar": 1, "jvar": 1, "vtype": "hol",
            "nugget": 0.0, "sill": 1.0, "at_k": 0.5,
            "product": True,
        },
    ]


def test_variogram_model_fit_spacetime_product_sum_and_convert_specs():
    truth = np.array([0.10, 0.06, -0.005, 5000.0, 9.0])
    hs, ht = np.meshgrid(
        np.linspace(250.0, 5250.0, 11),
        np.linspace(0.25, 12.75, 14),
    )
    gs = calc_vgm("sph", hs.ravel(), rng=truth[3])
    gt = calc_vgm("gau", ht.ravel(), rng=truth[4])
    gamma = truth[0] * gs + truth[1] * gt + truth[2] * gs * gt
    avg = pd.DataFrame({
        ("distance", "mean"): hs.ravel(),
        ("time_lag", "mean"): ht.ravel(),
        ("variogram", "mean"): gamma,
        ("variogram", "count"): 100.0,
    })

    model = VariogramModel()
    returned = model.fit_spacetime_product_sum(
        avg,
        starts=[(0.09, 0.05, -0.003, 4500.0, 8.0)],
        bounds=[
            (0.01, 0.20),
            (0.01, 0.15),
            (-0.05, 0.0),
            (1000.0, 8000.0),
            (1.0, 20.0),
        ],
    )

    assert returned is model
    assert model.spacetime_fit_result_.success
    assert len(model.spacetime_fit_results_) == 1
    np.testing.assert_allclose(
        model.calc_spacetime_variogram(hs, ht),
        gamma.reshape(hs.shape),
        rtol=2e-3,
        atol=2e-4,
    )

    specs = model.to_spacetime_kriging_specs(z_scale=5.0)
    assert specs["model"] == "product_sum"
    assert specs["spatial_spec"]["vtype"] == "sph"
    assert specs["temporal_spec"]["vtype"] == "gau"
    np.testing.assert_allclose(
        specs["spatial_spec"]["a_minor2"],
        model.spacetime_params_[3] / 5.0,
    )
    assert specs["k_ps"] >= 0.0
    assert specs["time_at"] > 0.0


def test_variogram_model_set_spacetime_params_validates_marginal_sills():
    model = VariogramModel()
    model.set_spacetime_params(
        [0.10, 0.06, -0.005, 5000.0, 9.0],
        spatial_vtype="sph",
        temporal_vtype="gau",
    )
    np.testing.assert_allclose(
        model.spacetime_params_,
        [0.10, 0.06, -0.005, 5000.0, 9.0],
    )

    with np.testing.assert_raises(ValueError):
        model.set_spacetime_params([0.10, 0.004, -0.005, 5000.0, 9.0])


def test_fit_vgm_returns_variogram_model_from_dict_template():
    h = np.linspace(0.2, 8.0, 30)
    true_model = VariogramModel()
    true_model.set_vgm(vtype="sph", nugget=0.1, sill=2.0, a_major=5.0)
    y = true_model.variogram(h)
    avg = pd.DataFrame({("distance", "mean"): h, ("variogram", "mean"): y})

    p, _, fitted = fit_vgm(
        avg,
        models=({"vtype": "sph"},),
        p0=(1.5, 4.0, 0.0),
        return_model=True,
    )

    np.testing.assert_allclose(p, [2.0, 5.0, 0.1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fitted.variogram(h), y, rtol=1e-6, atol=1e-6)


def test_fit_vgm_preserves_product_structure_template():
    h = np.linspace(0.2, 2.0, 30)
    template = (
        {"vtype": "exp"},
        {"vtype": "hol", "product": True},
    )
    true_model = VariogramModel()
    true_model.set_vgm(vtype="exp", sill=1.0, a_major=5.0)
    true_model.set_vgm(vtype="hol", sill=1.0, a_major=1.0, product=True)
    y = true_model.variogram(h)
    avg = pd.DataFrame({("distance", "mean"): h, ("variogram", "mean"): y})

    _, _, fitted = fit_vgm(
        avg,
        models=template,
        p0=(1.0, 4.5, 1.0, 1.2),
        return_model=True,
        maxfev=20000,
    )

    assert fitted.to_kriging_specs()[1]["product"]
    np.testing.assert_allclose(fitted.variogram(h), y, rtol=1e-5, atol=1e-5)


def test_fit_vgm_weight_col_matches_equivalent_sigma_col():
    h = np.linspace(0.2, 8.0, 30)
    true_model = VariogramModel()
    true_model.set_vgm(vtype="sph", nugget=0.1, sill=2.0, a_major=5.0)
    weights = np.linspace(1.0, 10.0, len(h))
    avg = pd.DataFrame({
        ("distance", "mean"): h,
        ("variogram", "mean"): true_model.variogram(h) + 0.02 * np.sin(h),
        ("variogram", "count"): weights,
        ("variogram", "sigma"): 1.0 / np.sqrt(weights),
    })

    p_weight, _ = fit_vgm(
        avg,
        models=({"vtype": "sph"},),
        p0=(1.5, 4.0, 0.0),
        weight_col=("variogram", "count"),
    )
    p_sigma, _ = fit_vgm(
        avg,
        models=({"vtype": "sph"},),
        p0=(1.5, 4.0, 0.0),
        sigma_col=("variogram", "sigma"),
    )

    np.testing.assert_allclose(p_weight, p_sigma, rtol=1e-10, atol=1e-10)


def test_variogram_model_set_obs_experimental_and_average():
    model = VariogramModel()
    model.set_obs(
        coord=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        value=np.array([0.0, 1.0, 2.0]),
    )

    cloud = model.experimental(calc_angle=True, verbose=False)
    avg = model.average(h_width=10.0)

    assert len(cloud) == 3
    assert model._raw is cloud
    assert model._avg is avg
    assert model.raw_variogram_ is cloud
    assert model.avg_variogram_ is avg
    assert avg[("variogram", "count")].sum() == 3


def test_variogram_model_verb_style_aliases():
    model = VariogramModel()
    model.set_obs(
        coord=np.array([[0.0], [1.0], [2.0]]),
        value=np.array([0.0, 1.0, 3.0]),
    )

    cloud = model.calc_experimental(verbose=False)
    avg = model.calc_average(h_width=10.0)

    assert model.raw_variogram_ is cloud
    assert model.avg_variogram_ is avg


def test_variogram_model_fit_method_returns_fitted_model():
    h = np.linspace(0.2, 8.0, 30)
    true_model = VariogramModel()
    true_model.set_vgm(vtype="sph", nugget=0.1, sill=2.0, a_major=5.0)
    avg = pd.DataFrame({
        ("distance", "mean"): h,
        ("variogram", "mean"): true_model.variogram(h),
    })

    template = VariogramModel()
    template.set_vgm(vtype="sph", nugget=0.0, sill=1.5, a_major=4.0)
    fitted, _, params = template.fit(avg, return_params=True)

    assert fitted is not template
    assert template._params is template.params_
    assert template._pcov is template.pcov_
    assert template.fitted_model_ is fitted
    np.testing.assert_allclose(fitted.params_, params)
    np.testing.assert_allclose(params, [2.0, 5.0, 0.1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fitted.variogram(h), true_model.variogram(h), rtol=1e-6, atol=1e-6)


def test_variogram_model_fit_inplace_updates_self():
    h = np.linspace(0.2, 8.0, 30)
    true_model = VariogramModel()
    true_model.set_vgm(vtype="sph", nugget=0.1, sill=2.0, a_major=5.0)
    avg = pd.DataFrame({
        ("distance", "mean"): h,
        ("variogram", "mean"): true_model.variogram(h),
    })

    template = VariogramModel()
    template.set_vgm(vtype="sph", nugget=0.0, sill=1.5, a_major=4.0)
    fitted, _ = template.fit(avg, inplace=True)

    assert fitted is template
    assert template.fitted_model_ is template
    np.testing.assert_allclose(template.params_, [2.0, 5.0, 0.1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(template.variogram(h), true_model.variogram(h), rtol=1e-6, atol=1e-6)


def test_variogram_model_plot_uses_cached_average():
    model = VariogramModel()
    model.set_obs(
        coord=np.array([[0.0], [1.0], [2.0]]),
        value=np.array([0.0, 1.0, 3.0]),
    )
    model.set_vgm(vtype="sph", sill=1.0, a_major=2.0)
    model.calc_experimental(verbose=False)
    model.calc_average(h_width=10.0)

    ax = model.plot()

    assert ax.get_xlabel() == "Lag"
    assert ax.get_ylabel() == "Semivariogram"
    plt.close(ax.figure)


def test_variogram_model_plot_map_uses_cached_raw_and_model_angle():
    model = VariogramModel()
    model.set_vgm(vtype="sph", sill=1.0, a_major=40.0, a_minor1=10.0, azimuth=90.0)
    model.raw_variogram_ = pd.DataFrame({
        "d0": [-20.0, -10.0, 10.0, 20.0, 0.0, 0.0],
        "d1": [0.0, 0.0, 0.0, 0.0, -10.0, 10.0],
        "distance": [20.0, 10.0, 10.0, 20.0, 10.0, 10.0],
        "variogram": [0.2, 0.1, 0.1, 0.2, 0.8, 0.8],
    })

    fig, ax = plt.subplots()
    out = model.plot_map(dx=10.0, dy=10.0, ax=ax, show_cbar=False)

    assert out is ax
    assert len(ax.lines) >= 3
    assert len(ax.patches) == 1
    assert "90.0" in ax.get_title()
    plt.close(fig)


def test_variogram_model_plot_map3d_uses_cached_raw_and_model_angle():
    rows = []
    for dx in (-20.0, -10.0, 0.0, 10.0, 20.0):
        for dy in (-20.0, -10.0, 0.0, 10.0, 20.0):
            for dz in (-10.0, 0.0, 10.0):
                if dx == 0.0 and dy == 0.0 and dz == 0.0:
                    continue
                distance = float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
                if distance > 30.0:
                    continue
                rows.append({
                    "d0": dx,
                    "d1": dy,
                    "d2": dz,
                    "distance": distance,
                    "angle_h": (90.0 - np.degrees(np.arctan2(dy, dx))) % 360.0,
                    "variogram": distance / 30.0,
                })

    model = VariogramModel()
    model.set_vgm(
        vtype="sph",
        sill=1.0,
        a_major=40.0,
        a_minor1=20.0,
        a_minor2=10.0,
        azimuth=90.0,
        dip=15.0,
    )
    model.raw_variogram_ = pd.DataFrame(rows)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    out = model.plot_map3d(
        dx=10.0,
        dy=10.0,
        dz=10.0,
        cutoff=30.0,
        ax=ax,
        show_cbar=False,
        rotate_fences=True,
    )

    assert out is ax
    assert ax.get_xlabel() == "X lag"
    assert "90.0" in ax.get_title()
    assert "15.0" in ax.get_title()
    plt.close(fig)


def test_variogram_model_plot_map3d_no_angle_h_required():
    """plot_map3d no longer requires angle_h; it uses direct projection."""
    model = VariogramModel()
    model.set_vgm(vtype="sph", sill=1.0, a_major=20.0, azimuth=0.0, dip=0.0)
    # Cloud without angle_h column — must not raise
    model.raw_variogram_ = pd.DataFrame({
        "d0": [5.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        "d1": [0.0, 0.0, 5.0, 10.0, 0.0, 0.0],
        "d2": [0.0, 0.0, 0.0, 0.0, 5.0, 10.0],
        "distance": [5.0, 10.0, 5.0, 10.0, 5.0, 10.0],
        "variogram": [0.3, 0.7, 0.3, 0.7, 0.3, 0.7],
    })
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    out = model.plot_map3d(ax=ax, dx=5.0, dz=5.0, show_cbar=False)
    assert out is ax
    plt.close(fig)


def test_plot_map3d_fill_nan_respects_support_mask():
    vv = np.array([
        [1.0, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ])
    support = np.array([
        [True, True, False],
        [False, True, False],
    ])

    filled = _fill_nan_nearest(vv, support_mask=support)

    assert np.isfinite(filled[0, 1])
    assert np.isfinite(filled[1, 1])
    assert np.isnan(filled[0, 2])
    assert np.isnan(filled[1, 0])
    assert np.isnan(filled[1, 2])


def test_variogram_system_calculates_direct_and_lmc_cross_clouds():
    coord = np.array([[0.0], [1.0], [2.0]])
    value1 = np.array([0.0, 1.0, 3.0])
    value2 = np.array([0.0, 2.0, 4.0])

    system = VariogramSystem(nvar=2)
    system.set_obs(1, coord, value1)
    system.set_obs(2, coord, value2)

    direct = system.calc_experimental(1, 1, verbose=False)
    cross = system.calc_experimental(1, 2, verbose=False)
    expected_cross = raw_cross_vgm(coord, value1, value2, verbose=False)

    np.testing.assert_allclose(direct["variogram"], raw_vgm(coord, value1, verbose=False)["variogram"])
    pd.testing.assert_frame_equal(cross, expected_cross)
    assert system.raw_variograms_[(1, 2)] is cross


def test_variogram_system_fit_pair_updates_pair_model():
    h = np.linspace(0.2, 8.0, 30)
    true_model = VariogramModel()
    true_model.set_vgm(vtype="sph", nugget=0.1, sill=2.0, a_major=5.0)
    avg = pd.DataFrame({
        ("distance", "mean"): h,
        ("variogram", "mean"): true_model.variogram(h),
    })

    system = VariogramSystem(nvar=1)
    system.set_vgm(1, 1, vtype="sph", nugget=0.0, sill=1.5, a_major=4.0)
    system.set_avg_vgm(1, 1, avg)
    fitted, _, params = system.fit_pair(1, 1, return_params=True)

    assert system.models[(1, 1)] is fitted
    np.testing.assert_allclose(params, [2.0, 5.0, 0.1], rtol=1e-6, atol=1e-6)


def test_variogram_system_fit_lmc_returns_psd_coregionalization_matrix():
    h = np.linspace(0.2, 8.0, 40)
    b11, b22, b12 = 2.0, 1.0, 0.8
    true_range = 5.0

    system = VariogramSystem(nvar=2)
    system.set_vgm(1, 1, vtype="sph", sill=1.6, a_major=4.0)
    system.set_vgm(2, 2, vtype="sph", sill=0.7, a_major=4.0)
    system.set_vgm(1, 2, vtype="sph", sill=0.2, a_major=4.0)
    for pair, sill in [((1, 1), b11), ((2, 2), b22), ((1, 2), b12)]:
        system.set_avg_vgm(*pair, pd.DataFrame({
            ("distance", "mean"): h,
            ("variogram", "mean"): calc_vgm("sph", h, psill=sill, rng=true_range),
            ("variogram", "count"): np.ones_like(h),
        }))

    fitted, result = system.fit_lmc(
        fit_nugget=False,
        weight_col=("variogram", "count"),
    )
    mats = fitted.get_lmc_matrices(include_nugget=False)
    eig = np.linalg.eigvalsh(mats[0])

    assert result.success
    assert np.min(eig) >= -1e-8
    np.testing.assert_allclose(mats[0], [[b11, b12], [b12, b22]], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        fitted.models[(1, 2)].variogram(h),
        calc_vgm("sph", h, psill=b12, rng=true_range),
        rtol=1e-5,
        atol=1e-5,
    )


def test_fit_lmc_enforces_psd_under_invalid_cross():
    """Even when the experimental cross-variogram is too strong for a valid LMC
    (|b12| > sqrt(b11*b22), as happens with a strongly correlated indicator +
    covariate), fit_lmc must return a positive-semidefinite coregionalization —
    the L@L.T parameterization clamps the cross-sill to a valid value."""
    h = np.linspace(0.2, 8.0, 40)
    rng = 5.0
    b11, b22, b12_bad = 1.0, 0.25, 0.8     # sqrt(b11*b22)=0.5 < 0.8 -> invalid

    # the naive (independent per-pair) matrix is indefinite
    naive = np.array([[b11, b12_bad], [b12_bad, b22]])
    assert np.min(np.linalg.eigvalsh(naive)) < 0.0

    system = VariogramSystem(nvar=2)
    system.set_vgm(1, 1, vtype="sph", sill=b11, a_major=rng)
    system.set_vgm(2, 2, vtype="sph", sill=b22, a_major=rng)
    system.set_vgm(1, 2, vtype="sph", sill=0.3, a_major=rng)
    for pair, sill in [((1, 1), b11), ((2, 2), b22), ((1, 2), b12_bad)]:
        system.set_avg_vgm(*pair, pd.DataFrame({
            ("distance", "mean"): h,
            ("variogram", "mean"): calc_vgm("sph", h, psill=sill, rng=rng),
            ("variogram", "count"): np.ones_like(h),
        }))

    fitted, result = system.fit_lmc(fit_nugget=False,
                                    weight_col=("variogram", "count"))
    B = fitted.get_lmc_matrices(include_nugget=False)[0]

    assert result.success
    assert np.min(np.linalg.eigvalsh(B)) >= -1e-8          # valid (PSD)
    assert abs(B[0, 1]) <= np.sqrt(B[0, 0] * B[1, 1]) + 1e-6  # Cauchy-Schwarz holds
    assert abs(B[0, 1]) < b12_bad                           # cross was pulled down


def test_fit_lmc_three_variable_nested_recovers_and_is_psd():
    """3 variables, 2 nested structures: recover both PSD sill matrices."""
    h = np.linspace(0.2, 12.0, 50)
    a0, a1 = 4.0, 20.0
    B0 = np.array([[1.0, 0.5, 0.3], [0.5, 0.9, 0.2], [0.3, 0.2, 0.7]])
    B1 = np.array([[0.5, 0.1, -0.1], [0.1, 0.4, 0.15], [-0.1, 0.15, 0.3]])
    assert np.min(np.linalg.eigvalsh(B0)) >= 0 and np.min(np.linalg.eigvalsh(B1)) >= 0

    system = VariogramSystem(nvar=3)
    for i in range(1, 4):
        for j in range(i, 4):
            system.set_vgm(i, j, vtype="sph", sill=0.5, a_major=a0)
            system.set_vgm(i, j, vtype="exp", sill=0.2, a_major=a1, append=True)
            curve = (calc_vgm("sph", h, psill=B0[i - 1, j - 1], rng=a0)
                     + calc_vgm("exp", h, psill=B1[i - 1, j - 1], rng=a1))
            system.set_avg_vgm(i, j, pd.DataFrame({
                ("distance", "mean"): h,
                ("variogram", "mean"): curve,
                ("variogram", "count"): np.ones_like(h),
            }))

    fitted, result = system.fit_lmc(fit_nugget=False, fit_ranges=False,
                                    weight_col=("variogram", "count"))
    mats = fitted.get_lmc_matrices(include_nugget=False)

    assert result.success
    np.testing.assert_allclose(mats[0], B0, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(mats[1], B1, rtol=1e-4, atol=1e-4)
    for B in mats:
        assert np.min(np.linalg.eigvalsh(B)) >= -1e-8


def test_set_markov_cross_formula_and_psd():
    """MM1 cross-sill is rho*sqrt(s_p*s_s) and the coregionalization is PSD."""
    s_p, s_s, rho = 0.25, 0.05, 0.9
    system = VariogramSystem(nvar=2)
    system.set_vgm(1, 1, vtype="exp", sill=s_p, a_major=6500.0)
    system.set_vgm(2, 2, vtype="exp", sill=s_s, a_major=6500.0)
    system.set_markov_cross(1, 2, corr=rho)

    cross = system.models[(1, 2)].structures[0].sill
    np.testing.assert_allclose(cross, rho * np.sqrt(s_p * s_s), rtol=1e-12)
    B = np.array([[s_p, cross], [cross, s_s]])
    assert np.min(np.linalg.eigvalsh(B)) >= -1e-12          # valid (PSD)
    # cross adopts the secondary's range
    assert system.models[(1, 2)].structures[0].a_major == 6500.0


def test_set_markov_cross_estimates_corr_from_collocated_obs():
    """With corr=None the collocated correlation is estimated from the data."""
    rng = np.random.default_rng(0)
    coord = rng.uniform(0.0, 100.0, size=(200, 2))
    a = rng.normal(size=200)
    b = 0.7 * a + 0.3 * rng.normal(size=200)          # correlated secondary
    rho = float(np.corrcoef(a, b)[0, 1])

    system = VariogramSystem(nvar=2)
    system.set_obs(1, coord, a)
    system.set_obs(2, coord, b)
    system.set_vgm(1, 1, vtype="exp", sill=float(np.var(a)), a_major=30.0)
    system.set_vgm(2, 2, vtype="exp", sill=float(np.var(b)), a_major=30.0)
    system.set_markov_cross(1, 2)                      # corr estimated

    np.testing.assert_allclose(system.markov_corr_[(1, 2)], rho, rtol=1e-10)
    expected = rho * np.sqrt(np.var(a) * np.var(b))
    np.testing.assert_allclose(system.models[(1, 2)].structures[0].sill,
                               expected, rtol=1e-10)


def test_set_markov_cross_stays_valid_where_strong_correlation():
    """Even at strong correlation the cross stays inside the PSD bound, unlike
    feeding a raw covariance computed against a mismatched variance."""
    s_p, s_s = 0.25, 0.05
    system = VariogramSystem(nvar=2)
    system.set_vgm(1, 1, vtype="exp", sill=s_p, a_major=6500.0)
    system.set_vgm(2, 2, vtype="exp", sill=s_s, a_major=6500.0)
    system.set_markov_cross(1, 2, corr=0.95)

    cross = system.models[(1, 2)].structures[0].sill
    assert abs(cross) <= np.sqrt(s_p * s_s) + 1e-12      # never exceeds the bound


def test_great_circle_distance_handles_antipodal_points():
    dist = _great_circle_dist(
        np.array([[0.0, 0.0]]),
        np.array([[180.0, 0.0]]),
    )
    np.testing.assert_allclose(dist, np.pi * 6371.2)


def _fortran_calc_rotmat(ang1, ang2, ang3, anis1, anis2):
    """Exact transcription of src/libkriging/rotation.f90 ``calc_rotmat``."""
    d2r = np.pi / 180.0
    rotmat = np.zeros((3, 3))
    rotmat[0, 0] = 1.0 / anis1
    rotmat[1, 1] = 1.0
    rotmat[2, 2] = 1.0 / anis2
    a = -ang3 * d2r          # rotate about Y (plunge)
    s, c = np.sin(a), np.cos(a)
    rotmat = rotmat @ np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    a = ang2 * d2r           # rotate about X (dip, positive down)
    s, c = np.sin(a), np.cos(a)
    rotmat = rotmat @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    a = ang1 * d2r           # rotate about Z (azimuth)
    s, c = np.sin(a), np.cos(a)
    rotmat = rotmat @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rotmat


def test_anisotropic_hr_matches_fortran_rotation_with_plunge():
    # Guards against the plunge-sign regression: the Python preview must agree
    # with the Fortran engine's calc_rotmat for arbitrary azimuth/dip/plunge,
    # which only bites when plunge != 0 and a_minor1 != a_minor2.
    a_major, a_minor1, a_minor2 = 100.0, 40.0, 10.0
    rng = np.random.default_rng(1)
    lags = rng.normal(0.0, 50.0, size=(6, 3))
    for az, dip, plunge in [(40, 25, 15), (120, -30, 60), (75, 10, -45)]:
        model = VariogramModel().set_vgm(
            "sph", a_major=a_major, a_minor1=a_minor1, a_minor2=a_minor2,
            azimuth=az, dip=dip, plunge=plunge,
        )
        comp = model.structures[0]
        rotmat = _fortran_calc_rotmat(az, dip, plunge,
                                      a_minor1 / a_major, a_minor2 / a_major)
        py = np.array([model._anisotropic_hr(comp, lag) for lag in lags])
        fortran = np.array([np.sqrt(np.sum((rotmat @ lag) ** 2)) / a_major
                            for lag in lags])
        np.testing.assert_allclose(py, fortran, atol=1e-12)


def test_directional_filter_treats_opposite_azimuths_as_same_axis():
    assert filter_vgm(
        dx=np.array([-1.0]),
        dy=np.array([0.0]),
        azimuth=np.array([270.0]),
        azimuth_anis=90.0,
        angle_tol=5.0,
    )[0]


def test_avg_vgm_directional_filter_keeps_opposite_pairs():
    cloud = raw_vgm(
        np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
        np.array([0.0, 1.0, 1.0]),
        calc_angle=True,
        verbose=False,
    )
    avg = avg_vgm(cloud, h_width=10.0, angleh=90.0, angleh_tor=5.0)

    assert avg[("variogram", "count")].sum() == len(cloud)


def test_avg_vgm_3d_axis_filter_flips_dip_with_opposite_azimuth():
    cloud = raw_vgm(
        np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],    # azimuth 90, dip -45 (up-going: negative down-dip)
            [-1.0, 0.0, -1.0],  # azimuth 270, dip +45: same axis reversed
            [-1.0, 0.0, 1.0],   # azimuth 270, dip -45: different axis
        ]),
        np.array([0.0, 1.0, 1.0, 2.0]),
        calc_angle=True,
        verbose=False,
    )
    first_node_pairs = cloud[cloud["index0"] == 0]

    avg = avg_vgm(
        first_node_pairs,
        h_width=10.0,
        angleh=90.0,
        anglev=-45.0,
        angleh_tor=1.0,
        anglev_tor=1.0,
    )

    assert avg[("variogram", "count")].sum() == 2


def test_cross_vgm_accepts_1d_coordinates():
    cloud = cross_vgm(
        coordsA=np.array([0.0, 1.0]),
        valsA=np.array([0.0, 1.0]),
        coordsB=np.array([0.0, 2.0]),
        valsB=np.array([0.0, 2.0]),
        verbose=False,
    )

    assert len(cloud) == 4
    assert set(["index0", "index1", "distance", "variogram", "d0"]).issubset(cloud.columns)


def test_directional_vgm_per_axis_bin_width():
    """Each direction should receive h_bins bins regardless of axis length.

    With a global h_width the short (Y) axis would get only ~1 bin while the
    long (X) axis fills h_bins; per-axis width gives equal resolution on both.
    """
    # Major axis along X, range 100.  Minor axis along Y, range 10.
    # Construct axis-aligned pairs: 50 along X (lag 2..100), 50 along Y (lag 0.2..10).
    n = 50
    h_bins = 5
    raw = pd.DataFrame({
        "d0": np.concatenate([np.linspace(2.0, 100.0, n), np.zeros(n)]),
        "d1": np.concatenate([np.zeros(n), np.linspace(0.2, 10.0, n)]),
        "distance": np.concatenate([np.linspace(2.0, 100.0, n),
                                     np.linspace(0.2, 10.0, n)]),
        "variogram": np.ones(2 * n) * 0.5,
    })
    directions = np.array([[1.0, 0.0], [0.0, 1.0]])  # X, Y

    avg = directional_vgm(raw, directions, h_bins=h_bins, angle_tol=5.0)

    # Each direction should fill ~h_bins bins (±1 for boundary pairs).
    for j, name in enumerate(["X", "Y"]):
        sub = avg[avg["direction"] == j]
        assert h_bins <= len(sub) <= h_bins + 1, (
            f"{name} axis: expected ~{h_bins} bins, got {len(sub)}.  "
            "Per-axis h_width not applied."
        )

    # The lag column must reflect the projected (short) axis range, not the long one.
    minor_max_lag = avg.loc[avg["direction"] == 1, "lag"].max()
    assert minor_max_lag < 15.0, (
        f"Minor-axis max lag {minor_max_lag:.1f} should be ~10, not the major-axis range. "
        "Check that hh uses projected lag."
    )
