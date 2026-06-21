"""Tests for the flat theoretical variogram component."""

import numpy as np
import pytest

from krigekit.variogram_component import VgmComponent
from krigekit.variogram_kernels import calc_cov


def test_component_normalizes_model_and_default_ranges():
    comp = VgmComponent("spherical", a_major=10.0)

    assert comp.vtype == "sph"
    assert comp.a_minor1 == 10.0
    assert comp.a_minor2 == 10.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a_major": 0.0},
        {"a_minor1": -1.0},
        {"a_minor2": 0.0},
        {"sill": np.nan},
        {"azimuth": np.inf},
    ],
)
def test_component_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        VgmComponent("sph", **kwargs)


def test_flat_dict_round_trip():
    comp = VgmComponent(
        "exp",
        nugget=0.1,
        sill=0.8,
        a_major=500.0,
        a_minor1=200.0,
        a_minor2=100.0,
        azimuth=45.0,
        dip=10.0,
        plunge=5.0,
        product=True,
    )

    restored = VgmComponent.from_flat_dict(comp.to_flat_dict())

    assert restored == comp
    with pytest.raises(TypeError, match="unknown VgmComponent"):
        VgmComponent.from_flat_dict({"vtype": "sph", "unexpected": 1})


def test_set_anisotropy_supports_ratios_and_is_transactional():
    comp = VgmComponent("sph", a_major=10.0)
    comp.set_anisotropy(a_major=20.0, anis1=0.5, anis2=0.25, azimuth=30.0)

    assert comp.a_major == 20.0
    assert comp.a_minor1 == 10.0
    assert comp.a_minor2 == 5.0
    assert comp.azimuth == 30.0

    before = comp.to_flat_dict()
    with pytest.raises(ValueError):
        comp.set_anisotropy(ratio_minor1=0.0)
    assert comp.to_flat_dict() == before


def test_anisotropic_distance_uses_component_ranges():
    comp = VgmComponent(
        "sph",
        a_major=10.0,
        a_minor1=5.0,
        a_minor2=10.0,
    )

    np.testing.assert_allclose(
        comp.calc_anisotropic_distance([[0.0, 5.0], [5.0, 0.0]]),
        [5.0, 10.0],
    )


def test_covariance_matches_kernel_and_adds_nugget_only_at_zero():
    comp = VgmComponent("exp", nugget=0.2, sill=0.8, a_major=10.0)
    distance = np.array([0.0, 2.0, 20.0])
    expected = calc_cov("exp", distance, psill=0.8, rng=10.0)
    expected[0] = 1.0

    np.testing.assert_allclose(comp.calc_covariance(distance), expected)
    np.testing.assert_allclose(
        comp.calc_variogram(distance),
        comp.cov0 - expected,
    )


def test_name_is_metadata_excluded_from_engine_transfer():
    comp = VgmComponent("sph", a_major=10.0, name="long range")
    assert comp.name == "long range"
    assert comp.display_name == "long range"
    assert "name" not in comp.to_flat_dict()          # engine transfer drops it
    assert comp.copy().name == "long range"           # copy preserves it
    assert VgmComponent("exp").display_name == "exp"   # falls back to vtype
    # from_flat_dict still accepts an explicit name
    assert VgmComponent.from_flat_dict({"vtype": "sph", "name": "x"}).name == "x"


def test_covariance_lag_applies_anisotropy():
    comp = VgmComponent(
        "sph",
        sill=2.0,
        a_major=10.0,
        a_minor1=5.0,
        a_minor2=10.0,
    )

    north = comp.calc_covariance_lag([0.0, 5.0])
    east = comp.calc_covariance_lag([5.0, 0.0])

    assert north > east
