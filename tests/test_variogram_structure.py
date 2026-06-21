"""Tests for the theoretical variogram structure and its engine equivalence."""

import numpy as np
import pytest

from krigekit.variogram_component import VgmComponent
from krigekit.variogram_model import VariogramModel
from krigekit.variogram_structure import VgmStructure


def _matched(builder):
    """Return a (VariogramModel, VgmStructure) pair built identically."""
    model = VariogramModel()
    structure = VgmStructure()
    builder(model.set_vgm)
    builder(structure.set_vgm)
    return model, structure


def test_empty_structure_is_zero():
    structure = VgmStructure()
    assert structure.ncomponent == 0
    np.testing.assert_array_equal(structure.covariance([0.0, 1.0, 2.0]), 0.0)
    np.testing.assert_array_equal(structure.variogram([0.0, 1.0, 2.0]), 0.0)


def test_construct_from_components_and_specs():
    comp = VgmComponent("sph", sill=0.6, a_major=300.0)
    from_obj = VgmStructure([comp])
    from_spec = VgmStructure([comp.to_flat_dict()])

    assert from_obj.ncomponent == from_spec.ncomponent == 1
    np.testing.assert_allclose(
        from_obj.covariance([0.0, 100.0]),
        from_spec.covariance([0.0, 100.0]),
    )


def test_append_false_clears_existing():
    structure = VgmStructure()
    structure.set_vgm("sph", sill=0.6, a_major=300.0)
    structure.set_vgm("nug", sill=0.1, append=False)
    assert structure.ncomponent == 1
    assert structure[0].vtype == "nug"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_isotropic_evaluation_matches_model(seed):
    rng = np.random.default_rng(seed)

    def build(set_vgm):
        set_vgm("sph", nugget=0.1, sill=0.6, a_major=300.0, append=False)
        set_vgm("exp", sill=0.3, a_major=1000.0)

    model, structure = _matched(build)
    h = rng.uniform(0.0, 1500.0, size=50)
    np.testing.assert_allclose(structure.covariance(h), model.covariance(h))
    np.testing.assert_allclose(structure.variogram(h), model.variogram(h))
    np.testing.assert_allclose(structure.cov0, model.cov0)


def test_product_group_matches_model():
    def build(set_vgm):
        set_vgm("sph", sill=1.0, a_major=200.0, append=False)
        set_vgm("gau", sill=1.0, a_major=400.0, product=True)

    model, structure = _matched(build)
    h = np.linspace(0.0, 800.0, 40)
    np.testing.assert_allclose(structure.covariance(h), model.covariance(h))


@pytest.mark.parametrize("seed", [3, 4])
def test_anisotropic_coordinate_evaluation_matches_model(seed):
    rng = np.random.default_rng(seed)

    def build(set_vgm):
        set_vgm("sph", nugget=0.05, sill=0.7, a_major=500.0,
                a_minor1=200.0, azimuth=45.0, append=False)
        set_vgm("exp", sill=0.3, a_major=1000.0, a_minor1=600.0, azimuth=45.0)

    model, structure = _matched(build)
    coord0 = rng.uniform(0.0, 1000.0, size=(30, 2))
    coord1 = rng.uniform(0.0, 1000.0, size=(30, 2))

    np.testing.assert_allclose(
        structure.calc_covariance(coord0, coord1),
        model.calc_covariance(coord0, coord1),
    )
    np.testing.assert_allclose(
        structure.calc_variogram(coord0, coord1, pairwise=True),
        model.calc_variogram(coord0, coord1, pairwise=True),
    )


def test_set_structure_params_and_anisotropy():
    structure = VgmStructure()
    structure.set_vgm("sph", sill=1.0, a_major=100.0, append=False)
    structure.set_structure_params(0, sill=2.0, a_major=250.0)
    assert structure[0].sill == 2.0 and structure[0].a_major == 250.0

    structure.set_anisotropy(anis1=0.5, azimuth=30.0)
    assert structure[0].a_minor1 == 125.0
    assert structure[0].azimuth == 30.0

    with pytest.raises(TypeError):
        structure.set_structure_params(0, bogus=1.0)


def test_to_kriging_specs_replace_flag():
    structure = VgmStructure()
    structure.set_vgm("sph", sill=0.6, a_major=300.0, append=False)
    structure.set_vgm("exp", sill=0.4, a_major=900.0)

    specs = structure.to_kriging_specs(replace=True)
    assert [s["append"] for s in specs] == [False, True]
    assert specs[0]["vtype"] == "sph"
    assert "product" in specs[0]  # flat engine fields are present


def test_structure_and_component_names():
    structure = VgmStructure(name="primary")
    structure.set_vgm("sph", sill=0.6, a_major=300.0, name="long", append=False)
    structure.set_vgm("exp", sill=0.3, a_major=900.0, name="short")

    assert structure.name == "primary"
    assert [comp.name for comp in structure] == ["long", "short"]
    # engine specs carry no name metadata
    assert all("name" not in spec for spec in structure.to_kriging_specs())
    # copy preserves structure and component names
    copy = structure.copy()
    assert copy.name == "primary" and copy[0].name == "long"


def test_variogram_model_owns_structure_and_delegates():
    model = VariogramModel()
    model.set_vgm("sph", nugget=0.1, sill=0.6, a_major=300.0, append=False)
    model.set_vgm("exp", sill=0.3, a_major=900.0)

    assert isinstance(model.structure, VgmStructure)
    assert model.structure.ncomponent == 2
    # the legacy parallel list must be gone
    assert not hasattr(model, "structures")

    # evaluation delegates to the owned structure
    h = np.linspace(0.0, 1200.0, 25)
    np.testing.assert_array_equal(model.covariance(h), model.structure.covariance(h))
    np.testing.assert_array_equal(model.variogram(h), model.structure.variogram(h))


def test_apply_to_emits_engine_calls():
    class _Recorder:
        def __init__(self):
            self.calls = []

        def set_vgm(self, **kwargs):
            self.calls.append(kwargs)

    structure = VgmStructure()
    structure.set_vgm("sph", sill=0.6, a_major=300.0, append=False)
    rec = _Recorder()
    structure.apply_to(rec, ivar=1, jvar=2, replace=True)

    assert len(rec.calls) == 1
    assert rec.calls[0]["ivar"] == 1 and rec.calls[0]["jvar"] == 2
    assert rec.calls[0]["append"] is False
