"""Tests for the symmetric 1-based ``system.vgm`` accessor."""

import numpy as np
import pytest

from krigekit import VariogramSystem, VgmStructure


def _configured(system, ivar, jvar, **kwargs):
    """Configure a pair through the accessor and return its structure."""
    kwargs.setdefault("sill", 1.0)
    kwargs.setdefault("a_major", 10.0)
    return system.vgm[ivar, jvar].set_vgm("sph", **kwargs)


def test_symmetric_object_identity():
    system = VariogramSystem(nvar=2)
    assert system.vgm[1, 2] is system.vgm[2, 1]
    assert system.vgm[1, 1] is system.vgm[1, 1]


def test_reading_a_pair_does_not_configure_it():
    system = VariogramSystem(nvar=2)
    _ = system.vgm[1, 2]                      # materialize by reading
    assert system.vgm.nmaterialized == 1
    assert system.vgm.nconfigured == 0
    assert system.vgm.configured_pairs() == []

    _configured(system, 1, 2)
    assert system.vgm.configured_pairs() == [(1, 2)]
    assert system.vgm.nconfigured == 1


def test_get_create_false_returns_none_for_unmaterialized():
    system = VariogramSystem(nvar=2)
    assert system.vgm.get(1, 2, create=False) is None
    assert system.vgm.nmaterialized == 0      # peeking did not materialize
    _configured(system, 1, 2)
    assert system.vgm.get(1, 2, create=False) is system.vgm[1, 2]


def test_auto_pair_via_get_and_subscript():
    system = VariogramSystem(nvar=2)
    _configured(system, 1, 1)
    assert system.vgm.get(1) is system.vgm[1, 1]


@pytest.mark.parametrize("bad", [(0, 1), (1, 0), (-1, 1)])
def test_non_positive_indices_teach_the_convention(bad):
    system = VariogramSystem(nvar=2)
    with pytest.raises(ValueError, match="1-based"):
        _ = system.vgm[bad]


def test_non_integer_and_malformed_indices():
    system = VariogramSystem(nvar=2)
    with pytest.raises(TypeError):
        _ = system.vgm[1.5, 1]
    with pytest.raises(TypeError):
        _ = system.vgm[1]                     # single index, not a pair


def test_fixed_nvar_is_a_strict_upper_bound():
    system = VariogramSystem(nvar=2)
    with pytest.raises(ValueError, match="exceeds nvar"):
        _ = system.vgm[3, 1]


def test_dynamic_nvar_grows_to_largest_index():
    system = VariogramSystem()                # dynamic
    _configured(system, 1, 1)
    assert system.nvar == 1
    _configured(system, 3, 2)                 # canonical (2, 3)
    assert system.nvar == 3
    assert system.vgm.configured_pairs() == [(1, 1), (2, 3)]


def test_setitem_assigns_structure_and_preserves_symmetry():
    system = VariogramSystem(nvar=2)
    struct = VgmStructure().set_vgm("sph", sill=0.4, a_major=300.0)
    system.vgm[1, 2] = struct
    assert system.vgm[2, 1] is struct
    assert system.vgm.configured_pairs() == [(1, 2)]

    with pytest.raises(TypeError, match="VgmStructure"):
        system.vgm[1, 2] = "not a structure"


def test_clear_entry_and_clear_components():
    system = VariogramSystem(nvar=2)
    _configured(system, 1, 2)

    # clearing the components unconfigures but leaves the pair materialized
    system.vgm[1, 2].clear()
    assert system.vgm.configured_pairs() == []
    assert system.vgm.nmaterialized == 1

    _configured(system, 1, 2)
    # clearing the entry removes it entirely
    system.vgm.clear(1, 2)
    assert system.vgm.nmaterialized == 0


def test_accessor_shares_storage_with_models_compat():
    system = VariogramSystem(nvar=2)
    _configured(system, 1, 2)
    # the legacy models view and the accessor see the same structure
    assert system.models[(1, 2)].structure is system.vgm[1, 2]
    np.testing.assert_array_equal(
        system.vgm[1, 2].covariance([0.0, 5.0]),
        system.models[(1, 2)].structure.covariance([0.0, 5.0]),
    )
