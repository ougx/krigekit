"""Tests for transactional VariogramSystem.apply (obs + variogram transfer)."""

import numpy as np
import pytest

from krigekit import VariogramSystem


class _Recorder:
    """Minimal kriging stand-in capturing transfer calls in order."""

    def __init__(self, ndim=2, nvar=2, ndrift=0):
        self.ndim, self.nvar, self.ndrift = ndim, nvar, ndrift
        self.obs_calls, self.drift_calls, self.vgm_calls = [], [], []

    def set_obs(self, ivar, coord, value, **kwargs):
        self.obs_calls.append((ivar, kwargs))

    def set_obs_drift(self, ivar, drift):
        self.drift_calls.append((ivar, drift))

    def set_vgm(self, ivar, jvar, **spec):
        self.vgm_calls.append((ivar, jvar, spec))


def _coords(n=6, dim=2):
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 10.0, size=(n, dim)), rng.normal(size=n)


def _full_system():
    """A valid 2-variable system with both autos and the cross configured."""
    system = VariogramSystem(nvar=2)
    coord, value = _coords()
    system.set_obs(1, coord, value)
    system.set_obs(2, coord, value)
    system.vgm[1, 1].set_vgm("sph", sill=1.0, a_major=5.0)
    system.vgm[2, 2].set_vgm("sph", sill=1.0, a_major=5.0)
    system.vgm[1, 2].set_vgm("sph", sill=0.3, a_major=5.0)
    return system


def test_apply_transfers_obs_then_variograms_in_order():
    system = _full_system()
    k = _Recorder(nvar=2)
    system.apply(k)
    assert [ivar for ivar, _ in k.obs_calls] == [1, 2]
    assert [(i, j) for i, j, _ in k.vgm_calls] == [(1, 1), (1, 2), (2, 2)]


def test_observation_only_and_variogram_only_modes():
    system = _full_system()
    ko = _Recorder(nvar=2)
    system.apply(ko, variograms=False)
    assert ko.obs_calls and ko.vgm_calls == []

    kv = _Recorder(nvar=2)
    system.apply(kv, observations=False)
    assert kv.vgm_calls and kv.obs_calls == []


def test_pairs_filter():
    system = _full_system()
    k = _Recorder(nvar=2)
    system.apply(k, observations=False, pairs=[(1, 1)])
    assert [(i, j) for i, j, _ in k.vgm_calls] == [(1, 1)]


def test_replace_flag_controls_append():
    system = _full_system()
    for replace, expected in [(True, False), (False, True)]:
        k = _Recorder(nvar=2)
        system.apply(k, observations=False, replace=replace)
        first_11 = next(spec for i, j, spec in k.vgm_calls if (i, j) == (1, 1))
        assert first_11["append"] is expected


def test_validation_failure_leaves_target_untouched():
    # cross pair without its auto-variograms is invalid
    system = VariogramSystem(nvar=2)
    coord, value = _coords()
    system.set_obs(1, coord, value)
    system.set_obs(2, coord, value)
    system.vgm[1, 2].set_vgm("sph", sill=0.3, a_major=5.0)

    k = _Recorder(nvar=2)
    with pytest.raises(ValueError, match="auto-variogram"):
        system.apply(k)
    assert k.obs_calls == [] and k.vgm_calls == []


def test_nvar_and_ndim_bounds():
    system = _full_system()
    with pytest.raises(ValueError, match="exceeds target nvar"):
        system.apply(_Recorder(nvar=1))

    s3 = VariogramSystem(nvar=1)
    coord, value = _coords(dim=3)
    s3.set_obs(1, coord, value)
    s3.vgm[1, 1].set_vgm("sph", sill=1.0, a_major=5.0)
    with pytest.raises(ValueError, match="3D|expected"):
        s3.apply(_Recorder(ndim=2, nvar=1))


def test_space_time_observations_rejected():
    system = VariogramSystem(nvar=1)
    coord, value = _coords()
    system.set_obs(1, coord, value, times=np.arange(6.0))
    system.vgm[1, 1].set_vgm("sph", sill=1.0, a_major=5.0)
    with pytest.raises(ValueError, match="space-time"):
        system.apply(_Recorder(nvar=1))


def test_drift_transfer_and_ndrift_mismatch():
    system = VariogramSystem(nvar=1)
    coord, value = _coords()
    system.set_obs(1, coord, value, drift=np.ones(6))
    system.vgm[1, 1].set_vgm("sph", sill=1.0, a_major=5.0)

    k = _Recorder(nvar=1, ndrift=1)
    system.apply(k)
    assert k.drift_calls and k.drift_calls[0][0] == 1

    with pytest.raises(ValueError, match="drift"):
        system.apply(_Recorder(nvar=1, ndrift=0))
