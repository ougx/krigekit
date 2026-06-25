"""Tests for the ObservationSet data/search-configuration object."""

import numpy as np
import pytest

from krigekit import ObservationSet


def _xy(n=5):
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 10.0, size=(n, 2)), rng.normal(size=n)


def test_set_validates_shapes_and_derives_sizes():
    coord, value = _xy(5)
    obs = ObservationSet()
    assert not obs.configured
    assert obs.nobs == 0 and obs.ndim == 0 and obs.ndrift == 0

    obs.set(coord, value)
    assert obs.configured
    assert obs.nobs == 5 and obs.ndim == 2

    with pytest.raises(ValueError):
        obs.set(coord, value[:-1])           # mismatched lengths
    with pytest.raises(ValueError):
        obs.set(np.zeros((3, 4)), np.zeros(3))   # 4D coordinates


def test_one_d_coord_is_reshaped():
    obs = ObservationSet().set(np.arange(4.0), np.arange(4.0))
    assert obs.ndim == 1 and obs.coord.shape == (4, 1)


def test_drift_normalized_to_two_d():
    coord, value = _xy(5)
    obs = ObservationSet().set(coord, value, drift=np.ones(5))
    assert obs.drift.shape == (5, 1) and obs.ndrift == 1
    obs.set(coord, value, drift=np.ones((5, 2)))
    assert obs.ndrift == 2


def test_set_search_validates_and_stores():
    coord, value = _xy()
    obs = ObservationSet().set(coord, value)
    obs.set_search(nmax=16, maxdist=500.0, anis1=0.5, azimuth=30.0, time_at=2.0)
    assert obs.nmax == 16 and obs.maxdist == 500.0
    assert obs.search_anis1 == 0.5 and obs.search_azimuth == 30.0
    assert obs.search_time_at == 2.0
    with pytest.raises(ValueError):
        obs.set_search(anis1=0.0)
    with pytest.raises(ValueError):
        obs.set_search(time_at=-1.0)


def test_is_collocated_with():
    coord, value = _xy(5)
    a = ObservationSet().set(coord, value)
    b = ObservationSet().set(coord.copy(), np.random.default_rng(1).normal(size=5))
    assert a.is_collocated_with(b)

    c = ObservationSet().set(coord + 1.0, value)
    assert not a.is_collocated_with(c)

    # times must match too
    t = np.arange(5.0)
    at = ObservationSet().set(coord, value, times=t)
    assert not a.is_collocated_with(at)       # one has times, the other not
    assert at.is_collocated_with(ObservationSet().set(coord, value, times=t.copy()))


def test_clear_resets_everything():
    coord, value = _xy()
    obs = ObservationSet().set(coord, value, variance=np.ones(5))
    obs.set_search(nmax=8, anis1=0.5)
    obs.clear()
    assert not obs.configured
    assert obs.nmax is None and obs.search_anis1 == 1.0


def test_apply_to_emits_engine_calls():
    class _Recorder:
        def __init__(self):
            self.obs = None
            self.drift = None

        def set_obs(self, ivar, coord, value, **kwargs):
            self.obs = (ivar, coord, value, kwargs)

        def set_obs_drift(self, ivar, drift):
            self.drift = (ivar, drift)

    coord, value = _xy(5)
    obs = ObservationSet().set(coord, value, variance=np.ones(5),
                               sk_mean=1.5, drift=np.ones(5))
    obs.set_search(nmax=12, maxdist=300.0)
    rec = _Recorder()
    obs.apply_to(rec, ivar=2)
    ivar, c, v, kwargs = rec.obs
    assert ivar == 2
    assert "nmax" not in kwargs and "maxdist" not in kwargs
    assert kwargs["sk_mean"] == 1.5
    assert obs.nmax == 12 and obs.maxdist == 300.0
    assert rec.drift[0] == 2 and rec.drift[1].shape == (5, 1)

    with pytest.raises(RuntimeError):
        ObservationSet().apply_to(rec, ivar=1)    # unconfigured
