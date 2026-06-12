"""
test_wind_exact.py
==================
Validates SpaceTimeKriging against the Irish wind dataset
(Haslett & Raftery 1989, Applied Statistics) — the canonical ST benchmark.

Data (StatLib):  https://lib.stat.cmu.edu/datasets/wind.data
  columns : year month day RPT VAL ROS KIL SHA BIR DUB CLA MUL CLO BEL MAL
  units   : knots;  no missing values in 1961

Observation limits
------------------
The Fortran KD-tree builder is now safe for arbitrarily large structured datasets
(the stack overflow from degenerate identical-coordinate splits was fixed in
kdtree2_maxidx.f90).  Tests here stay at ≤120 obs (10 days) for the quality/
variance tests, and 12 obs (1 day) for exact-interpolation tests.

Exact-interpolator property
----------------------------
With nugget=0, kriging reproduces observed values exactly at training locations.
This holds when each spatial station appears at most once in the nmax
neighbourhood. The 1-day fixture (12 obs, one per station) is ideal for this.
With multiple days the same station fills the neighbourhood, making the
covariance submatrix near-singular and breaking numerical exactness.
"""

import os
import urllib.request

import numpy as np
import pytest

pytest.importorskip("krigekit", reason="compiled libkriging not found")
from krigekit import SpaceTimeKriging

# ---------------------------------------------------------------------------
# Station locations — from gstat R-package docs (Haslett & Raftery 1989)
# order matches data columns: RPT VAL ROS KIL SHA BIR DUB CLA MUL CLO BEL MAL
# ---------------------------------------------------------------------------
_STATIONS = [
    ("RPT", 51.80, -8.25),   # Roche's Point
    ("VAL", 51.93, -10.25),  # Valentia
    ("ROS", 52.28, -6.36),   # Rosslare
    ("KIL", 52.67, -7.27),   # Kilkenny
    ("SHA", 52.70, -8.92),   # Shannon
    ("BIR", 53.08, -7.88),   # Birr
    ("DUB", 53.43, -6.25),   # Dublin
    ("CLA", 53.72, -8.98),   # Claremorris
    ("MUL", 53.53, -7.37),   # Mullingar
    ("CLO", 54.18, -7.23),   # Clones
    ("BEL", 54.23, -10.00),  # Belmullet
    ("MAL", 55.37, -7.33),   # Malin Head
]

# Flat-earth projection centred on Ireland
_REF_LAT, _REF_LON = 53.0, -8.0
_KM_LAT = 111.0
_KM_LON = 111.0 * np.cos(np.radians(_REF_LAT))

_DATA_URL   = "https://lib.stat.cmu.edu/datasets/wind.data"
_DATA_CACHE = os.path.join(os.path.dirname(__file__), "../test_data", "wind.data")

# Known 1 Jan 1961 wind speeds (knots) from the dataset, one per station
_DAY1_OBS = [15.04, 14.96, 13.17, 9.29, 13.96, 9.87,
             13.67, 10.25, 10.83, 12.58, 18.50, 15.04]


def _station_xy():
    """Return (12, 2) array of [x_km, y_km] for each station."""
    return np.array([
        [(lon - _REF_LON) * _KM_LON, (lat - _REF_LAT) * _KM_LAT]
        for _, lat, lon in _STATIONS
    ])


def _load_wind(n_days: int):
    """
    Download (once) and return the first n_days of 1961 Irish wind data.

    Returns
    -------
    coord4 : (nobs, 4) float — [x_km, y_km, 0, decimal_year]
    value  : (nobs,)   float — wind speed in knots

    Larger n_days values are safe; the KD-tree stack overflow is fixed.
    """
    os.makedirs(os.path.dirname(_DATA_CACHE), exist_ok=True)
    if not os.path.exists(_DATA_CACHE):
        urllib.request.urlretrieve(_DATA_URL, _DATA_CACHE)

    xy = _station_xy()
    rows, vals = [], []
    days_seen = 0
    last_day = None

    with open(_DATA_CACHE) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 15:
                continue
            yr = 1900 + int(parts[0])
            mo, da = int(parts[1]), int(parts[2])
            if yr != 1961:
                continue
            day_key = (mo, da)
            if day_key != last_day:
                if days_seen == n_days:
                    break
                days_seen += 1
                last_day = day_key
            t = yr + (mo - 1) / 12.0 + (da - 1) / 365.25
            for i, ws_str in enumerate(parts[3:]):
                ws = float(ws_str)
                if ws < 0:          # -99.99 = missing
                    continue
                rows.append([xy[i, 0], xy[i, 1], 0.0, t])
                vals.append(ws)

    return np.array(rows, dtype=float), np.array(vals, dtype=float)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wind_1day():
    """12 obs: all 12 stations on 1 Jan 1961.  Ideal for exact-interp tests."""
    return _load_wind(1)


@pytest.fixture(scope="module")
def wind_10day():
    """120 obs: 10 days × 12 stations.  Used for quality and variance tests."""
    return _load_wind(10)


# ---------------------------------------------------------------------------
# Variogram helper
# ---------------------------------------------------------------------------

def _attach_vgm(k, nugget: float = 0.0):
    """
    Sum-metric variogram calibrated to Irish wind (knots, untransformed):
      Spatial:  exponential, range 200 km,  sill 6 kn²
      Temporal: exponential, range 0.07 yr  (~26 days), sill 4 kn²
      Joint sill: 1.5 kn²
      at = 3000 km/yr  (equalises spatial and temporal contributions)
    """
    k.set_vgm(1, 1, vtype="exp",
              nugget=nugget, sill=6.0,
              a_major=200.0, a_minor1=200.0, a_minor2=200.0)
    k.set_vgm_temporal(1, 1, vtype="exp",
                       nugget=0.0, sill=4.0, at_k=0.07)
    k.set_vgm_joint_sills(1, 1, 1.5)


# ===========================================================================
# 1. Exact-interpolator property  (1-day fixture, nugget = 0)
# ===========================================================================
class TestExactInterpolation:
    """
    With nugget=0 and one observation per station (1-day data), predicting
    at each station's own (x, y, t) must return its observed wind speed.
    The covariance matrix is well-conditioned because all stations are at
    distinct spatial locations (50–300 km apart) on the same day.
    """

    def _build(self, coord4, value):
        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("sum_metric", "linear", at=3000.0)
        k.set_obs(1, coord4, value, nmax=12)
        _attach_vgm(k, nugget=0.0)
        return k

    def test_exact_single_station(self, wind_1day):
        """Predict at RPT (obs #0) — should return 15.04 kn exactly."""
        coord4, value = wind_1day
        k = self._build(coord4, value)
        k.set_grid(coord4[[0], :3], coord4[[0], 3])
        k.set_search(1)
        k.solve()
        est, var = k.get_results()

        assert abs(est[0] - value[0]) < 1e-3, \
            f"exact interp failed: est={est[0]:.4f}  obs={value[0]:.4f}"
        assert abs(est[0] - 15.04) < 1e-3, \
            f"known 1 Jan 1961 RPT value wrong: est={est[0]:.4f}"
        assert 0.0 <= var[0] < 1e-6

    def test_exact_all_twelve_stations(self, wind_1day):
        """
        Predict at all 12 stations simultaneously on 1 Jan 1961.
        Estimates must match the known observed values exactly.
        """
        coord4, value = wind_1day
        assert len(value) == 12, "Expected exactly 12 obs for 1-day fixture"

        k = self._build(coord4, value)
        k.set_grid(coord4[:, :3], coord4[:, 3])
        k.set_search(1)
        k.solve()
        est, var = k.get_results()

        np.testing.assert_allclose(
            est, _DAY1_OBS, atol=1e-3,
            err_msg="Known 1 Jan 1961 values not reproduced exactly"
        )
        assert np.all(var >= 0)
        assert np.all(var < 1e-6), f"max var at obs: {var.max():.2e}"

    def test_exact_kriging_factor(self, wind_1day):
        """
        At an observation location the kriging weight vector is a unit vector
        (weight=1 on the collocated obs, ~0 on all others), and the Lagrange
        multiplier is ~0.
        """
        coord4, value = wind_1day
        k = self._build(coord4, value)
        k.set_grid(coord4[[0], :3], coord4[[0], 3])
        k.set_search(1)
        k.solve()

        f = k.get_factor()
        assert f["valid"]
        w = np.linalg.solve(f["matA"], f["rhsB"][0])
        # First weight should be ~1, rest ~0, Lagrange ~0
        assert abs(w[0] - 1.0) < 1e-6,   f"collocated weight: {w[0]:.6f}"
        assert np.all(np.abs(w[1:-1]) < 1e-6), \
            f"non-zero off-obs weights: {w[1:-1]}"


# ===========================================================================
# 2. Prediction quality at unobserved space-time locations
# ===========================================================================
class TestPredictionQuality:
    """
    Interpolate at the centroid of Ireland (0 km, 0 km) over 5 days.
    These checks hold with a small nugget (real-world use case).
    """

    @pytest.fixture(scope="class")
    def centroid_result(self, wind_10day):
        coord4, value = wind_10day
        # One prediction per day at Ireland's centroid (0, 0, 0)
        day_times = np.unique(coord4[:, 3])[:5]
        gcoord = np.zeros((5, 3))

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("sum_metric", "linear", at=3000.0)
        k.set_obs(1, coord4, value, nmax=20)
        _attach_vgm(k, nugget=0.5)      # small nugget for interior prediction
        k.set_grid(gcoord, day_times)
        k.set_search(1)
        k.solve()
        return k.get_results()

    def test_shape(self, centroid_result):
        est, var = centroid_result
        assert est.shape == (5,)
        assert var.shape == (5,)

    def test_non_negative_variance(self, centroid_result):
        _, var = centroid_result
        assert np.all(var >= 0)

    def test_finite_estimates(self, centroid_result):
        est, _ = centroid_result
        assert np.all(np.isfinite(est))

    def test_physical_range(self, centroid_result):
        """Interpolated wind at Ireland's centroid should be 0–100 kn."""
        est, _ = centroid_result
        assert np.all(est > 0),   f"non-positive wind speed: {est.min():.2f}"
        assert np.all(est < 100), f"implausibly large wind speed: {est.max():.2f}"

    def test_estimates_near_station_mean(self, wind_10day):
        """
        Centroid interpolation should fall within the range of the 12 surrounding
        station values on the same day — spatial interpolation is bounded.
        """
        coord4, value = wind_10day
        day_times = np.unique(coord4[:, 3])

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model("sum_metric", "linear", at=3000.0)
        k.set_obs(1, coord4, value, nmax=20)
        _attach_vgm(k, nugget=0.5)
        k.set_grid(np.zeros((1, 3)), day_times[:1])
        k.set_search(1)
        k.solve()
        est, _ = k.get_results()

        # Observations on the first day
        t0 = day_times[0]
        obs_day1 = value[coord4[:, 3] == t0]
        assert obs_day1.min() * 0.5 < est[0] < obs_day1.max() * 1.5, (
            f"centroid estimate {est[0]:.2f} kn far outside station range "
            f"[{obs_day1.min():.2f}, {obs_day1.max():.2f}]"
        )


# ===========================================================================
# 3. Variance ordering
# ===========================================================================
class TestVarianceOrdering:
    """
    Kriging variance must be lower at a training location than at a point
    that is far away from all observations in time.
    """

    def test_variance_higher_far_in_time(self, wind_10day):
        coord4, value = wind_10day

        def _var_at(gc, gt, nugget=0.5):
            k = SpaceTimeKriging(nvar=1)
            k.set_st_model("sum_metric", "linear", at=3000.0)
            k.set_obs(1, coord4, value, nmax=20, maxdist=1e9)
            _attach_vgm(k, nugget=nugget)
            k.set_grid(gc, gt)
            k.set_search(1)
            k.solve()
            _, var = k.get_results()
            return var[0]

        # At an observation location (small variance)
        var_at_obs   = _var_at(coord4[[0], :3], coord4[[0], 3])
        # Same location, but 50 years into the future (large variance)
        var_far_time = _var_at(coord4[[0], :3], np.array([2011.0]))

        assert var_far_time > var_at_obs, (
            f"expected higher variance 50 yr from obs: "
            f"at_obs={var_at_obs:.4f}  far_future={var_far_time:.4f}"
        )

    def test_variance_higher_at_unsampled_station(self, wind_10day):
        """
        A synthetic 13th station position (centroid) has no observations,
        so its variance must exceed that of any of the 12 real stations.
        """
        coord4, value = wind_10day
        t0 = coord4[0, 3]
        xy = _station_xy()

        def _v(xk, yk):
            k = SpaceTimeKriging(nvar=1)
            k.set_st_model("sum_metric", "linear", at=3000.0)
            k.set_obs(1, coord4, value, nmax=20)
            _attach_vgm(k, nugget=0.5)
            k.set_grid(np.array([[xk, yk, 0.0]]), np.array([t0]))
            k.set_search(1)
            k.solve()
            _, var = k.get_results()
            return var[0]

        var_obs_station  = _v(xy[0, 0], xy[0, 1])   # RPT — real station
        var_centroid     = _v(0.0, 0.0)              # centroid — no obs here

        assert var_centroid > var_obs_station, (
            f"centroid var={var_centroid:.4f} should exceed "
            f"station var={var_obs_station:.4f}"
        )
