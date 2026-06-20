import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from krigekit.variogram_binning import calculate_lag_edges
from krigekit.variogram_empirical import raw_vgm, avg_vgm, _dowd_estimator, _genton_estimator
from krigekit.variogram_fitting import fit_vgm, calc_fit_metrics
from krigekit.variogram_model import VariogramModel

def test_calculate_lag_edges():
    np.random.seed(42)
    lags = np.random.uniform(0, 100, 500)
    
    sturges = calculate_lag_edges(lags, method="sturges")
    assert len(sturges) > 5
    assert sturges[0] <= np.min(lags)
    assert sturges[-1] >= np.max(lags)
    
    fd = calculate_lag_edges(lags, method="fd")
    assert len(fd) > 5
    
    scott = calculate_lag_edges(lags, method="scott")
    assert len(scott) > 5
    
    kmeans = calculate_lag_edges(lags, method="kmeans", n_bins=10)
    assert len(kmeans) == 11
    
    # uniform binning test removed since it is handled by 'h_bins=integer' directly

def test_robust_estimators():
    # True semivariance is 2.5, meaning variance of differences is 5.0
    np.random.seed(42)
    dz = np.random.normal(0, np.sqrt(5.0), 100000)
    g = 0.5 * dz**2
    
    dowd = _dowd_estimator(g)
    assert_almost_equal(dowd, 2.5, decimal=1)
    
    genton = _genton_estimator(g)
    assert_almost_equal(genton, 2.5, decimal=1)

def test_raw_vgm_custom_metric():
    coords = np.array([[0, 0], [0, 3], [4, 0]])
    vals = np.array([1, 2, 3])
    
    def manhattan(u, v):
        return np.sum(np.abs(u - v))
        
    cloud = raw_vgm(coords, vals, metric=manhattan, verbose=False)
    # Distances should be 3, 4, 7 (manhattan) instead of 3, 4, 5 (euclidean)
    dists = sorted(cloud['distance'].tolist())
    assert_allclose(dists, [3, 4, 7])

def test_avg_vgm_string_estimator_and_binning():
    coords = np.random.rand(100, 2) * 10
    vals = np.random.randn(100)
    cloud = raw_vgm(coords, vals, verbose=False)
    
    # Test string binning
    avg_fd = avg_vgm(cloud, h_width="fd")
    assert not avg_fd.empty
    
    # Test estimators
    avg_cressie = avg_vgm(cloud, estimator="cressie", h_bins=10)
    assert not avg_cressie.empty
    
    avg_dowd = avg_vgm(cloud, estimator="dowd", h_bins=10)
    assert not avg_dowd.empty
    
    avg_genton = avg_vgm(cloud, estimator="genton", h_bins=10)
    assert not avg_genton.empty

def test_calc_fit_metrics():
    coords = np.random.rand(50, 2) * 10
    vals = np.random.randn(50)
    cloud = raw_vgm(coords, vals, verbose=False)
    avg = avg_vgm(cloud, h_bins=10)
    
    params, cov, model, metrics = fit_vgm(avg, models=("spherical",), p0=(1.0, 5.0, 0.0), return_model=True, return_metrics=True)
    assert isinstance(model, VariogramModel)
    assert "RMSE" in metrics
    assert "MSE" in metrics
    assert "MAE" in metrics
    assert "R2" in metrics
    assert metrics["RMSE"] >= 0
    assert metrics["R2"] <= 1.0
