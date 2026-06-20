"""Statistical binning algorithms for variogram lags."""

import numpy as np
from scipy.cluster.vq import kmeans2

def _sturges_rule(distances):
    """Calculate lag edges using Sturges' rule."""
    n = len(distances)
    if n == 0:
        return np.array([0.0, 1.0])
    n_bins = int(np.ceil(np.log2(n) + 1))
    return np.linspace(np.min(distances), np.max(distances), n_bins + 1)

def _fd_rule(distances):
    """Calculate lag edges using the Freedman-Diaconis rule."""
    n = len(distances)
    if n == 0:
        return np.array([0.0, 1.0])
    q75, q25 = np.percentile(distances, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return _sturges_rule(distances)
    bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
    if bin_width == 0:
        return np.array([np.min(distances), np.max(distances)])
    n_bins = int(np.ceil((np.max(distances) - np.min(distances)) / bin_width))
    return np.linspace(np.min(distances), np.max(distances), n_bins + 1)

def _scott_rule(distances):
    """Calculate lag edges using Scott's normal reference rule."""
    n = len(distances)
    if n == 0:
        return np.array([0.0, 1.0])
    std = np.std(distances)
    if std == 0:
        return np.array([np.min(distances), np.max(distances)])
    bin_width = 3.49 * std * (n ** (-1.0 / 3.0))
    if bin_width == 0:
        return np.array([np.min(distances), np.max(distances)])
    n_bins = int(np.ceil((np.max(distances) - np.min(distances)) / bin_width))
    return np.linspace(np.min(distances), np.max(distances), n_bins + 1)

def _kmeans_bins(distances, n_bins=15):
    """Calculate lag edges using k-means clustering."""
    distances = np.asarray(distances).reshape(-1, 1)
    if len(distances) <= n_bins:
        n_bins = max(1, len(distances) - 1)
        if n_bins == 0:
            return np.array([0.0, 1.0])
    
    centers, _ = kmeans2(
        distances,
        n_bins,
        iter=100,
        minit="++",
        missing="raise",
        seed=42,
    )
    centers = np.sort(centers.ravel())
    
    # Calculate edges as midpoints between centers
    edges = np.zeros(n_bins + 1)
    edges[0] = np.min(distances)
    edges[-1] = np.max(distances)
    for i in range(1, n_bins):
        edges[i] = (centers[i - 1] + centers[i]) / 2.0
        
    return edges

def calculate_lag_edges(distances, method="sturges", n_bins=15):
    """Calculate lag edges based on a specified statistical method."""
    method = method.lower().strip()
    if method == "sturges":
        return _sturges_rule(distances)
    elif method == "fd" or method == "freedman-diaconis":
        return _fd_rule(distances)
    elif method == "scott":
        return _scott_rule(distances)
    elif method == "kmeans":
        return _kmeans_bins(distances, n_bins=n_bins)
    else:
        raise ValueError(f"Unknown binning method: {method}")
