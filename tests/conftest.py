"""
conftest.py
===========
Shared pytest fixtures for pykriging tests.

All tests that need data files use the fixtures defined here.
The shared library is expected to be compiled before running tests:

    python build_lib.py
    pytest
"""

import os
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")


def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


# ---------------------------------------------------------------------------
# Fixtures: raw data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def simple_obs():
    """5 observations in 2D with known z values (obs_simple.csv)."""
    df = pd.read_csv(data_path("obs_simple.csv"))
    return df[["x", "y"]].values, df["z"].values


@pytest.fixture(scope="session")
def simple_grid():
    """3 grid points in 2D (grid_simple.csv)."""
    df = pd.read_csv(data_path("grid_simple.csv"))
    return df[["x", "y"]].values


@pytest.fixture(scope="session")
def pc2d_obs():
    """62 percent-coarse observations in 2D (pc2d.csv)."""
    df = pd.read_csv(data_path("pc2d.csv"))
    return df[["x", "y"]].values, df["pc"].values


@pytest.fixture(scope="session")
def pc2d_grid():
    """4800 grid nodes in 2D with reference kriging results (grid2d.csv)."""
    df = pd.read_csv(data_path("grid2d.csv"))
    return df[["x", "y"]].values, df["result"].values


@pytest.fixture(scope="session")
def walker_obs():
    """470 Walker Lake observations with primary (V) and secondary (U) variables."""
    df = pd.read_csv(data_path("walker.csv"))
    # Drop rows where secondary variable U == -999 (not observed)
    valid = df[df["U"] != -999].copy()
    obs_primary   = valid[["X", "Y"]].values, valid["V"].values
    obs_secondary = valid[["X", "Y"]].values, valid["U"].values
    return obs_primary, obs_secondary


@pytest.fixture(scope="session")
def head2d_obs():
    """29 hydraulic head observations used for kriging with drift (head2d.csv)."""
    df = pd.read_csv(data_path("head2d.csv"))
    return df[["x", "y"]].values, df["head"].values


@pytest.fixture(scope="session")
def sgsim_path_sample():
    """Pre-computed random path and samples for 4800-node SGSIM test."""
    path   = pd.read_csv(data_path("path4800.csv"))["randpath"].values.astype(np.int32)
    sample = pd.read_csv(data_path("sample4800.csv"))["sample"].values
    # sample is 1D (nsim=1); reshape to (1, nblocks)
    return path, sample.reshape(1, -1)
