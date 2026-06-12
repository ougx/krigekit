"""
conftest.py
===========
Shared pytest fixtures for krigekit tests.

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
def aem2d():
    """AEM data (aem2d.csv)."""
    df = pd.read_csv(data_path("aem2d.csv"))
    return df[["x", "y"]].values, df["logRho"].values


@pytest.fixture(scope="session")
def aem2d_small(aem2d):
    """Reproducible 80-point subset of AEM data (aem2d.csv).

    Uses a fixed-seed Generator so the selected indices are identical
    regardless of the global numpy random state (which depends on test
    execution order).  Changing this seed may require updating tolerances
    in TestCoKrigingButte if the new subset is numerically less well-conditioned.
    """
    coord, value = aem2d
    rng = np.random.default_rng(seed=42)
    iloc = rng.choice(len(value), 80, replace=False)
    return coord[iloc], value[iloc]


@pytest.fixture(scope="session")
def pc2d_grid():
    """4800 grid nodes in 2D with reference kriging results (grid2d.csv)."""
    df = pd.read_csv(data_path("grid2d.csv"))
    # The reference estimate column may be named 'result' or 'estimate'
    ref_col = "estimate" if "estimate" in df.columns else "result"
    return df[["x", "y"]].values, df[ref_col].values


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
    return path, sample

@pytest.fixture(scope="session")
def pc2d_loo():
    """Reference leave-one-out CV estimates for the pc2d dataset (loo-cv column)."""
    df = pd.read_csv(data_path("pc2d.csv"))
    return df["loo-cv"].values



@pytest.fixture(scope="session")
def block_data(pc2d_obs):
    """Load block sub-nodes and build block description arrays."""
    coord, value = pc2d_obs

    # gridblockpnt2d.csv: 16 Gaussian quadrature sub-nodes for one block
    df_pnt = pd.read_csv(os.path.join(DATA_DIR, "gridblockpnt2d.csv"))
    sub_coords  = df_pnt[["x", "y"]].values          # (16, 2)
    sub_weights = df_pnt["weight"].values              # (16,)

    # gridblock2d.csv: block centroid and sub-node count
    # Column header is "#pnts" — read without treating # as comment char
    df_blk = pd.read_csv(
        os.path.join(DATA_DIR, "gridblock2d.csv"),
    )
    # Rename the first column which may be read as "#pnts" or "# pnts"
    df_blk.columns = [c.strip().lstrip('#').strip() for c in df_blk.columns]
    nblockpnt = df_blk["pnts"].values.astype(np.int32)  # [16]
    nblock = len(nblockpnt)   # 1

    return {
        "coord": coord,
        "value": value,
        "sub_coords":  sub_coords,
        "sub_weights": sub_weights,
        "nblockpnt":   nblockpnt,
        "nblock":      nblock,
        "block_centroid": np.array([[df_blk["x"].iloc[0],
                                        df_blk["y"].iloc[0]]]),
    }
