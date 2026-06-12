"""
test_sector_search.py
======================
Unit and integration tests for the sector search feature.
Verify that:
1. Spatial Kriging with sector_search=True partitions neighbors into quadrants (2D) or octants (3D).
2. SpaceTimeKriging with sector_search=True partitions neighbors along spatial dimensions only, ignoring temporal lag.
3. The selected neighbors are limited to `nmax` per sector, up to a global limit of 2^ndim * nmax.
"""

import numpy as np
import pytest
from krigekit import Kriging, SpaceTimeKriging


# ---------------------------------------------------------------------------
# 1. 2D Spatial Kriging Sector Search Tests
# ---------------------------------------------------------------------------

def test_2d_spatial_sector_search():
    """
    Test 2D sector search (4 quadrants).
    Target node: (0.0, 0.0)
    Quadrant 1 (+x, +y): (0.1, 0.1) [index 1], (0.2, 0.2) [index 2]
    Quadrant 2 (-x, +y): (-5.0, 5.0) [index 3]
    Quadrant 3 (-x, -y): (-5.0, -5.0) [index 4]
    Quadrant 4 (+x, -y): (5.0, -5.0) [index 5]

    With sector_search=True and nmax=1:
    - Quadrant 1 should only keep the closest neighbor: (0.1, 0.1) (index 1).
    - Quadrants 2, 3, 4 should keep their single neighbor.
    - Total selected neighbors should be 4.
    - Neighbors selected should be [1, 3, 4, 5] (1-based indices).
    """
    obs_coord = np.array([
        [0.1, 0.1],   # Q1 (close) -> index 1
        [0.2, 0.2],   # Q1 (close but further than first) -> index 2
        [-5.0, 5.0],  # Q2 (far) -> index 3
        [-5.0, -5.0], # Q3 (far) -> index 4
        [5.0, -5.0]   # Q4 (far) -> index 5
    ])
    obs_value = np.array([10.0, 10.0, 20.0, 20.0, 20.0])
    grid_coord = np.array([[0.0, 0.0]])

    k = Kriging(ndim=2, nvar=1, store_weight=True)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=1)
    k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=100.0)
    k.set_grid(coord=grid_coord)
    k.set_search(ivar=1, sector_search=True)
    k.solve()

    w = k.get_weights()
    nnear = w["nnear"][0, 0]
    inear = w["inear"][0, 0, :nnear]

    assert nnear == 4
    # Sort for index comparison
    sorted_inear = sorted(list(inear))
    assert sorted_inear == [1, 3, 4, 5]


# ---------------------------------------------------------------------------
# 2. 3D Spatial Kriging Sector Search Tests
# ---------------------------------------------------------------------------

def test_3d_spatial_sector_search():
    """
    Test 3D sector search (8 octants).
    Target node: (0.0, 0.0, 0.0)
    We place:
    - 2 observations in Octant 1 (+x, +y, +z) at (0.1, 0.1, 0.1) [index 1], (0.2, 0.2, 0.2) [index 2]
    - 1 observation in each of the remaining 7 octants at coordinates with sign configurations matching the octants.
    """
    # Octant signs:
    # 1: + + +
    # 2: - + +
    # 3: + - +
    # 4: - - +
    # 5: + + -
    # 6: - + -
    # 7: + - -
    # 8: - - -
    obs_coord = np.array([
        [0.1, 0.1, 0.1],     # Octant 1 (close) -> index 1
        [0.2, 0.2, 0.2],     # Octant 1 (close but further) -> index 2
        [-5.0, 5.0, 5.0],    # Octant 2 -> index 3
        [5.0, -5.0, 5.0],    # Octant 3 -> index 4
        [-5.0, -5.0, 5.0],   # Octant 4 -> index 5
        [5.0, 5.0, -5.0],    # Octant 5 -> index 6
        [-5.0, 5.0, -5.0],   # Octant 6 -> index 7
        [5.0, -5.0, -5.0],   # Octant 7 -> index 8
        [-5.0, -5.0, -5.0]   # Octant 8 -> index 9
    ])
    obs_value = np.ones(9) * 10.0
    grid_coord = np.array([[0.0, 0.0, 0.0]])

    k = Kriging(ndim=3, nvar=1, store_weight=True)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=1)
    k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=100.0)
    k.set_grid(coord=grid_coord)
    k.set_search(ivar=1, sector_search=True)
    k.solve()

    w = k.get_weights()
    nnear = w["nnear"][0, 0]
    inear = w["inear"][0, 0, :nnear]

    assert nnear == 8
    sorted_inear = sorted(list(inear))
    assert sorted_inear == [1, 3, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# 3. Space-Time Kriging Sector Search Tests
# ---------------------------------------------------------------------------

def test_spacetime_sector_search():
    """
    Test spatiotemporal sector search.
    Coordinates partition along spatial dimensions only, ignoring temporal lag.
    Target node: (0.0, 0.0, 0.0) at time t=0.0
    Observations coordinates in space-time (x, y, z, t):
    - Q1 (+x, +y): (0.1, 0.1, 0.0, 0.0) [index 1], (0.2, 0.2, 0.0, 0.0) [index 2]
    - Q2 (-x, +y): (-5.0, 5.0, 0.0, 0.0) [index 3]
    - Q3 (-x, -y): (-5.0, -5.0, 0.0, 0.0) [index 4]
    - Q4 (+x, -y): (5.0, -5.0, 0.0, 0.0) [index 5]

    With sector_search=True and nmax=1:
    - Quadrant 1 should only keep the closest neighbor: index 1.
    - Quadrants 2, 3, 4 should keep their single neighbor.
    - Total selected neighbors should be 4.
    - Neighbors selected should be [1, 3, 4, 5] (1-based indices).
    """
    obs_coord = np.array([
        [0.1, 0.1, 0.0, 0.0],   # Q1 -> index 1
        [0.2, 0.2, 0.0, 0.0],   # Q1 -> index 2
        [-5.0, 5.0, 0.0, 0.0],  # Q2 -> index 3
        [-5.0, -5.0, 0.0, 0.0], # Q3 -> index 4
        [5.0, -5.0, 0.0, 0.0]   # Q4 -> index 5
    ])
    obs_value = np.array([10.0, 10.0, 20.0, 20.0, 20.0])
    grid_coord = np.array([[0.0, 0.0, 0.0]])
    grid_time = np.array([0.0])

    k = SpaceTimeKriging(nvar=1, neglect_error=True)
    k.set_st_model("sum_metric", "linear", at=10.0)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=1)
    k.set_vgm(1, 1, vtype="sph", nugget=0, sill=0.8, a_major=500, a_minor1=300, a_minor2=100)
    k.set_vgm_temporal(1, 1, vtype="exp", nugget=0, sill=0.5, at_k=10.0)
    k.set_vgm_joint_sills(1, 1, 0.3)
    k.set_grid(grid_coord, grid_time)
    k.set_search(ivar=1, sector_search=True)
    k.solve()

    f = k.get_factor()
    assert f["valid"]
    assert f["npp"] == 4

    # Now run the same configuration with sector_search=False and nmax=2
    k_std = SpaceTimeKriging(nvar=1, neglect_error=True)
    k_std.set_st_model("sum_metric", "linear", at=10.0)
    k_std.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=2)
    k_std.set_vgm(1, 1, vtype="sph", nugget=0, sill=0.8, a_major=500, a_minor1=300, a_minor2=100)
    k_std.set_vgm_temporal(1, 1, vtype="exp", nugget=0, sill=0.5, at_k=10.0)
    k_std.set_vgm_joint_sills(1, 1, 0.3)
    k_std.set_grid(grid_coord, grid_time)
    k_std.set_search(ivar=1, sector_search=False)
    k_std.solve()

    f_std = k_std.get_factor()
    assert f_std["valid"]
    assert f_std["npp"] == 2


# ---------------------------------------------------------------------------
# 4. regression against standard search when data is uniform
# ---------------------------------------------------------------------------

def test_sector_search_vs_standard_search():
    """
    When the number of observations in each quadrant is less than or equal to nmax,
    sector search and standard search should select the exact same neighbors.
    """
    obs_coord = np.array([
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0]
    ])
    obs_value = np.array([1.0, 2.0, 3.0, 4.0])
    grid_coord = np.array([[0.0, 0.0]])

    # Standard
    k_std = Kriging(ndim=2, nvar=1, store_weight=True)
    k_std.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=4)
    k_std.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=10.0)
    k_std.set_grid(coord=grid_coord)
    k_std.set_search(ivar=1, sector_search=False)
    k_std.solve()
    w_std = k_std.get_weights()

    # Sector
    k_sec = Kriging(ndim=2, nvar=1, store_weight=True)
    k_sec.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=1)  # 1 per quadrant
    k_sec.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=10.0)
    k_sec.set_grid(coord=grid_coord)
    k_sec.set_search(ivar=1, sector_search=True)
    k_sec.solve()
    w_sec = k_sec.get_weights()

    assert w_std["nnear"][0, 0] == w_sec["nnear"][0, 0]
    assert np.all(np.sort(w_std["inear"][0, 0]) == np.sort(w_sec["inear"][0, 0]))
    # Compare estimates and variances
    est_std, var_std = k_std.get_results()
    est_sec, var_sec = k_sec.get_results()
    assert est_std[0] == pytest.approx(est_sec[0])
    assert var_std[0] == pytest.approx(var_sec[0])


# ---------------------------------------------------------------------------
# 5. SGSIM sector search smoke test
# ---------------------------------------------------------------------------

def test_sgsim_sector_search():
    """
    Verify that sector_search=True works inside the SGSIM path.

    The SGSIM tree contains both obs (indices 1:nobs) and grid block centres
    (indices nobs+1:end) in a single combined array.  With sector_search the
    code should:
      1. Query a larger candidate pool via kdtree2_n_nearest_maxidx (respecting
         the 'already-simulated blocks only' constraint).
      2. Apply the per-sector nmax limit across the combined pool.
      3. Split accepted candidates into obs (inear) and prior-blocks (inearb).

    We place 5 observations symmetrically in all 4 quadrants around the origin
    (2D, ndim=2) and simulate 8 blocks in a 2x4 grid centred on the origin.
    With nmax=1 and sector_search=True each simulated block should see at most
    4 obs-quadrant neighbours, so the run must complete without error and
    produce finite, in-range estimates.
    """
    rng = np.random.default_rng(42)

    obs_coord = np.array([
        [ 1.0,  1.0],
        [-1.0,  1.0],
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 0.0,  0.0],   # at origin — assigned to one quadrant deterministically
    ])
    obs_value = rng.uniform(0, 10, size=5)

    grid_coord = np.array([
        [ 0.5,  1.5], [-0.5,  1.5],
        [ 0.5, -1.5], [-0.5, -1.5],
        [ 1.5,  0.5], [-1.5,  0.5],
        [ 1.5, -0.5], [-1.5, -0.5],
    ])

    nsim = 3
    k = Kriging(ndim=2, nvar=1, nsim=nsim, neglect_error=True, seed=1)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=1)
    k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=10.0)
    k.set_grid(coord=grid_coord)
    k.set_sim()
    k.set_search(ivar=1, sector_search=True)
    k.solve()

    est, var = k.get_results()
    nblocks = len(grid_coord)
    assert est.shape == (nblocks, nsim)
    assert var.shape == (nblocks,)   # kriging variance is per-block, not per-realization
    assert np.all(np.isfinite(est))
    assert np.all(var >= 0.0)
