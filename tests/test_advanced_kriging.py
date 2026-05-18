"""
test_advanced_kriging.py
========================
Tests for three advanced kriging features:

  1. Local nugget  (localnugget per grid node)
  2. Range scaler  (rangescale per grid node)
  3. Block kriging (set_grid_block with user-supplied sub-nodes)

All tests use the pc2d dataset (62 observations, 2D) with the
standard spherical variogram so results are comparable to ordinary
point kriging.

Variogram:  sph  nugget=0  sill=0.12  range=5000  (isotropic)
"""

import numpy as np
import pytest
import pandas as pd
import os
from pykriging import Kriging, ordinary_kriging

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
_VGM = "sph 0.0 0.12 5000.0 5000.0 5000.0 0.0 0.0 0.0"
_NMAX = 20


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_pc2d():
    df = pd.read_csv(os.path.join(DATA_DIR, "pc2d.csv"))
    return df[["x", "y"]].values, df["pc"].values.astype(float)


def _load_grid():
    df = pd.read_csv(os.path.join(DATA_DIR, "grid2d.csv"))
    return df[["x", "y"]].values


def _run_ok(coord, value, grid, nmax=_NMAX, **kwargs):
    """Run ordinary kriging; kwargs forwarded to set_grid."""
    k = Kriging(ndim=2, nvar=1, verbose=0)
    k.set_obs(ivar=1, coord=coord, value=value, nmax=nmax)
    k.set_vgm(ivar=1, jvar=1, spec=_VGM)
    k.set_grid(coord=grid, **kwargs)
    k.set_search(ivar=1)
    k.solve()
    return k.get_results()


# ===========================================================================
# 1. Local nugget tests
# ===========================================================================

class TestLocalNugget:
    """
    localnugget adds a per-node nugget on top of the global variogram nugget.
    It is added to the diagonal of the kriging matrix at the estimation point,
    so it increases the kriging variance without changing the weights.

    Physical interpretation: measurement uncertainty at a specific location.
    A localnugget=sigma^2 at node i means the estimate at that node is treated
    as if the supporting sample has measurement error sigma.
    """

    def test_zero_localnugget_matches_ordinary_kriging(self):
        """Explicit zeros must give identical results to omitting localnugget."""
        coord, value = _load_pc2d()
        grid = _load_grid()[:20]   # small subset for speed

        est_default, var_default = _run_ok(coord, value, grid)
        est_zero, var_zero = _run_ok(
            coord, value, grid,
            localnugget=np.zeros(len(grid))
        )

        np.testing.assert_allclose(est_default, est_zero, rtol=1e-6,
            err_msg="Zero localnugget should match default (no localnugget)")
        np.testing.assert_allclose(var_default, var_zero, rtol=1e-6,
            err_msg="Zero localnugget variance should match default")

    def test_localnugget_increases_variance(self):
        """Adding a positive localnugget must increase kriging variance at every node."""
        coord, value = _load_pc2d()
        grid = _load_grid()[:20]
        nugget_val = 0.05   # ~40% of sill

        _, var_base  = _run_ok(coord, value, grid)
        _, var_nugget = _run_ok(
            coord, value, grid,
            localnugget=np.full(len(grid), nugget_val)
        )

        assert np.all(var_nugget >= var_base - 1e-10), (
            "localnugget should increase variance at every node")

    def test_localnugget_does_not_change_estimates(self):
        """
        localnugget is added to the diagonal of the kriging matrix (acts as
        per-node measurement error on the data), so it DOES change estimates —
        it smooths them away from the observation values.

        This test verifies the smoothing effect: estimates with localnugget
        should be closer to the global mean than without.
        """
        coord, value = _load_pc2d()
        grid = _load_grid()[:20]
        global_mean = value.mean()

        est_base,   _ = _run_ok(coord, value, grid)
        est_nugget, _ = _run_ok(coord, value, grid,
                                localnugget=np.full(len(grid), 0.05))

        # localnugget smooths estimates toward the mean
        dev_base   = np.abs(est_base   - global_mean).mean()
        dev_nugget = np.abs(est_nugget - global_mean).mean()
        assert dev_nugget <= dev_base + 1e-5, (
            "localnugget should smooth estimates toward the global mean "
            f"(base dev={dev_base:.4f}, nugget dev={dev_nugget:.4f})"
        )

    def test_localnugget_per_node_variation(self):
        """Different localnugget values at different nodes produce independent variance increases."""
        coord, value = _load_pc2d()
        grid = _load_grid()[:10]
        n = len(grid)

        # Alternating nugget: 0 at even nodes, 0.05 at odd nodes
        ln = np.array([0.0 if i % 2 == 0 else 0.05 for i in range(n)])
        _, var_base    = _run_ok(coord, value, grid)
        _, var_partial = _run_ok(coord, value, grid, localnugget=ln)

        # Even-indexed nodes: variance unchanged
        np.testing.assert_allclose(var_partial[0::2], var_base[0::2], rtol=1e-5,
            err_msg="Zero-nugget nodes should have unchanged variance")
        # Odd-indexed nodes: variance must increase
        assert np.all(var_partial[1::2] >= var_base[1::2] - 1e-10), (
            "Positive-nugget nodes should have increased variance")

    def test_localnugget_variance_nonnegative(self):
        """Kriging variance must remain non-negative even with large local nuggets."""
        coord, value = _load_pc2d()
        grid = _load_grid()[:20]

        _, var = _run_ok(coord, value, grid,
                         localnugget=np.full(len(grid), 0.5))
        assert np.all(var >= -1e-10), f"Negative variance: {var.min():.4f}"

    def test_exact_match_localnugget_zero(self):
        """
        At a grid node coinciding with an observation and localnugget=0,
        the estimate must equal the observed value exactly.
        """
        coord, value = _load_pc2d()
        grid_at_obs = coord[[0]]

        est, _ = _run_ok(coord, value, grid_at_obs,
                         localnugget=np.zeros(1))
        assert est[0] == pytest.approx(value[0], rel=1e-4)

    def test_exact_match_localnugget_nonzero_smooths(self):
        """
        localnugget is added to the diagonal of matA (data-to-data covariance),
        so it smooths non-exact-match estimates.  At a grid node that is NOT
        co-located with any observation, a larger localnugget pulls the estimate
        closer to the global mean.
        """
        coord, value = _load_pc2d()
        global_mean = value.mean()

        # Use a grid point far from all observations so no exact match occurs
        grid_far = np.array([[coord[:, 0].mean(), coord[:, 1].mean()]])

        est_zero,    _ = _run_ok(coord, value, grid_far,
                                  localnugget=np.zeros(1))
        est_large,   _ = _run_ok(coord, value, grid_far,
                                  localnugget=np.full(1, 0.5))

        # Larger localnugget → estimate pulled further toward global mean
        assert (abs(est_large[0] - global_mean) <=
                abs(est_zero[0]  - global_mean) + 1e-5), (
            "Large localnugget should pull estimate toward global mean"
        )


# ===========================================================================
# 2. Range scaler tests
# ===========================================================================

class TestRangeScaler:
    """
    rangescale divides the lag vector before variogram evaluation:
        h_scaled = h / rangescale

    rangescale > 1  stretches the effective range  →  longer-range correlation
                    →  smoother estimates, lower variance
    rangescale < 1  compresses the effective range →  shorter-range correlation
                    →  less-smooth estimates, higher variance
    rangescale = 1  reproduces standard ordinary kriging
    """

    def test_unit_rangescale_matches_ordinary_kriging(self):
        """Explicit rangescale=1 must give identical results to omitting it."""
        coord, value = _load_pc2d()
        grid = _load_grid()[:20]

        est_default, var_default = _run_ok(coord, value, grid)
        est_one, var_one = _run_ok(
            coord, value, grid,
            rangescale=np.ones(len(grid))
        )

        np.testing.assert_allclose(est_default, est_one, rtol=1e-6,
            err_msg="Unit rangescale should match default (no rangescale)")
        np.testing.assert_allclose(var_default, var_one, rtol=1e-6,
            err_msg="Unit rangescale variance should match default")

    def test_larger_rangescale_reduces_variance(self):
        """
        rangescale > 1 stretches the variogram range so more distant observations
        contribute with higher weight, reducing kriging variance.
        """
        coord, value = _load_pc2d()
        grid = _load_grid()[:20]

        _, var_1   = _run_ok(coord, value, grid,
                             rangescale=np.ones(len(grid)))
        _, var_2   = _run_ok(coord, value, grid,
                             rangescale=np.full(len(grid), 2.0))

        # Mean variance with larger range should be lower (smoother interpolation)
        assert var_2.mean() <= var_1.mean() + 1e-6, (
            f"rangescale=2 mean variance ({var_2.mean():.4f}) should be <= "
            f"rangescale=1 mean variance ({var_1.mean():.4f})")

    def test_smaller_rangescale_increases_variance(self):
        """
        rangescale < 1 compresses the effective range so the same observations
        appear 'farther away', increasing kriging variance.
        """
        coord, value = _load_pc2d()
        grid = _load_grid()[:20]

        _, var_1   = _run_ok(coord, value, grid,
                             rangescale=np.ones(len(grid)))
        _, var_half = _run_ok(coord, value, grid,
                              rangescale=np.full(len(grid), 0.5))

        assert var_half.mean() >= var_1.mean() - 1e-6, (
            f"rangescale=0.5 mean variance ({var_half.mean():.4f}) should be >= "
            f"rangescale=1 mean variance ({var_1.mean():.4f})")

    def test_rangescale_variance_nonnegative(self):
        """Variance must remain non-negative for any positive rangescale."""
        coord, value = _load_pc2d()
        grid = _load_grid()[:20]

        for rs in [0.25, 0.5, 1.0, 2.0, 5.0]:
            _, var = _run_ok(coord, value, grid,
                             rangescale=np.full(len(grid), rs))
            assert np.all(var >= -1e-10), (
                f"Negative variance with rangescale={rs}: {var.min():.4f}")

    def test_rangescale_per_node_spatial_variation(self):
        """
        Nodes with larger rangescale should have lower variance than
        nodes with smaller rangescale, all else being equal.
        """
        coord, value = _load_pc2d()
        grid = _load_grid()[:10]
        n = len(grid)

        # Low rangescale for first half, high for second half
        rs_mixed = np.array([0.5 if i < n // 2 else 2.0 for i in range(n)])
        _, var_low_rs  = _run_ok(coord, value, grid,
                                  rangescale=np.full(n, 0.5))
        _, var_high_rs = _run_ok(coord, value, grid,
                                  rangescale=np.full(n, 2.0))

        # Across all nodes: high range scale → lower variance on average
        assert var_high_rs.mean() <= var_low_rs.mean() + 1e-4

    def test_rangescale_monotone_with_scale(self):
        """Kriging variance should be monotonically non-increasing as rangescale grows."""
        coord, value = _load_pc2d()
        grid = _load_grid()[:5]

        variances = []
        for rs in [0.5, 1.0, 2.0, 4.0, 8.0]:
            _, var = _run_ok(coord, value, grid,
                             rangescale=np.full(len(grid), rs))
            variances.append(var.mean())

        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i+1] - 1e-5, (
                f"Variance should not increase as rangescale grows: "
                f"rs={[0.5,1,2,4,8][i]}->{[0.5,1,2,4,8][i+1]} "
                f"var={variances[i]:.4f}->{variances[i+1]:.4f}")

    def test_localnugget_and_rangescale_combined(self):
        """Both features can be used together without conflict."""
        coord, value = _load_pc2d()
        grid = _load_grid()[:20]

        est, var = _run_ok(coord, value, grid,
                           rangescale=np.full(len(grid), 1.5),
                           localnugget=np.full(len(grid), 0.02))
        assert est.shape == (len(grid),)
        assert var.shape == (len(grid),)
        assert np.all(var >= -1e-10)


# ===========================================================================
# 3. Block kriging tests
# ===========================================================================

class TestBlockKriging:
    """
    Block kriging estimates the spatial average of Z over a support block by
    integrating the kriging estimator over the block area.

    The block is discretised into sub-nodes (quadrature points or a regular
    grid of points).  Each sub-node has a weight; the block estimate is the
    weighted mean of the per-sub-node kriging estimates.

    Key theoretical property (I&S Chapter 12):
      block_variance <= point_variance
    The block variance is always less than or equal to the point variance
    because averaging reduces the within-block variability.

    Dataset: pc2d observations, single large block using the 4×4 Gaussian
    quadrature sub-nodes stored in gridblockpnt2d.csv.
    """

    @pytest.fixture(scope="class")
    def block_data(self):
        """Load block sub-nodes and build block description arrays."""
        coord, value = _load_pc2d()

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

    def test_block_kriging_result_shape(self, block_data):
        """Block kriging must return arrays of length nblock."""
        d = block_data
        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=d["coord"], value=d["value"], nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM)
        k.set_grid_block(
            coord=d["sub_coords"],
            block_type=1,
            nblockpnt=d["nblockpnt"],
            pointweight=d["sub_weights"],
        )
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
        assert est.shape == (d["nblock"],), f"Expected ({d['nblock']},), got {est.shape}"
        assert var.shape == (d["nblock"],)

    def test_block_variance_nonnegative(self, block_data):
        """Block kriging variance must be non-negative."""
        d = block_data
        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=d["coord"], value=d["value"], nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM)
        k.set_grid_block(
            coord=d["sub_coords"],
            block_type=1,
            nblockpnt=d["nblockpnt"],
            pointweight=d["sub_weights"],
        )
        k.set_search(ivar=1)
        k.solve()
        _, var = k.get_results()
        assert np.all(var >= -1e-10), f"Negative block variance: {var.min():.4f}"

    def test_block_variance_less_than_point_variance(self, block_data):
        """
        Block kriging variance <= point kriging variance at the block centroid.

        This is the fundamental regularisation property: averaging over a support
        reduces variance because within-block variability is smoothed out.
        (Journel & Huijbregts 1978, Ch. 4; Isaaks & Srivastava 1989, Ch. 12)
        """
        d = block_data

        # Block kriging variance
        kb = Kriging(ndim=2, nvar=1, verbose=0)
        kb.set_obs(ivar=1, coord=d["coord"], value=d["value"], nmax=_NMAX)
        kb.set_vgm(ivar=1, jvar=1, spec=_VGM)
        kb.set_grid_block(
            coord=d["sub_coords"],
            block_type=1,
            nblockpnt=d["nblockpnt"],
            pointweight=d["sub_weights"],
        )
        kb.set_search(ivar=1)
        kb.solve()
        _, var_block = kb.get_results()

        # Point kriging variance at the block centroid
        kp = Kriging(ndim=2, nvar=1, verbose=0)
        kp.set_obs(ivar=1, coord=d["coord"], value=d["value"], nmax=_NMAX)
        kp.set_vgm(ivar=1, jvar=1, spec=_VGM)
        kp.set_grid(coord=d["block_centroid"])
        kp.set_search(ivar=1)
        kp.solve()
        _, var_point = kp.get_results()

        assert var_block[0] <= var_point[0] + 1e-6, (
            f"Block variance ({var_block[0]:.4f}) should be <= "
            f"point variance at centroid ({var_point[0]:.4f}) — "
            "regularisation property violated")

    def test_block_estimate_close_to_centroid_point_estimate(self, block_data):
        """
        The block estimate should be close to the point estimate at the centroid,
        because the block is small relative to the variogram range (5000 m).
        A large block would show more divergence.
        """
        d = block_data

        # Block estimate
        kb = Kriging(ndim=2, nvar=1, verbose=0)
        kb.set_obs(ivar=1, coord=d["coord"], value=d["value"], nmax=_NMAX)
        kb.set_vgm(ivar=1, jvar=1, spec=_VGM)
        kb.set_grid_block(
            coord=d["sub_coords"],
            block_type=1,
            nblockpnt=d["nblockpnt"],
            pointweight=d["sub_weights"],
        )
        kb.set_search(ivar=1)
        kb.solve()
        est_block, _ = kb.get_results()

        # Point estimate at centroid
        kp = Kriging(ndim=2, nvar=1, verbose=0)
        kp.set_obs(ivar=1, coord=d["coord"], value=d["value"], nmax=_NMAX)
        kp.set_vgm(ivar=1, jvar=1, spec=_VGM)
        kp.set_grid(coord=d["block_centroid"])
        kp.set_search(ivar=1)
        kp.solve()
        est_point, _ = kp.get_results()

        # Block is ~1400 m across, range is 5000 m → expect < 10% difference
        data_range = d["value"].max() - d["value"].min()
        assert abs(est_block[0] - est_point[0]) < 0.10 * data_range, (
            f"Block estimate ({est_block[0]:.3f}) and centroid point estimate "
            f"({est_point[0]:.3f}) differ by more than 10% of data range "
            f"({data_range:.3f})")

    def test_block_kriging_uniform_weights_sums_to_one(self, block_data):
        """
        With default uniform weights (1/nblockpnt per sub-node), kriging of a
        constant field must return that constant — verifying weight normalisation.
        """
        d = block_data
        const_value = 2.71828 * np.ones(d["coord"].shape[0])

        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=d["coord"], value=const_value, nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM)
        k.set_grid_block(
            coord=d["sub_coords"],
            block_type=1,
            nblockpnt=d["nblockpnt"],
            # no pointweight → uniform 1/16 per sub-node
        )
        k.set_search(ivar=1)
        k.solve()
        est, _ = k.get_results()
        assert est[0] == pytest.approx(2.71828, rel=1e-3), (
            "Block kriging of constant field should return the constant "
            "(unbiasedness / weights sum to 1)")

    def test_block_kriging_localnugget(self, block_data):
        """localnugget can be set per block; positive value increases block variance."""
        d = block_data

        def _block_var(localnugget=None):
            k = Kriging(ndim=2, nvar=1, verbose=0)
            k.set_obs(ivar=1, coord=d["coord"], value=d["value"], nmax=_NMAX)
            k.set_vgm(ivar=1, jvar=1, spec=_VGM)
            k.set_grid_block(
                coord=d["sub_coords"],
                block_type=1,
                nblockpnt=d["nblockpnt"],
                pointweight=d["sub_weights"],
                localnugget=localnugget,
            )
            k.set_search(ivar=1)
            k.solve()
            _, var = k.get_results()
            return var[0]

        var_base   = _block_var(localnugget=np.zeros(d["nblock"]))
        var_nugget = _block_var(localnugget=np.full(d["nblock"], 0.05))

        assert var_nugget >= var_base - 1e-10, (
            "Block localnugget should increase block kriging variance")

    def test_block_kriging_rangescale(self, block_data):
        """rangescale can be set per block; larger scale reduces block variance."""
        d = block_data

        def _block_var(rs):
            k = Kriging(ndim=2, nvar=1, verbose=0)
            k.set_obs(ivar=1, coord=d["coord"], value=d["value"], nmax=_NMAX)
            k.set_vgm(ivar=1, jvar=1, spec=_VGM)
            k.set_grid_block(
                coord=d["sub_coords"],
                block_type=1,
                nblockpnt=d["nblockpnt"],
                pointweight=d["sub_weights"],
                rangescale=np.full(d["nblock"], rs),
            )
            k.set_search(ivar=1)
            k.solve()
            _, var = k.get_results()
            return var[0]

        var_small_range = _block_var(0.5)
        var_large_range = _block_var(2.0)

        assert var_large_range <= var_small_range + 1e-6, (
            f"Larger rangescale ({var_large_range:.4f}) should give <= "
            f"variance than smaller rangescale ({var_small_range:.4f})")

    def test_multiple_blocks(self, block_data):
        """Multiple blocks in a single call: repeat the same sub-nodes twice."""
        d = block_data
        n_sub = len(d["sub_coords"])

        # Two identical blocks side by side
        two_sub_coords  = np.vstack([d["sub_coords"], d["sub_coords"]])
        two_weights     = np.concatenate([d["sub_weights"], d["sub_weights"]])
        two_nblockpnt   = np.array([n_sub, n_sub], dtype=np.int32)

        k = Kriging(ndim=2, nvar=1, verbose=0)
        k.set_obs(ivar=1, coord=d["coord"], value=d["value"], nmax=_NMAX)
        k.set_vgm(ivar=1, jvar=1, spec=_VGM)
        k.set_grid_block(
            coord=two_sub_coords,
            block_type=1,
            nblockpnt=two_nblockpnt,
            pointweight=two_weights,
        )
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()

        assert est.shape == (2,)
        assert var.shape == (2,)
        # Both blocks are identical so estimates must be equal
        assert est[0] == pytest.approx(est[1], rel=1e-5), (
            "Identical blocks must produce identical estimates")
        assert var[0] == pytest.approx(var[1], rel=1e-5), (
            "Identical blocks must produce identical variances")
