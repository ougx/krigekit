"""
test_bk.py
========================
Tests for three advanced kriging features:

  1. Block kriging (set_grid_block with user-supplied sub-nodes)

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




# ===========================================================================
# Block kriging tests
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
