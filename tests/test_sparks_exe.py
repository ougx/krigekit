"""
test_sparks_exe.py
==================
Integration tests for the sparks.exe CLI executable.

Each test launches the binary as a subprocess, captures stdout/stderr, and
checks output values, error messages, and regression results against known-good
data stored in test_data/.

Prerequisites
-------------
  - bin/sparks.exe (Windows) or bin/sparks (Linux/macOS) must be compiled.
  - test_data/ must contain: obs_simple.csv, grid_simple.csv, pc2d.csv,
    grid2d.csv, path_simple.csv, sample_simple.csv.

Run with::

    pytest tests/test_sparks_exe.py
"""

import io
import os
import subprocess
import textwrap

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN_DIR = os.path.join(_ROOT, "bin")
_DATA    = os.path.join(_ROOT, "test_data")


def _exe_path():
    for name in ("sparks.exe", "sparks"):
        p = os.path.join(_BIN_DIR, name)
        if os.path.isfile(p):
            return p
    return None


def _data(filename):
    return os.path.join(_DATA, filename)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sparks(request):
    """Absolute path to sparks executable; skips the whole session if absent."""
    p = _exe_path()
    if p is None:
        pytest.skip("sparks executable not found in bin/")
    return p


def _run(sparks_path, *args, timeout=60):
    """Invoke sparks with *args and return the CompletedProcess."""
    cmd = [sparks_path] + [str(a) for a in args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _parse_xy(stdout):
    """Parse CSV output produced by -xy into a DataFrame."""
    return pd.read_csv(io.StringIO(stdout.strip()))


def _parse_values(stdout):
    """
    Parse default (no -xy) output: one float per line.
    Returns a 1-D float array.
    """
    lines = [l.strip() for l in stdout.strip().splitlines() if l.strip()]
    return np.array([float(l) for l in lines])


# Variogram tokens used repeatedly
_VGM_SIMPLE = ("sph", "100.0", "0.09", "0.01")   # fitted to obs_simple
_VGM_PC2D   = ("sph", "5000.0", "0.12", "0.0")   # fitted to pc2d


# ===========================================================================
# 1.  Help / smoke
# ===========================================================================

class TestHelpAndVersion:

    def test_help_exits_cleanly(self, sparks, tmp_path):
        cp = _run(sparks, "-h")
        assert cp.returncode == 0
        assert "SPARKS" in cp.stdout or "sparks" in cp.stdout.lower()

    def test_help_lists_required_flags(self, sparks):
        cp = _run(sparks, "-h")
        assert "-d" in cp.stdout or "--dim" in cp.stdout
        assert "-of" in cp.stdout or "--obsfile1" in cp.stdout
        assert "-v1" in cp.stdout or "--vario1" in cp.stdout


# ===========================================================================
# 2.  Simple ordinary kriging (obs_simple / grid_simple)
# ===========================================================================

class TestSimpleOrdinaryKriging:
    """Five observations → three grid nodes, 2-D ordinary kriging."""

    _DIM = ("-d", "2", "5", "3", "0", "0")

    def _ok(self, sparks, *extra):
        return _run(
            sparks,
            *self._DIM,
            "-of", _data("obs_simple.csv"),
            "-bf", _data("grid_simple.csv"),
            "-v1", *_VGM_SIMPLE,
            "-u",
            *extra,
        )

    def test_produces_three_estimate_lines(self, sparks):
        cp = self._ok(sparks)
        assert cp.returncode == 0
        vals = _parse_values(cp.stdout)
        assert vals.shape == (3,)

    def test_estimates_in_observation_range(self, sparks):
        obs = pd.read_csv(_data("obs_simple.csv"))
        lo, hi = obs["z"].min(), obs["z"].max()
        cp = self._ok(sparks)
        vals = _parse_values(cp.stdout)
        margin = 0.10 * (hi - lo)
        assert vals.min() >= lo - margin
        assert vals.max() <= hi + margin

    def test_writexy_produces_header_and_three_rows(self, sparks):
        cp = self._ok(sparks, "-xy")
        assert cp.returncode == 0
        df = _parse_xy(cp.stdout)
        assert df.shape[0] == 3
        assert "estimate" in df.columns
        assert "variance" in df.columns

    def test_variance_nonnegative(self, sparks):
        cp = self._ok(sparks, "-xy")
        df = _parse_xy(cp.stdout)
        assert (df["variance"] >= 0).all(), \
            f"Negative variance: {df['variance'].tolist()}"

    def test_exact_match_at_observation_location(self, sparks):
        """Grid node co-located with an observation should return that value."""
        obs = pd.read_csv(_data("obs_simple.csv"))
        # Write a one-node grid at obs[0]
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, dir=_DATA
        ) as fh:
            fh.write("igrid,x,y\n")
            fh.write(f"1,{obs['x'].iloc[0]},{obs['y'].iloc[0]}\n")
            tmp = fh.name
        try:
            cp = _run(
                sparks,
                "-d", "2", "5", "1", "0", "0",
                "-of", _data("obs_simple.csv"),
                "-bf", tmp,
                "-v1", *_VGM_SIMPLE,
                "-u",
            )
            assert cp.returncode == 0
            vals = _parse_values(cp.stdout)
            assert vals[0] == pytest.approx(obs["z"].iloc[0], rel=1e-3)
        finally:
            os.remove(tmp)

    def test_constant_field_returns_constant(self, sparks):
        """
        With a constant observation field, ordinary kriging must reproduce
        that constant at every grid node (weights sum to 1 under -u).
        """
        import tempfile
        constant = 3.14
        obs_df = pd.read_csv(_data("obs_simple.csv"))
        obs_df = obs_df.copy()
        obs_df["z"] = constant
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, dir=_DATA
        ) as fh:
            obs_df.to_csv(fh, index=False)
            tmp = fh.name
        try:
            cp = _run(
                sparks,
                "-d", "2", "5", "3", "0", "0",
                "-of", tmp,
                "-bf", _data("grid_simple.csv"),
                "-v1", *_VGM_SIMPLE,
                "-u",
            )
            assert cp.returncode == 0
            vals = _parse_values(cp.stdout)
            assert vals == pytest.approx(constant, rel=1e-4)
        finally:
            os.remove(tmp)

    def test_simple_kriging_without_unbias_flag(self, sparks):
        """Without -u (simple kriging, sk_mean=0) the program should still run."""
        cp = _run(
            sparks,
            "-d", "2", "5", "3", "0", "0",
            "-of", _data("obs_simple.csv"),
            "-bf", _data("grid_simple.csv"),
            "-v1", *_VGM_SIMPLE,
        )
        assert cp.returncode == 0
        vals = _parse_values(cp.stdout)
        assert vals.shape == (3,)

    def test_ordinary_differs_from_simple_kriging(self, sparks):
        """OK and SK must produce different estimates (SK is biased toward 0 here)."""
        cp_ok = self._ok(sparks)
        cp_sk = _run(
            sparks,
            "-d", "2", "5", "3", "0", "0",
            "-of", _data("obs_simple.csv"),
            "-bf", _data("grid_simple.csv"),
            "-v1", *_VGM_SIMPLE,
        )
        ok_vals = _parse_values(cp_ok.stdout)
        sk_vals = _parse_values(cp_sk.stdout)
        assert not np.allclose(ok_vals, sk_vals), \
            "OK and SK estimates should differ when obs mean != 0"


# ===========================================================================
# 3.  Error handling
# ===========================================================================

class TestErrorHandling:
    """sparks.exe must print a clear error message when arguments are invalid."""

    def test_missing_variogram_message(self, sparks):
        cp = _run(
            sparks,
            "-d", "2", "5", "3", "0", "0",
            "-of", _data("obs_simple.csv"),
            "-bf", _data("grid_simple.csv"),
        )
        assert "Error" in cp.stderr or "Error" in cp.stdout

    def test_missing_obsfile_flag_message(self, sparks):
        cp = _run(
            sparks,
            "-d", "2", "5", "3", "0", "0",
            "-bf", _data("grid_simple.csv"),
            "-v1", *_VGM_SIMPLE,
            "-u",
        )
        combined = cp.stdout + cp.stderr
        assert "Error" in combined

    def test_missing_d_flag_message(self, sparks):
        cp = _run(
            sparks,
            "-of", _data("obs_simple.csv"),
            "-bf", _data("grid_simple.csv"),
            "-v1", *_VGM_SIMPLE,
            "-u",
        )
        combined = cp.stdout + cp.stderr
        assert "Error" in combined

    def test_missing_blockfile_message(self, sparks):
        """Without -bf (and not in LOO-CV mode) sparks must report an error."""
        cp = _run(
            sparks,
            "-d", "2", "5", "3", "0", "0",
            "-of", _data("obs_simple.csv"),
            "-v1", *_VGM_SIMPLE,
            "-u",
        )
        combined = cp.stdout + cp.stderr
        assert "Error" in combined

    def test_nobs1_zero_message(self, sparks):
        """nobs1=0 is invalid — sparks must print an error."""
        cp = _run(
            sparks,
            "-d", "2", "0", "3", "0", "0",
            "-of", _data("obs_simple.csv"),
            "-bf", _data("grid_simple.csv"),
            "-v1", *_VGM_SIMPLE,
            "-u",
        )
        combined = cp.stdout + cp.stderr
        assert "Error" in combined


# ===========================================================================
# 4.  pc2d regression
# ===========================================================================

class TestPC2DRegression:
    """
    Regression tests against pre-computed reference values in grid2d.csv
    (column "estimate") and pc2d.csv (column "loo-cv").

    Use only the first 20 grid nodes to keep run time short while exercising
    the full kriging solver.
    """

    _NBLOCK = 20   # first N rows of grid2d used for regression

    def _run_ok(self, sparks, *extra):
        return _run(
            sparks,
            "-d", "2", "62", str(self._NBLOCK), "0", "0",
            "-of", _data("pc2d.csv"),
            "-bf", _data("grid2d.csv"),
            "-v1", *_VGM_PC2D,
            "-u",
            "-n1", "62",
            "-xy",
            *extra,
        )

    def test_output_shape(self, sparks):
        cp = self._run_ok(sparks)
        assert cp.returncode == 0
        df = _parse_xy(cp.stdout)
        assert df.shape[0] == self._NBLOCK

    def test_estimate_matches_reference(self, sparks):
        ref = pd.read_csv(_data("grid2d.csv"))
        ref_col = "estimate" if "estimate" in ref.columns else "result"
        ref_vals = ref[ref_col].values[:self._NBLOCK]

        cp = self._run_ok(sparks)
        df = _parse_xy(cp.stdout)
        corr = np.corrcoef(df["estimate"].values, ref_vals)[0, 1]
        assert corr > 0.999, \
            f"Correlation with reference = {corr:.5f} (expected > 0.999)"

    def test_variance_nonnegative(self, sparks):
        cp = self._run_ok(sparks)
        df = _parse_xy(cp.stdout)
        assert (df["variance"] >= 0).all()

    def test_estimate_in_data_range(self, sparks):
        obs = pd.read_csv(_data("pc2d.csv"))
        lo, hi = obs["pc"].min(), obs["pc"].max()
        margin = 0.10 * (hi - lo)
        cp = self._run_ok(sparks)
        df = _parse_xy(cp.stdout)
        assert df["estimate"].min() >= lo - margin
        assert df["estimate"].max() <= hi + margin


# ===========================================================================
# 5.  Leave-one-out cross-validation (LOO-CV)
# ===========================================================================

class TestLeaveOneOutCV:
    """
    Leave-one-out CV mode (-cv): grid equals observations; each obs is
    estimated from its neighbours only.
    """

    def test_loocv_produces_one_row_per_obs(self, sparks):
        cp = _run(
            sparks,
            "-d", "2", "62", "0", "0", "0",
            "-of", _data("pc2d.csv"),
            "-v1", *_VGM_PC2D,
            "-u",
            "-n1", "62",
            "-cv",
            "-xy",
        )
        assert cp.returncode == 0
        df = _parse_xy(cp.stdout)
        assert df.shape[0] == 62

    def test_loocv_has_observed_column(self, sparks):
        cp = _run(
            sparks,
            "-d", "2", "62", "0", "0", "0",
            "-of", _data("pc2d.csv"),
            "-v1", *_VGM_PC2D,
            "-u",
            "-n1", "62",
            "-cv",
            "-xy",
        )
        df = _parse_xy(cp.stdout)
        assert "observed" in df.columns
        assert "estimate" in df.columns

    def test_loocv_estimates_match_reference(self, sparks):
        ref = pd.read_csv(_data("pc2d.csv"))
        cp = _run(
            sparks,
            "-d", "2", "62", "0", "0", "0",
            "-of", _data("pc2d.csv"),
            "-v1", *_VGM_PC2D,
            "-u",
            "-n1", "62",
            "-cv",
            "-xy",
        )
        df = _parse_xy(cp.stdout)
        corr = np.corrcoef(df["estimate"].values, ref["loo-cv"].values)[0, 1]
        assert corr > 0.999, \
            f"LOO-CV correlation with reference = {corr:.5f}"

    def test_loocv_simple_dataset_variance_nonnegative(self, sparks):
        cp = _run(
            sparks,
            "-d", "2", "5", "0", "0", "0",
            "-of", _data("obs_simple.csv"),
            "-v1", *_VGM_SIMPLE,
            "-u",
            "-cv",
            "-xy",
        )
        assert cp.returncode == 0
        df = _parse_xy(cp.stdout)
        assert (df["variance"] >= 0).all()


# ===========================================================================
# 6.  Sequential Gaussian Simulation (SGSIM)
# ===========================================================================

class TestSGSIM:
    """Tests for the -s (simulation) mode on the small synthetic dataset."""

    _DIM = ("-d", "2", "5", "3", "0", "0")
    _OBS = ("-of", _data("obs_simple.csv"))
    _BLK = ("-bf", _data("grid_simple.csv"))
    _VGM = ("-v1", *_VGM_SIMPLE)
    _PATH = ("-pf", _data("path_simple.csv"))
    _SAM  = ("-sf", _data("sample_simple.csv"))

    def _sgsim(self, sparks, seed=42, *extra):
        return _run(
            sparks,
            *self._DIM, *self._OBS, *self._BLK, *self._VGM,
            "-u", "-s", "-sd", str(seed), "-n1", "5",
            *self._PATH, *self._SAM,
            *extra,
        )

    def test_sgsim_produces_three_lines(self, sparks):
        cp = self._sgsim(sparks)
        assert cp.returncode == 0
        vals = _parse_values(cp.stdout)
        assert vals.shape == (3,)

    def test_sgsim_writexy_has_estimate1_column(self, sparks):
        cp = self._sgsim(sparks, 42, "-xy")
        df = _parse_xy(cp.stdout)
        assert "estimate1" in df.columns

    def test_sgsim_values_in_physical_range(self, sparks):
        """SGSIM values shouldn't be wildly outside the conditioning data."""
        obs = pd.read_csv(_data("obs_simple.csv"))
        lo, hi = obs["z"].min() - 1.0, obs["z"].max() + 1.0
        cp = self._sgsim(sparks)
        vals = _parse_values(cp.stdout)
        assert vals.min() >= lo
        assert vals.max() <= hi

    def test_sgsim_seed_reproducibility(self, sparks):
        """Same seed + same path/sample → identical output."""
        cp1 = self._sgsim(sparks, 7)
        cp2 = self._sgsim(sparks, 7)
        v1 = _parse_values(cp1.stdout)
        v2 = _parse_values(cp2.stdout)
        np.testing.assert_array_equal(v1, v2,
            err_msg="SGSIM with same seed must reproduce identical results")

    def test_sgsim_differs_from_kriging(self, sparks):
        """A simulation realisation should differ from the kriging estimate."""
        cp_ok = _run(
            sparks,
            *self._DIM, *self._OBS, *self._BLK, *self._VGM, "-u",
        )
        cp_sg = self._sgsim(sparks)
        ok_vals = _parse_values(cp_ok.stdout)
        sg_vals = _parse_values(cp_sg.stdout)
        assert not np.allclose(ok_vals, sg_vals), \
            "SGSIM realisation should differ from kriging estimate"


# ===========================================================================
# 7.  Bounds clipping (-bd)
# ===========================================================================

class TestBoundsClipping:

    _DIM = ("-d", "2", "5", "3", "0", "0")

    def _run_bounded(self, sparks, lower, upper):
        return _run(
            sparks,
            *self._DIM,
            "-of", _data("obs_simple.csv"),
            "-bf", _data("grid_simple.csv"),
            "-v1", *_VGM_SIMPLE,
            "-u",
            "-bd", str(lower), str(upper),
            "-xy",
        )

    def test_upper_bound_clamps_high_estimates(self, sparks):
        upper = 1.1
        cp = self._run_bounded(sparks, 0.0, upper)
        df = _parse_xy(cp.stdout)
        assert df["estimate"].max() <= upper + 1e-6, \
            f"estimate {df['estimate'].max():.4f} exceeds upper bound {upper}"

    def test_lower_bound_clamps_low_estimates(self, sparks):
        lower = 1.0
        cp = self._run_bounded(sparks, lower, 2.0)
        df = _parse_xy(cp.stdout)
        assert df["estimate"].min() >= lower - 1e-6, \
            f"estimate {df['estimate'].min():.4f} below lower bound {lower}"

    def test_bounds_do_not_affect_variance(self, sparks):
        """Variance should be the same regardless of clipping."""
        cp_unb = _run(
            sparks,
            *self._DIM,
            "-of", _data("obs_simple.csv"),
            "-bf", _data("grid_simple.csv"),
            "-v1", *_VGM_SIMPLE,
            "-u", "-xy",
        )
        cp_bnd = self._run_bounded(sparks, 0.9, 1.2)
        df_u = _parse_xy(cp_unb.stdout)
        df_b = _parse_xy(cp_bnd.stdout)
        np.testing.assert_allclose(df_u["variance"].values, df_b["variance"].values,
            rtol=1e-5, err_msg="Bounds clipping must not change kriging variance")


# ===========================================================================
# 8.  Anisotropy flags
# ===========================================================================

class TestAnisotropy:
    """Verify anisotropy options run without error and affect results."""

    _BASE = [
        "-d", "2", "5", "3", "0", "0",
        "-of", _data("obs_simple.csv"),
        "-bf", _data("grid_simple.csv"),
        "-v1", *_VGM_SIMPLE,
        "-u",
    ]

    def test_azimuth_runs_cleanly(self, sparks):
        cp = _run(sparks, *self._BASE, "-a1", "45")
        assert cp.returncode == 0
        vals = _parse_values(cp.stdout)
        assert vals.shape == (3,)

    def test_anis_ratio_runs_cleanly(self, sparks):
        cp = _run(sparks, *self._BASE, "-a1", "30", "-s1", "0.5")
        assert cp.returncode == 0
        vals = _parse_values(cp.stdout)
        assert vals.shape == (3,)

    def test_anisosearch_flag_runs_cleanly(self, sparks):
        cp = _run(sparks, *self._BASE, "-a1", "30", "-s1", "0.5", "-as")
        assert cp.returncode == 0
        vals = _parse_values(cp.stdout)
        assert vals.shape == (3,)

    def test_anisotropy_changes_estimates(self, sparks):
        """Rotating the search ellipse should change estimates."""
        cp_iso = _run(sparks, *self._BASE, "-xy")
        cp_ani = _run(sparks, *self._BASE, "-a1", "45", "-s1", "0.3", "-xy")
        df_iso = _parse_xy(cp_iso.stdout)
        df_ani = _parse_xy(cp_ani.stdout)
        assert not np.allclose(
            df_iso["estimate"].values, df_ani["estimate"].values
        ), "Anisotropy should change at least some estimates"


# ===========================================================================
# 9.  Output format options
# ===========================================================================

class TestOutputFormat:
    """Tests for -xy, output to file, and custom -fm format string."""

    _DIM = ("-d", "2", "5", "3", "0", "0")
    _COMMON = [
        "-of", _data("obs_simple.csv"),
        "-bf", _data("grid_simple.csv"),
        "-v1", *_VGM_SIMPLE,
        "-u",
    ]

    def test_default_output_is_values_only(self, sparks):
        """Without -xy there must be no header and no commas."""
        cp = _run(sparks, *self._DIM, *self._COMMON)
        for line in cp.stdout.strip().splitlines():
            assert "," not in line, "Default output should not contain commas"
            float(line.strip())  # must be parseable as a single float

    def test_writexy_output_has_five_columns(self, sparks):
        """With -xy: igrid, x, y, estimate, variance (5 columns)."""
        cp = _run(sparks, *self._DIM, *self._COMMON, "-xy")
        df = _parse_xy(cp.stdout)
        assert list(df.columns) == ["igrid", "x", "y", "estimate", "variance"]

    def test_writexy_coordinates_match_grid_file(self, sparks):
        """x/y values in -xy output must equal the grid file coordinates."""
        grid = pd.read_csv(_data("grid_simple.csv"))
        cp = _run(sparks, *self._DIM, *self._COMMON, "-xy")
        df = _parse_xy(cp.stdout)
        np.testing.assert_allclose(df["x"].values, grid["x"].values, rtol=1e-5)
        np.testing.assert_allclose(df["y"].values, grid["y"].values, rtol=1e-5)

    def test_output_to_file(self, sparks, tmp_path):
        """Specifying an output file path must write results there."""
        out_file = str(tmp_path / "out.txt")
        cp = _run(sparks, *self._DIM, *self._COMMON, out_file)
        assert cp.returncode == 0
        assert os.path.isfile(out_file)
        with open(out_file) as fh:
            content = fh.read()
        vals = _parse_values(content)
        assert vals.shape == (3,)

    def test_output_to_file_matches_stdout(self, sparks, tmp_path):
        """File output and stdout output must contain the same estimates."""
        out_file = str(tmp_path / "out2.txt")
        cp_file = _run(sparks, *self._DIM, *self._COMMON, out_file)
        cp_std  = _run(sparks, *self._DIM, *self._COMMON)
        with open(out_file) as fh:
            file_vals = _parse_values(fh.read())
        std_vals = _parse_values(cp_std.stdout)
        np.testing.assert_allclose(file_vals, std_vals, rtol=1e-8)


# ===========================================================================
# 10. Namelist mode (-nl)
# ===========================================================================

class TestNamelistMode:
    """
    Verify that the namelist (-nl) input path produces the same result as
    the equivalent CLI invocation on the obs_simple / grid_simple dataset.
    """

    def test_namelist_matches_cli_result(self, sparks, tmp_path):
        # Reference: run via CLI
        cp_cli = _run(
            sparks,
            "-d", "2", "5", "3", "0", "0",
            "-of", _data("obs_simple.csv"),
            "-bf", _data("grid_simple.csv"),
            "-v1", *_VGM_SIMPLE,
            "-u",
        )
        cli_vals = _parse_values(cp_cli.stdout)

        # Write a namelist file that encodes the same configuration
        nml_content = textwrap.dedent(f"""\
            &input_output
              obsfile1 = '{_data("obs_simple.csv")}'
              blockfile = '{_data("grid_simple.csv")}'
            /
            &dims
              ndim = 2
              nobs1 = 5
              nblock = 3
              nobs2 = 0
              ndrift = 0
            /
            &krige_opt
              unbias = 1
            /
            &variograms
              vgm_spec1(1) = 'sph 100.0 0.09 0.01'
            /
            &anisotropy
            /
            &flags
            /
        """)
        nml_file = tmp_path / "sparks.nml"
        nml_file.write_text(nml_content)

        cp_nml = _run(sparks, "-nl", str(nml_file))
        assert cp_nml.returncode == 0
        nml_vals = _parse_values(cp_nml.stdout)
        np.testing.assert_allclose(nml_vals, cli_vals, rtol=1e-6,
            err_msg="Namelist mode must reproduce CLI results exactly")
