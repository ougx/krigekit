# krigekit Test Suite

This directory contains the automated test suite for the **krigekit** library and its associated CLI executable (`sparks`). The suite uses `pytest` and provides comprehensive coverage for core geostatistical functions, input validation, object reuse, and Fortran/C-backend integration.

## Requirements and Prerequisites

To run these tests, you need the following dependencies installed in your Python environment:

* `pytest`
* `numpy`
* `pandas`

**Backend Requirements:**

* **Compiled Library:** The Python wrapper relies on a compiled Fortran/C backend (e.g., `libkriging`). Tests will fail or skip if this is not present in your environment.
* **CLI Executable:** `test_sparks_exe.py` requires the compiled `sparks.exe` (Windows) or `sparks` (Linux/macOS) binary to be located in the `../bin/` directory relative to this folder.
* **Test Data:** Most tests rely on reference datasets (e.g., `pc2d.csv`, `walker.csv`, `obs_simple.csv`) located in the `../test_data/` directory.

---

## Test Modules Overview

The test suite is divided into specific modules covering different geostatistical capabilities and API features.

| Test File | Coverage & Features |
| --- | --- |
| **`test_api.py`** | Input validation, edge cases, error handling, bounds clipping, universal kriging (drift), and memory safety during object reuse. |
| **`test_weight_store.py`** | In-memory weight store: array shapes/dtypes, weight correctness (sum-to-1, estimate reconstruction), factor-file round-trip, `set_weights()` in-memory reuse (including variance storage and round-trip), and error handling. |
| **`test_ordinary_kriging.py`** | Core 2D ordinary kriging functionality, testing both the `Kriging` class and the `ordinary_kriging` convenience function. |
| **`test_bk.py`** | Block kriging integration, Gaussian quadrature, block variance regularization, and per-block properties. |
| **`test_cokriging.py`** | Ordinary co-kriging (Primary/Secondary variables) using the Walker Lake dataset and validating the Linear Model of Coregionalization (LMC). |
| **`test_spacetime_kriging.py`** | 3D + Time kriging (`SpaceTimeKriging`), covering `sum_metric` and `product_sum` models, and spatio-temporal co-kriging. |
| **`test_sgsim.py`** | Sequential Gaussian Simulation (SGSIM) deterministic reproducibility, seed consistency, and ensemble mean convergence. |
| **`test_sva.py`** | Spatial Varying Anisotropy (SVA) features, specifically evaluating the impact of `localnugget` and `rangescale` on kriging variance. |
| **`test_variogram.py`** | Coverage for all parsed variogram types, varying/multi-structure variograms, and exact-match interpolation behavior. |
| **`test_cross_validation.py`** | Leave-one-out cross-validation (LOO-CV) accuracy, residual distributions, and correlation against pre-computed references. |
| **`test_sparks_exe.py`** | Subprocess integration tests for the `sparks` CLI. Validates flags, namelist (`-nl`) mode, bounds clamping, and I/O handling. |

---

## Running the Tests

You can run the entire test suite from the root of the repository using `pytest`:

```bash
# Run all tests in the tests/ directory
pytest tests/

# Run tests with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_ordinary_kriging.py

# Run a specific test class or function
pytest tests/test_api.py::TestInputValidation

```

---

## Key Testing Principles

* **Geostatistical Theory Validation:** Tests explicitly check mathematical properties, such as ensuring kriging variances are strictly non-negative, block variance is less than or equal to point variance, and ordinary kriging weights sum to one (by predicting constant fields).
* **Exact Interpolation:** Multiple modules ensure that predicting exactly at a sampled location returns the exact observation value and a variance equal to the observation error (or zero).
* **Memory and Object Reuse:** Tests rigorously check that `Kriging` objects can be reused with new data or grids without memory leaks, stale array bounds, or residual state from previous solves.
* **Deterministic Simulation:** SGSIM tests utilize pre-computed random paths and samples to ensure that specific random seeds produce identical, platform-independent realizations.

---

## Appendix: Complete List of Individual Tests

### `test_api.py`

* `test_coord_wrong_ndim_raises`: Asserts passing a 3D coordinate array to a 2D kriging object raises a dimension error.
* `test_coord_transposed_raises`: Ensures that passing transposed coordinate arrays (e.g., shape `(2, N)`) raises an assertion error.
* `test_missing_library_error_message`: Verifies the module loads and the backend C/Fortran library is present.
* `test_repr`: Checks that the `__repr__` method outputs correct string information (dimension, variables, data count).
* `test_set_obs_value_wrong_length_raises`: Validates that providing a value array of incorrect length relative to coordinates raises a `ValueError`.
* `test_set_obs_variance_wrong_length_raises`: Validates that providing a variance array of incorrect length raises a `ValueError`.
* `test_set_grid_block_nblockpnt_sum_mismatch_raises`: Ensures block sub-node counts match the provided coordinate rows.
* `test_set_sim_randpath_wrong_length_raises`: Checks that SGSIM random path lengths match the grid size.
* `test_set_sim_randpath_must_be_permutation`: Ensures the SGSIM path is a valid 1-based numerical permutation.
* `test_set_sim_sample_wrong_shape_raises`: Validates the SGSIM sample matrix strictly adheres to the `(nsim, nblocks)` shape.
* `test_set_grid_before_obs_raises_runtime_error`: Verifies that setting a grid before setting observations fails gracefully.
* `test_set_search_before_obs_raises_runtime_error`: Verifies that configuring search parameters before observations fails gracefully.
* `test_solve_without_search_raises_runtime_error`: Ensures `solve()` cannot be called without configuring search parameters first.
* `test_obs_drift_before_obs_raises_runtime_error`: Ensures universal drift setup fails if observations haven't been loaded.
* `test_write_mat_with_openmp_writes_debug_files`: Checks that debug matrices are correctly written to disk safely under OpenMP execution.
* `test_simple_kriging_with_sk_mean`: Validates that simple kriging configured with a known mean computes correctly.
* `test_bounds_clip_upper`: Ensures the upper bound clipping parameter strictly restricts maximum estimates.
* `test_bounds_clip_lower`: Ensures the lower bound clipping parameter strictly restricts minimum estimates.
* `test_kriging_with_linear_drift`: Tests universal kriging correctly applies a linear drift (x, y coordinates) on hydraulic head data.
* `test_obs_drift_wrong_shape_raises`: Confirms a Fortran backend error is raised if the drift array shape does not match `ndrift`.
* `test_second_run_differs_with_different_obs`: Verifies that reusing a Kriging object with a new observation dataset successfully changes the output.
* `test_second_run_differs_with_different_grid`: Verifies that reusing a Kriging object with a new estimation grid successfully changes the output.
* `test_third_run_reproduces_first`: Ensures reloading original data into a previously reused object perfectly reproduces the initial results.
* `test_reuse_with_smaller_then_larger_obs`: Confirms memory safety when reusing an object sequentially with a smaller, then a larger dataset.
* `test_reuse_variance_nonnegative_across_runs`: Ensures kriging variance remains strictly non-negative across multiple continuous object reuses.
* `test_set_vgm_accumulates_structures`: Documents that calling `set_vgm` repeatedly adds nested structures, doubling the sill if the inputs are identical.
* `test_nthread_1_is_bitwise_reproducible`: Confirms `solve(nthread=1)` called twice on the same object returns bitwise-identical estimates and variances — no OpenMP scheduling variation.
* `test_nthread_1_agrees_with_parallel`: Verifies that serial (`nthread=1`) and default parallel `solve()` results agree within floating-point noise, confirming the thread-count argument does not alter the mathematical result.
* `test_nthread_restores_omp_setting`: Checks that `solve(nthread=1)` saves and restores the ambient `omp_get_max_threads()` value, leaving the runtime unmodified for subsequent calls.
* `test_nthread_0_accepted_without_error`: Ensures the `nthread=0` no-op sentinel completes without error and returns finite, non-negative results.

### `test_bk.py`

* `test_block_kriging_result_shape`: Verifies that block kriging returns output array lengths matching the number of defined blocks.
* `test_block_variance_nonnegative`: Ensures block kriging variance is strictly non-negative.
* `test_block_variance_less_than_point_variance`: Confirms regularisation mathematically holds (block variance must be ≤ point variance at the block's centroid).
* `test_block_estimate_close_to_centroid_point_estimate`: Checks that a relatively small block's estimate closely matches the point estimate at its centroid.
* `test_block_kriging_uniform_weights_sums_to_one`: Verifies that block kriging a constant field perfectly returns the constant value.
* `test_gaussian_quadrature_block_matches_explicit_nodes`: Validates that internal Gaussian quadrature (`block_type=-4`) perfectly matches explicitly provided sub-nodes.
* `test_gaussian_quadrature_multiple_blocks_allow_different_sizes`: Tests using unique Gaussian quadrature block sizes for multiple blocks evaluated simultaneously.
* `test_gaussian_quadrature_3d_smoke`: Ensures Gaussian quadrature block kriging executes successfully and cleanly in 3D.
* `test_block_kriging_localnugget`: Confirms that establishing a `localnugget` effectively increases block kriging variance.
* `test_block_kriging_rangescale`: Checks that artificially increasing the `rangescale` effectively reduces block kriging variance.
* `test_multiple_blocks`: Verifies that passing identical discrete blocks yields identical estimates and variances for both.

### `test_cokriging.py`

* `test_lmc_validity`: Checks that the provided linear model of coregionalization (LMC) satisfies mathematical validity constraints.
* `test_result_shapes`: Ensures co-kriging output array structures correctly match the dimensions of the estimation grid.
* `test_estimate_all_variables_and_covariance_matrix`: Confirms co-kriging fills all target-variable estimates and returns the full `(nblock, nvar, nvar)` conditional covariance matrix.
* `test_variance_nonnegative`: Validates that co-kriging variance is strictly non-negative.
* `test_variance_bounded_by_total_sill`: Confirms co-kriging variance does not ever exceed the total sill of the primary variogram.
* `test_ok_variance_bounded_by_u_total_sill`: Ensures ordinary kriging variance on a secondary variable alone respects its isolated total sill.
* `test_cokriging_reduces_variance_vs_kriging`: Demonstrates that co-kriging with an abundant primary variable mathematically reduces variance vs ordinary kriging on a sparse secondary variable.
* `test_exact_match_reduces_variance`: Checks that a grid node coinciding with an observation strictly reduces the co-kriging variance below the total sill — the coincident observation conditions the estimate even when variance does not reach exactly zero (as it does in ordinary kriging with no nugget).

### `test_cross_validation.py`

* `test_cross_validation_returns_nobs_estimates`: Verifies cross-validation mode returns an output array length perfectly equal to the number of observations.
* `test_cross_validation_residuals_unbiased`: Checks that mean cross-validation residuals remain highly constrained close to zero.
* `test_output_shape`: Confirms LOO-CV produces one discrete estimate and variance output per individual observation.
* `test_variance_positive`: Ensures LOO-CV variance is strictly greater than zero (proves it is not self-conditioning).
* `test_correlation_with_reference`: Checks that LOO-CV estimates strongly correlate (`>0.999`) with known external reference values.
* `test_rmse_vs_reference`: Ensures root mean square error measured against a reference LOO-CV set remains suitably small.
* `test_mean_error_near_zero`: Validates that generated LOO-CV estimates are mathematically unbiased compared to true observed values.
* `test_standardised_residuals_distribution`: Checks that standardized residuals point to a well-calibrated variance structure (MSSE < 4.0).
* `test_within_3_sigma_fraction`: Ensures at least 85% of all LOO-CV residuals fall cleanly within ±3 standard deviations.
* `test_estimates_within_data_range`: Confirms LOO-CV estimates stay within a reasonable ±10% margin of the source data range.
* `test_loocv_rerun_reproduces_first`: Verifies deterministic reproducibility when executing LOO-CV consecutively on the exact same object.
* `test_estimate_equals_obs_no_error`: Confirms the exact interpolator property functions correctly at observation locations.
* `test_variance_zero_no_obs_error`: Checks that internal variance drops exactly to zero at exactly matched grid nodes.
* `test_variance_equals_obs_error`: Validates that kriging variance rises exactly to match the provided observation error at matched nodes.
* `test_multiple_exact_matches_simultaneously`: Ensures all coincident nodes existing within a mixed grid perfectly reproduce their observations.
* `test_exact_match_among_non_exact_nodes`: Tests that mixed node grids handle exactly-matched and interpolated nodes simultaneously without failure.
* `test_exact_match_with_nugget_variogram`: Confirms exact matching constraints override variogram nugget effects at specific colocated nodes.
* `test_synthetic_exact_match_all_obs`: Verifies total 1:1 reproduction of an observation set when used exclusively as the estimation grid.

### `test_ordinary_kriging.py`

* `test_estimate_shape`: Asserts that `estimate` and `variance` output arrays scale exactly with the grid size.
* `test_variance_nonnegative`: Ensures kriging variance is strictly non-negative at all evaluated grid nodes.
* `test_exact_match`: Verifies that a generated grid node perfectly co-located with an observation exactly reproduces its value.
* `test_weights_sum_to_one`: Validates ordinary kriging weights properly sum to 1 by successfully interpolating a uniform constant field.
* `test_destructor_does_not_crash`: Ensures object garbage collection (`__del__`) processes cleanly even if `solve()` was never invoked.
* `test_returns_two_arrays`: Checks that the `ordinary_kriging` convenience function safely unpacks and returns both an estimate and variance array.
* `test_estimate_in_reasonable_range`: Asserts that ordinary kriging estimates fall logically within the bounds of the provided input data.
* `test_bad_coord_shape_raises`: Ensures that improperly shaped user coordinate arrays predictably raise assertion errors.
* `test_bad_value_shape_raises`: Confirms that value arrays that mismatch the number of provided coordinates raise immediate exceptions.
* `test_estimate_correlation_with_reference`: Checks output spatial correlation directly against a known 2D benchmark dataset.
* `test_pc2d_variance_nonnegative`: Verifies strictly positive variance on a full, realistic 2D field dataset.
* `test_pc2d_estimate_in_data_range`: Confirms estimates on the full 2D field predictably stay within the global data range.

### `test_sgsim.py`

* `test_sgsim_shape_single_realisation`: Ensures a single simulation returns an array of shape `(ngrid,)` matching the grid size.
* `test_sgsim_shape_multiple_realisations`: Checks that multiple simulations return a 2D array of shape `(ngrid, nsim)` — grid nodes in the first axis, realisations in the second.
* `test_realisations_differ`: Confirms that two consecutively generated simulations utilizing the same parameters yield disparate spatial fields.
* `test_realisations_differ_seperate_seeds`: Checks that configuring distinct random seeds guarantees different spatial fields.
* `test_seed_reproducibility`: Validates that invoking the exact same random seed identically reproduces the exact simulation arrays.
* `test_ensemble_mean_close_to_kriging`: Ensures the computed average of many independent SGSIM realizations converges directly toward the ordinary kriging estimate.
* `test_class_interface_with_precomputed_path_sample`: Verifies deterministic execution behavior across platforms using pre-computed random paths and statistical samples.
* `test_joint_cosim_get_estimate_all_shape`: Confirms joint co-simulation (`nvar>1`, `nsim>0`) returns `get_estimate_all()` with shape `(nblock, nvar, nsim)` and `get_variance_all()` with shape `(nblock, nvar, nvar)`.

### `test_spacetime_kriging.py`

* `test_basic_shape`: Checks generated array dimensions safely handle spacetime sum-metric kriging.
* `test_variance_non_negative`: Ensures spacetime kriging variance outputs remain strictly non-negative.
* `test_estimate_finite`: Confirms all computed spacetime estimates return valid, finite numerical values (no NaNs).
* `test_exact_match`: Verifies exact interpolator behavior functions effectively in the multi-dimensional space-time domain.
* `test_bounded_transform`: Validates the bounded geometric transform model processes cleanly without crashing.
* `test_maxtlag_reduces_neighbours`: Confirms that mechanically tightening the temporal search window appropriately yields fewer valid neighbors.
* `test_product_sum`: Verifies that the alternative product-sum spacetime model evaluates successfully.
* `test_spacetime_kriging_sum_metric`: Tests the robustness of the convenience wrapper utilizing the sum-metric model.
* `test_spacetime_kriging_product_sum`: Tests the robustness of the convenience wrapper utilizing the product-sum model.
* `test_cv_shape`: Ensures spacetime cross-validation accurately returns arrays matching the total observation count.
* `test_cokriging_shape`: Validates array dimensional outputs for spacetime co-kriging (2 complex variables).
* `test_cokriging_convenience`: Tests the general stability of the spacetime co-kriging convenience function wrapper.
* `test_nested_spatial`: Validates the use of multiple nested spatial structures interacting with one temporal structure.
* `test_nested_temporal`: Validates the use of one spatial structure interacting with multiple nested temporal structures.
* `test_sgsim_shape`: Verifies that spacetime SGSIM logic correctly generates multiple structurally sound simulation arrays.

### `test_sparks_exe.py`

* `test_help_exits_cleanly`: Checks that executing with the `-h` flag exits properly with code 0 and outputs formatted help text.
* `test_help_lists_required_flags`: Ensures the generated help text specifically outlines core mandatory flags (`-of`, `-d`, `-v1`).
* `test_produces_three_estimate_lines`: Confirms the default CLI operation accurately formats and outputs standard kriging estimates.
* `test_estimates_in_observation_range`: Asserts output CLI estimates fall logically inside the numerical bounds of the input observation file.
* `test_writexy_produces_header_and_three_rows`: Checks that the `-xy` output flag formats a valid CSV string with clear headers.
* `test_variance_nonnegative`: Ensures the `-xy` generated CSV contains strictly positive variance values.
* `test_exact_match_at_observation_location`: Validates that perfectly matched coordinates mapped in the CLI yield exact observation values.
* `test_constant_field_returns_constant`: Ensures OK of a uniform constant field via the CLI correctly sums internal weights to one.
* `test_simple_kriging_without_unbias_flag`: Confirms executing without `-u` accurately performs Simple Kriging instead of Ordinary.
* `test_ordinary_differs_from_simple_kriging`: Verifies that OK (`-u`) and SK yield mathematically distinct results via CLI execution.
* `test_missing_variogram_message`: Checks that omitting the primary variogram flag correctly triggers a clear stderr message.
* `test_missing_obsfile_flag_message`: Checks that omitting the observation file flag correctly triggers a clear stderr message.
* `test_missing_d_flag_message`: Checks that omitting the specific dimension flag correctly triggers a clear stderr message.
* `test_missing_blockfile_message`: Checks that omitting the block/grid file (when not in CV mode) properly triggers a crash error.
* `test_nobs1_zero_message`: Asserts that explicitly specifying zero observations results in a clear parameter error.
* `test_output_shape`: Confirms the standard 2D benchmark regression test returns the exact correct number of text lines.
* `test_estimate_matches_reference`: Verifies parsed CLI output perfectly correlates with the internal `pc2d` reference dataset.
* `test_variance_nonnegative` (Regression): Validates strictly positive variance for the larger `pc2d` regression execution.
* `test_estimate_in_data_range`: Confirms `pc2d` CLI generated estimates remain strictly within dataset bounds.
* `test_loocv_produces_one_row_per_obs`: Checks that the `-cv` evaluation flag strictly produces one output row for every observation.
* `test_loocv_has_observed_column`: Ensures CLI CV output injects the original `observed` value as a dedicated column.
* `test_loocv_estimates_match_reference`: Validates CLI LOO-CV evaluation outputs tightly match pre-computed textbook reference values.
* `test_loocv_simple_dataset_variance_nonnegative`: Confirms LOO-CV variance output via the CLI evaluates to strictly positive.
* `test_sgsim_produces_three_lines`: Verifies CLI SGSIM (`-s`) strictly outputs the expected number of structural rows.
* `test_sgsim_writexy_has_estimate1_column`: Checks that SGSIM `-xy` formatted output titles columns sequentially (e.g., `estimate1`).
* `test_sgsim_values_in_physical_range`: Confirms CLI simulated numerical values stay within physically logical dataset bounds.
* `test_sgsim_seed_reproducibility`: Validates that injecting a `-sd` specific seed guarantees wholly reproducible CLI simulations.
* `test_sgsim_differs_from_kriging`: Ensures CLI stochastic simulation output mathematically diverges from deterministic kriging output.
* `test_upper_bound_clamps_high_estimates`: Tests the `-bd` parameter flag properly clips any estimates processed above a maximum limit.
* `test_lower_bound_clamps_low_estimates`: Tests the `-bd` parameter flag properly clips any estimates processed below a minimum limit.
* `test_bounds_do_not_affect_variance`: Verifies that establishing CLI bounding flags does not artificially alter the evaluated node variance.
* `test_azimuth_runs_cleanly`: Checks that establishing a geometric anisotropy angle (`-a1`) processes successfully in the CLI.
* `test_anis_ratio_runs_cleanly`: Checks that scaling the geometric anisotropy ratio (`-s1`) processes successfully in the CLI.
* `test_anisosearch_flag_runs_cleanly`: Tests the focused anisotropic search flag (`-as`) executes without raising internal errors.
* `test_anisotropy_changes_estimates`: Confirms that implementing global anisotropy fundamentally and visibly alters the resulting estimates.
* `test_default_output_is_values_only`: Verifies default standard CLI output is purely a simple column of floats devoid of headers.
* `test_writexy_output_has_five_columns`: Checks that `-xy` output rigidly formats 5 required structural columns (igrid, x, y, est, var).
* `test_writexy_coordinates_match_grid_file`: Confirms X/Y coordinates output directly by `-xy` perfectly map back to the input block file.
* `test_output_to_file`: Ensures the CLI safely redirects data streams to a designated file path instead of the standard stdout.
* `test_output_to_file_matches_stdout`: Validates that captured file output perfectly mirrors the raw standard stdout.
* `test_namelist_matches_cli_result`: Confirms the `-nl` (namelist) configuration path structurally replicates pure command-line flag execution.

### `test_sva.py`

* `test_zero_localnugget_matches_ordinary_kriging`: Verifies that evaluating a `localnugget` identically equal to 0 behaves identically to OK.
* `test_localnugget_increases_variance`: Confirms injecting a per-node local nugget explicitly scales kriging variance at that specific node.
* `test_localnugget_does_not_change_estimates`: Verifies that local nuggets mathematically smooth non-exact estimates toward the global mean.
* `test_localnugget_per_node_variation`: Demonstrates that generating node-specific nugget values distinctly scales variance per location.
* `test_localnugget_variance_nonnegative`: Ensures highly scaled local nuggets do not invert and push global variance negative.
* `test_exact_match_localnugget_zero`: Validates exact interpolation occurs definitively when the localnugget is explicitly 0.
* `test_exact_match_localnugget_nonzero_smooths`: Confirms exact interpolation intentionally fails (smooths) when localnugget is set > 0.
* `test_unit_rangescale_matches_ordinary_kriging`: Verifies that a structural `rangescale` of 1.0 yields outputs identical to standard OK.
* `test_larger_rangescale_reduces_variance`: Confirms scaling the range up (`>1`) includes further peripheral data, mathematically lowering variance.
* `test_smaller_rangescale_increases_variance`: Confirms compressing the range (`<1`) simulates data distance, mathematically raising variance.
* `test_rangescale_variance_nonnegative`: Ensures extremely scaled rangescales maintain mathematically sound, positive variance values.
* `test_rangescale_per_node_spatial_variation`: Checks that scaling unique rangescales by node applies targeted variance impacts directly to those nodes.
* `test_rangescale_monotone_with_scale`: Validates that node variance strictly decreases or holds effectively steady as rangescale increases.
* `test_localnugget_and_rangescale_combined`: Ensures both spatial varying anisotropy features execute simultaneously without crashing the backend solver.

### `test_variogram.py`

* `test_set_vgm_preserves_requested_type`: Ensures variogram type string models (e.g., 'sph', 'exp', 'gau') map functionally into Fortran/C backend components.
* `test_unknown_vtype_raises_fortran_error`: Confirms that invalid variogram string IDs logically throw a catchable backend error immediately.
* `test_set_vgm_block_matches_separate_per_block_solves`: Verifies that `varying_vgm` execution mode safely allows defining unique variogram parameters per individual node.
* `test_varying_vgm_sgsim_smoke`: Ensures block-varying localized variograms execute safely during standard SGSIM generation runs.
* `test_output_shape`: Confirms LOO-CV execution mode produces proper length structural output arrays without truncation.
* `test_variance_positive`: Ensures field variance evaluates strictly positive during standard CV testing modes.
* `test_correlation_with_reference`: Checks LOO-CV spatial estimates correlate with extreme accuracy against established reference outputs.
* `test_rmse_vs_reference`: Checks the global root mean square error against a validated reference LOO-CV set.
* `test_mean_error_near_zero`: Validates LOO-CV statistical bias averages efficiently out to zero.
* `test_standardised_residuals_distribution`: Checks that standardized residual plotting confirms a highly normalized variance structure (MSSE < 4.0).
* `test_within_3_sigma_fraction`: Asserts over 85% of evaluated residuals fall naturally within a ±3 sigma window.
* `test_estimates_within_data_range`: Confirms LOO-CV extrapolated values stay mathematically inside logical data boundaries.
* `test_loocv_rerun_reproduces_first`: Validates re-executing CV on the identical object safely reproduces results perfectly.
* `test_nugget_increases_variance`: Confirms generating a pure nugget structural component inherently scales overall field variance.
* `test_additive_split_same_range_reproducibility`: Verifies that manually breaking a variogram into additive parts upholds identical mathematical equivalency.
* `test_short_range_structure_changes_estimates`: Checks that injecting a localized short-range structure substantially alters localized spatial predictions.
* `test_short_range_structure_increases_variance`: Confirms evaluating short-range structures reliably elevates total computational sill and evaluated variance.
* `test_three_structure_model_variance_nonnegative`: Ensures highly complex, layered multi-structure variograms do not invert mathematical matrices to negative variance.
* `test_multi_struct_estimate_in_data_range`: Validates evaluation estimates stay bounded properly when leveraging advanced multi-structure models.
* `test_constant_field_multi_struct`: Verifies ordinary kriging run on a constant field preserves the field constant under complex multi-structure variograms.
* `test_estimate_equals_obs_no_error`: Confirms pure exact matching holds completely without actively defined observation errors.
* `test_variance_zero_no_obs_error`: Checks that calculated kriging variance hits 0.0 without rounding limits at exact nodes.
* `test_variance_equals_obs_error`: Validates total variance scales perfectly to match the manually provided observation error at coincident nodes.
* `test_multiple_exact_matches_simultaneously`: Ensures all coincident evaluation points within a grid safely match data simultaneously.
* `test_exact_match_among_non_exact_nodes`: Verifies internal interpolators handle layered exactly-matched and inter-point nodes natively and accurately.
* `test_exact_match_with_nugget_variogram`: Confirms exact match behavior explicitly overrides generalized nugget behavior isolated at specific evaluation nodes.
* `test_synthetic_exact_match_all_obs`: Validates entire field sets match perfectly if generated grid coordinates explicitly copy observation coordinates.

### `test_weight_store.py`

* `test_nnear_shape`: Asserts `nnear` has shape `(nblock, ngroups)` after a `store_weight=True` solve.
* `test_inear_shape`: Asserts `inear` has shape `(nblock, ngroups, nmax)`.
* `test_weight_shape`: Asserts `weight` has shape `(nblock, ngroups, nmax)`.
* `test_dtypes`: Confirms `nnear`/`inear` are `int32` and `weight` is `float64`.
* `test_ngroups_kriging`: Checks `ngroups == nvar` (== 1) for ordinary kriging without simulation.
* `test_ngroups_sgsim`: Checks `ngroups == 2*nvar` (== 2) for SGSIM, covering the obs and sim-block groups.
* `test_nnear_in_range`: Verifies every `nnear` value lies in `[0, nmax]`.
* `test_inear_valid_indices`: Confirms all active (non-zero) `inear` entries are valid 1-based observation indices.
* `test_unused_slots_are_zero`: Ensures padding slots beyond `nnear[ib, ig]` are zero in both `inear` and `weight`.
* `test_ok_weights_sum_to_one`: For ordinary kriging, checks that obs weights sum to exactly 1 at every block.
* `test_estimate_consistency`: Reconstructs the estimate from stored weights and confirms it matches `get_results()`.
* `test_weights_unchanged_by_get_weights`: Confirms that calling `get_weights()` twice returns identical arrays.
* `test_file_written`: Verifies a factor file is created on disk when `weight_file` is provided.
* `test_reuse_gives_same_estimates`: Confirms `use_old_weight=True` reproduces estimates and variances from the original `store_weight=True` run.
* `test_roundtrip_matches_plain_solve`: Verifies that the full round-trip (plain solve → store → reload) gives bit-identical estimates. *(Moved from `test_api.py::TestOperationalModes`.)*
* `test_cokriging_reuse_gives_same_estimates`: Confirms `use_old_weight=True` reproduces co-kriging estimates and variances for a two-variable model.
* `test_memory_only_matches_plain_solve`: Confirms that `store_weight=True` without a `weight_file` gives bit-identical estimates to a plain solve (verifying the weight store does not perturb the solver).
* `test_memory_only_no_file_created`: Confirms that omitting `weight_file` keeps weights in memory only and writes no file.
* `test_get_weights_without_store_weight_raises`: Checks that `get_weights()` raises `RuntimeError` when the store was never allocated.
* `test_free_then_get_raises`: Checks that `get_weights()` raises `RuntimeError` after `free_weight_store()` is called.
* `test_use_old_weight_reproduces_all_realizations`: Joint co-simulation with explicit path/samples: `use_old_weight` reproduces the complete `(nsim, nblock, nvar)` result.
* `test_use_old_weight_same_seed_reproduces_all_realizations`: Joint co-simulation with auto-generated path/samples: the same integer seed ensures the Fortran RNG produces identical samples in both runs, so `use_old_weight` gives bit-identical realizations.
* `test_use_old_weight_reads_full_est_var_from_factor_file`: Confirms factor-file replay reads the full persisted covariance matrix rather than recomputing only diagonal variances.
* `test_different_seed_gives_different_realizations`: Sanity check that different seeds produce different co-simulation realizations.
* `test_set_weights_reproduces_estimates`: Full round-trip — `store_weight=True` solve → `get_weights()` → `set_weights()` → `solve()` — reproduces the original estimates and variance exactly.
* `test_set_weights_different_obs_values`: Confirms that `set_weights()` re-estimates with updated observation values while reusing the stored neighbour indices and kriging weights; doubling all obs values doubles the estimates (linearity check).
* `test_set_weights_variance_key_in_get_weights`: Verifies that `get_weights()` includes a `'variance'` key of shape `(nblock,)` for nvar=1 when called after a `store_weight=True` solve.
* `test_set_weights_variance_matches_get_results`: Confirms the variance stored inside the weight store (`get_weights()['variance']`) is identical to the variance returned by `get_results()`.
* `test_set_weights_without_variance_gives_zero_variance`: Checks that passing a dict without the `'variance'` key to `set_weights()` results in zero variance output — a deliberate zero-default rather than stale data.
* `test_set_weights_with_store_weight_raises`: Verifies that calling `set_weights()` on a `store_weight=True` object raises `ValueError`, since the two modes are mutually exclusive.
