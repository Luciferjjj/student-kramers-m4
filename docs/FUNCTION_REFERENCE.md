# Function reference for the M4 Student Kramers project

This document is a lookup reference for the Python implementation. It records
where each function lives, what it accepts, what it returns, and how it is used
in the M4 study. It is not a Python tutorial.

## Package boundary

The repository has two Python packages.

```text
student_kramers/
    General model definitions, likelihoods, estimation, simulation,
    recovery studies, discrimination studies, and reusable bootstrap code.

greenland_application/
    Greenland Ca2+ data access, real-data runners, IOS analysis,
    application diagnostics, saved-result summaries, and figures.
```

The dependency direction is:

```text
greenland_application -> student_kramers
```

The core package does not import the Greenland application package.

## Parameter and data conventions

All models use the same full parameter order:

```text
[eta, a, b, c, d, alpha, beta, gamma, delta, epsilon, zeta]
```

The common SDE is

\[
dX_t = V_t\,dt,
\]

\[
dV_t =
\left[-\eta V_t + aX_t^3+bX_t^2+cX_t+d\right]dt
+ \sqrt{q(X_t,V_t)}\,dW_t,
\]

with

\[
q(x,v)=
\alpha v^2+\beta v+\gamma+
\delta x^2+\epsilon xv+\zeta x.
\]

M1 to M3 fix the position-dependent coefficients to zero. M4 estimates all
eleven parameters.

The partial-observation data array has shape `(n_pseudo_states, 2)`:

```text
column 0: X_t
column 1: Vhat_t = (X_{t+1} - X_t) / h
```

An array with 2500 pseudo-states contains 2499 likelihood transitions.

# Core package: `student_kramers`

## `config.py`

This module stores numerical defaults and project paths.

Important constants:

| Name | Meaning |
|---|---|
| `H_OBS` | Observation spacing used by the likelihood |
| `H_SIM` | Internal Euler-Maruyama simulation step |
| `H_SIM_VALIDATION` | Finer step used in simulation validation |
| `CORRECTION_FACTOR` | Partial-observation log-variance correction, fixed at `2/3` |
| `PENALTY` | Objective value returned for invalid numerical proposals |
| `LBFGS_MAXITER`, `LBFGS_TOL` | Default optimization settings |
| `M4_CHOLESKY_*` | Finite-difference and stopping settings for M4 |
| `BOOTSTRAP_SEED` | Base seed for appendable bootstrap runs |
| `IOS_MAXITER` | Maximum iterations for leave-one-out refits |
| `REFERENCE_PARAMS_BY_MODEL` | Simulation truth for M1 to M4 |
| `RECOVERY_INIT_PARAMS_BY_MODEL` | Starting values for recovery studies |

Functions:

### `run_dir(run_name=DEFAULT_RUN_NAME)`

Returns the path `results/runs/<run_name>/`. It rejects empty names, path
separators, and `..`.

### `make_result_dirs(run_name=DEFAULT_RUN_NAME)`

Creates the directory returned by `run_dir`.

## `models.py`

This module defines the model registry, deterministic force, potential,
diffusion variance, parameter constraints, and the M4 Cholesky coordinates.

### Model registry

| Function | Input | Return |
|---|---|---|
| `get_model(model_name)` | `"M1"` to `"M4"` | Registry dictionary for one model |
| `free_names(model_name)` | Model name | Names of free coefficients |
| `free_bounds(model_name)` | Model name | L-BFGS-B bounds for free coefficients |
| `embed_params(free_params, model_name)` | Model-specific free vector | Full 11-vector |
| `extract_free_params(params, model_name)` | Full 11-vector | Model-specific free vector |
| `parameter_row(model_name, params, nll=None)` | Model, full parameters, optional NLL | Flat dictionary for a result table |
| `validate_model_registry()` | None | Raises if a model definition is inconsistent |

### Mathematical functions

| Function | Definition or purpose |
|---|---|
| `force(x, params)` | \(a x^3+b x^2+c x+d\) |
| `potential(x, params)` | Potential \(U\) satisfying \(F=-U'\) |
| `velocity_drift(x, v, params)` | \(-\eta v+F(x)\) |
| `diffusion_variance(x, v, params)` | \(q(x,v)\) |
| `diffusion_quadratic_matrix(params)` | Two-dimensional quadratic matrix in `(x, v)` |
| `diffusion_augmented_matrix(params, floor=0.0)` | Homogeneous 3 by 3 matrix for `q-floor` |
| `diffusion_minimum(params)` | Global minimizer of `q` and its minimum value |
| `diffusion_rectangle_bounds(data, margin=0.0)` | State-space rectangle determined by data |
| `diffusion_rectangle_minimum(params, data, margin=0.0)` | Exact minimum of `q` on that rectangle |

### Constraint and Cholesky functions

| Function | Purpose |
|---|---|
| `constraints_valid(free_params, model_name, data=None)` | Checks shared stability and global diffusion positivity |
| `m4_constraint_values(free_params)` | Returns M4 augmented-matrix eigenvalues and the `2*eta-alpha` margin |
| `m4_to_cholesky_params(free_params, jitter=1e-10)` | Converts direct M4 coefficients to optimization coordinates |
| `m4_from_cholesky_params(cholesky_params)` | Converts Cholesky coordinates back to direct coefficients |

For M4, the augmented matrix for \(q(x,v)-q_{\mathrm{floor}}\) is written as

\[
H=LL^\top.
\]

This representation keeps every optimization proposal globally feasible. The
parameter `rho` also maps through a logistic function so that
\(0<\alpha<2\eta\).

## `likelihoods.py`

This module contains the complete-observation and corrected
partial-observation Strang objectives.

### Public likelihood functions

#### `partial_transition_nlls(...)`

```python
partial_transition_nlls(
    free_params,
    data,
    model_name,
    h=H_OBS,
    moment_mask=None,
    eval_mask=None,
)
```

Returns one negative log pseudo-likelihood contribution for every selected
transition.

- `moment_mask` selects transitions used to estimate empirical splitting
  moments.
- `eval_mask` selects transitions to score.
- Exact IOS uses a leave-one-out training mask for `moment_mask` and the held
  out transition for `eval_mask`.

#### `partial_neg_log_lik(free_params, data, model_name, h=H_OBS)`

Returns the sum of all corrected partial-observation transition
contributions. Invalid parameters return `PENALTY`.

#### `masked_partial_neg_log_lik(free_params, data, model_name, mask, h=H_OBS)`

Returns the training objective over the transitions selected by `mask`.

#### `complete_neg_log_lik(free_params, data, model_name, h, branch="plus")`

Returns the complete-observation two-dimensional Gaussian Strang objective.
This function is used in simulation validation and recovery studies.

### Internal moment functions

| Function | Role |
|---|---|
| `_A_mat(params, drift_const)` | Linear drift matrix |
| `_check_alpha(params)` | Quadratic diffusion map for second moments |
| `_check_beta(params)` | Linear diffusion map |
| `_check_gamma(params)` | Constant diffusion term |
| `_shift_cross_matrices(b)` | Cross-moment maps after state translation |
| `_precompute_matrices(...)` | Matrix-exponential terms `I1` to `I5` |
| `_step_omega(...)` | Conditional covariance for one midpoint |
| `_step_omega_batch(...)` | Vectorized covariance calculation |
| `_branch_partial_moments(...)` | Branch-specific means and covariances |
| `_f_step(...)` | Nonlinear half-step of the Strang splitting |
| `_data_moments(X)` | Empirical complete-data moments |
| `_splitting_terms(...)` | Drift matrix, shift, slope, and branch root |
| `_moment_matrix(...)` | Seven-dimensional first/second moment ODE matrix |
| `_linear_moments(...)` | Exact linear moment propagation |
| `_branch_linear_moments(...)` | Complete-data branch propagation |

## `estimation.py`

This module binds likelihood functions to numerical optimizers.

| Function | Purpose |
|---|---|
| `run_estimator_lbfgs(...)` | Standard bounded L-BFGS-B fit for M1 to M3 |
| `minimize_m4_cholesky(...)` | Low-level M4 fit in Cholesky coordinates |
| `run_estimator_m4(...)` | M4 wrapper returning parameters, NLL, and convergence flag |
| `make_loss_fn(model_name, data, h=H_OBS)` | Binds model and data to a one-argument objective |
| `make_random_starts(...)` | Builds valid boundary, warm, and interior starting points |
| `estimate_model(...)` | Partial-observation fit with optional multiple starts |
| `estimate_complete_model(...)` | Complete-observation fit |
| `estimate_models(...)` | Fits several registered models and returns NLL, AIC, BIC, constraints, and timing |

`estimate_model` returns:

```text
params_hat : full eleven-parameter vector
nll        : final objective value
conv       : 0 for accepted optimizer convergence, 1 otherwise
```

For M4, all candidate starts are converted to Cholesky coordinates before
optimization.

## `simulation.py`

### Path simulation

| Function | Return |
|---|---|
| `simulate_trajectory(...)` | `(n_obs, 2)` latent `[X, V]` path |
| `simulate_x(...)` | Simulated observed `X` series |
| `simulate_partial_data(...)` | Simulated `X` converted to `[X, Vhat]` pseudo-states |

`simulate_trajectory` uses Euler-Maruyama with internal step `h_sim`.

### Mechanism and regime summaries

| Function | Purpose |
|---|---|
| `potential_extrema(params)` | Returns lower well, barrier, and upper well |
| `compute_derived_params(params)` | Symmetric double-well derived quantities |
| `classify_regime(x, barrier=0.0)` | Labels observations below or above a barrier |
| `extract_waiting_times(x, h=H_OBS, barrier=0.0)` | Extracts consecutive regime durations |
| `summarize_waiting_times(waits, source)` | Counts and distribution summaries |

### Simulation diagnostics

| Function | Purpose |
|---|---|
| `simulate_waiting_times(...)` | Pools waiting times across simulated paths |
| `simulate_first_passage(...)` | Simulates first hitting times to a barrier |
| `path_summary_matrix(X_paths, Vhat_paths)` | One diagnostic row per path |
| `observed_summary(x, vhat)` | Observed counterpart of path summaries |
| `simulate_model_check(...)` | Simulates arrays used by predictive checks |
| `simulate_diffusion_check(...)` | Records minimum M4 diffusion along simulated paths |

## `bootstrap.py`

This module contains reusable leave-one-out and bootstrap calculations. It
does not create figures.

### Exact IOS

| Function | Purpose |
|---|---|
| `fit_leave_one_out(...)` | Omits one transition, refits the model, and scores the omitted transition |
| `select_ios_pilot_transitions(...)` | Chooses a deterministic diagnostic subset shared by M2 to M4 |
| `run_exact_ios(...)` | Runs or resumes all requested leave-one-out fits |
| `summarize_ios(table, expected_n_transitions=None)` | Computes \(T_N\) only when the table is complete and valid |
| `influential_transitions(table, data, top_n=20)` | Adds observed states to the largest IOS contributions |

The transition contribution is

\[
\mathrm{IOS}_k=
\ell_k(\widehat\theta_{-k})-\ell_k(\widehat\theta).
\]

The formal statistic is

\[
T_N=\sum_k \mathrm{IOS}_k.
\]

### Parametric and contrast bootstrap

| Function | Purpose |
|---|---|
| `run_parametric_bootstrap(...)` | Simulates and refits one model with appendable checkpoints |
| `run_contrast_bootstrap(...)` | Simulates under a null model and refits null and alternative |
| `summarize_bootstrap(...)` | Mean, SD, quantiles, and finite-sample tail probabilities |

### Internal IOS validation

| Function | Purpose |
|---|---|
| `_start_candidate_rows(...)` | Labels and removes duplicate LOO starts |
| `_finite_ios_row(row)` | Determines whether a checkpoint row enters the statistic |
| `_safe_diffusion_rectangle_minimum(...)` | Handles semidefinite boundary cases in diagnostic columns |

An optimizer exit code is retained as a diagnostic. A leave-one-out row enters
the IOS sum only if its held-out score is finite, its training objective is not
worse than the starting value beyond numerical tolerance, and its parameters
satisfy the model constraints.

## `recovery.py`

| Function | Purpose |
|---|---|
| `run_recovery_study(...)` | Extendable complete/partial same-model recovery |
| `summarize_recovery(...)` | Parameter bias, RMSE, and empirical intervals |
| `summarize_recovery_diagnostics(...)` | Convergence, NLL, diffusion recovery, and runtime |
| `diffusion_recovery_surfaces(...)` | Common-grid truth and fitted diffusion surfaces |

`_q_recovery_metrics` compares the fitted and true diffusion variance along the
latent simulated path.

## `discrimination.py`

| Function | Purpose |
|---|---|
| `discrimination_truth(truth)` | Returns M3 or weak/moderate/strong M4 generating parameters |
| `run_discrimination_study(...)` | Simulates partial data and fits M3 and M4 |
| `summarize_discrimination(table)` | Summarizes convergence and likelihood contrasts |

The study is intended to check whether the implementation can separate M3 and
M4 under controlled simulation. Its small pilot runs are not a calibrated
power analysis.

## Core command modules

| Command | Role |
|---|---|
| `python -m student_kramers.run_validation` | One known-parameter complete/partial validation |
| `python -m student_kramers.run_recovery` | Extendable repeated recovery study |
| `python -m student_kramers.run_discrimination` | M3/M4 simulation discrimination |

# Greenland package: `greenland_application`

## `config.py`

This is an application-facing facade over `student_kramers.config`. The shared
location keeps numerical settings and result paths identical across the core
and application code.

## `data_loading.py`

This is an application-facing facade over the existing data and result I/O
implementation. It exposes:

```text
ensure_official_excel
load_raw_excel
preprocess_ca2
build_partial_data
load_real_data
result_path
save_table
load_table
load_result
load_model_fits
save_model_fits
checkpoint_context
prepare_checkpoint
```

The implementation was left byte-for-byte in its original core file during
the repository reorganization so completed long-running result provenance
remains verifiable.

## `pre_ios.py`

| Function | Purpose |
|---|---|
| `audit_diffusion_minima(...)` | Compares global, observed-rectangle, protected-rectangle, and path minima |
| `transition_improvement_table(...)` | Decomposes the real-data M3-to-M4 NLL improvement by transition |
| `run_optimization_stability(...)` | Records every M4 start across independent start sets |
| `run_tolerance_stability(...)` | Refits the same M4 solution under several stopping tolerances |
| `run_predictive_checks_checkpointed(...)` | Appendable predictive simulations and density summaries |
| `run_predictive_checks(...)` | In-memory predictive check |
| `run_nested_m3_m4_bootstrap(...)` | M3-null likelihood-contrast bootstrap |
| `summarize_nested_bootstrap(...)` | Finite-sample contrast quantiles and upper-tail probability |

The helpers `_path_behavior`, `_density_band`, `_fixed_density_rows`, and
`summarize_predictive_density` construct the saved predictive-check tables.

## `ios_analysis.py`

| Function | Purpose |
|---|---|
| `build_ios_summary(...)` | Comparable M2/M3/M4 IOS rows |
| `build_ios_pairwise_comparison(...)` | Pearson, Spearman, and influential-set overlap |
| `build_ios_regime_summary(...)` | IOS split by switch and non-switch transitions |
| `build_ios_transition_table(...)` | Joins IOS values to age and pseudo-state coordinates |
| `build_ios_parameter_shift_table(...)` | Parameter movement across leave-one-out fits |
| `validate_ios_tables(...)` | Explicit completeness and validity audit |

## `bootstrap_analysis.py`

### M4 parameter bootstrap

| Function | Purpose |
|---|---|
| `successful_bootstrap_rows(table)` | Selects successful refits |
| `build_parametric_bootstrap_overview(table)` | Success rate, runtime, and NLL distribution |
| `build_parameter_bootstrap_summary(table, observed_params)` | Parameter quantiles and relative SD |
| `build_diffusion_bootstrap_diagnostics(...)` | Diffusion minima and path summaries per replication |
| `summarize_diffusion_bootstrap(...)` | Observed values and bootstrap quantiles |
| `build_path_diffusion_band(...)` | Pointwise bootstrap band along the observed path |
| `safe_diffusion_rectangle_minimum(...)` | Rectangle minimum robust to singular quadratic matrices |

### Model-wise IOS bootstrap

| Function | Purpose |
|---|---|
| `successful_modelwise_ios_rows(table)` | Selects complete refit plus exact-IOS replications |
| `build_modelwise_ios_bootstrap_summary(...)` | Finite-sample IOS reference and tail probabilities |
| `build_modelwise_ios_cumulative(...)` | Tail-probability stability as replications accumulate |

## `figures.py`

Figures are separated from numerical calculations. Every plotting function
accepts already computed tables and an optional `save_path`.

| Function | Figure content |
|---|---|
| `plot_real_data_state_space(...)` | Calcium coordinate, reconstructed velocity, pathwise M4 change |
| `plot_real_data_mechanisms(...)` | NLL/AIC/BIC, potential, diffusion ratios, phase-space change |
| `plot_transition_improvement(...)` | Transition-level M4 likelihood gains |
| `plot_m4_diffusion_audit(...)` | Global versus observed diffusion minima |
| `plot_optimization_stability(...)` | Start and tolerance sensitivity |
| `plot_predictive_densities(...)` | Observed densities and simulation bands |
| `plot_predictive_percentiles(...)` | Observed diagnostics in simulated distributions |
| `plot_waiting_time_comparison(...)` | Observed and simulated regime durations |
| `plot_discrimination(...)` | M3/M4 simulation contrasts |
| `plot_nested_bootstrap(...)` | M3-null likelihood-contrast distribution |
| `plot_recovery_study(...)` | Complete and partial recovery |
| `plot_ios_overview(...)` | IOS magnitude, concentration, agreement, and timing |
| `plot_ios_phase_space(...)` | IOS values over time and state space |
| `plot_ios_numerical_diagnostics(...)` | LOO parameter movement and optimizer effort |
| `plot_m4_parametric_bootstrap_parameters(...)` | Eleven parameter bootstrap summaries |
| `plot_m4_parametric_bootstrap_diffusion(...)` | Diffusion uncertainty and pathwise bands |
| `plot_modelwise_ios_bootstrap(...)` | IOS calibration, p-value stability, correlations, and runtime |

Internal helpers:

- `_style(ax)` applies a common axis style.
- `_finish(fig, save_path=None, tight=True)` saves PNG and PDF versions.
- `_fit_params(fits)` converts a model-fit table to a parameter dictionary.

## Greenland command modules

| Command | Role |
|---|---|
| `python -m greenland_application.run_single` | Recomputes one fitted NLL |
| `python -m greenland_application.run_application` | Fits registered models to the real data |
| `python -m greenland_application.run_analysis` | Saves derived and simulation-based diagnostics |
| `python -m greenland_application.run_pre_ios` | Runs pre-IOS audits and nested bootstrap |
| `python -m greenland_application.run_bootstrap` | Runs IOS, parametric bootstrap, or contrast bootstrap |
| `python -m greenland_application.run_ios_analysis` | Builds observed IOS tables and figures |
| `python -m greenland_application.run_bootstrap_analysis` | Summarizes M4 parameter bootstrap |
| `python -m greenland_application.run_modelwise_ios_bootstrap` | Parallel strict IOS calibration |
| `python -m greenland_application.run_modelwise_ios_analysis` | Summarizes strict IOS calibration |
| `python -m greenland_application.run_figures` | Rebuilds saved application figures |

# Result and provenance conventions

All generated results use:

```text
results/runs/<run_name>/
```

Long-running CSV files have a neighboring `.meta.json` file containing:

```text
workflow
model
parameter hash
data hash
code hash
observation and simulation steps
workflow settings
```

`prepare_checkpoint` rejects resume attempts when the saved context no longer
matches the current model, data, code, or settings. Target counts such as the
number of bootstrap replications are omitted where a run is designed to grow
from a pilot to a larger run.

Current formal run names:

| Run | Content |
|---|---|
| `m4_real_data_cholesky` | Formal M2/M3/M4 real-data fits |
| `ios_pilot_cholesky` | Shared 48-transition IOS pilot |
| `ios_observed` | Complete observed exact IOS |
| `m4_parametric_bootstrap` | M4 parameter and diffusion uncertainty |
| `m4_modelwise_ios_bootstrap` | Finite-sample M4 IOS calibration |

# Notebook

`notebooks/greenland_m4_analysis.ipynb` is the main interactive research
surface. It reads saved formal results and calls the functions above. Expensive
leave-one-out and bootstrap calculations remain in command modules so they can
resume from checkpoints.
