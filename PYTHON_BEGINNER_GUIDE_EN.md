---
title: "Student Kramers Python code guide, fourth edition"
subtitle: "A beginner-oriented guide to every research module, parameter, and workflow"
lang: en
toc: true
toc-depth: 3
number-sections: true
code-fold: false
code-copy: true
format:
  html:
    theme: cosmo
    html-math-method: mathjax
    smooth-scroll: true
---

# Reading goal

This guide explains the Python code used by the Student Kramers research project. It is written for a
reader who is still learning Python and needs to understand how mathematical formulas move through a
real codebase.

The goal is not to memorize every line. The goal is to answer four questions whenever you open a
function:

1. What does this function receive?
2. Where did those inputs come from?
3. What does the function return?
4. Which mathematical or research step does it implement?

The fourth edition includes the unified M1-M4 implementation, global M4 diffusion positivity,
nonzero interior starts, warm starts, recovery and discrimination workflows, and simulation stability
checks.

## Three levels of understanding

Some parts of this project are straightforward Python. Other parts, especially matrix exponentials and
Strang splitting, are mathematically dense. Learn them in layers:

1. Use the function correctly.
2. Explain the purpose and input-output contract.
3. Modify the implementation and verify it with tests.

You can begin research after reaching the first two levels. The third level takes more time.

## A useful reading routine

For each function:

1. Open the corresponding `.py` file.
2. Read the function signature.
3. Find the `return` statement.
4. Run a small example.
5. Change one harmless input and observe the output.
6. Explain the function in one sentence.

For example, after reading `build_partial_data`, you should be able to say:

> It receives an observed position series and time step, reconstructs velocity by forward differences,
> and returns pseudo-states with columns `[X, Vhat]`.

# Model and parameter conventions

The project studies a partially observed Student Kramers stochastic differential equation:

$$
\begin{aligned}
dX_t &= V_t\,dt,\\
dV_t &= [-\eta V_t+F(X_t)]\,dt+\sqrt{q(X_t,V_t)}\,dW_t.
\end{aligned}
$$

The force is:

$$
F(x)=ax^3+bx^2+cx+d.
$$

The M4 squared diffusion is:

$$
q(x,v)=\alpha v^2+\beta v+\gamma
       +\delta x^2+\epsilon xv+\zeta x.
$$

Every model uses the same complete parameter order:

$$
\theta=(\eta,a,b,c,d,\alpha,\beta,\gamma,\delta,\epsilon,\zeta).
$$

This order is fixed across the entire project. `params[0]` always means $\eta$, `params[1]` always
means $a$, and so on.

## Python arguments and model parameters are different

The word "parameter" has two meanings here:

- A Python function argument, such as `seed` in `simulate_x(params, n_obs, seed=42)`.
- A statistical model parameter, such as $\eta$ or $\delta$.

This guide uses:

- `params` for the complete eleven-dimensional model vector;
- `free_params` for the coordinates optimized by one named model;
- "function argument" for values written inside Python function parentheses.

# Project map

## Main data flow

```text
config.py
  -> paths and numerical settings

models.py
  -> parameter structure, F(x), U(x), q(x,v), constraints

data_loading.py
  -> raw Excel data becomes [X, Vhat]

likelihoods.py
  -> candidate parameters receive an NLL score

estimation.py
  -> optimization searches for lower NLL

simulation.py
  -> fitted parameters generate new stochastic paths

recovery.py / discrimination.py / bootstrap.py
  -> repeated research experiments

run_*.py
  -> command-line entry points
```

## File responsibilities

| File | Responsibility |
|---|---|
| `config.py` | Shared paths, step sizes, seeds, experiment sizes, reference parameters |
| `models.py` | Model registry, parameter conversion, force, potential, diffusion, constraints |
| `data_loading.py` | Download, preprocessing, result paths, CSV I/O, provenance |
| `likelihoods.py` | Complete and partial Strang pseudo-likelihoods |
| `estimation.py` | L-BFGS-B, multiple starts, warm starts, model comparison tables |
| `simulation.py` | SDE simulation and model diagnostics |
| `recovery.py` | Repeated complete and partial observation recovery |
| `discrimination.py` | M3/M4 truth scenarios and comparison |
| `bootstrap.py` | Exact IOS and bootstrap workflows |
| `run_*.py` | Short command-line programs that call the modules |
| `tests/` | Permanent checks that protect formulas and workflows |

# Python foundations used by the project

## Variables and objects

Python assignment gives a name to an object:

```python
h = 0.02
model_name = "M4"
params = np.array([1.0, 2.0, 3.0])
```

`h` refers to a floating-point number, `model_name` refers to a string, and `params` refers to a NumPy
array.

Two names can refer to the same mutable object:

```python
a = np.array([1.0, 2.0])
b = a
b[0] = 99.0
print(a)
```

The first value of `a` also changes. Use `.copy()` when you need an independent array:

```python
b = a.copy()
```

## Arrays and shapes

An array shape tells you how many values exist along each axis:

```python
params.shape       # (11,)
data.shape         # (N, 2)
trajectory.shape   # (N, 2)
```

For partial observations:

```text
data[:, 0] = X
data[:, 1] = Vhat
```

For a complete simulated trajectory:

```text
trajectory[:, 0] = X
trajectory[:, 1] = V
```

`Vhat` is reconstructed from observed positions. It is not the same object as the latent simulated
velocity `V`.

## Functions

A function signature defines required and optional arguments:

```python
def simulate_trajectory(params, n_obs, h_obs=config.H_OBS,
                        h_sim=config.H_SIM, init_state=(0.0, 0.0), seed=42):
```

Required arguments:

```text
params
n_obs
```

Optional arguments already have defaults:

```text
h_obs
h_sim
init_state
seed
```

Call optional arguments by name when their meaning matters:

```python
simulate_trajectory(params, n_obs=501, seed=123)
```

## Multiple return values

Python packs multiple returned values into a tuple:

```python
return params, nll, conv
```

The caller can unpack them:

```python
params, nll, conv = estimate_model("M4", data)
```

## Lists, dictionaries, and comprehensions

A list preserves order:

```python
MODELS = ["M2", "M3", "M4"]
```

A dictionary maps keys to values:

```python
COLORS = {"M2": "blue", "M3": "orange", "M4": "green"}
```

A comprehension builds a collection from a loop:

```python
free_names = [PARAM_NAMES[i] for i in free_indices]
```

## Imports

This:

```python
from student_kramers.models import diffusion_variance
```

makes `diffusion_variance` available in the current file.

An underscore at the start of a name, such as `_A_mat`, means "internal helper." Python does not make
it truly private, but callers should normally use the public function instead.

## Exceptions

The simulator raises an exception when the squared diffusion becomes invalid:

```python
raise FloatingPointError("Diffusion variance became invalid")
```

The likelihood catches expected numerical failures and returns a large penalty:

```python
except EXPECTED_NUMERICAL_ERRORS:
    return config.PENALTY
```

This allows the optimizer to reject invalid candidates without terminating the whole run.

# `config.py`: shared settings

`config.py` is a settings module. It should not contain likelihood or estimation logic.

## Paths

```python
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
```

Using `Path` avoids fragile string concatenation.

## Important numerical settings

| Setting | Meaning |
|---|---|
| `H_OBS` | Observation interval used by the likelihood |
| `H_SIM` | Fine Euler-Maruyama simulation step |
| `H_SIM_VALIDATION` | Finer step for validation |
| `PENALTY` | Objective value assigned to invalid parameters |
| `EPS` | Small numerical floor |
| `M4_DIAGNOSTIC_MARGIN` | Enlarged data rectangle used for diagnostics only |
| `LBFGS_MAXITER` | Maximum optimizer iterations |
| `LBFGS_TOL` | Optimizer tolerance |
| `N_RANDOM_STARTS` | Default number of optimization starts |

`M4_DIAGNOSTIC_MARGIN` does not replace global diffusion positivity. It only controls diagnostic
rectangles used in plots and simulation summaries.

## Named result directories

```python
def run_dir(run_name=DEFAULT_RUN_NAME):
```

returns:

```text
results/runs/<run_name>/
```

Every distinct experiment should use a distinct run name.

# `models.py`: model mathematics and structure

## Why all models use eleven coordinates

M1-M4 share one complete parameter vector. Simpler models fix some coordinates to zero.

The registry stores:

- which coordinates are free;
- which coordinates are fixed;
- initial values for free coordinates;
- a human-readable description.

Example:

```python
"M3": {
    "free_indices": list(range(8)),
    "fixed": {8: 0.0, 9: 0.0, 10: 0.0},
}
```

M3 optimizes the first eight coordinates and fixes M4's three new diffusion coefficients to zero.

## `get_model`, `free_names`, and `free_bounds`

```python
get_model("M4")
```

returns the M4 registry entry.

```python
free_names("M3")
```

translates free parameter indices into names.

```python
free_bounds("M4")
```

returns the box bounds passed to L-BFGS-B. These bounds control the search scale. They do not replace
the nonlinear positivity constraint.

## `embed_params`

```python
def embed_params(free_params, model_name):
```

reconstructs the complete eleven-dimensional vector.

For M2:

```text
free_params = [eta, a, c, alpha, beta, gamma]

complete params =
[eta, a, 0, c, 0, alpha, beta, gamma, 0, 0, 0]
```

Likelihood and simulation functions use complete parameters so they do not need separate formulas for
every model.

## `extract_free_params`

```python
def extract_free_params(params, model_name):
```

performs the reverse operation. It takes a complete vector and keeps only the coordinates optimized by
one model.

This is important for warm starts:

```python
free_m4_start = extract_free_params(previous_m4_params, "M4")
```

## Force and potential

```python
def force(x, params):
```

computes:

$$
F(x)=ax^3+bx^2+cx+d.
$$

```python
def potential(x, params):
```

returns $U(x)$ satisfying:

$$
F(x)=-U'(x).
$$

The implemented formula is:

$$
U(x)=-\left(\frac{a}{4}x^4+\frac{b}{3}x^3+\frac{c}{2}x^2+dx\right).
$$

## Diffusion variance

```python
def diffusion_variance(x, v, params):
```

computes:

$$
q(x,v)=\alpha v^2+\beta v+\gamma+\delta x^2+\epsilon xv+\zeta x.
$$

M1-M3 have $\delta=\epsilon=\zeta=0$, so the same function automatically reduces to their diffusion.

## Matrix representation of M4 diffusion

Let:

$$
y=
\begin{bmatrix}x\\v\end{bmatrix},
\qquad
Q=
\begin{bmatrix}
\delta & \epsilon/2\\
\epsilon/2 & \alpha
\end{bmatrix},
\qquad
\ell=
\begin{bmatrix}\zeta\\\beta\end{bmatrix}.
$$

Then:

$$
q(x,v)=y^\mathsf{T}Qy+\ell^\mathsf{T}y+\gamma.
$$

`diffusion_quadratic_matrix(params)` returns $Q$.

`diffusion_augmented_matrix(params, floor)` returns:

$$
H_{\text{floor}}=
\begin{bmatrix}
\delta & \epsilon/2 & \zeta/2\\
\epsilon/2 & \alpha & \beta/2\\
\zeta/2 & \beta/2 & \gamma-\text{floor}
\end{bmatrix}.
$$

If this matrix is positive semidefinite, then:

$$
q(x,v)\geq\text{floor}
$$

for all real $x,v$.

Semidefinite is the correct word. Strict positive definiteness would exclude the M3 boundary.

## Global and regional minima

```python
def diffusion_minimum(params):
```

returns a global minimizer and the global minimum value of $q$.

When $Q$ is invertible:

$$
y_{\min}=-\frac12Q^{-1}\ell.
$$

The implementation uses a pseudoinverse so it also works for M1-M3.

```python
def diffusion_rectangle_bounds(data, margin=0.0):
def diffusion_rectangle_minimum(params, data, margin=0.0):
```

define an enlarged data rectangle and calculate the exact minimum of $q$ on that rectangle. They are
diagnostic functions. Formal M4 estimation still requires global positivity.

## `constraints_valid`

```python
def constraints_valid(free_params, model_name, data=None):
```

checks:

$$
\eta>0,\qquad a<0,\qquad 0\leq\alpha<2\eta,
$$

and global non-negativity of the squared diffusion through the augmented matrix.

The `data` argument remains in the interface because likelihood and start-generation code share the
same validation function. The formal constraint is global whether or not data is supplied.

## Registry validation

```python
def validate_model_registry():
```

runs when `models.py` is imported. It checks that free and fixed coordinates form a complete
non-overlapping partition, initial values have the correct shape, and bounds are consistent.

# `data_loading.py`: external data and result files

This module handles the outside world: downloads, Excel files, CSV files, paths, hashes, and
provenance.

## Raw-data pipeline

```python
ensure_official_excel()
```

downloads the official workbook if it is missing.

```python
load_raw_excel()
```

reads the required columns.

```python
preprocess_ca2(raw)
```

applies the research preprocessing:

1. Restrict age to 17-90 ka and interpolate missing Ca2+.
2. Average duplicate ages.
3. Transform Ca2+ by $-\log$.
4. Restrict to 30-80 ka.
5. Center the transformed series.

## Partial observations

```python
def build_partial_data(x, h=config.H_OBS):
```

constructs:

$$
\widehat V_t=\frac{X_{t+h}-X_t}{h}.
$$

The returned array contains:

```text
[X_t, Vhat_t]
```

Because one difference uses two positions, an input position series of length $N+1$ produces partial
data of length $N$.

## Loading real data

```python
df, age, x, data = load_real_data()
```

returns:

| Object | Contents |
|---|---|
| `df` | Processed DataFrame |
| `age` | Age sequence |
| `x` | Oldest-to-youngest position series |
| `data` | Partial pseudo-states `[X, Vhat]` |

## Result paths

```python
result_path(name, model_name=None, suffix="csv", run_name="development")
```

builds a path inside one named run.

`save_table` and `load_table` write and read DataFrames.

`save_model_fits` writes the shared model-comparison table and a provenance JSON file.

`load_model_fits(..., data=data)` verifies that the saved parameters, current data, and current code
fingerprint match.

## Why provenance matters

Long computations can outlive the code version that created them. A CSV without provenance can be
silently mistaken for a current result.

The project records:

- parameter hash;
- data hash;
- code hash;
- workflow name;
- time steps;
- experiment settings.

When these change, resumable workflows reject stale checkpoints.

# `likelihoods.py`: scoring candidate parameters

## Strang splitting overview

The partial-observation likelihood separates the drift into a linear Pearson part and a nonlinear
remainder. A transition applies:

```text
half nonlinear flow
  -> exact linear moment propagation
  -> inverse half nonlinear flow
```

For each transition, the partial likelihood scores the reconstructed-velocity residual.

The corrected contribution is:

$$
\ell_k=
\frac{r_k^2}{\Omega_k(h)}
+\frac23\log\Omega_k(3h/2).
$$

## Check matrices

The squared diffusion matrix is represented as:

$$
\operatorname{vec}(\Sigma\Sigma^\mathsf{T}(Y))
=\check\alpha\operatorname{vec}(YY^\mathsf{T})
+\check\beta Y+\check\gamma.
$$

Internal helpers:

| Function | Purpose |
|---|---|
| `_A_mat` | Linear drift matrix |
| `_check_alpha` | Quadratic diffusion terms |
| `_check_beta` | Linear diffusion terms |
| `_check_gamma` | Constant diffusion term |
| `_shift_cross_matrices` | Cross terms after branch shifting |
| `_precompute_matrices` | Block-exponential evaluation of `I1-I5` |
| `_step_omega` | Conditional covariance for one midpoint state |

M4 changes both `_check_alpha` and `_check_beta`. The $xv$ term also requires separate shifted cross
integrals `I2` and `I3`.

## `partial_transition_nlls`

```python
def partial_transition_nlls(free_params, data, model_name, h=config.H_OBS,
                            moment_mask=None, eval_mask=None):
```

This is the main source of truth for partial-observation scoring.

It:

1. validates the candidate;
2. reconstructs complete parameters;
3. computes empirical splitting moments;
4. selects positive or negative splitting branches;
5. propagates conditional means and covariances;
6. returns one NLL contribution per evaluated transition.

`moment_mask` controls which transitions estimate empirical moments.

`eval_mask` controls which transitions receive scores.

Exact IOS needs both masks.

## Total partial NLL

```python
def partial_neg_log_lik(free_params, data, model_name, h=config.H_OBS):
```

sums transition contributions:

$$
\operatorname{NLL}(\theta)=\sum_k\ell_k(\theta).
$$

It returns a scalar because optimizers need one number.

## Complete-observation likelihood

```python
def complete_neg_log_lik(free_params, data, model_name, h, branch="plus"):
```

uses complete `[X,V]` observations and a two-dimensional Gaussian transition objective. It is mainly
used for simulation validation and complete-data parameter recovery.

# `estimation.py`: optimization and starting values

## L-BFGS-B

```python
def run_estimator_lbfgs(params_init, loss_fn, bounds, ...):
```

calls SciPy's L-BFGS-B optimizer.

The optimizer knows nothing about Student Kramers models. It only knows:

```text
candidate vector -> loss_fn(candidate) -> scalar loss
```

`conv == 0` means the result is accepted. `conv == 1` means the run failed or returned an invalid
objective.

## Binding a loss function

```python
def make_loss_fn(model_name, data, h=config.H_OBS):
```

returns a function that accepts only `free_params` while keeping the model name, data, and time step
fixed.

## Multiple starts

```python
def make_random_starts(model_name, start, n_starts, seed=42,
                       data=None, extra_starts=None):
```

returns a list of valid starts:

1. the requested primary start;
2. valid, non-duplicate `extra_starts`;
3. globally valid nonzero M4 interior starts;
4. random valid perturbations for other models.

## Why M4 needs interior starts

M3 is embedded in M4 at:

$$
\delta=\epsilon=\zeta=0.
$$

This is a boundary point of the globally feasible M4 set. Invalid candidates receive a large fixed
penalty. A finite-difference optimizer starting only on the boundary can see many penalized directions
and incorrectly remain at M3.

The current code adds nonzero, globally valid M4 starts such as:

```python
candidate[8:11] = [delta, 0.0, 0.0]
```

and additional valid candidates with small nonzero cross and linear terms.

This change solved the first real-data M4 collapse without weakening global positivity.

## Estimating one model

```python
def estimate_model(model_name, data, h=config.H_OBS, start=None,
                   maxiter=None, tol=None, verbose=True,
                   n_starts=1, seed=42, extra_starts=None):
```

It:

1. reads the registry;
2. selects a primary start;
3. binds the partial likelihood;
4. generates valid starts;
5. optimizes from every start;
6. keeps the result with the lowest NLL;
7. returns the complete parameter vector.

## Estimating several models

```python
def estimate_models(model_names, data, h=config.H_OBS, verbose=True,
                    n_starts=1, seed=42, warm_starts=None):
```

returns one DataFrame row per model, including:

- complete parameters;
- NLL;
- number of free parameters;
- AIC and BIC;
- observed-region, diagnostic-region, and global minimum diffusion;
- convergence status;
- runtime.

When M3 is fitted before M4, its complete parameter vector supplies the M4 boundary baseline. A
previously fitted M4 can also be supplied through `warm_starts`.

# `simulation.py`: generating and checking paths

## Euler-Maruyama update

For fine step $\Delta t$:

$$
X_{n+1}=X_n+V_n\Delta t,
$$

$$
V_{n+1}
=V_n+[-\eta V_n+F(X_n)]\Delta t
+\sqrt{q(X_n,V_n)\Delta t}\,Z_n.
$$

`simulate_trajectory` performs these updates and retains states on the observation grid.

## `simulate_trajectory`

```python
def simulate_trajectory(params, n_obs, h_obs=config.H_OBS,
                        h_sim=config.H_SIM, init_state=(0.0, 0.0), seed=42):
```

returns an `(n_obs, 2)` array with columns `[X,V]`.

It stops with `FloatingPointError` if:

- diffusion becomes nonpositive;
- a state becomes non-finite;
- a path exceeds the explosion threshold.

## Convenience simulation functions

```python
simulate_x(...)
```

returns only positions.

```python
simulate_partial_data(...)
```

simulates positions and reconstructs partial pseudo-states `[X,Vhat]`.

## Potential and regime functions

`potential_extrema` identifies lower well, central barrier, and upper well.

`compute_derived_params` calculates interpretable quantities such as well positions, barrier heights,
global diffusion minimum, and a tail proxy.

`classify_regime`, `extract_waiting_times`, and `summarize_waiting_times` convert position paths into
lower and upper regime durations.

`simulate_first_passage` estimates the first time a path reaches the central barrier.

## Model checks

`path_summary_matrix` calculates summary statistics for many simulated paths.

`observed_summary` calculates the matching statistics for the real data.

`simulate_model_check` returns simulated position and reconstructed-velocity arrays.

## Diffusion stability check

```python
def simulate_diffusion_check(params, data, n_rep=100, seed=42, ...):
```

records one row per simulated path:

| Column | Meaning |
|---|---|
| `success` | Whether the full path completed |
| `q_min_true` | Minimum diffusion on latent `[X,V]` states |
| `q_min_partial` | Minimum diffusion on reconstructed `[X,Vhat]` states |
| `outside_fraction` | Fraction outside the enlarged diagnostic rectangle |
| `max_abs_x`, `max_abs_v` | Largest absolute state values |

The final global M4 fit completed 500 out of 500 same-length paths. A regional-only sensitivity fit
completed 499 out of 500 and failed once because diffusion became negative.

# `recovery.py`: repeated parameter recovery

One successful simulated fit is a code check. Repeated recovery asks whether the estimator behaves
consistently.

The intended same-model studies are:

```text
M2 truth -> fit M2
M3 truth -> fit M3
M4 truth -> fit M4
```

Each path is fitted under:

- complete observations `[X,V]`;
- partial observations `[X,Vhat]`.

Main functions:

| Function | Purpose |
|---|---|
| `_fit_one_observation_scheme` | Fit one path under one observation scheme |
| `run_recovery_study` | Simulate, fit, and checkpoint repeated paths |
| `summarize_recovery` | Parameter bias and RMSE |
| `summarize_recovery_diagnostics` | Success rates and function-level errors |
| `diffusion_recovery_surfaces` | Truth, complete, and partial diffusion surfaces |

Old recovery pilots created before the corrected M4 interior-start strategy are stale and were deleted.

# `discrimination.py`: separating M3 and M4

Discrimination studies fit both M3 and M4 to each simulated partial-observation path.

The comparison statistic is:

$$
C=2\{\operatorname{NLL}(M3)-\operatorname{NLL}(M4)\}.
$$

Positive values favor M4.

Scenarios include:

```text
M3 truth
weak M4 truth
moderate M4 truth
strong M4 truth
```

Main functions:

| Function | Purpose |
|---|---|
| `scenario_params` | Construct a valid truth scenario |
| `_fit_models` | Fit M3 and M4 to one partial dataset |
| `run_discrimination_study` | Repeat simulation and comparison |

These studies must be repeated after meaningful changes to constraints or starting values.

# `bootstrap.py`: IOS and calibration

## Exact IOS

For transition $k$:

$$
\operatorname{IOS}_k
=\ell_k(\widehat\theta_{-k})
-\ell_k(\widehat\theta).
$$

The model is refitted without transition $k$, then the held-out transition is scored using the
leave-one-out estimate.

Main functions include:

- `fit_leave_one_out`;
- `run_exact_ios`;
- `summarize_ios`;
- `influential_transitions`.

Exact IOS is expensive because it refits once per transition.

## Parametric bootstrap

Parametric bootstrap repeats:

```text
simulate from fitted model
  -> reconstruct partial data
  -> refit model
  -> save parameters and statistics
```

It estimates uncertainty under the fitted model.

## Contrast bootstrap

Contrast bootstrap simulates under a null model and fits both null and alternative models to each
replication. It calibrates an observed nested-model likelihood contrast.

# Command-line entry points

Run entry points from the project root:

```bash
python3 -m student_kramers.run_application
```

Do not run `python3 run_application.py` from inside the package directory.

## Cheap checks

```bash
python3 -m student_kramers.run_single --help
python3 -m unittest discover -s tests -v
```

## Real-data fitting

```bash
python3 -m student_kramers.run_application \
  --models M2 M3 M4 \
  --run-name real_fit_01 \
  --n-starts 16 \
  --seed 42
```

Use a previous run as additional starts:

```bash
python3 -m student_kramers.run_application \
  --models M2 M3 M4 \
  --run-name real_fit_02 \
  --warm-start-run real_fit_01 \
  --n-starts 16
```

`--warm-start-run` does not copy old results. It uses old parameters as starts and re-optimizes with
current code and data.

## Validation, recovery, and discrimination

```bash
python3 -m student_kramers.run_validation --help
python3 -m student_kramers.run_recovery --help
python3 -m student_kramers.run_discrimination --help
```

## IOS and bootstrap

```bash
python3 -m student_kramers.run_bootstrap --help
```

These workflows can be much more expensive than a real-data fit.

# Tests

Tests answer different questions.

| Test file | Main protection |
|---|---|
| `test_models.py` | Registry, parameter conversion, constraints, starts, diagnostics |
| `test_m4.py` | M4-to-M3 reduction and independent moment propagation |
| `test_regression.py` | Stable known NLL and IOS values |
| `test_workflow.py` | Result isolation and provenance |
| `test_recovery.py` | Recovery checkpoint extension |
| `test_discrimination.py` | Scenario validity and nested M3 start |

Run everything:

```bash
python3 -m unittest discover -s tests -v
```

A passing test suite does not prove that M4 is scientifically better. It proves that the code satisfies
the properties encoded by the tests.

# The M4 optimization lesson

The first real-data M4 fit returned exactly the M3 result. That result was initially compatible with
several explanations, including no M4 signal, overly restrictive constraints, or incorrect
implementation.

The diagnostic sequence was:

1. Confirm M4 reduces to M3 across formulas, likelihoods, and simulation.
2. Inspect the supervisor's public implementation.
3. Temporarily test regional positivity to see whether useful M4 directions exist.
4. Observe that regional fits improve NLL but can fail simulation outside the protected region.
5. Restore global positivity.
6. Start from nonzero, globally valid M4 interior points.
7. Warm-start repeated polishing from the best valid M4.
8. Verify global minimum diffusion and 500 simulated paths.

The final global M4 fit has:

| Model | NLL | AIC | BIC | Global minimum $q$ |
|---|---:|---:|---:|---:|
| M2 | 8524.348 | 17060.695 | 17095.637 | 4176.668 |
| M3 | 8524.059 | 17064.118 | 17110.707 | 4167.946 |
| M4 | 8506.650 | 17035.299 | 17099.359 | 57.190 |

The lesson is specific:

> A mathematically correct hard constraint can still produce a failed optimization when the optimizer
> starts only on the boundary of the feasible set.

Do not respond by automatically removing the constraint. First test feasible interior starts.

# Suggested learning sequence

## Stage 1: arrays and model structure

Run:

```python
from student_kramers.models import MODELS, embed_params, extract_free_params

print(MODELS["M3"])
params = embed_params(MODELS["M3"]["init"], "M3")
print(params)
print(extract_free_params(params, "M3"))
```

Goal: explain complete and free parameter vectors.

## Stage 2: data shapes

```python
from student_kramers.data_loading import load_real_data

df, age, x, data = load_real_data()
print(x.shape)
print(data.shape)
print(data[:3])
```

Goal: explain why `data` has two columns and one fewer row than `x`.

## Stage 3: one NLL evaluation

```python
from student_kramers.likelihoods import partial_transition_nlls, partial_neg_log_lik
from student_kramers.models import MODELS

free = MODELS["M2"]["init"]
values = partial_transition_nlls(free, data, "M2")
total = partial_neg_log_lik(free, data, "M2")
print(values.shape)
print(values.sum(), total)
```

Goal: explain transition contributions and total NLL.

## Stage 4: one simulation

```python
from student_kramers.simulation import simulate_trajectory

traj = simulate_trajectory(params, n_obs=101, seed=42)
print(traj.shape)
```

Goal: explain observation count, fine step, seed, and two-dimensional state.

## Stage 5: one optimization

```python
from student_kramers.estimation import estimate_model

params_hat, nll, conv = estimate_model("M2", data, n_starts=2, seed=42)
print(nll, conv)
```

Goal: explain what changes across starts and what remains fixed.

## Stage 6: research workflows

Only after understanding the earlier stages, inspect:

- repeated recovery;
- discrimination;
- IOS;
- parametric bootstrap;
- contrast bootstrap.

These workflows repeat the same core functions many times. They are easier to understand after one
data load, one likelihood evaluation, one simulation, and one fit are clear.

# Function-reading checklist

When you encounter an unfamiliar function:

1. Read the signature.
2. Identify required and optional arguments.
3. Check expected shapes.
4. Find conversions such as `np.asarray` or `to_numpy`.
5. Identify lower-level functions it calls.
6. Read the `return` statement.
7. Search for callers.
8. Match every array column and parameter coordinate to the mathematics.
9. Check which tests protect the function.

The three relationships to keep in mind are:

$$
\theta=(\eta,a,b,c,d,\alpha,\beta,\gamma,\delta,\epsilon,\zeta),
$$

$$
\texttt{data}_t=(X_t,\widehat V_t),
$$

and:

$$
\widehat\theta
=\arg\min_\theta\operatorname{NLL}(\theta;\texttt{data}).
$$

# Coverage checklist

The English fourth edition covers:

```text
student_kramers/config.py
student_kramers/models.py
student_kramers/data_loading.py
student_kramers/likelihoods.py
student_kramers/estimation.py
student_kramers/simulation.py
student_kramers/recovery.py
student_kramers/discrimination.py
student_kramers/bootstrap.py
student_kramers/run_*.py
tests/
```

Coverage does not mean mastery. Track your progress with three labels:

| Level | Standard |
|---|---|
| Seen | You know the function exists |
| Explain | You can describe inputs, outputs, callers, and formula |
| Modify | You can change it and verify the change with tests |

Before running expensive formal research, aim to explain `models.py`, `data_loading.py`,
`estimation.py`, `simulation.py`, and the relevant command-line entry point. For M4 work, also explain
the relevant `likelihoods.py` matrices and the difference between a boundary start and a feasible
interior start.

