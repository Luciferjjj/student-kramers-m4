---
title: "M4 research workflow and audit log"
subtitle: "From model extension and failed optimization to the current real-data fit"
lang: en
toc: true
toc-depth: 3
number-sections: true
format:
  html:
    theme: cosmo
    html-math-method: mathjax
    smooth-scroll: true
---

# Purpose of this document

This is not a formal paper or a final scientific conclusion. It is a working audit log that records
what we tried, what failed, what evidence changed our interpretation, and what still needs to be done.

The main questions are:

1. What changed from M3 to M4?
2. Which Python modules changed, and why?
3. Why did the first real-data M4 fit collapse to M3?
4. How did we separate a model problem from a constraint or optimization problem?
5. Which results are current, and which experiments must be repeated?

Each investigation follows the same pattern:

```text
problem
  -> initial explanations
  -> checks performed
  -> evidence
  -> current decision
  -> unfinished work
```

Failed attempts stay in the record because they explain later decisions.

# Research question and M4 definition

The M4 change supplied by the supervisor was:

$$
q_{M3}(x,v)=\alpha v^2+\beta v+\gamma,
$$

$$
q_{M4}(x,v)
=\alpha v^2+\beta v+\gamma
+\delta x^2+\epsilon xv+\zeta x.
$$

The supervisor also stated that everything else remains unchanged and that the implementation change
mainly affects `I1-I5`, through the matrices based on $\check\alpha$ and $\check\beta$.

The drift is therefore unchanged:

$$
F(x)=ax^3+bx^2+cx+d,
$$

and the state equation remains:

$$
\begin{aligned}
dX_t &= V_t\,dt,\\
dV_t &= [-\eta V_t+F(X_t)]\,dt+\sqrt{q(X_t,V_t)}\,dW_t.
\end{aligned}
$$

The complete M4 parameter order is:

$$
\theta=(\eta,a,b,c,d,\alpha,\beta,\gamma,\delta,\epsilon,\zeta).
$$

M4 must reduce exactly to M3 when:

$$
\delta=\epsilon=\zeta=0.
$$

# Stage 1: unified M1-M4 implementation

## Goal

We did not create a separate copy of the code for M4. The research requires repeated comparison of
M1-M4, so model differences now live in one registry and share the same likelihood, simulation, and
estimation interfaces.

## Main code changes

| File | Main responsibility |
|---|---|
| `student_kramers/models.py` | Unified eleven-parameter vector, M4 diffusion, matrices, and constraints |
| `student_kramers/likelihoods.py` | M4 terms in `check_alpha`, `check_beta`, and `I1-I5` |
| `student_kramers/simulation.py` | State-dependent diffusion and simulation stability diagnostics |
| `student_kramers/estimation.py` | Unified estimation, multiple starts, and warm starts |
| `student_kramers/config.py` | Shared model parameters and experiment settings |
| `student_kramers/recovery.py` | Repeated complete and partial observation recovery |
| `student_kramers/discrimination.py` | M3/M4 discrimination scenarios |
| `tests/` | Nested-model, moment propagation, likelihood, and workflow protection |

## Coding style

The implementation keeps names familiar from the supervisor's GitHub repository:

```text
A, Ak, check_alpha, check_beta, check_gamma, I1, I2, I3, I4, I5
```

This makes review easier, but similar names are not treated as evidence of correctness. M4-specific
mathematics is checked independently.

# Stage 2: M4-to-M3 sanity checks

## Why these checks came first

Setting three coefficients to zero is easy on paper. Code can still fail to reduce correctly in a
matrix integral, simulation step, or likelihood branch. The unified implementation was accepted only
after all computational layers reduced to M3.

## Checks that passed

`tests/test_m4.py` checks:

| Check | Requirement |
|---|---|
| Diffusion function | `q_M4(x,v) == q_M3(x,v)` |
| Simulation | Same path for the same parameters, step size, and seed |
| Partial likelihood | Same transition-level NLL values |
| Complete likelihood | Same total NLL |
| Genuine M4 | Finite likelihood when new coefficients are nonzero |
| Independent moments | `I1-I5` agree with a direct 7 by 7 moment equation |

Current test status:

```text
27 tests passed
```

The real-data fitting problem was therefore unlikely to be caused by simply omitting M4 from the
likelihood.

# Stage 3: first simulation studies and real-data fit

## Initial plan

The first research plan was:

```text
repeated parameter recovery
  -> M2 truth fit M2
  -> M3 truth fit M3
  -> M4 truth fit M4
  -> complete and partial data

M3/M4 discrimination
  -> M3 truth fit M3 and M4
  -> weak/moderate/strong M4 truth fit M3 and M4

real-data fitting
  -> fit M2, M3 and M4
```

The pilot workflows ran, but the M4 start strategy later changed materially. The old pilot outputs were
deleted and must not be treated as current evidence. Recovery and discrimination need to be repeated
using the current globally valid interior-start strategy.

## First real-data result

The first M4 fit started from the embedded M3 boundary:

$$
(\delta,\epsilon,\zeta)=(0,0,0).
$$

It returned:

$$
\operatorname{NLL}_{M4}=\operatorname{NLL}_{M3}.
$$

M4 appeared to collapse completely to M3.

## Explanations considered at the time

We considered several possibilities:

1. The real data does not support M4.
2. The M4 constraints are too restrictive.
3. The M4 starting point is poor.
4. The M4 likelihood or constraint implementation is wrong.
5. The supervisor's better M4 result used different data, complete observations, or different optimizer settings.

At this point, "the data does not support M4" was not justified because the optimizer might not have
explored M4 at all.

# Stage 4: supervisor code and constraint review

## What the public reference code showed

We inspected the public Greenland application and Student Kramers estimation code:

- The public repository contains M1-M3, but not M4.
- Its public JAX L-BFGS estimator does not use our L-BFGS-B parameter bounds.
- It does not state formal M4 constraints.
- The email specifies the M4 diffusion formula and the `I1-I5` change.

## Mathematical positivity requirement

The M4 diffusion can be written as:

$$
q(x,v)=
\begin{bmatrix}x&v&1\end{bmatrix}
H
\begin{bmatrix}x&v&1\end{bmatrix}^{\mathsf T},
$$

with:

$$
H=
\begin{bmatrix}
\delta & \epsilon/2 & \zeta/2\\
\epsilon/2 & \alpha & \beta/2\\
\zeta/2 & \beta/2 & \gamma
\end{bmatrix}.
$$

Requiring $H$ to be positive semidefinite guarantees:

$$
q(x,v)\geq0
\qquad\text{for all }(x,v)\in\mathbb R^2.
$$

Semidefiniteness matters because M3 lies on the boundary of M4. Strict positive definiteness would
incorrectly remove the nested M3 case.

# Stage 5: regional positivity sensitivity study

## Why regional positivity was tested

To test whether the global condition was hiding useful M4 directions, we temporarily required
positivity only on the observed state rectangle and enlarged versions of that rectangle.

This was a diagnostic experiment, not a final model decision.

## Evidence

M4 improved on M3 under every tested regional margin:

| Margin | M4 NLL | NLL improvement over M3 |
|---:|---:|---:|
| 10% | 8487.360 | 36.699 |
| 50% | 8500.600 | 23.459 |
| 75% | 8500.197 | 23.862 |
| 100% | 8502.559 | 21.499 |
| 200% | 8513.196 | 10.863 |

This showed that the new M4 directions contain information. The first boundary result could not be
interpreted as an absence of all M4 signal.

## Why regional positivity was rejected as the formal constraint

The 100% regional fit had a low NLL, but it was not a globally valid SDE. In 500 simulations with the
same length as the real data:

```text
499 succeeded
1 failed because q(x,v) became negative
```

The regional study supports a limited conclusion:

> The first constrained boundary fit failed to explore useful M4 directions.

It does not support:

> The regional-only model is the final correct M4.

# Stage 6: identifying the boundary-start problem

## Why optimization became stuck

M3 is embedded in M4 at:

$$
\delta=\epsilon=\zeta=0.
$$

This is a boundary point of the globally feasible M4 set. Invalid candidates receive:

```python
PENALTY = 1e15
```

When L-BFGS-B estimates finite-difference directions from this boundary, many small moves leave the
positive-semidefinite cone and receive the same large penalty. The optimizer may then report the M3
boundary as the M4 optimum without exploring the feasible interior.

## Decisive check

We retained global positivity but initialized M4 from multiple nonzero, globally valid interior points.

Several starts improved on M3. Repeated warm-start polishing produced:

$$
\widehat\theta_{M4}=
\begin{aligned}[t]
(&68.1151,-97.4728,8.0575,67.7974,-8.7130,\\
 &63.4467,231.9098,4368.4149,84.0535,-142.6631,-9.2392).
\end{aligned}
$$

The new coefficients are:

$$
(\widehat\delta,\widehat\epsilon,\widehat\zeta)
=(84.0535,-142.6631,-9.2392).
$$

They do not collapse to zero.

## Current formal real-data comparison

| Model | NLL | AIC | BIC | Global minimum $q$ |
|---|---:|---:|---:|---:|
| M2 | 8524.348 | 17060.695 | 17095.637 | 4176.668 |
| M3 | 8524.059 | 17064.118 | 17110.707 | 4167.946 |
| M4 | **8506.650** | **17035.299** | **17099.359** | **57.190** |

Relative to M3:

$$
\Delta\operatorname{NLL}=-17.409,
$$

$$
\Delta\operatorname{AIC}=-28.819,
$$

$$
\Delta\operatorname{BIC}=-11.348.
$$

NLL, AIC, and BIC all favor the current M4 fit.

# Stage 7: final stability checks

## Global diffusion check

The formal M4 fit has:

$$
\min_{x,v}q_{M4}(x,v)=57.190>0.
$$

The minimizer is approximately:

$$
(x_{\min},v_{\min})=(-32.6,-38.5).
$$

The position is far outside the observed $x$ range, but the global minimum remains positive.

## Simulation check

Using the final M4 parameters, we simulated 500 paths with the same length as the real data:

```text
500 / 500 paths succeeded
```

The minimum diffusion values along successful paths also remained well above zero.

## Current interpretation

The current M4 fit has:

- lower NLL than M3;
- lower AIC and BIC than M3;
- globally positive diffusion;
- 500 successful simulations out of 500;
- passing M4-to-M3 reduction tests.

The main cause of the first failed M4 fit was not global positivity itself. It was:

> Starting only from the M3 boundary under a hard global constraint, which prevented the optimizer from entering the feasible M4 interior.

# Current files and results

## Formal results

```text
results/runs/m4_real_data_final/
```

| File | Contents |
|---|---|
| `model_fits.csv` | Current formal M2/M3/M4 real-data fits |
| `m4_diffusion_check.csv` | 500-path stability check for final M4 |
| `m4_constraint_diagnostics.csv` | Boundary, regional, and global interior-start comparison |
| `m4_margin_sensitivity.csv` | Regional positivity margin sensitivity |
| `figures/` | Diagnostic figures generated by the notebook |

## Research notebook

```text
notebooks/new project.ipynb
```

It has been executed from top to bottom locally. All 16 cells completed without errors. It contains:

- NLL, AIC, and BIC comparisons;
- parameter changes;
- potential comparison;
- common-scale diffusion heatmaps;
- M4 minus M3 diffusion differences;
- boundary, regional, and global interior-start diagnosis;
- global diffusion minimum and 500-path simulation stability.

## Teaching guide

```text
PYTHON_BEGINNER_GUIDE.md
PYTHON_BEGINNER_GUIDE.html
```

The fourth edition includes the constraint and optimization investigation.

# Completed audit checklist

| Item | Status | Evidence |
|---|---|---|
| M4 formula added to unified code | Complete | `models.py`, `likelihoods.py` |
| M4 reduces to M3 when new coefficients are zero | Complete | `tests/test_m4.py` |
| `I1-I5` agree with independent moment propagation | Complete | `tests/test_m4.py` |
| M4 complete and partial likelihoods run | Complete | Tests and recovery code |
| Cause of first real-data collapse diagnosed | Complete | Boundary and interior-start comparison |
| Regional positivity sensitivity checked | Complete | `m4_margin_sensitivity.csv` |
| Formal M4 is globally positive | Complete | `q_min_global = 57.190` |
| Formal M4 simulation stability checked | Complete | 500/500 |
| Formal M2/M3/M4 real-data comparison completed | Complete | `model_fits.csv` |
| Notebook executed and figures reviewed | Complete | `new project.ipynb` |
| Old boundary fits and stale pilots removed | Complete | Only the final run remains in `results/runs/` |

# Current milestone and next steps

## Completed milestone

The formal M2, M3, and M4 real-data fits are complete. Their fitted parameters, NLL, AIC, BIC,
global positivity diagnostics, 500-path M4 simulation check, and comparison figures are saved.
No additional real-data fit is required before starting IOS.

## Why recovery and discrimination still appear as future work

Both studies were run earlier, but their M4 fits used only the M3 boundary start. After diagnosing
that problem and adopting globally valid interior starts, the stale simulation pilots were deleted
as requested. They need to be repeated with the corrected optimizer before they can serve as
reliable method-validation evidence.

This does not mean that the real-data fit is unfinished, and these repeats are not hard
prerequisites for computing IOS.

## Next formal analysis track: start IOS now

1. Compute IOS goodness-of-fit for M2, M3, and M4 using the locked real-data fits.
2. Inspect IOS contributions and influential observations.
3. Calibrate each model's IOS with a model-wise parametric bootstrap, while assessing parameter
   and diffusion-surface uncertainty.
4. Compare observed and simulated densities, waiting times, and transition behavior.
5. Run a formal M3 versus M4 nested-model comparison bootstrap.

## Independent method-validation track: repeat later

1. Repeat M2/M3/M4 parameter recovery with the current global interior-start strategy.
2. Repeat M3 truth and weak/moderate/strong M4 discrimination.
3. Check stability across seeds, interior-start sets, and optimizer tolerances.

Both tracks matter, but the method-validation track does not block the next IOS analysis. Successful
and failed experiments will continue to be recorded in this audit log.

# Template for future runs

Append this template for each new experiment:

```markdown
## YYYY-MM-DD: experiment name

### Question

What does this experiment try to answer?

### Code and settings

- Git/code state:
- run name:
- truth model:
- fitted models:
- n starts:
- seed:
- observation scheme:
- positivity rule:

### Results

- successful/failed fits:
- NLL/AIC/BIC:
- parameter or function recovery:
- minimum q:
- simulation stability:

### Interpretation

What does the evidence directly support? Which explanations remain tentative?

### Decision

Keep, repeat, modify, or stop?
```
