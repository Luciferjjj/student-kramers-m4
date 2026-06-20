---
title: "Current status of the M4 Student Kramers extension"
author: "Zisen Pan"
date: "2026-06-20"
date-format: "MMMM D, YYYY"
format:
  html:
    toc: true
    toc-depth: 2
    html-math-method: mathjax
    theme: cosmo
    embed-resources: true
  pdf:
    toc: true
    toc-depth: 2
    pdf-engine: xelatex
    geometry:
      - margin=22mm
execute:
  enabled: false
---

> **Current summary**
>
> - M4 adds position dependence to the M3 diffusion while keeping the drift
>   unchanged.
> - The first direct-coefficient optimizer was unreliable at the M3 boundary.
> - The current optimizer represents the augmented diffusion matrix as
>   $H=LL^\top$, so every M4 proposal is globally feasible.
> - On the Greenland data, the current NLL is 8524.059 for M3 and 8499.312 for
>   M4. AIC and BIC are also lower for M4.
> - The 500-replication M4 parametric bootstrap has 496 successful fits. The
>   diffusion function is more stable over the observed region than its
>   individual polynomial coefficients.
> - Exact IOS is complete for all 2499 transitions in M2, M3, and M4. A strict
>   200-replication M4 IOS bootstrap gives upper-tail $p=0.965$.
> - M4 has not yet been formally selected over M3. The remaining model
>   comparison is a new M3-null nested bootstrap using the final optimizer.

## 1. Research question

The previous Greenland analysis compared three nested Student Kramers models.
M2 allowed diffusion to depend on velocity, and M3 added asymmetry to the
deterministic force. The present extension asks whether the diffusion should
also depend on position.

M3 uses

$$
q_{M3}(v)=\alpha v^2+\beta v+\gamma,
$$

whereas M4 uses

$$
q_{M4}(x,v)=\alpha v^2+\beta v+\gamma
+\delta x^2+\epsilon xv+\zeta x.
$$

The M3 drift and partial-observation Strang pseudo-likelihood are retained.
M4 reduces to M3 when $\delta=\epsilon=\zeta=0$.

The data pipeline is unchanged from the previous project. Calcium is
negative-log transformed and centered in the 30 to 80 kyr BP window. Velocity
is reconstructed from the observed position series by forward differences.
The likelihood uses 2500 pseudo-states and 2499 transitions.

![Observed Greenland series, reconstructed velocity, and phase space](figures/data_and_state_space.png)

**Figure 1. Data used by the partial-observation likelihood.** The panels show
the transformed calcium coordinate, reconstructed velocity, the fitted M4
diffusion adjustment along the path, and the observed phase space.

## 2. Why positivity became an optimization problem

The M4 diffusion is a quadratic surface in $(x,v)$. Define

$$
y=
\begin{pmatrix}
x\\v\\1
\end{pmatrix},
\qquad
H=
\begin{pmatrix}
\delta & \epsilon/2 & \zeta/2\\
\epsilon/2 & \alpha & \beta/2\\
\zeta/2 & \beta/2 & \gamma-q_{\mathrm{floor}}
\end{pmatrix}.
$$

Then

$$
q(x,v)-q_{\mathrm{floor}}=y^\top Hy.
$$

Global diffusion validity follows from $H\succeq0$. The first implementation
optimized the direct diffusion coefficients and assigned a large objective to
invalid proposals. Starting from fitted M3 placed the optimizer on the M4
boundary. Finite-difference steps frequently left the feasible set, and the
optimizer could return the M3 solution even when better M4 solutions existed.

Interior starts showed that the equality between M3 and M4 was an optimizer
artifact. The best direct-coefficient fit reached NLL 8506.650. This was a
useful diagnostic, but the hard boundary remained unsuitable for thousands of
related leave-one-out and bootstrap fits.

The final optimizer uses

$$
H=LL^\top
$$

and

$$
\alpha=2\eta\,\mathrm{logistic}(\rho).
$$

Every optimizer proposal therefore satisfies global diffusion positivity and
$0<\alpha<2\eta$. Direct coefficients are reconstructed after optimization,
so reported parameters retain the original scientific interpretation.

## 3. Current real-data fit

The formal run is `m4_real_data_cholesky`.

| Model | Free parameters | NLL | AIC | BIC |
|---|---:|---:|---:|---:|
| M2 | 6 | 8524.348 | 17060.695 | 17095.637 |
| M3 | 8 | 8524.059 | 17064.118 | 17110.707 |
| M4 | 11 | 8499.312 | 17020.623 | 17084.683 |

The observed M3 versus M4 contrast is

$$
C_{\mathrm{obs}}=
2\{\mathrm{NLL}_{M3}-\mathrm{NLL}_{M4}\}=
49.495.
$$

This is a descriptive improvement in the corrected partial
pseudo-likelihood. AIC and BIC point in the same direction, but they are not a
replacement for finite-sample calibration because M3 is a boundary submodel
of M4.

The fitted potential remains broadly similar across M2, M3, and M4. The main
change is in the diffusion surface. M4 raises or lowers conditional variance
according to both position and reconstructed velocity.

![Current model comparison and fitted mechanisms](figures/real_data_mechanisms.png)

**Figure 2. What changes under M4.** M4 has the lowest descriptive NLL, AIC,
and BIC. The fitted potential remains double-well, while the diffusion changes
across the observed phase space.

The likelihood gain is not produced by a single transition. Transition-level
decomposition shows where M4 improves or worsens the local score and how the
gain accumulates through the record.

![Transition-level likelihood improvement](figures/transition_improvement.png)

**Figure 3. Location of the M3 to M4 likelihood gain.** Positive values favor
M4. The figure connects local score changes with age, reconstructed state, and
changes in the diffusion surface.

## 4. Global boundary and observed-region behavior

The fitted M4 quadratic surface has a global minimum near the imposed floor:

$$
\min_{x,v}q_{M4}(x,v)=0.000276.
$$

The minimizer is far outside the observed state region. Its standardized
distance from the center of the data is about 178. The corresponding minima
are

$$
\min_{\text{observed rectangle}}q_{M4}=3867.032
$$

and

$$
\min_{\text{observed path}}q_{M4}=3869.180.
$$

Thus the current M4 fit is close to the positive semidefinite boundary in
remote state space, but it does not obtain its real-data fit by approaching
zero diffusion along the observed path.

![Global and data-supported diffusion minima](figures/diffusion_domain_audit.png)

**Figure 4. Diffusion-domain audit.** The figure separates the remote global
minimum from the observed rectangle, protected rectangle, and realized path.
This distinction matters for both interpretation and asymptotic arguments.

## 5. Predictive checks

Current predictive checks use 100 simulations from each fitted model. They
compare marginal spread, occupancy, switching counts, and regime durations.

M4 improves some aspects of the simulated dynamics, but the improvement is
limited. The observed position standard deviation is 1.001 and lies above all
100 simulations for each model. The observed number of switches is 45,
compared with medians of 63.5 for M2, 63.5 for M3, and 61 for M4. M4 moves the
switching distribution in the observed direction, but all models usually
switch too often and generate paths that are not persistent enough.

![Observed and simulated marginal densities](figures/predictive_density_bands.png)

**Figure 5. Marginal predictive densities.** The observed position and
reconstructed-velocity densities are compared with pointwise simulation
bands. The figure shows where local distributional fit remains incomplete.

![Predictive percentile map](figures/predictive_check_map.png)

**Figure 6. Predictive-check summary.** Each observed statistic is placed in
its simulated distribution. M4 modestly improves some switching summaries,
but the observed position spread and long waiting times remain difficult for
all three models.

These checks suggest that any next model extension should not be motivated by
local likelihood alone. Long-run occupancy and persistence remain distinct
targets.

## 6. M4 parametric bootstrap

The M4 parametric bootstrap performs

```text
simulate from fitted M4
    -> refit M4 with the Cholesky optimizer
    -> store coefficients and diffusion summaries
```

Of 500 target replications, 496 produced successful fits. The median runtime
was 3.23 seconds per replication.

The direct coefficients show substantial compensation. In particular,
intervals for several drift and diffusion coefficients cross zero, and the
three position-dependent coefficients have broad distributions. This makes
coefficient-by-coefficient interpretation unreliable.

![Bootstrap intervals for the M4 coefficients](figures/m4_parameter_intervals.png)

**Figure 7. Parameter uncertainty.** Bootstrap intervals are shown relative
to the observed M4 estimate, with the three new diffusion coefficients
displayed separately. The width of these intervals motivates a
function-focused interpretation.

The diffusion function is more stable in the data-supported region than its
individual coefficients. The 95% bootstrap interval for the minimum on the
observed rectangle is approximately $[2190,4884]$, and the corresponding
interval for the minimum along the observed path is approximately
$[2285,4897]$.

![M4 diffusion uncertainty under parametric bootstrap](figures/m4_diffusion_bootstrap.png)

**Figure 8. Diffusion-function uncertainty.** The panels report refit
stability, global and observed-region minima, uncertainty along the observed
path, and the relation between pathwise scale and the rectangle minimum.

The bootstrap supports interpretation of the fitted function over observed
states. It gives much weaker support for interpreting isolated polynomial
coefficients or extrapolating the quadratic surface far beyond the data.

## 7. Exact IOS and finite-sample calibration

For transition $k$, the information-omission contribution is

$$
\mathrm{IOS}_k=
\ell_k(\widehat\theta_{-k})-\ell_k(\widehat\theta),
$$

and the exact statistic is

$$
T_N=\sum_{k=1}^{2499}\mathrm{IOS}_k.
$$

All 2499 transitions produced valid results in all three models:

| Model | Valid transitions | Observed $T_N$ | Optimization time |
|---|---:|---:|---:|
| M2 | 2499/2499 | 8.589 | 123.1 s |
| M3 | 2499/2499 | 11.549 | 182.2 s |
| M4 | 2499/2499 | 21.876 | 3793.9 s |

M4 has a larger total IOS, but its positive contributions are less
concentrated. M2 and M3 reach 80% of positive IOS with 2.4% and 3.4% of
transitions, whereas M4 requires 8.6%.

![Observed exact IOS comparison](figures/observed_exact_ios.png)

**Figure 9. Exact observed IOS.** The panels compare total statistics,
transition timing, concentration, and agreement between M3 and M4. The usual
M4 asymptotic reference is marked as nonregular because the fitted augmented
diffusion matrix is close to the PSD boundary.

The strict model-wise bootstrap recalculates the complete statistic for every
simulated sample:

```text
simulate from fitted M4
    -> refit M4
    -> run all 2499 leave-one-out fits
    -> calculate T_N*
```

All 200 replications produced complete, valid exact-IOS results. The summary
is:

| Quantity | Value |
|---|---:|
| Observed $T_N$ | 21.876 |
| Bootstrap median | 39.829 |
| Bootstrap 95% interval | [21.089, 154.801] |
| Upper-tail probability | 0.965 |
| Lower-tail probability | 0.040 |
| Observed percentile | 3.5% |

![Finite-sample M4 IOS calibration](figures/m4_modelwise_ios_bootstrap.png)

**Figure 10. Model-wise IOS bootstrap.** The observed statistic is not
unusually large under fitted M4. It is near the lower edge of the bootstrap
distribution, which remains a diagnostic feature rather than evidence of poor
fit in the planned upper-tail check.

The IOS result addresses goodness of fit under fitted M4. It does not compare
M4 with M3 and does not prove that M4 is correct.

## 8. What is established

The current evidence supports the following statements.

1. M4 is implemented consistently with the M1 to M3 model family.
2. The Cholesky coordinates maintain global diffusion feasibility throughout
   M4 optimization.
3. The current M4 fit has a lower corrected partial pseudo-likelihood than M3.
4. M4 mainly changes the diffusion mechanism rather than the qualitative
   double-well potential.
5. The fitted diffusion remains far from zero in the observed state region.
6. Parametric-bootstrap uncertainty is smaller for the diffusion function
   than for its individual polynomial coefficients.
7. The observed exact IOS statistic is not unusually large under the
   finite-sample distribution generated by fitted M4.

The current evidence does not establish that M4 should replace M3. The old
M3-null nested bootstrap used the pre-Cholesky M4 optimizer and the old
observed contrast 34.819. It cannot calibrate the current contrast 49.495.

## 9. Proposed next calculation

The next formal calculation is:

```text
simulate under current fitted M3
    -> refit M3
    -> refit M4 with the Cholesky optimizer
    -> calculate 2(NLL_M3* - NLL_M4*)
```

The run should first reach 100 successful replications. If the finite-sample
tail probability is clearly away from the decision boundary, 100
replications are sufficient for the next research discussion. If the result
is borderline, the same checkpointed run can be extended to at most 300
replications.

Current-optimizer M4 recovery and M3/M4 discrimination remain useful follow-up
experiments. They are secondary to the nested calibration because the latter
directly addresses the present model-selection question.

## 10. Questions for discussion

1. Is the position-dependent diffusion scientifically interpretable enough to
   justify M4 beyond its descriptive likelihood improvement?
2. Should the remote near-boundary behavior of the quadratic surface be
   treated as a modelling limitation, even though the observed region is well
   separated from zero diffusion?
3. Is a 100-replication nested bootstrap sufficient for the next decision,
   with extension to 300 only if the result is borderline?
4. Should the next model revision target state persistence and occupancy
   rather than adding more local diffusion flexibility?
5. Which current-optimizer simulation study is most useful after the nested
   bootstrap: functional recovery, discrimination, or both?
