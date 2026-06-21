---
title: "Implementation note for the M4 Student Kramers extension"
author: "Zisen Pan"
date: "2026-06-21"
date-format: "MMMM D, YYYY"
format:
  html:
    toc: true
    toc-depth: 3
    html-math-method: mathjax
    theme: cosmo
    embed-resources: true
  pdf:
    toc: true
    toc-depth: 3
    pdf-engine: xelatex
    geometry:
      - margin=22mm
execute:
  enabled: false
---

## 1. Scope

This note describes how the M4 extension is represented in the current Python
implementation. It focuses on the connection between the model equations,
optimization coordinates, likelihood calculation, and numerical validation.
It is not a catalogue of every function in the repository. The complete
function lookup remains in `CODE_REFERENCE.md`.

The implementation has two packages:

```text
student_kramers/
    reusable model definitions, likelihoods, estimation, simulation,
    recovery, discrimination, IOS, and bootstrap algorithms

greenland_application/
    Greenland data loading, application runners, diagnostics,
    result summaries, and figures
```

The dependency direction is
`greenland_application -> student_kramers`. The mathematical package does not
depend on the Greenland application.

## 2. Common model representation

All models use the same full parameter order:

$$
\theta=
(\eta,a,b,c,d,\alpha,\beta,\gamma,\delta,\epsilon,\zeta).
$$

The state equation is

$$
\begin{aligned}
dX_t&=V_t\,dt,\\
dV_t&=
\left[-\eta V_t+aX_t^3+bX_t^2+cX_t+d\right]dt
+\sqrt{q(X_t,V_t)}\,dW_t.
\end{aligned}
$$

The deterministic force is

$$
F(x)=ax^3+bx^2+cx+d.
$$

Each model is registered by the indices of its free coefficients. Functions
such as `extract_free_params(...)` and `embed_params(...)` convert between a
model-specific optimization vector and the common eleven-dimensional vector.
This avoids maintaining separate likelihood and simulation implementations
for M1, M2, M3, and M4.

The shared representation is also the basis of the nesting tests. A parameter
vector can be evaluated as M3 or embedded in M4 without changing the order of
the coefficients used by the mathematical functions.

## 3. M4 diffusion and the nesting of M3

M3 uses velocity-dependent diffusion:

$$
q_{M3}(v)=\alpha v^2+\beta v+\gamma.
$$

M4 adds position-dependent terms:

$$
q_{M4}(x,v)=\alpha v^2+\beta v+\gamma
+\delta x^2+\epsilon xv+\zeta x.
$$

Thus M4 changes the stochastic forcing but retains the M3 drift. Setting

$$
\delta=\epsilon=\zeta=0
$$

recovers M3. This equality is enforced by the model registry and tested at
the levels of the diffusion function, simulated paths, complete-observation
likelihood, and partial-observation transition contributions.

The function `diffusion_variance(x, v, params)` evaluates the same formula for
all registered models. M1 to M3 have zeros in the M4-only positions of the
full parameter vector.

## 4. Quadratic and augmented matrix representations

The pure quadratic part of M4 can be written with

$$
z=
\begin{pmatrix}
x\\v
\end{pmatrix},
\qquad
Q=
\begin{pmatrix}
\delta & \epsilon/2\\
\epsilon/2 & \alpha
\end{pmatrix},
\qquad
r=
\begin{pmatrix}
\zeta\\\beta
\end{pmatrix},
$$

so that

$$
q(x,v)=z^\top Qz+r^\top z+\gamma.
$$

The matrix $Q$ is useful for stationary-point and rectangle-minimum
calculations, but it is not sufficient for global positivity because it does
not include the linear and constant terms.

For the positivity constraint, define

$$
y=
\begin{pmatrix}
x\\v\\1
\end{pmatrix}
$$

and

$$
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

If $H$ is positive semidefinite, then

$$
y^\top Hy\geq0
$$

for every $y$, and therefore

$$
q(x,v)\geq q_{\mathrm{floor}}
\qquad
\text{for all }(x,v)\in\mathbb R^2.
$$

This is stronger than checking positivity at observed states or on a bounded
rectangle. A rectangle check remains useful as a diagnostic because it
separates model behavior in the data-supported region from remote behavior of
the quadratic surface. It is not the formal feasibility constraint used by
the current optimizer.

The functions `diffusion_quadratic_matrix(...)`,
`diffusion_augmented_matrix(...)`, `diffusion_minimum(...)`, and
`diffusion_rectangle_minimum(...)` implement these distinct calculations.

## 5. Cholesky coordinates used by the optimizer

The first M4 implementation optimized the direct coefficients
$(\alpha,\beta,\gamma,\delta,\epsilon,\zeta)$ and returned a large penalty for
an invalid diffusion surface. This approach was sensitive to the starting
point. In particular, starting from fitted M3 places the optimizer on the M4
boundary, where finite-difference proposals can leave the feasible set.

The current implementation generates the augmented matrix as

$$
H=LL^\top,
$$

with

$$
L=
\begin{pmatrix}
l_{11} & 0 & 0\\
l_{21} & l_{22} & 0\\
l_{31} & l_{32} & l_{33}
\end{pmatrix}.
$$

Any matrix of this form produces a positive semidefinite $H$. The direct
diffusion coefficients obtained by expanding $LL^\top$ are

$$
\begin{aligned}
\delta &= l_{11}^2,\\
\epsilon &= 2l_{11}l_{21},\\
\zeta &= 2l_{11}l_{31},\\
\alpha &= l_{21}^2+l_{22}^2,\\
\beta &= 2(l_{21}l_{31}+l_{22}l_{32}),\\
\gamma &= q_{\mathrm{floor}}+l_{31}^2+l_{32}^2+l_{33}^2.
\end{aligned}
$$

The Student Kramers tail condition also requires
$0<\alpha<2\eta$. The code parameterizes the second row of $L$ by

$$
\begin{aligned}
\alpha&=2\eta\,\mathrm{logistic}(\rho),\\
l_{21}&=\sqrt{\alpha}\cos\phi,
\qquad
l_{22}=\sqrt{\alpha}\sin\phi.
\end{aligned}
$$

This gives

$$
l_{21}^2+l_{22}^2=\alpha
$$

and enforces the tail condition for every optimizer proposal.

The actual M4 optimization vector is

```text
[eta, a, b, c, d, l11, rho, phi, l31, l32, l33]
```

The model output is converted back to the common direct coefficient vector
before it is returned or saved. The functions
`m4_to_cholesky_params(...)` and `m4_from_cholesky_params(...)` perform the
two conversions.

An exact M3 boundary matrix can be semidefinite rather than positive definite.
The conversion to a numerical Cholesky factor therefore applies a small
eigenvalue floor when an M3 estimate is used as an M4 starting point. This
jitter stabilizes the coordinate conversion. It does not redefine the M3
model or change the direct coefficient representation.

## 6. Where M4 enters the likelihood

The real-data likelihood is a corrected partial-observation Strang
pseudo-likelihood. Only $X_t$ is observed. The velocity used by the likelihood
is reconstructed as

$$
\widehat V_{t_k}=
\frac{X_{t_{k+1}}-X_{t_k}}{h}.
$$

The input array therefore contains pseudo-states
$(X_{t_k},\widehat V_{t_k})$. The Greenland analysis has 2500 pseudo-states
and 2499 likelihood transitions.

The Strang construction separates a nonlinear force step from a linear
Pearson moment-propagation step. M4 changes the covariance propagation in the
linear step because the diffusion now contains $x^2$, $xv$, and $x$ terms.

For a two-dimensional state $Y=(X,V)^\top$, the diffusion covariance is

$$
\Sigma\Sigma^\top(Y)=
\begin{pmatrix}
0&0\\
0&q(X,V)
\end{pmatrix}.
$$

The implementation writes its vectorization as

$$
\mathrm{vec}\{\Sigma\Sigma^\top(Y)\}=
\check\alpha\,\mathrm{vec}(YY^\top)
+\check\beta\,Y+\check\gamma.
$$

For M4,

$$
\check\alpha=
\begin{pmatrix}
0&0&0&0\\
0&0&0&0\\
0&0&0&0\\
\delta&\epsilon/2&\epsilon/2&\alpha
\end{pmatrix},
$$

The linear and constant maps are

$$
\check\beta=
\begin{pmatrix}
0&0\\
0&0\\
0&0\\
\zeta&\beta
\end{pmatrix},
\qquad
\check\gamma=
\begin{pmatrix}
0\\0\\0\\\gamma
\end{pmatrix}.
$$

The two $\epsilon/2$ entries account for the retained $XV$ and $VX$ terms in
the vectorized second moment.

The linear step is translated around a branch-dependent point
$b=(\kappa,0)^\top$. Expanding $YY^\top$ for $Y=Z+b$ produces quadratic,
two cross, linear, and constant contributions. These are propagated through
the matrix-exponential terms named `I1` to `I5`. The M4 implementation retains
the same notation and overall construction as the earlier models, but the
position-dependent diffusion makes the shifted cross terms and
$q(\kappa,0)$ contribution nonzero.

The main public likelihood functions are:

| Function | Role |
|---|---|
| `partial_transition_nlls(...)` | one corrected partial-observation contribution per transition |
| `partial_neg_log_lik(...)` | sum of all transition contributions |
| `masked_partial_neg_log_lik(...)` | training objective used by leave-one-out IOS |
| `complete_neg_log_lik(...)` | complete-observation objective used in validation and recovery |

## 7. Estimation workflow

M1 to M3 use bounded L-BFGS-B in their direct free parameters. M4 uses
L-BFGS-B after conversion to the Cholesky coordinates above.

`estimate_model(...)` has one interface for all models:

```text
model name + partial data + starts
    -> model-specific optimizer
    -> full eleven-dimensional parameter vector
    -> NLL and convergence diagnostic
```

For an M3 versus M4 fit, the fitted M3 vector is included as an M4 boundary
start. Additional globally feasible interior starts or a saved M4 warm start
can also be supplied. The best finite objective is retained.

The current formal real-data result comes from
`results/runs/m4_real_data_cholesky/`. Older direct-coefficient M4 fits are
kept as development history because the optimizer changed the observed M4
objective enough to affect model comparison and bootstrap calibration.

## 8. Validation checks

The test suite protects the implementation at several levels.

1. Diffusion nesting: M4 with
   $\delta=\epsilon=\zeta=0$ equals M3 at supplied states.
2. Fixed-seed simulation nesting: M3 and nested M4 generate the same path.
3. Partial likelihood nesting: every transition contribution agrees.
4. Complete likelihood nesting: total complete-observation objectives agree.
5. Moment propagation: the `I1` to `I5` covariance construction agrees with an
   independent seven-dimensional first and second moment system.
6. Coordinate conversion: direct and Cholesky representations satisfy the
   registered constraints and return the expected coefficient order.
7. Workflow tests: recovery, IOS, bootstrap checkpointing, and result
   provenance retain consistent inputs and outputs.

The current suite contains 44 tests. Numerical results are additionally
separated by evidence class in `docs/results/`:

```text
docs/results/current/       current Cholesky M4 evidence
docs/results/development/   pre-Cholesky historical experiments
```

## 9. Technical decisions requested

The Cholesky representation solves the feasibility problem, but several parts
of the current formulation still require a modelling decision.

### 9.1 Domain of the positivity constraint

The current model requires

$$
q(x,v)\geq q_{\mathrm{floor}}
\qquad\text{for every }(x,v)\in\mathbb R^2.
$$

This global condition is stronger than positivity on the observed path or on
a scientifically relevant phase-space region. The current M4 fit is close to
the PSD boundary only in a remote extrapolation region; its diffusion remains
far from zero on the observed path. The first decision is therefore whether
global PSD positivity is part of the intended M4 model or a conservative
computational choice.

### 9.2 Optimizer and parameterization

The current estimator uses Cholesky coordinates, finite-difference gradients,
multistart L-BFGS-B, an M3 boundary start, and optional M4 warm starts. This
combination is reproducible and keeps all proposals feasible. The open
question is whether it should be treated as the final estimator or compared
with another constrained optimizer or parameterization as a sensitivity
analysis.

### 9.3 Theoretical restrictions and numerical safeguards

Three types of restriction should be distinguished:

1. the Student Kramers tail condition $0<\alpha<2\eta$;
2. the numerical diffusion floor $q_{\mathrm{floor}}=10^{-8}$;
3. finite search bounds on the Cholesky coordinates, including
   $0\leq l_{11}\leq 50$.

The first is a model condition. The latter two primarily make optimization
well-defined. Before treating the current parameterization as final, it
should be decided whether the floor and search bounds have a scientific
interpretation or should instead be justified through a pre-specified
sensitivity analysis. Because M4 bounds are imposed in Cholesky coordinates,
their implied ranges for the reported coefficients do not coincide with the
direct-coordinate bounds used by M1 to M3. For example,
$0\leq l_{11}\leq 50$ implies $0\leq\delta\leq2500$.

### 9.4 Near-boundary behavior in the nested bootstrap

M3 is a boundary submodel of M4. The current M3-null bootstrap completed 100
valid replications with a corrected upper-tail probability of 0.1386. Two
replications reached the $l_{11}$ search bound, and several upper-tail fits
approached the diffusion floor on their simulated data rectangle.

These replications are feasible outputs of the stated estimator and remain in
the formal null distribution. They must not be removed after inspection. The
technical decision is whether this behavior should motivate wider bounds,
regularization, or a narrower scientifically defined M4 alternative.

### 9.5 Interpretation target

Current recovery and parametric bootstrap results show that the diffusion
function is more stable than the three direct position-dependent
coefficients. Interpretation should therefore focus on $q(x,v)$ over relevant
states. If individual coefficient signs are scientifically important, the
present evidence does not identify them precisely enough.

The concise list of questions for the next meeting is given in
[`M4_RESEARCH_UPDATE_FOR_PREDRAG.md`](M4_RESEARCH_UPDATE_FOR_PREDRAG.md).
