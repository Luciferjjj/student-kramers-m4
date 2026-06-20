"""
models.py - Student Kramers equations and nested model definitions

All models share the full parameter order

    theta = [eta, a, b, c, d, alpha, beta, gamma, delta, epsilon, zeta].

The common SDE family is

    dX_t = V_t dt
    dV_t = [-eta V_t + F(X_t)] dt + sqrt(q(X_t, V_t)) dW_t,

    F(x) = a x^3 + b x^2 + c x + d,
    q(x,v) = alpha v^2 + beta v + gamma
             + delta x^2 + epsilon x v + zeta x.

M1-M3 fix the three position-dependent diffusion coefficients to zero.  M4
releases all eleven parameters, so M3 is a boundary submodel of M4.
"""
import numpy as np


PARAM_NAMES = [
    "eta", "a", "b", "c", "d", "alpha", "beta", "gamma",
    "delta", "epsilon", "zeta",
]
PARAM_LABELS = {
    "eta": r"$\eta$",
    "a": r"$a$",
    "b": r"$b$",
    "c": r"$c$",
    "d": r"$d$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
    "delta": r"$\delta$",
    "epsilon": r"$\epsilon$",
    "zeta": r"$\zeta$",
}
PARAM_COUNT = len(PARAM_NAMES)
DIFFUSION_FLOOR = 1e-8
CONSTRAINT_TOL = 1e-10

FULL_BOUNDS = [
    (1e-6, 200.0),
    (-500.0, -1e-6),
    (-500.0, 500.0),
    (-500.0, 500.0),
    (-500.0, 500.0),
    (0.0, 200.0),
    (-1000.0, 1000.0),
    (1e-6, 20000.0),
    (-1000.0, 1000.0),
    (-1000.0, 1000.0),
    (-1000.0, 1000.0),
]

# M4 is optimized through a Cholesky representation of the augmented
# diffusion matrix.  These are numerical search bounds for
# [eta, a, b, c, d, l11, rho, phi, l31, l32, l33].
M4_CHOLESKY_BOUNDS = [
    *FULL_BOUNDS[:5],
    (0.0, 50.0),
    (-20.0, 20.0),
    (0.0, np.pi),
    (-200.0, 200.0),
    (-200.0, 200.0),
    (0.0, 200.0),
]

MODELS = {
    "M1": {
        "description": "constant diffusion symmetric model",
        "free_indices": [0, 1, 3, 7],
        "fixed": {
            2: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
            8: 0.0, 9: 0.0, 10: 0.0,
        },
        "init": np.array([57.6876, -130.0412, 115.2705, 8322.5068]),
    },
    "M2": {
        "description": "symmetric Student Kramers model",
        "free_indices": [0, 1, 3, 5, 6, 7],
        "fixed": {2: 0.0, 4: 0.0, 8: 0.0, 9: 0.0, 10: 0.0},
        "init": np.array([
            67.8737, -96.6383, 65.4232, 64.2695, 233.1185, 4388.0601,
        ]),
    },
    "M3": {
        "description": "asymmetric drift with velocity-dependent diffusion",
        "free_indices": list(range(8)),
        "fixed": {8: 0.0, 9: 0.0, 10: 0.0},
        "init": np.array([
            67.8844, -97.8274, 8.4228, 66.2782,
            -9.1246, 64.4645, 228.2274, 4369.9513,
        ]),
    },
    "M4": {
        "description": "asymmetric drift with position-velocity diffusion",
        "free_indices": list(range(PARAM_COUNT)),
        "fixed": {},
        "init": np.array([
            67.8844, -97.8274, 8.4228, 66.2782, -9.1246,
            64.4645, 228.2274, 4369.9513, 1.0, 0.0, 0.0,
        ]),
    },
}


def get_model(model_name):
    """Return one named model specification."""
    try:
        return MODELS[model_name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {model_name!r}. Available: {list(MODELS)}") from exc


def free_names(model_name):
    """Return the free parameter names for one model."""
    return [PARAM_NAMES[i] for i in get_model(model_name)["free_indices"]]


def free_bounds(model_name):
    """Return L-BFGS-B bounds for one model."""
    cfg = get_model(model_name)
    if "bounds" in cfg:
        return list(cfg["bounds"])
    return [FULL_BOUNDS[i] for i in cfg["free_indices"]]


def embed_params(free_params, model_name):
    """Reconstruct the full eleven-parameter vector."""
    cfg = get_model(model_name)
    free_params = np.asarray(free_params, dtype=float)
    if free_params.shape != (len(cfg["free_indices"]),):
        raise ValueError(f"{model_name} expects {len(cfg['free_indices'])} free parameters")

    params = np.zeros(PARAM_COUNT, dtype=float)
    for idx, value in cfg["fixed"].items():
        params[idx] = value
    for pos, idx in enumerate(cfg["free_indices"]):
        params[idx] = free_params[pos]
    return params


def extract_free_params(params, model_name):
    """Extract one model's free parameters from a full parameter vector."""
    params = np.asarray(params, dtype=float)
    if params.shape != (PARAM_COUNT,):
        raise ValueError(f"Full parameter vector must have length {PARAM_COUNT}")
    return params[get_model(model_name)["free_indices"]]


def force(x, params):
    """Return F(x) = a*x^3 + b*x^2 + c*x + d."""
    _, a, b, c, d = np.asarray(params, dtype=float)[:5]
    x = np.asarray(x)
    return a*x**3 + b*x**2 + c*x + d


def potential(x, params):
    """Return the potential U satisfying F = -dU/dx."""
    _, a, b, c, d = np.asarray(params, dtype=float)[:5]
    x = np.asarray(x)
    return -(a*x**4/4.0 + b*x**3/3.0 + c*x**2/2.0 + d*x)


def diffusion_variance(x, v, params):
    """Return q(x,v), the squared diffusion coefficient."""
    params = np.asarray(params, dtype=float)
    alpha, beta, gamma, delta, epsilon, zeta = params[5:11]
    x, v = np.asarray(x), np.asarray(v)
    return alpha*v**2 + beta*v + gamma + delta*x**2 + epsilon*x*v + zeta*x


def diffusion_quadratic_matrix(params):
    """Return Q such that q(y) = y.T Q y + l.T y + gamma for y=(x,v)."""
    params = np.asarray(params, dtype=float)
    alpha, delta, epsilon = params[5], params[8], params[9]
    return np.array([[delta, epsilon/2.0], [epsilon/2.0, alpha]], dtype=float)


def diffusion_augmented_matrix(params, floor=0.0):
    """
    Return the homogeneous matrix representing ``q(x,v) - floor``.

        [x, v, 1] H [x, v, 1].T = q(x,v) - floor,

        H = [[delta,     epsilon/2, zeta/2],
             [epsilon/2, alpha,     beta/2],
             [zeta/2,    beta/2,    gamma-floor]].

    Positive semidefiniteness of this matrix is equivalent to global
    non-negativity of ``q(x,v) - floor``.  Unlike requiring Q to be positive
    definite, this condition allows M1-M3 to remain boundary cases of M4.
    """
    params = np.asarray(params, dtype=float)
    beta, gamma, zeta = params[6], params[7], params[10]
    Q = diffusion_quadratic_matrix(params)
    return np.array([
        [Q[0, 0], Q[0, 1], zeta/2.0],
        [Q[1, 0], Q[1, 1], beta/2.0],
        [zeta/2.0, beta/2.0, gamma - floor],
    ])


def diffusion_minimum(params):
    """Return the minimum-norm minimizer of q and its global minimum value."""
    params = np.asarray(params, dtype=float)
    Q = diffusion_quadratic_matrix(params)
    linear = np.array([params[10], params[6]], dtype=float)
    minimizer = -0.5*np.linalg.pinv(Q, rcond=1e-12) @ linear
    q_min = float(diffusion_variance(minimizer[0], minimizer[1], params))
    return minimizer, q_min


def diffusion_rectangle_bounds(data, margin=0.0):
    """Return lower and upper corners of the enlarged data rectangle."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) == 0:
        raise ValueError("data must have shape (n, 2)")
    lo = np.min(data, axis=0)
    hi = np.max(data, axis=0)
    span = np.maximum(hi - lo, 1e-8)
    return lo - margin*span, hi + margin*span


def diffusion_rectangle_minimum(params, data, margin=0.0):
    """
    Return the exact minimum of q(x,v) on a data-defined rectangle.

    The rectangle spans the observed x and v ranges and is enlarged by
    ``margin`` times each range on both sides.  A quadratic reaches its
    rectangle minimum at an interior stationary point, a boundary stationary
    point, or a corner, so only those candidates need evaluation.
    """
    params = np.asarray(params, dtype=float)
    lo, hi = diffusion_rectangle_bounds(data, margin)
    x_lo, v_lo = lo
    x_hi, v_hi = hi
    alpha, beta, gamma, delta, epsilon, zeta = params[5:11]

    candidates = [
        (x_lo, v_lo), (x_lo, v_hi), (x_hi, v_lo), (x_hi, v_hi),
    ]

    # Minima on x-fixed boundaries.
    if alpha > 0.0:
        for x in (x_lo, x_hi):
            v_star = -(beta + epsilon*x)/(2.0*alpha)
            if v_lo <= v_star <= v_hi:
                candidates.append((x, v_star))

    # Minima on v-fixed boundaries.
    if delta > 0.0:
        for v in (v_lo, v_hi):
            x_star = -(zeta + epsilon*v)/(2.0*delta)
            if x_lo <= x_star <= x_hi:
                candidates.append((x_star, v))

    # Interior stationary point.
    Q = diffusion_quadratic_matrix(params)
    linear = np.array([zeta, beta], dtype=float)
    if abs(np.linalg.det(Q)) > 1e-12:
        x_star, v_star = -0.5*np.linalg.solve(Q, linear)
        if x_lo <= x_star <= x_hi and v_lo <= v_star <= v_hi:
            candidates.append((x_star, v_star))

    values = np.array([
        diffusion_variance(x, v, params) for x, v in candidates
    ], dtype=float)
    index = int(np.argmin(values))
    return np.asarray(candidates[index], dtype=float), float(values[index])


def velocity_drift(x, v, params):
    """Return -eta*v + F(x)."""
    return -float(params[0])*np.asarray(v) + force(x, params)


def constraints_valid(free_params, model_name, data=None):
    """
    Check shared stability and diffusion-positivity constraints.

    The original tail condition ``0 <= alpha < 2*eta`` is retained.  M1-M3
    retain their global one-dimensional diffusion positivity condition.

    All models require global non-negativity of the squared diffusion.  For
    M4 this is equivalent to positive semidefiniteness of the augmented
    quadratic matrix.  The optional ``data`` argument is accepted so this
    function has one common interface for likelihood and start validation.
    """
    try:
        params = embed_params(free_params, model_name)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(params)):
        return False

    eta, a, alpha = params[0], params[1], params[5]
    if eta <= 0.0 or a >= 0.0 or alpha < 0.0 or alpha >= 2.0*eta:
        return False

    augmented = diffusion_augmented_matrix(params, floor=DIFFUSION_FLOOR)
    if np.min(np.linalg.eigvalsh(augmented)) < -CONSTRAINT_TOL:
        return False
    return True


def m4_constraint_values(free_params):
    """
    Return smooth inequality values used by the constrained M4 optimizer.

    Feasible M4 parameters satisfy every returned value >= 0.  The first
    three values are the eigenvalues of the augmented diffusion matrix for
    ``q(x,v) - DIFFUSION_FLOOR``; the final value is the retained tail
    condition ``2*eta - alpha > 0``.
    """
    params = embed_params(free_params, "M4")
    eigenvalues = np.linalg.eigvalsh(
        diffusion_augmented_matrix(params, floor=DIFFUSION_FLOOR),
    )
    tail_margin = 2.0*params[0] - params[5] - CONSTRAINT_TOL
    return np.r_[eigenvalues, tail_margin]


def m4_to_cholesky_params(free_params, jitter=1e-10):
    """
    Map direct M4 coefficients to stable optimization coordinates.

    The augmented matrix for ``q(x,v) - DIFFUSION_FLOOR`` is represented as
    ``L L.T``.  A small eigenvalue floor permits stable conversion of boundary
    cases such as M3 while remaining negligible relative to the fitted scale.
    """
    params = embed_params(free_params, "M4")
    augmented = diffusion_augmented_matrix(params, floor=DIFFUSION_FLOOR)
    eigenvalues, eigenvectors = np.linalg.eigh(augmented)
    positive = eigenvectors @ np.diag(np.maximum(eigenvalues, jitter)) @ eigenvectors.T
    L = np.linalg.cholesky(positive)

    eta = params[0]
    alpha = L[1, 0]**2 + L[1, 1]**2
    fraction = np.clip(alpha/(2.0*eta), 1e-10, 1.0 - 1e-10)
    rho = np.log(fraction/(1.0 - fraction))
    phi = np.arctan2(L[1, 1], L[1, 0])
    return np.r_[params[:5], L[0, 0], rho, phi, L[2, 0], L[2, 1], L[2, 2]]


def m4_from_cholesky_params(cholesky_params):
    """
    Map stable M4 optimization coordinates to direct model coefficients.

    This construction guarantees both global diffusion positivity and
    ``0 < alpha < 2*eta`` throughout optimization:

        H = L L.T,
        alpha = 2*eta*logistic(rho).
    """
    values = np.asarray(cholesky_params, dtype=float)
    if values.shape != (PARAM_COUNT,):
        raise ValueError(f"M4 Cholesky vector must have length {PARAM_COUNT}")
    eta, a, b, c, d, l11, rho, phi, l31, l32, l33 = values
    fraction = 1.0/(1.0 + np.exp(-np.clip(rho, -50.0, 50.0)))
    alpha = 2.0*eta*fraction
    l21 = np.sqrt(alpha)*np.cos(phi)
    l22 = np.sqrt(alpha)*np.sin(phi)
    L = np.array([
        [l11, 0.0, 0.0],
        [l21, l22, 0.0],
        [l31, l32, l33],
    ])
    augmented = L @ L.T
    return np.array([
        eta, a, b, c, d,
        augmented[1, 1],
        2.0*augmented[1, 2],
        augmented[2, 2] + DIFFUSION_FLOOR,
        augmented[0, 0],
        2.0*augmented[0, 1],
        2.0*augmented[0, 2],
    ])


def parameter_row(model_name, params, nll=None):
    """Flatten one model name, parameter vector, and optional NLL for a CSV row."""
    row = {"model": model_name, "description": get_model(model_name)["description"]}
    row.update(dict(zip(PARAM_NAMES, np.asarray(params, dtype=float))))
    if nll is not None:
        row["nll"] = float(nll)
    return row


def validate_model_registry():
    """Fail early when a registered nested model is internally inconsistent."""
    expected = set(range(PARAM_COUNT))
    for name, cfg in MODELS.items():
        free = list(cfg["free_indices"])
        fixed = set(cfg["fixed"])
        if len(free) != len(set(free)):
            raise ValueError(f"{name}: free_indices contains duplicates")
        if set(free) | fixed != expected or set(free) & fixed:
            raise ValueError(f"{name}: free_indices and fixed must partition all parameters")
        if len(cfg["init"]) != len(free):
            raise ValueError(f"{name}: init must contain one value per free parameter")
        bounds = free_bounds(name)
        if len(bounds) != len(free):
            raise ValueError(f"{name}: bounds must contain one pair per free parameter")
        if any(not lo <= value <= hi for value, (lo, hi) in zip(cfg["init"], bounds)):
            raise ValueError(f"{name}: initial values must lie inside bounds")
        if not constraints_valid(cfg["init"], name):
            raise ValueError(f"{name}: initial values violate model constraints")


validate_model_registry()
