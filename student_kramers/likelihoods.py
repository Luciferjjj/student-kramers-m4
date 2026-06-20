"""
likelihoods.py — Complete and partial Strang pseudo-likelihoods

Data conventions
----------------
Complete observations have shape ``(N+1, 2)`` with columns ``[X_t, V_t]``.
Partial pseudo-states have shape ``(N, 2)`` with columns ``[X_t, Vhat_t]``,
where ``Vhat_t = (X_{t+h} - X_t)/h``.

Strang splitting
----------------
The drift is split into a linear Pearson part and nonlinear remainder:

    A = [[0, 1], [L, -eta]],
    L = 3*a*E[X^2] + 2*b*E[X] + c,
    N(x) = F(x) - L*(x-kappa).

One transition applies half the nonlinear flow, the exact linear Pearson
moment flow, then the inverse half nonlinear flow.

Partial-observation objective
-----------------------------
Only the reconstructed-velocity residual is scored.  For transition k:

    ell_k = r_k^2 / Omega_k(h) + (2/3)*log Omega_k(3h/2).

``partial_transition_nlls`` is the single source of truth used by estimation,
exact IOS, and all bootstrap calculations.

M4 diffusion representation
----------------------------
The reference implementation's check-matrix notation is retained:

    vec(SigmaSigmaT(Y))
        = check_alpha vec(Y Y.T) + check_beta Y + check_gamma.

After shifting Y = Z + b, the five matrix integrals I1-I5 propagate the
quadratic, two cross, linear, and constant terms.  A separate direct 7x7
moment ODE remains below as an independent numerical check.
"""
import numpy as np
from scipy.linalg import expm

from . import config
from .models import constraints_valid, embed_params, force


EXPECTED_NUMERICAL_ERRORS = (
    FloatingPointError,
    ValueError,
    OverflowError,
    np.linalg.LinAlgError,
)


def _A_mat(params, drift_const):
    """Return the 2x2 linear drift matrix A."""
    return np.array([[0.0, 1.0], [drift_const, -params[0]]], dtype=float)


def _check_alpha(params):
    """Return check_alpha, mapping vec(YY.T) to vec(SigmaSigmaT(Y))."""
    alpha, _, _, delta, epsilon, _ = params[5:11]
    check_alpha = np.zeros((4, 4))
    check_alpha[3] = [delta, epsilon/2.0, epsilon/2.0, alpha]
    return check_alpha


def _check_beta(params):
    """Return check_beta, mapping Y to the linear part of vec(SigmaSigmaT(Y))."""
    _, beta, _, _, _, zeta = params[5:11]
    check_beta = np.zeros((4, 2))
    check_beta[3] = [zeta, beta]
    return check_beta


def _check_gamma(params):
    """Return check_gamma, the constant part of vec(SigmaSigmaT(Y))."""
    gamma = params[7]
    return np.array([0.0, 0.0, 0.0, gamma])


def _shift_cross_matrices(b):
    """Return maps from z to vec(z b.T) and vec(b z.T), respectively."""
    b0, b1 = b
    R_zb = np.array([
        [b0, 0.0],
        [b1, 0.0],
        [0.0, b0],
        [0.0, b1],
    ])
    R_bz = np.array([
        [b0, 0.0],
        [0.0, b0],
        [b1, 0.0],
        [0.0, b1],
    ])
    return R_zb, R_bz


def _precompute_matrices(params, drift_const, kappa, h):
    """
    Precompute the M4 versions of I1-I5 for one splitting branch.

    The layout follows the matrix-integral implementation in the reference
    Greenland application.  I2 and I3 are separate because M4's x*v term
    makes the two shifted quadratic cross terms non-zero.  I5_G5 uses
    SigmaSigmaT(b), which now depends on branch-specific ``kappa``.
    """
    A = _A_mat(params, drift_const)
    I = np.eye(2)
    Ak = np.kron(A, I) + np.kron(I, A)
    al = _check_alpha(params)
    be = _check_beta(params)
    ga = _check_gamma(params)
    b = np.array([kappa, 0.0])
    R_zb, R_bz = _shift_cross_matrices(b)

    # One block exponential evaluates all five matrix integrals.
    P = np.zeros((18, 18))
    P[0:4, 0:4] = Ak + al
    P[0:4, 4:8] = al
    P[4:8, 4:8] = Ak
    P[0:4, 8:10] = al @ R_zb
    P[8:10, 8:10] = A
    P[0:4, 10:12] = al @ R_bz
    P[10:12, 10:12] = A
    P[0:4, 12:14] = be
    P[12:14, 12:14] = A
    P[0:4, 14:18] = np.eye(4)
    eP = expm(h*P)

    I1 = eP[0:4, 4:8]
    I2 = eP[0:4, 8:10]
    I3 = eP[0:4, 10:12]
    I4 = eP[0:4, 12:14]
    I5 = eP[0:4, 14:18]
    bb = np.outer(b, b).reshape(4)
    G5 = al @ bb + be @ b + ga
    I5_G5 = I5 @ G5
    return expm(h*A), I1, I2, I3, I4, I5_G5


def _step_omega(Y_mid, kappa, I1, I2, I3, I4, I5_G5):
    """Return Omega for one midpoint state and one splitting branch."""
    z = np.array([Y_mid[0] - kappa, Y_mid[1]])
    z_out = np.outer(z, z).reshape(4)
    flat = I1 @ z_out + (I2 + I3 + I4) @ z + I5_G5
    return flat.reshape((2, 2))


def _step_omega_batch(Y_mid, kappa, I1, I2, I3, I4, I5_G5):
    """Vectorized ``_step_omega`` for all states in one splitting branch."""
    z = np.asarray(Y_mid, dtype=float) - np.array([kappa, 0.0])
    z_out = np.einsum("ni,nj->nij", z, z).reshape(-1, 4)
    flat = (
        z_out @ I1.T
        + z @ (I2 + I3 + I4).T
        + I5_G5
    )
    return flat.reshape((-1, 2, 2))


def _branch_partial_moments(Y_mid, h, params, drift_const, kappa):
    """Return branch-specific linear means and covariance matrices."""
    mean = np.empty_like(Y_mid)
    cov = np.empty((len(Y_mid), 2, 2), dtype=float)
    for branch in np.unique(kappa):
        mask = kappa == branch
        mats = _precompute_matrices(params, drift_const, branch, h)
        eAh, I1, I2, I3, I4, I5_G5 = mats
        b = np.array([branch, 0.0])
        mean[mask] = (Y_mid[mask] - b) @ eAh.T + b
        cov[mask] = _step_omega_batch(
            Y_mid[mask], branch, I1, I2, I3, I4, I5_G5,
        )
    return mean, cov


def _f_step(Y, h_step, params, drift_const, kappa):
    """Apply the nonlinear flow ``V <- V + h_step*N(X)``."""
    X = Y[:, 0]
    V_new = Y[:, 1] + h_step*(force(X, params) - drift_const*(X - kappa))
    return np.column_stack([X, V_new])


def partial_transition_nlls(free_params, data, model_name, h=config.H_OBS,
                            moment_mask=None, eval_mask=None):
    """
    Return one corrected partial-observation NLL contribution per transition.

    ``moment_mask`` controls which transitions estimate the splitting moments;
    ``eval_mask`` controls which transitions are scored.  Exact IOS uses a
    training mask for moment construction and a one-element held-out mask for
    evaluation.
    """
    data = np.asarray(data, dtype=float)
    if not constraints_valid(free_params, model_name, data):
        raise FloatingPointError("invalid parameter constraints")
    params = embed_params(free_params, model_name)

    Y_old_all, Y_new_all = data[:-1], data[1:]
    n = len(Y_old_all)
    moment_mask = np.ones(n, dtype=bool) if moment_mask is None else np.asarray(moment_mask, dtype=bool)
    eval_mask = np.ones(n, dtype=bool) if eval_mask is None else np.asarray(eval_mask, dtype=bool)
    if moment_mask.shape != (n,) or eval_mask.shape != (n,):
        raise ValueError("Masks must match the number of transitions")
    if moment_mask.sum() < 5:
        raise FloatingPointError("too few transitions in moment mask")
    if eval_mask.sum() == 0:
        return np.array([], dtype=float)

    a, b, c = params[1], params[2], params[3]
    X_moment = Y_old_all[moment_mask, 0]
    m = float(np.mean(X_moment))
    m2 = float(np.mean(X_moment**2))
    var_x = float(np.var(X_moment, ddof=1))
    drift_const = 3.0*a*m2 + 2.0*b*m + c
    root_arg = (3.0*a*m + b)**2 + 9.0*a**2*var_x
    if not np.isfinite(root_arg) or root_arg < 0.0 or abs(3.0*a) < 1e-12:
        raise FloatingPointError("invalid splitting branch")

    root = np.sqrt(root_arg)
    kappa_pos = (-b - root)/(3.0*a)
    kappa_neg = (-b + root)/(3.0*a)
    Y_old, Y_new = Y_old_all[eval_mask], Y_new_all[eval_mask]
    kappa = np.where(Y_old[:, 0] > 0.0, kappa_pos, kappa_neg)

    f_new = _f_step(Y_new, -h/2.0, params, drift_const, kappa)
    Y_mid = _f_step(Y_old, h/2.0, params, drift_const, kappa)
    mu, cov = _branch_partial_moments(Y_mid, h, params, drift_const, kappa)
    residual = (f_new - mu)[:, 1]
    omega = cov[:, 1, 1]

    Y_mid_15 = _f_step(Y_old, 0.75*h, params, drift_const, kappa)
    _, cov_15 = _branch_partial_moments(
        Y_mid_15, 1.5*h, params, drift_const, kappa,
    )
    omega1 = cov_15[:, 1, 1]
    if np.any(omega <= 0.0) or np.any(omega1 <= 0.0):
        raise FloatingPointError("invalid partial covariance")

    nlls = residual**2/omega + config.CORRECTION_FACTOR*np.log(omega1)
    if not np.all(np.isfinite(nlls)):
        raise FloatingPointError("non-finite transition NLL")
    return nlls


def partial_neg_log_lik(free_params, data, model_name, h=config.H_OBS):
    """Return ``sum_k ell_k`` or a large penalty for invalid parameters."""
    try:
        values = partial_transition_nlls(free_params, data, model_name, h)
        total = float(np.sum(values))
        return total if np.isfinite(total) else config.PENALTY
    except EXPECTED_NUMERICAL_ERRORS:
        return config.PENALTY


def masked_partial_neg_log_lik(free_params, data, model_name, mask, h=config.H_OBS):
    """Return ``sum_{k in mask} ell_k`` for leave-one-out estimation."""
    try:
        values = partial_transition_nlls(
            free_params, data, model_name, h, moment_mask=mask, eval_mask=mask,
        )
        total = float(np.sum(values))
        return total if np.isfinite(total) else config.PENALTY
    except EXPECTED_NUMERICAL_ERRORS:
        return config.PENALTY


def _data_moments(X):
    """Return empirical E[X], E[X^2], and Var(X) for the complete likelihood."""
    return {
        "mean_x": float(np.mean(X)),
        "mean_x2": float(np.mean(X**2)),
        "var_x": float(np.var(X, ddof=0)),
    }


def _splitting_terms(params, moments, branch):
    """Construct ``A``, the shift ``b_bar``, slope ``L``, and branch root kappa."""
    a, b, c = params[1], params[2], params[3]
    mx, mx2, vx = moments["mean_x"], moments["mean_x2"], moments["var_x"]
    L = 3.0*a*mx2 + 2.0*b*mx + c
    inside = vx + (mx + b/(3.0*a))**2
    if inside < 0.0 or abs(a) < 1e-12:
        raise FloatingPointError("invalid splitting terms")
    sign = 1.0 if branch == "plus" else -1.0
    bx = -b/(3.0*a) + sign*np.sqrt(inside)
    A = np.array([[0.0, 1.0], [L, -params[0]]])
    return A, np.array([bx, 0.0]), L, bx


def _moment_matrix(params, A, b_bar):
    """Return the 7x7 ODE matrix for first and second Pearson moments."""
    al, be, ga = _check_alpha(params), _check_beta(params), _check_gamma(params)
    d_off = -A @ b_bar
    K = np.zeros((7, 7))
    for j in range(7):
        z = np.zeros(7)
        z[j] = 1.0
        m = z[:2]
        md = A @ m + d_off*z[6]
        M = np.array([[z[2], z[3]], [z[4], z[5]]])
        sigma_sigma_T = (al @ z[2:6] + be @ z[:2] + ga*z[6]).reshape(2, 2)
        Md = A @ M + M @ A.T + np.outer(d_off, m) + np.outer(m, d_off) + sigma_sigma_T
        K[:, j] = [md[0], md[1], Md[0, 0], Md[0, 1], Md[1, 0], Md[1, 1], 0.0]
    return K


def _linear_moments(Y, h, params, A, b_bar):
    """Propagate exact linear conditional means and covariances over step h."""
    T = expm(_moment_matrix(params, A, b_bar)*h)
    X, V = Y[:, 0], Y[:, 1]
    z0 = np.column_stack([X, V, X*X, X*V, V*X, V*V, np.ones(len(X))])
    z1 = z0 @ T.T
    mean = z1[:, :2]
    M = np.zeros((len(X), 2, 2))
    M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1] = z1[:, 2], z1[:, 3], z1[:, 4], z1[:, 5]
    cov = M - mean[:, :, None]*mean[:, None, :]
    return mean, 0.5*(cov + np.swapaxes(cov, 1, 2))


def _branch_linear_moments(Y, h, params, A, kappa):
    """Propagate moments for rows assigned to either splitting branch."""
    Y = np.asarray(Y, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    if Y.shape[0] != kappa.shape[0]:
        raise ValueError("Y and kappa must have the same number of rows")

    mean = np.empty_like(Y)
    cov = np.empty((len(Y), 2, 2), dtype=float)
    for branch in np.unique(kappa):
        mask = kappa == branch
        mean[mask], cov[mask] = _linear_moments(
            Y[mask], h, params, A, np.array([branch, 0.0]),
        )
    return mean, cov


def complete_neg_log_lik(free_params, data, model_name, h, branch="plus"):
    """
    Return the complete-data two-dimensional Gaussian Strang objective.

    The additive constant and factor 1/2 are omitted because they do not
    affect parameter estimation.
    """
    if not constraints_valid(free_params, model_name, data):
        return config.PENALTY
    try:
        params = embed_params(free_params, model_name)
        data = np.asarray(data, dtype=float)
        old, new = data[:-1], data[1:]
        A, b_bar, L, bx = _splitting_terms(params, _data_moments(data[:, 0]), branch)

        old_mid = old.copy()
        new_mid = new.copy()
        old_mid[:, 1] += h/2.0*(force(old[:, 0], params) - L*(old[:, 0] - bx))
        new_mid[:, 1] -= h/2.0*(force(new[:, 0], params) - L*(new[:, 0] - bx))
        mean, cov = _linear_moments(old_mid, h, params, A, b_bar)
        residual = new_mid - mean

        c11, c12, c22 = cov[:, 0, 0] + config.EPS, cov[:, 0, 1], cov[:, 1, 1] + config.EPS
        det = c11*c22 - c12**2
        if np.any(det <= 0.0):
            return config.PENALTY
        r1, r2 = residual[:, 0], residual[:, 1]
        quad = (c22*r1**2 - 2.0*c12*r1*r2 + c11*r2**2)/det
        total = float(np.sum(quad + np.log(det)))
        return total if np.isfinite(total) else config.PENALTY
    except EXPECTED_NUMERICAL_ERRORS:
        return config.PENALTY
