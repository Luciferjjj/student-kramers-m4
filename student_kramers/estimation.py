"""
estimation.py — L-BFGS-B wrappers for all registered models

Every model is estimated through the same interface:

    estimate_model(model_name, partial_data)
    estimate_complete_model(model_name, complete_data, h)

The model registry supplies free parameters, fixed parameters, initial values,
and bounds.  The likelihood module supplies the objective.
"""
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config
from .likelihoods import complete_neg_log_lik, partial_neg_log_lik
from .models import (
    M4_CHOLESKY_BOUNDS,
    MODELS,
    constraints_valid,
    diffusion_minimum,
    diffusion_rectangle_minimum,
    embed_params,
    extract_free_params,
    free_bounds,
    get_model,
    m4_from_cholesky_params,
    m4_to_cholesky_params,
    parameter_row,
)


def run_estimator_lbfgs(params_init, loss_fn, bounds, maxiter=None, tol=None, verbose=True):
    """Minimise one objective and return params, final loss, and convergence flag."""
    maxiter = maxiter or config.LBFGS_MAXITER
    tol = tol or config.LBFGS_TOL
    result = minimize(
        loss_fn,
        np.asarray(params_init, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-5, "maxls": 40},
    )
    params = np.asarray(result.x, dtype=float)
    loss = float(loss_fn(params))
    conv = int(not result.success or not np.isfinite(loss) or loss >= config.PENALTY)
    if verbose:
        print(f"NLL={loss:.6f} | iterations={getattr(result, 'nit', -1)} | {result.message}")
    return params, loss, conv


def minimize_m4_cholesky(params_init, loss_fn, maxiter=None, tol=None):
    """
    Minimize an M4 objective in globally valid Cholesky coordinates.

    Direct L-BFGS-B cannot represent the positive-semidefinite M4 diffusion
    constraint.  Optimizing a Cholesky factor keeps every trial value globally
    valid and avoids unreliable hard-penalty or near-boundary SLSQP steps.
    """
    params_init = np.asarray(params_init, dtype=float)
    maxiter = maxiter or config.LBFGS_MAXITER
    tol = tol or config.M4_CHOLESKY_FTOL
    cholesky_start = m4_to_cholesky_params(
        params_init, jitter=config.M4_CHOLESKY_JITTER,
    )
    scale = np.maximum(np.abs(cholesky_start), 1.0)
    scaled_start = cholesky_start/scale
    scaled_bounds = [
        (lower/value, upper/value)
        for (lower, upper), value in zip(M4_CHOLESKY_BOUNDS, scale)
    ]

    result = minimize(
        lambda scaled: loss_fn(m4_from_cholesky_params(
            np.asarray(scaled, dtype=float)*scale,
        )),
        scaled_start,
        method="L-BFGS-B",
        jac="2-point",
        bounds=scaled_bounds,
        options={
            "maxiter": maxiter,
            "ftol": tol,
            "gtol": config.M4_CHOLESKY_GTOL,
            "maxls": 50,
            "finite_diff_rel_step": config.M4_CHOLESKY_REL_STEP,
        },
    )
    result.cholesky_x = np.asarray(result.x, dtype=float)*scale
    result.x = m4_from_cholesky_params(result.cholesky_x)
    result.fun = float(loss_fn(result.x))
    return result


def run_estimator_m4(params_init, loss_fn, maxiter=None, tol=None, verbose=True):
    """Minimise one M4 objective while preserving global diffusion positivity."""
    result = minimize_m4_cholesky(params_init, loss_fn, maxiter=maxiter, tol=tol)
    params = np.asarray(result.x, dtype=float)
    loss = float(result.fun)
    conv = int(
        not result.success
        or not np.isfinite(loss)
        or loss >= config.PENALTY
        or not constraints_valid(params, "M4")
    )
    if verbose:
        print(f"NLL={loss:.6f} | iterations={getattr(result, 'nit', -1)} | {result.message}")
    return params, loss, conv


def make_loss_fn(model_name, data, h=config.H_OBS):
    """Bind model name, data, and h to obtain ``loss(free_params)``."""
    return lambda free_params: partial_neg_log_lik(free_params, data, model_name, h)


def make_random_starts(model_name, start, n_starts, seed=42, data=None,
                       extra_starts=None):
    """Return valid targeted and perturbed starts, including ``start`` first."""
    start = np.asarray(start, dtype=float)
    bounds = np.asarray(free_bounds(model_name), dtype=float)
    starts = [start.copy()]
    for candidate in extra_starts or []:
        candidate = np.asarray(candidate, dtype=float)
        if (
            candidate.shape == start.shape
            and constraints_valid(candidate, model_name, data)
            and not any(np.allclose(candidate, saved) for saved in starts)
        ):
            starts.append(candidate.copy())
        if len(starts) >= n_starts:
            return starts
    if n_starts <= 1:
        return starts

    rng = np.random.default_rng(seed)
    if model_name == "M4":
        base = embed_params(start, "M4")
        for delta in (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0):
            candidate = base.copy()
            candidate[8:11] = [delta, 0.0, 0.0]
            free = extract_free_params(candidate, "M4")
            if constraints_valid(free, "M4", data):
                starts.append(free)
            if len(starts) >= n_starts:
                return starts

        attempts = 0
        while len(starts) < n_starts and attempts < 100*n_starts:
            candidate = base.copy()
            candidate[8] = rng.uniform(5.0, 500.0)
            candidate[9:11] = rng.normal(0.0, 20.0, size=2)
            free = extract_free_params(candidate, "M4")
            if constraints_valid(free, "M4", data):
                starts.append(free)
            attempts += 1
        return starts

    span = bounds[:, 1] - bounds[:, 0]
    scale = np.maximum(np.abs(start)*0.20, span*0.02)
    attempts = 0
    while len(starts) < n_starts and attempts < 100*n_starts:
        candidate = np.clip(
            start + rng.normal(size=len(start))*scale,
            bounds[:, 0],
            bounds[:, 1],
        )
        if constraints_valid(candidate, model_name, data):
            starts.append(candidate)
        attempts += 1
    return starts


def estimate_model(model_name, data, h=config.H_OBS, start=None,
                   maxiter=None, tol=None, verbose=True, n_starts=1, seed=42,
                   extra_starts=None):
    """Estimate one model and keep the best result across valid starting values."""
    cfg = get_model(model_name)
    start = cfg["init"] if start is None else np.asarray(start, dtype=float)
    loss_fn = make_loss_fn(model_name, data, h)
    starts = make_random_starts(
        model_name, start, n_starts, seed, data, extra_starts,
    )
    results = []
    for candidate in starts:
        if model_name == "M4":
            result = run_estimator_m4(
                candidate, loss_fn, maxiter=maxiter, tol=tol, verbose=False,
            )
        else:
            result = run_estimator_lbfgs(
                candidate, loss_fn, free_bounds(model_name),
                maxiter=maxiter, tol=tol, verbose=False,
            )
        results.append(result)
    free_hat, nll, conv = min(results, key=lambda item: item[1])
    if verbose:
        print(f"{model_name}: best NLL={nll:.6f} from {len(results)} start(s)")
    return embed_params(free_hat, model_name), nll, conv


def estimate_complete_model(model_name, data, h, branch="plus", start=None,
                            maxiter=None, tol=None, verbose=True, n_starts=1, seed=42):
    """Estimate complete data and keep the best result across starting values."""
    cfg = get_model(model_name)
    start = cfg["init"] if start is None else np.asarray(start, dtype=float)
    loss_fn = lambda free: complete_neg_log_lik(free, data, model_name, h, branch)
    results = []
    for candidate in make_random_starts(model_name, start, n_starts, seed, data):
        if model_name == "M4":
            result = run_estimator_m4(
                candidate, loss_fn, maxiter=maxiter, tol=tol, verbose=False,
            )
        else:
            result = run_estimator_lbfgs(
                candidate, loss_fn, free_bounds(model_name),
                maxiter=maxiter, tol=tol, verbose=False,
            )
        results.append(result)
    free_hat, nll, conv = min(results, key=lambda item: item[1])
    if verbose:
        print(f"{model_name} complete: best NLL={nll:.6f} from {len(results)} start(s)")
    return embed_params(free_hat, model_name), nll, conv


def estimate_models(model_names, data, h=config.H_OBS, verbose=True,
                    n_starts=1, seed=42, warm_starts=None):
    """
    Estimate several models and return one summary table.

    When M3 has already been fitted, its exact boundary estimate supplies the
    first M4 starting value.  Since M4 contains M3, this makes the M3 objective
    value available to the M4 optimization while all eleven M4 parameters
    remain free.
    """
    rows = []
    fitted = {}
    warm_starts = warm_starts or {}
    for index, model_name in enumerate(model_names):
        t0 = time.perf_counter()
        start = None
        if model_name == "M4" and "M3" in fitted:
            start = extract_free_params(fitted["M3"], "M4")
        params, nll, conv = estimate_model(
            model_name, data, h, start=start, verbose=verbose,
            n_starts=n_starts, seed=seed + index,
            extra_starts=[
                extract_free_params(warm_starts[model_name], model_name)
            ] if model_name in warm_starts else None,
        )
        fitted[model_name] = params
        row = parameter_row(model_name, params, nll)
        n_free = len(get_model(model_name)["free_indices"])
        n_transition = len(data) - 1
        _, q_min_observed = diffusion_rectangle_minimum(params, data)
        _, q_min_protected = diffusion_rectangle_minimum(
            params, data, margin=config.M4_DIAGNOSTIC_MARGIN,
        )
        _, q_min_global = diffusion_minimum(params)
        row.update({
            "n_free": n_free,
            "aic": 2.0*nll + 2.0*n_free,
            "bic": 2.0*nll + np.log(n_transition)*n_free,
            "q_min_observed": q_min_observed,
            "q_min_protected": q_min_protected,
            "q_min_global": q_min_global,
            "convergence": conv,
            "time_sec": time.perf_counter() - t0,
        })
        rows.append(row)
    return pd.DataFrame(rows)
