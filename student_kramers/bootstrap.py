"""
bootstrap.py — Exact IOS and resumable parametric bootstrap workflows

For transition ``k``, exact leave-one-out information-omission sensitivity is

    IOS_k = ell_k(theta_hat_{-k}) - ell_k(theta_hat),
    T_N   = sum_k IOS_k,

where ``theta_hat_{-k}`` is estimated after removing transition ``k``.

The parametric bootstrap repeats:

    simulate from fitted model -> refit model -> optionally recompute T_N.

Every long run writes a CSV checkpoint after each replication and can resume.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config
from .data_loading import (
    checkpoint_context,
    load_table,
    prepare_checkpoint,
    save_table,
)
from .estimation import estimate_model, minimize_m4_cholesky
from .likelihoods import masked_partial_neg_log_lik, partial_transition_nlls
from .models import (
    PARAM_NAMES,
    constraints_valid,
    diffusion_minimum,
    diffusion_quadratic_matrix,
    diffusion_rectangle_bounds,
    diffusion_variance,
    embed_params,
    extract_free_params,
    free_bounds,
    free_names,
    parameter_row,
)
from .simulation import simulate_partial_data


def _start_candidate_rows(free_hat, start_candidates):
    """Return labelled, unique leave-one-out starting values."""
    candidates = [("full fit warm start", np.asarray(free_hat, dtype=float))]
    for label, candidate in start_candidates or []:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != candidates[0][1].shape:
            raise ValueError(f"IOS start {label!r} has the wrong parameter dimension")
        if not any(np.allclose(candidate, saved) for _, saved in candidates):
            candidates.append((str(label), candidate))
    return candidates


def _finite_ios_row(row):
    """Return whether one saved leave-one-out row is usable."""
    return (
        bool(row.get("loo_valid", row.get("optimizer_success", False)))
        and np.isfinite(row.get("heldout_nll_under_loo", np.nan))
        and np.isfinite(row.get("ios_contribution", np.nan))
    )


def _safe_diffusion_rectangle_minimum(params, data, margin=0.0):
    """
    Return the rectangle minimum of ``q`` for diagnostic columns.

    Bootstrap M4 refits can land exactly on a semidefinite diffusion boundary.
    The IOS definition does not require an invertible quadratic matrix, so a
    singular interior stationary equation should not invalidate the whole
    leave-one-out fit.  In that case the minimum is still checked over corners
    and valid boundary stationary points.
    """
    params = np.asarray(params, dtype=float)
    lo, hi = diffusion_rectangle_bounds(data, margin)
    x_lo, v_lo = lo
    x_hi, v_hi = hi
    alpha, beta, _, delta, epsilon, zeta = params[5:11]
    candidates = [
        (x_lo, v_lo), (x_lo, v_hi), (x_hi, v_lo), (x_hi, v_hi),
    ]

    if alpha > 0.0:
        for x in (x_lo, x_hi):
            v_star = -(beta + epsilon*x)/(2.0*alpha)
            if v_lo <= v_star <= v_hi:
                candidates.append((x, v_star))
    if delta > 0.0:
        for v in (v_lo, v_hi):
            x_star = -(zeta + epsilon*v)/(2.0*delta)
            if x_lo <= x_star <= x_hi:
                candidates.append((x_star, v))

    Q = diffusion_quadratic_matrix(params)
    linear = np.array([zeta, beta], dtype=float)
    try:
        x_star, v_star = -0.5*np.linalg.solve(Q, linear)
    except np.linalg.LinAlgError:
        pass
    else:
        if x_lo <= x_star <= x_hi and v_lo <= v_star <= v_hi:
            candidates.append((x_star, v_star))

    values = np.array([
        diffusion_variance(x, v, params) for x, v in candidates
    ], dtype=float)
    index = int(np.argmin(values))
    return np.asarray(candidates[index], dtype=float), float(values[index])


def fit_leave_one_out(model_name, data, free_hat, k, maxiter=config.IOS_MAXITER,
                      start_candidates=None):
    """
    Refit one model after omitting transition ``k`` and score that transition.

    The full-data estimate is always the first start.  Optional starts are
    compared by their leave-one-out training NLL, never by the held-out score.
    This preserves the IOS definition while exposing whether M4 needs a
    boundary or interior alternative start.
    """
    n_trans = data.shape[0] - 1
    train_mask = np.ones(n_trans, dtype=bool)
    train_mask[k] = False
    heldout_mask = ~train_mask
    free_hat = np.asarray(free_hat, dtype=float)
    loss_fn = lambda free: masked_partial_neg_log_lik(
        free, data, model_name, train_mask,
    )
    candidates = _start_candidate_rows(free_hat, start_candidates)
    rows = []
    t_all = time.perf_counter()

    for label, candidate in candidates:
        t0 = time.perf_counter()
        start_nll = float(loss_fn(candidate))
        if model_name == "M4":
            result = minimize_m4_cholesky(
                candidate, loss_fn, maxiter=maxiter,
            )
        else:
            result = minimize(
                loss_fn,
                candidate,
                method="L-BFGS-B",
                bounds=free_bounds(model_name),
                options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-5, "maxls": 40},
            )
        free_minus = np.asarray(result.x, dtype=float)
        train_nll = float(loss_fn(free_minus))
        try:
            heldout = float(partial_transition_nlls(
                free_minus, data, model_name,
                moment_mask=train_mask, eval_mask=heldout_mask,
            )[0])
        except Exception:
            heldout = np.nan

        params_minus = embed_params(free_minus, model_name)
        _, q_min_global = diffusion_minimum(params_minus)
        _, q_min_observed = _safe_diffusion_rectangle_minimum(params_minus, data)
        allowed_increase = max(1e-6, 1e-8*abs(start_nll))
        numerically_usable = (
            np.isfinite(start_nll)
            and np.isfinite(train_nll)
            and train_nll < config.PENALTY
            and np.isfinite(heldout)
            and constraints_valid(free_minus, model_name, data)
            and train_nll <= start_nll + allowed_increase
        )
        rows.append({
            "start_kind": label,
            "free_minus": free_minus,
            "train_nll_start": start_nll,
            "train_nll_final": train_nll,
            "training_improvement": start_nll - train_nll,
            "heldout_nll_under_loo": heldout,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "nit": int(getattr(result, "nit", -1)),
            "nfev": int(getattr(result, "nfev", -1)),
            "numerically_usable": bool(numerically_usable),
            "loo_valid": bool(numerically_usable),
            "q_min_global": q_min_global,
            "q_min_observed": q_min_observed,
            "parameter_shift_l2": float(np.linalg.norm(free_minus - free_hat)),
            "parameter_shift_relative": float(np.linalg.norm(
                (free_minus - free_hat)/np.maximum(np.abs(free_hat), 1.0)
            )),
            "seconds_selected_start": time.perf_counter() - t0,
        })

    # L-BFGS-B may report a line-search or iteration-limit status after it has
    # already found a finite, feasible, lower training loss.  The exit status is
    # kept as a diagnostic column, but IOS usability is based on the objective
    # value and constraints because those are what enter the statistic.
    valid_rows = [row for row in rows if row["loo_valid"]]
    selectable = valid_rows or [
        row for row in rows if np.isfinite(row["train_nll_final"])
    ] or rows
    best = min(selectable, key=lambda row: row["train_nll_final"]).copy()
    warm = rows[0]
    alternatives = rows[1:]
    best_alt = (
        min(alternatives, key=lambda row: row["train_nll_final"])
        if alternatives else None
    )
    best.update({
        "n_starts": len(rows),
        "n_optimizer_successful_starts": sum(row["optimizer_success"] for row in rows),
        "n_valid_starts": len(valid_rows),
        "warm_start_train_nll_final": warm["train_nll_final"],
        "best_alternative_train_nll_final": (
            best_alt["train_nll_final"] if best_alt is not None else np.nan
        ),
        "warm_start_excess_nll": (
            warm["train_nll_final"] - best["train_nll_final"]
        ),
        "seconds": time.perf_counter() - t_all,
    })
    return best


def select_ios_pilot_transitions(fits, data, n_indices=48):
    """
    Select one deterministic, diagnostically diverse real-data IOS pilot.

    The same transitions are used for every model.  Selection balances time
    coverage, locally difficult transitions, M3/M4 disagreement, and extreme
    pseudo-states.  The resulting IOS contributions are diagnostics only and
    must not be summed or interpreted as the formal observed statistic.
    """
    data = np.asarray(data, dtype=float)
    n_trans = len(data) - 1
    n_indices = min(max(int(n_indices), 1), n_trans)
    available = set(fits["model"].astype(str))
    required = {"M2", "M3", "M4"}
    if missing := required - available:
        raise ValueError(f"IOS pilot selection requires M2/M3/M4 fits; missing {missing}")

    nlls = {}
    for model_name in sorted(required):
        row = fits.loc[fits["model"] == model_name].iloc[0]
        params = row[PARAM_NAMES].to_numpy(dtype=float)
        nlls[model_name] = partial_transition_nlls(
            extract_free_params(params, model_name), data, model_name,
        )

    states = data[:-1]
    state_scale = np.maximum(np.std(states, axis=0, ddof=1), 1e-12)
    state_score = np.max(np.abs((states - np.mean(states, axis=0))/state_scale), axis=1)
    nll_matrix = np.column_stack([nlls[name] for name in ("M2", "M3", "M4")])
    nll_scale = np.maximum(np.std(nll_matrix, axis=0, ddof=1), 1e-12)
    difficult_score = np.max(
        (nll_matrix - np.median(nll_matrix, axis=0))/nll_scale,
        axis=1,
    )
    disagreement = np.abs(nlls["M3"] - nlls["M4"])
    switch = states[:, 0]*data[1:, 0] <= 0.0

    n_per_group = max(1, int(np.ceil(n_indices/4)))
    candidate_groups = {
        "time coverage": np.unique(
            np.linspace(0, n_trans - 1, n_per_group, dtype=int)
        ).tolist(),
        "high transition NLL": np.argsort(difficult_score)[::-1].tolist(),
        "large M3/M4 disagreement": np.argsort(disagreement)[::-1].tolist(),
        "extreme state or switch": np.lexsort((
            -state_score, ~switch,
        )).tolist(),
    }
    reasons = {k: [] for k in range(n_trans)}
    for reason, candidates in candidate_groups.items():
        for k in candidates[:n_per_group]:
            reasons[int(k)].append(reason)

    selected = []
    max_length = max(len(candidates) for candidates in candidate_groups.values())
    for rank in range(max_length):
        for candidates in candidate_groups.values():
            if rank < len(candidates):
                k = int(candidates[rank])
                if k not in selected:
                    selected.append(k)
                    if len(selected) == n_indices:
                        break
        if len(selected) == n_indices:
            break

    rows = []
    for k in sorted(selected):
        rows.append({
            "k": k,
            "selection_reason": "; ".join(reasons[k]) or "balanced fill",
            "X_old": states[k, 0],
            "X_new": data[k + 1, 0],
            "Vhat_old": states[k, 1],
            "Vhat_new": data[k + 1, 1],
            "state_extremeness": state_score[k],
            "regime_switch": bool(switch[k]),
            "full_nll_m2": nlls["M2"][k],
            "full_nll_m3": nlls["M3"][k],
            "full_nll_m4": nlls["M4"][k],
            "abs_m3_m4_nll_difference": disagreement[k],
        })
    return pd.DataFrame(rows)


def run_exact_ios(model_name, data, params_hat, output_path=None,
                  maxiter=config.IOS_MAXITER, save_every=config.IOS_SAVE_EVERY,
                  indices=None, resume=True, verbose=True,
                  start_candidates=None):
    """Run or resume all leave-one-out refits and return transition-level IOS."""
    data = np.asarray(data, dtype=float)
    free_hat = extract_free_params(params_hat, model_name)
    n_trans = data.shape[0] - 1
    full_mask = np.ones(n_trans, dtype=bool)
    full_nlls = partial_transition_nlls(
        free_hat, data, model_name, moment_mask=full_mask, eval_mask=full_mask,
    )
    indices = list(range(n_trans)) if indices is None else list(indices)

    output_path = Path(output_path) if output_path else None
    if output_path:
        candidates = _start_candidate_rows(free_hat, start_candidates)
        context = checkpoint_context(
            "exact_ios", model_name, params_hat, data,
            maxiter=maxiter,
            start_candidates=[
                {"label": label, "values": values.tolist()}
                for label, values in candidates
            ],
        )
        # ``indices`` is the current target, not the experiment identity.
        # Omitting it allows a sampled IOS run to be extended to more
        # transitions without rerunning completed leave-one-out fits.
        prepare_checkpoint(output_path, context, resume=resume)
    table = load_table(output_path, deduplicate="k") if output_path and resume else pd.DataFrame()
    completed = (
        set(table.loc[
            table.apply(_finite_ios_row, axis=1), "k"
        ].astype(int))
        if len(table) else set()
    )
    rows = table.to_dict("records") if len(table) else []

    for count, k in enumerate(indices, start=1):
        if k in completed:
            continue
        previous = [row for row in rows if int(row["k"]) == int(k)]
        fit = fit_leave_one_out(
            model_name, data, free_hat, k, maxiter, start_candidates,
        )
        row = {
            "k": int(k),
            "attempt": 1 + max(
                [int(row.get("attempt", 1)) for row in previous],
                default=0,
            ),
            "full_nll_k": float(full_nlls[k]),
            "heldout_nll_under_loo": fit["heldout_nll_under_loo"],
            "ios_contribution": fit["heldout_nll_under_loo"] - float(full_nlls[k]),
        }
        row.update({
            key: value for key, value in fit.items() if key != "free_minus"
        })
        row.update(dict(zip(free_names(model_name), fit["free_minus"])))
        rows.append(row)

        if verbose:
            print(f"IOS {model_name}: {count}/{len(indices)} | k={k} | "
                  f"valid={row['loo_valid']} | contribution={row['ios_contribution']:.6f}")
        if output_path and count % save_every == 0:
            checkpoint = pd.DataFrame(rows).drop_duplicates(subset=["k"], keep="last")
            save_table(checkpoint.sort_values("k"), output_path)

    table = pd.DataFrame(rows).drop_duplicates(subset=["k"], keep="last").sort_values("k")
    table = table.reset_index(drop=True)
    if output_path:
        save_table(table, output_path)
    return table


def summarize_ios(table, expected_n_transitions=None):
    """Return ``T_N = sum(IOS_k)`` and leave-one-out optimizer diagnostics."""
    valid_mask = table.apply(_finite_ios_row, axis=1)
    valid = table.loc[valid_mask].copy()
    complete = (
        expected_n_transitions is not None
        and len(table) == int(expected_n_transitions)
        and len(valid) == int(expected_n_transitions)
    )
    return {
        "T_N": float(valid["ios_contribution"].sum()) if complete else np.nan,
        "sampled_ios_sum": float(valid["ios_contribution"].sum()),
        "formal_T_N_complete": bool(complete),
        "expected_n_transitions": (
            int(expected_n_transitions) if expected_n_transitions is not None else np.nan
        ),
        "n_transitions": int(len(table)),
        "n_valid": int(len(valid)),
        "n_failed": int((~valid_mask).sum()),
        "success_rate": float(valid_mask.mean()),
        "median_ios_contribution": float(valid["ios_contribution"].median()),
        "max_ios_contribution": float(valid["ios_contribution"].max()),
        "min_ios_contribution": float(valid["ios_contribution"].min()),
        "total_seconds": float(table["seconds"].sum()),
    }


def influential_transitions(table, data, top_n=20):
    """Attach observed pseudo-states to the largest IOS contributions."""
    valid = table.loc[table.apply(_finite_ios_row, axis=1)].copy()
    out = valid.nlargest(top_n, "ios_contribution").copy()
    k = out["k"].to_numpy(dtype=int)
    out["X_old"], out["X_new"] = data[k, 0], data[k + 1, 0]
    out["Vhat_old"], out["Vhat_new"] = data[k, 1], data[k + 1, 1]
    out["share_of_total_IOS"] = out["ios_contribution"]/valid["ios_contribution"].sum()
    return out


def run_parametric_bootstrap(model_name, params_hat, data, output_path,
                             n_boot=config.N_BOOTSTRAP, seed=config.BOOTSTRAP_SEED,
                             compute_ios=False, details_dir=None, resume=True,
                             verbose=True):
    """
    Run or resume parameter and optional IOS bootstrap for one model.

    Each successful row contains the refitted full parameter vector, NLL, and,
    when ``compute_ios=True``, the exact bootstrap IOS statistic ``T_N``.
    """
    output_path = Path(output_path)
    context = checkpoint_context(
        "parametric_bootstrap", model_name, params_hat, data,
        seed=seed, compute_ios=compute_ios,
    )
    # ``n_boot`` is intentionally omitted so a pilot can be extended.
    prepare_checkpoint(output_path, context, resume=resume)
    table = load_table(output_path, deduplicate="rep") if resume else pd.DataFrame()
    completed = (
        set(table.loc[table["success"].astype(bool), "rep"].astype(int))
        if len(table) and "success" in table.columns else set()
    )
    rows = table.to_dict("records") if len(table) else []

    init_state = np.asarray(data[0], dtype=float)
    n_obs = data.shape[0] + 1
    free_start = extract_free_params(params_hat, model_name)

    for rep in range(n_boot):
        if rep in completed:
            continue
        t0 = time.perf_counter()
        row = {"rep": rep, "success": False, "seed": seed + rep, "error": ""}
        try:
            boot_data = simulate_partial_data(
                params_hat, n_obs, init_state=init_state, seed=seed + rep,
            )
            params_star, nll, conv = estimate_model(
                model_name, boot_data, start=free_start,
                maxiter=config.BOOTSTRAP_MAXITER, verbose=False,
            )
            row.update(parameter_row(model_name, params_star, nll))
            row["success"] = conv == 0
            row["convergence"] = conv

            if compute_ios and row["success"]:
                details_dir = Path(details_dir or output_path.parent / f"{model_name.lower()}_ios_details")
                detail_path = details_dir / f"rep_{rep:04d}.csv"
                ios_table = run_exact_ios(
                    model_name, boot_data, params_star, detail_path,
                    maxiter=config.IOS_MAXITER, save_every=1, verbose=False,
                )
                ios_summary = summarize_ios(
                    ios_table, expected_n_transitions=len(boot_data) - 1,
                )
                row.update({f"ios_{key}": value for key, value in ios_summary.items()})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"

        row["time_sec"] = time.perf_counter() - t0
        rows.append(row)
        save_table(pd.DataFrame(rows), output_path)
        if verbose:
            print(f"Bootstrap {model_name}: {rep + 1}/{n_boot} | success={row['success']}")
    return pd.DataFrame(rows)


def run_contrast_bootstrap(null_model, alt_model, null_params, data, output_path,
                           n_boot=config.N_BOOTSTRAP, seed=config.BOOTSTRAP_SEED,
                           resume=True, verbose=True):
    """
    Bootstrap the nested-model likelihood contrast.

        contrast = 2 * [NLL(null) - NLL(alternative)].
    """
    output_path = Path(output_path)
    context = checkpoint_context(
        "contrast_bootstrap", f"{null_model}_vs_{alt_model}", null_params, data,
        seed=seed,
    )
    # ``n_boot`` is intentionally omitted so a pilot can be extended.
    prepare_checkpoint(output_path, context, resume=resume)
    table = load_table(output_path, deduplicate="rep") if resume else pd.DataFrame()
    completed = (
        set(table.loc[table["success"].astype(bool), "rep"].astype(int))
        if len(table) and "success" in table.columns else set()
    )
    rows = table.to_dict("records") if len(table) else []
    init_state = np.asarray(data[0], dtype=float)
    n_obs = data.shape[0] + 1

    for rep in range(n_boot):
        if rep in completed:
            continue
        row = {"rep": rep, "success": False, "seed": seed + rep, "error": ""}
        try:
            boot_data = simulate_partial_data(
                null_params, n_obs, init_state=init_state, seed=seed + rep,
            )
            _, nll_null, conv_null = estimate_model(null_model, boot_data, verbose=False)
            _, nll_alt, conv_alt = estimate_model(alt_model, boot_data, verbose=False)
            row.update({
                "nll_null": nll_null,
                "nll_alt": nll_alt,
                "contrast": 2.0*(nll_null - nll_alt),
                "success": conv_null == 0 and conv_alt == 0,
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        save_table(pd.DataFrame(rows), output_path)
        if verbose:
            print(f"Contrast bootstrap: {rep + 1}/{n_boot} | success={row['success']}")
    return pd.DataFrame(rows)


def summarize_bootstrap(table, observed=None, value_col="ios_T_N"):
    """Return mean, SD, quantiles, and optional finite-sample tail probabilities."""
    values = table.loc[table["success"].astype(bool), value_col].dropna().to_numpy(dtype=float)
    out = {
        "n_valid": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "q025": float(np.quantile(values, 0.025)),
        "q50": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
    }
    if observed is not None:
        out["p_upper"] = float((np.sum(values >= observed) + 1)/(len(values) + 1))
        out["p_lower"] = float((np.sum(values <= observed) + 1)/(len(values) + 1))
    return out
