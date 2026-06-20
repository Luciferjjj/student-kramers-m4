"""
pre_ios.py - Diagnostics required before the IOS goodness-of-fit study

This module keeps the pre-IOS research checks separate from the likelihood
implementation.  It audits the fitted diffusion surface, records every M4
optimization start, performs predictive checks, and bootstraps the nested
M3-versus-M4 likelihood contrast.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loading import (
    build_partial_data,
    checkpoint_context,
    load_table,
    prepare_checkpoint,
    save_table,
)
from .estimation import (
    estimate_model,
    make_loss_fn,
    make_random_starts,
    run_estimator_m4,
)
from .likelihoods import partial_transition_nlls
from .models import (
    PARAM_NAMES,
    diffusion_augmented_matrix,
    diffusion_minimum,
    diffusion_quadratic_matrix,
    diffusion_rectangle_bounds,
    diffusion_rectangle_minimum,
    diffusion_variance,
    embed_params,
    extract_free_params,
    parameter_row,
)
from .simulation import (
    classify_regime,
    extract_waiting_times,
    path_summary_matrix,
    potential_extrema,
    simulate_partial_data,
    simulate_trajectory,
)


def audit_diffusion_minima(fits, data, margin=config.M4_DIAGNOSTIC_MARGIN):
    """Describe where each fitted model reaches its minimum diffusion."""
    data = np.asarray(data, dtype=float)
    center = np.mean(data, axis=0)
    scale = np.std(data, axis=0, ddof=1)
    rows = []

    for _, fit in fits.iterrows():
        params = fit[PARAM_NAMES].to_numpy(dtype=float)
        point, q_min = diffusion_minimum(params)
        observed_point, q_min_observed = diffusion_rectangle_minimum(params, data)
        protected_point, q_min_protected = diffusion_rectangle_minimum(
            params, data, margin=margin,
        )
        q_path = diffusion_variance(data[:, 0], data[:, 1], params)
        standardized = (data - point)/scale
        nearest_index = int(np.argmin(np.sum(standardized**2, axis=1)))
        point_z = (point - center)/scale
        Q_eigenvalues = np.linalg.eigvalsh(diffusion_quadratic_matrix(params))
        H_eigenvalues = np.linalg.eigvalsh(diffusion_augmented_matrix(params))

        rows.append({
            "model": fit["model"],
            "q_min_global": q_min,
            "q_min_x": point[0],
            "q_min_v": point[1],
            "q_min_x_z": point_z[0],
            "q_min_v_z": point_z[1],
            "q_min_standardized_distance": float(np.linalg.norm(point_z)),
            "q_min_observed_rectangle": q_min_observed,
            "q_min_observed_x": observed_point[0],
            "q_min_observed_v": observed_point[1],
            "q_min_protected_rectangle": q_min_protected,
            "q_min_protected_x": protected_point[0],
            "q_min_protected_v": protected_point[1],
            "q_path_min": float(np.min(q_path)),
            "q_path_q01": float(np.quantile(q_path, 0.01)),
            "q_path_median": float(np.median(q_path)),
            "q_path_q99": float(np.quantile(q_path, 0.99)),
            "nearest_observation_index": nearest_index,
            "nearest_observation_x": data[nearest_index, 0],
            "nearest_observation_v": data[nearest_index, 1],
            "nearest_standardized_distance": float(
                np.linalg.norm(standardized[nearest_index])
            ),
            "q_at_nearest_observation": float(q_path[nearest_index]),
            "quadratic_eigenvalue_min": float(Q_eigenvalues[0]),
            "quadratic_eigenvalue_max": float(Q_eigenvalues[-1]),
            "augmented_eigenvalue_min": float(H_eigenvalues[0]),
        })
    return pd.DataFrame(rows)


def transition_improvement_table(fits, data, age=None):
    """
    Return transition-level M3-to-M4 likelihood gains on the real data.

    Positive ``gain_m4_over_m3`` means the transition is scored better by M4.
    The cumulative column shows whether the total improvement is diffuse or
    driven by a small number of observations.
    """
    data = np.asarray(data, dtype=float)
    params = {
        str(row["model"]): row[PARAM_NAMES].to_numpy(dtype=float)
        for _, row in fits.iterrows()
    }
    missing = {"M3", "M4"} - set(params)
    if missing:
        raise ValueError(f"Transition comparison requires M3 and M4; missing {missing}")

    nll_m3 = partial_transition_nlls(
        extract_free_params(params["M3"], "M3"), data, "M3",
    )
    nll_m4 = partial_transition_nlls(
        extract_free_params(params["M4"], "M4"), data, "M4",
    )
    gain = nll_m3 - nll_m4
    states = data[:-1]
    q_m3 = diffusion_variance(states[:, 0], states[:, 1], params["M3"])
    q_m4 = diffusion_variance(states[:, 0], states[:, 1], params["M4"])
    transition_age = (
        np.arange(len(gain), dtype=float)
        if age is None else np.asarray(age, dtype=float)[:len(gain)]
    )

    return pd.DataFrame({
        "transition": np.arange(len(gain)),
        "age_ka": transition_age,
        "x": states[:, 0],
        "vhat": states[:, 1],
        "nll_m3": nll_m3,
        "nll_m4": nll_m4,
        "gain_m4_over_m3": gain,
        "cumulative_gain": np.cumsum(gain),
        "q_m3": q_m3,
        "q_m4": q_m4,
        "log10_q_ratio": np.log10(q_m4/q_m3),
    })


def _start_kind(candidate, boundary, warm):
    """Label one M4 starting point for the optimization audit."""
    if np.allclose(candidate, boundary):
        return "M3 boundary"
    if warm is not None and np.allclose(candidate, warm):
        return "final M4 warm start"
    return "global interior"


def run_optimization_stability(data, m3_params, m4_params, output_path,
                               seeds=(42, 314, 2026), n_starts=12,
                               include_warm=True, maxiter=None, tol=None,
                               resume=True, verbose=True):
    """Record the result from every M4 start across independent start sets."""
    data = np.asarray(data, dtype=float)
    output_path = Path(output_path)
    boundary = extract_free_params(m3_params, "M4")
    warm = extract_free_params(m4_params, "M4") if include_warm else None
    context = checkpoint_context(
        "optimization_stability", "M4", m4_params, data,
        seeds=list(seeds), n_starts=n_starts, include_warm=include_warm,
        maxiter=maxiter, tol=tol,
    )
    prepare_checkpoint(output_path, context, resume=resume)
    table = load_table(output_path) if resume else pd.DataFrame()
    completed = (
        set(zip(table["seed"].astype(int), table["start_index"].astype(int)))
        if len(table) else set()
    )
    rows = table.to_dict("records") if len(table) else []
    loss_fn = make_loss_fn("M4", data)

    for seed in seeds:
        starts = make_random_starts(
            "M4", boundary, n_starts, seed=seed, data=data,
            extra_starts=[warm] if warm is not None else None,
        )
        for start_index, candidate in enumerate(starts):
            key = (int(seed), int(start_index))
            if key in completed:
                continue
            t0 = time.perf_counter()
            free_hat, nll, conv = run_estimator_m4(
                candidate, loss_fn,
                maxiter=maxiter, tol=tol, verbose=False,
            )
            params = embed_params(free_hat, "M4")
            _, q_min_global = diffusion_minimum(params)
            _, q_min_observed = diffusion_rectangle_minimum(params, data)
            row = {
                "seed": int(seed),
                "start_index": int(start_index),
                "start_kind": _start_kind(candidate, boundary, warm),
                "start_delta": candidate[8],
                "start_epsilon": candidate[9],
                "start_zeta": candidate[10],
                "nll": nll,
                "convergence": conv,
                "q_min_global": q_min_global,
                "q_min_observed": q_min_observed,
                "distance_to_final_m4": float(np.linalg.norm(params - m4_params)),
                "time_sec": time.perf_counter() - t0,
            }
            row.update({
                name: value for name, value in zip(PARAM_NAMES, params)
            })
            rows.append(row)
            save_table(pd.DataFrame(rows), output_path)
            if verbose:
                print(
                    f"M4 stability seed={seed} start={start_index + 1}/{len(starts)} "
                    f"| {row['start_kind']} | NLL={nll:.6f}"
                )
    return pd.DataFrame(rows).sort_values(["seed", "start_index"]).reset_index(drop=True)


def run_tolerance_stability(data, m4_params, output_path,
                            settings=((400, 1e-6), (800, 1e-8), (1600, 1e-10)),
                            resume=True, verbose=True):
    """Polish the same M4 fit under several stopping settings."""
    data = np.asarray(data, dtype=float)
    output_path = Path(output_path)
    start = extract_free_params(m4_params, "M4")
    context = checkpoint_context(
        "tolerance_stability", "M4", m4_params, data,
        settings=[list(setting) for setting in settings],
    )
    prepare_checkpoint(output_path, context, resume=resume)
    table = load_table(output_path) if resume else pd.DataFrame()
    completed = (
        set(zip(table["maxiter"].astype(int), table["tol"].astype(float)))
        if len(table) else set()
    )
    rows = table.to_dict("records") if len(table) else []
    loss_fn = make_loss_fn("M4", data)

    for maxiter, tol in settings:
        if (int(maxiter), float(tol)) in completed:
            continue
        t0 = time.perf_counter()
        free_hat, nll, conv = run_estimator_m4(
            start, loss_fn, maxiter=maxiter, tol=tol,
            verbose=False,
        )
        params = embed_params(free_hat, "M4")
        _, q_min_global = diffusion_minimum(params)
        row = {
            "maxiter": int(maxiter),
            "tol": float(tol),
            "nll": nll,
            "convergence": conv,
            "q_min_global": q_min_global,
            "distance_to_final_m4": float(np.linalg.norm(params - m4_params)),
            "time_sec": time.perf_counter() - t0,
        }
        row.update({name: value for name, value in zip(PARAM_NAMES, params)})
        rows.append(row)
        save_table(pd.DataFrame(rows), output_path)
        if verbose:
            print(f"M4 tolerance maxiter={maxiter} tol={tol:g} | NLL={nll:.6f}")
    return pd.DataFrame(rows).sort_values(["maxiter", "tol"]).reset_index(drop=True)


def _path_behavior(x, barrier):
    """Return occupancy, switching, and waiting-time summaries for one path."""
    states = classify_regime(x, barrier)
    waits = extract_waiting_times(x, barrier=barrier)
    switches = int(np.sum(states[1:] != states[:-1])) if len(states) else 0
    return {
        "lower_occupancy": float(np.mean(states < 0)) if len(states) else np.nan,
        "upper_occupancy": float(np.mean(states > 0)) if len(states) else np.nan,
        "n_switches": switches,
        "lower_wait_mean": (
            float(np.mean(waits["lower"])) if len(waits["lower"]) else np.nan
        ),
        "upper_wait_mean": (
            float(np.mean(waits["upper"])) if len(waits["upper"]) else np.nan
        ),
        "lower_wait_median": (
            float(np.median(waits["lower"])) if len(waits["lower"]) else np.nan
        ),
        "upper_wait_median": (
            float(np.median(waits["upper"])) if len(waits["upper"]) else np.nan
        ),
    }


def _density_band(observed, simulated, variable, model, n_bins=60):
    """Return observed density and simulated pointwise density bands."""
    combined = np.concatenate([observed, simulated.ravel()])
    lo, hi = np.quantile(combined, [0.001, 0.999])
    bins = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5*(bins[:-1] + bins[1:])
    observed_density, _ = np.histogram(observed, bins=bins, density=True)
    simulated_density = np.array([
        np.histogram(values, bins=bins, density=True)[0]
        for values in simulated
    ])
    return pd.DataFrame({
        "model": model,
        "variable": variable,
        "value": centers,
        "observed_density": observed_density,
        "simulated_q025": np.quantile(simulated_density, 0.025, axis=0),
        "simulated_median": np.quantile(simulated_density, 0.5, axis=0),
        "simulated_q975": np.quantile(simulated_density, 0.975, axis=0),
    })


def _fixed_density_rows(values, observed, variable, model, source, rep, n_bins=60):
    """Return one fixed-grid density so predictive replications can be appended."""
    observed = np.asarray(observed, dtype=float)
    values = np.asarray(values, dtype=float)
    span = max(float(np.ptp(observed)), 1e-8)
    bins = np.linspace(
        float(np.min(observed) - span),
        float(np.max(observed) + span),
        n_bins + 1,
    )
    counts, _ = np.histogram(values, bins=bins, density=False)
    total = counts.sum()
    density = (
        counts/(total*np.diff(bins))
        if total > 0 else np.zeros_like(counts, dtype=float)
    )
    return pd.DataFrame({
        "model": model,
        "variable": variable,
        "source": source,
        "rep": rep,
        "value": 0.5*(bins[:-1] + bins[1:]),
        "density": density,
    })


def summarize_predictive_density(density_rep):
    """Aggregate appendable per-replication densities into prediction bands."""
    rows = []
    for (model, variable), group in density_rep.groupby(["model", "variable"]):
        observed = (
            group[group["source"] == "observed"][["value", "density"]]
            .drop_duplicates("value").sort_values("value")
        )
        simulated = (
            group[group["source"] == "simulated"]
            .pivot_table(index="rep", columns="value", values="density")
            .sort_index(axis=1)
        )
        if simulated.empty:
            continue
        values = simulated.columns.to_numpy(dtype=float)
        observed_density = observed.set_index("value").reindex(values)["density"].to_numpy()
        rows.append(pd.DataFrame({
            "model": model,
            "variable": variable,
            "value": values,
            "observed_density": observed_density,
            "simulated_q025": simulated.quantile(0.025, axis=0).to_numpy(),
            "simulated_median": simulated.quantile(0.5, axis=0).to_numpy(),
            "simulated_q975": simulated.quantile(0.975, axis=0).to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_predictive_checks_checkpointed(fits, data, output_paths, n_rep=100,
                                       seed=20260612, n_bins=60, resume=True,
                                       verbose=True):
    """
    Run appendable predictive checks and save every completed replication.

    ``n_rep`` is intentionally omitted from provenance, so a 100-replication
    run can later be extended to 500 or 1000 without rerunning the first 100.
    """
    data = np.asarray(data, dtype=float)
    output_paths = {name: Path(path) for name, path in output_paths.items()}
    params_matrix = fits[PARAM_NAMES].to_numpy(dtype=float)
    context = checkpoint_context(
        "predictive_checks", ",".join(fits["model"].astype(str)),
        params_matrix, data, seed=seed, n_bins=n_bins,
    )
    prepare_checkpoint(output_paths["summary"], context, resume=resume)

    summary = load_table(output_paths["summary"]) if resume else pd.DataFrame()
    behavior = load_table(output_paths["behavior"]) if resume else pd.DataFrame()
    waiting = load_table(output_paths["waiting"]) if resume else pd.DataFrame()
    density_rep = load_table(output_paths["density_rep"]) if resume else pd.DataFrame()
    completed = (
        set(zip(
            summary.loc[summary["source"] == "simulated", "model"].astype(str),
            summary.loc[summary["source"] == "simulated", "rep"].astype(int),
        ))
        if len(summary) else set()
    )

    observed_x, observed_v = data[:, 0], data[:, 1]
    reference_model = "M3" if "M3" in set(fits["model"]) else str(fits.iloc[0]["model"])
    reference_params = (
        fits.loc[fits["model"] == reference_model, PARAM_NAMES]
        .iloc[0].to_numpy(dtype=float)
    )
    _, common_barrier, _ = potential_extrema(reference_params)
    common_barrier = common_barrier if np.isfinite(common_barrier) else 0.0

    for model_index, (_, fit) in enumerate(fits.iterrows()):
        model = str(fit["model"])
        params = fit[PARAM_NAMES].to_numpy(dtype=float)
        observed_saved = (
            len(summary)
            and ((summary["model"] == model) & (summary["source"] == "observed")).any()
        )
        if not observed_saved:
            observed_summary = path_summary_matrix(
                observed_x[None, :], observed_v[None, :],
            ).iloc[0].to_dict()
            summary = pd.concat([summary, pd.DataFrame([{
                "model": model, "source": "observed", "rep": -1,
                **observed_summary,
            }])], ignore_index=True)
            behavior = pd.concat([behavior, pd.DataFrame([{
                "model": model, "source": "observed", "rep": -1,
                "barrier": common_barrier,
                **_path_behavior(observed_x, common_barrier),
            }])], ignore_index=True)
            observed_waits = extract_waiting_times(observed_x, barrier=common_barrier)
            wait_rows = []
            for regime, values in observed_waits.items():
                wait_rows.extend({
                    "model": model, "source": "observed", "rep": -1,
                    "event": event, "barrier": common_barrier, "regime": regime,
                    "waiting_time_years": value,
                } for event, value in enumerate(values))
            waiting = pd.concat([waiting, pd.DataFrame(wait_rows)], ignore_index=True)
            density_rep = pd.concat([
                density_rep,
                _fixed_density_rows(
                    observed_x, observed_x, "X", model, "observed", -1, n_bins,
                ),
                _fixed_density_rows(
                    observed_v, observed_v, "Vhat", model, "observed", -1, n_bins,
                ),
            ], ignore_index=True)
            save_table(behavior, output_paths["behavior"])
            save_table(waiting, output_paths["waiting"])
            save_table(density_rep, output_paths["density_rep"])
            save_table(summary, output_paths["summary"])

        for rep in range(n_rep):
            if (model, rep) in completed:
                continue
            traj = simulate_trajectory(
                params, len(data) + 1, init_state=tuple(data[0]),
                seed=seed + 10000*model_index + rep,
            )
            partial = build_partial_data(traj[:, 0])
            simulated_summary = path_summary_matrix(
                partial[:, 0][None, :], partial[:, 1][None, :],
            ).iloc[0].to_dict()
            summary_row = {
                "model": model, "source": "simulated", "rep": rep,
                **simulated_summary,
            }
            behavior_row = {
                "model": model, "source": "simulated", "rep": rep,
                "barrier": common_barrier,
                **_path_behavior(partial[:, 0], common_barrier),
            }
            simulated_waits = extract_waiting_times(
                partial[:, 0], barrier=common_barrier,
            )
            wait_rows = []
            for regime, values in simulated_waits.items():
                wait_rows.extend({
                    "model": model, "source": "simulated", "rep": rep,
                    "event": event, "barrier": common_barrier, "regime": regime,
                    "waiting_time_years": value,
                } for event, value in enumerate(values))

            behavior = pd.concat([behavior, pd.DataFrame([behavior_row])], ignore_index=True)
            waiting = pd.concat([waiting, pd.DataFrame(wait_rows)], ignore_index=True)
            density_rep = pd.concat([
                density_rep,
                _fixed_density_rows(
                    partial[:, 0], observed_x, "X", model, "simulated", rep, n_bins,
                ),
                _fixed_density_rows(
                    partial[:, 1], observed_v, "Vhat", model, "simulated", rep, n_bins,
                ),
            ], ignore_index=True)
            summary = pd.concat([summary, pd.DataFrame([summary_row])], ignore_index=True)

            # Summary is saved last: a completed summary row certifies that all
            # other tables for the same replication have already been saved.
            save_table(behavior, output_paths["behavior"])
            save_table(waiting, output_paths["waiting"])
            save_table(density_rep, output_paths["density_rep"])
            save_table(summary, output_paths["summary"])
            completed.add((model, rep))
            if verbose and (rep + 1) % 25 == 0:
                print(f"Predictive check {model}: {rep + 1}/{n_rep}", flush=True)

    return {
        "predictive_summary": summary,
        "predictive_behavior": behavior,
        "predictive_waiting": waiting,
        "predictive_density": summarize_predictive_density(density_rep),
    }


def run_predictive_checks(fits, data, n_rep=100, seed=20260612, verbose=True):
    """
    Simulate every fitted model once per replicate and return comparison tables.

    Regime-based summaries use one common fitted-M3 barrier.  A shared barrier
    keeps occupancy, switching, and waiting-time comparisons attributable to
    the fitted model rather than to different classification thresholds.
    """
    data = np.asarray(data, dtype=float)
    observed_x = data[:, 0]
    observed_v = data[:, 1]
    summary_rows = []
    behavior_rows = []
    waiting_rows = []
    density_tables = []
    reference_model = "M3" if "M3" in set(fits["model"]) else str(fits.iloc[0]["model"])
    reference_params = (
        fits.loc[fits["model"] == reference_model, PARAM_NAMES]
        .iloc[0].to_numpy(dtype=float)
    )
    _, common_barrier, _ = potential_extrema(reference_params)
    common_barrier = common_barrier if np.isfinite(common_barrier) else 0.0

    for model_index, (_, fit) in enumerate(fits.iterrows()):
        model = str(fit["model"])
        params = fit[PARAM_NAMES].to_numpy(dtype=float)
        barrier = common_barrier
        X_paths, V_paths = [], []

        observed_summary = path_summary_matrix(
            observed_x[None, :], observed_v[None, :],
        ).iloc[0].to_dict()
        summary_rows.append({"model": model, "source": "observed", "rep": -1,
                             **observed_summary})
        behavior_rows.append({
            "model": model, "source": "observed", "rep": -1,
            "barrier": barrier,
            **_path_behavior(observed_x, barrier),
        })
        observed_waits = extract_waiting_times(observed_x, barrier=barrier)
        for regime, values in observed_waits.items():
            waiting_rows.extend({
                "model": model, "source": "observed", "rep": -1,
                "barrier": barrier, "regime": regime, "waiting_time_years": value,
            } for value in values)

        for rep in range(n_rep):
            traj = simulate_trajectory(
                params, len(data) + 1, init_state=tuple(data[0]),
                seed=seed + 10000*model_index + rep,
            )
            partial = build_partial_data(traj[:, 0])
            X_paths.append(partial[:, 0])
            V_paths.append(partial[:, 1])
            behavior_rows.append({
                "model": model, "source": "simulated", "rep": rep,
                "barrier": barrier,
                **_path_behavior(partial[:, 0], barrier),
            })
            simulated_waits = extract_waiting_times(partial[:, 0], barrier=barrier)
            for regime, values in simulated_waits.items():
                waiting_rows.extend({
                    "model": model, "source": "simulated", "rep": rep,
                    "barrier": barrier, "regime": regime,
                    "waiting_time_years": value,
                } for value in values)
            if verbose and (rep + 1) % 25 == 0:
                print(f"Predictive check {model}: {rep + 1}/{n_rep}")

        X_paths = np.asarray(X_paths)
        V_paths = np.asarray(V_paths)
        simulated_summary = path_summary_matrix(X_paths, V_paths)
        simulated_summary.insert(0, "rep", np.arange(n_rep))
        simulated_summary.insert(0, "source", "simulated")
        simulated_summary.insert(0, "model", model)
        summary_rows.extend(simulated_summary.to_dict("records"))
        density_tables.extend([
            _density_band(observed_x, X_paths, "X", model),
            _density_band(observed_v, V_paths, "Vhat", model),
        ])

    return {
        "predictive_summary": pd.DataFrame(summary_rows),
        "predictive_behavior": pd.DataFrame(behavior_rows),
        "predictive_waiting": pd.DataFrame(waiting_rows),
        "predictive_density": pd.concat(density_tables, ignore_index=True),
    }


def run_nested_m3_m4_bootstrap(m3_params, m4_params, data, output_path,
                               n_boot=100, seed=20260612, n_starts_m3=2,
                               n_starts_m4=12, resume=True, verbose=True):
    """Bootstrap the M3-versus-M4 contrast using corrected M4 start sets."""
    data = np.asarray(data, dtype=float)
    output_path = Path(output_path)
    context = checkpoint_context(
        "nested_m3_m4_bootstrap", "M3_vs_M4", m3_params, data,
        m4_params=m4_params.tolist(), seed=seed,
        n_starts_m3=n_starts_m3, n_starts_m4=n_starts_m4,
    )
    # n_boot is intentionally omitted so a completed pilot can be extended.
    prepare_checkpoint(output_path, context, resume=resume)
    table = load_table(output_path, deduplicate="rep") if resume else pd.DataFrame()
    completed = (
        set(table.loc[table["success"].astype(bool), "rep"].astype(int))
        if len(table) and "success" in table else set()
    )
    rows = table.to_dict("records") if len(table) else []
    init_state = tuple(data[0])
    n_obs = len(data) + 1

    for rep in range(n_boot):
        if rep in completed:
            continue
        row = {"rep": rep, "seed": seed + rep, "success": False, "error": ""}
        t0 = time.perf_counter()
        try:
            boot_data = simulate_partial_data(
                m3_params, n_obs, init_state=init_state, seed=seed + rep,
            )
            m3_hat, nll_m3, conv_m3 = estimate_model(
                "M3", boot_data, start=extract_free_params(m3_params, "M3"),
                n_starts=n_starts_m3, seed=seed + 2*rep, verbose=False,
            )
            m4_hat, nll_m4, conv_m4 = estimate_model(
                "M4", boot_data, start=extract_free_params(m3_hat, "M4"),
                n_starts=n_starts_m4, seed=seed + 2*rep + 1, verbose=False,
                extra_starts=[extract_free_params(m4_params, "M4")],
            )
            _, q_min_m4 = diffusion_minimum(m4_hat)
            row.update({
                "nll_m3": nll_m3,
                "nll_m4": nll_m4,
                "contrast": 2.0*(nll_m3 - nll_m4),
                "convergence_m3": conv_m3,
                "convergence_m4": conv_m4,
                "nested_violation": bool(nll_m4 > nll_m3 + 1e-6),
                "q_min_m4": q_min_m4,
                "success": conv_m3 == 0 and conv_m4 == 0 and nll_m4 <= nll_m3 + 1e-6,
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["time_sec"] = time.perf_counter() - t0
        rows.append(row)
        table = pd.DataFrame(rows).drop_duplicates("rep", keep="last").sort_values("rep")
        save_table(table, output_path)
        if verbose:
            print(
                f"Nested M3/M4 bootstrap {rep + 1}/{n_boot} | "
                f"success={row['success']} | contrast={row.get('contrast', np.nan):.4f}"
            )
    return table.reset_index(drop=True)


def summarize_nested_bootstrap(table, observed_contrast):
    """Return finite-sample nested-comparison bootstrap diagnostics."""
    valid = table.loc[table["success"].astype(bool), "contrast"].to_numpy(dtype=float)
    return pd.DataFrame([{
        "observed_contrast": float(observed_contrast),
        "n_boot": int(len(table)),
        "n_valid": int(len(valid)),
        "success_rate": float(len(valid)/len(table)) if len(table) else np.nan,
        "contrast_mean": float(np.mean(valid)) if len(valid) else np.nan,
        "contrast_median": float(np.median(valid)) if len(valid) else np.nan,
        "contrast_q95": float(np.quantile(valid, 0.95)) if len(valid) else np.nan,
        "contrast_q99": float(np.quantile(valid, 0.99)) if len(valid) else np.nan,
        "p_upper": (
            float((np.sum(valid >= observed_contrast) + 1)/(len(valid) + 1))
            if len(valid) else np.nan
        ),
    }])
