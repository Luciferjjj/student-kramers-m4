"""
recovery.py - Complete- and partial-observation parameter recovery studies

For each simulated path, the same known parameter vector generates

    (X_t, V_t)  -> complete-data fit,
    X_t         -> Vhat_t -> partial-data fit.

Setting ``n_traj=1`` gives a smoke validation.  Increasing ``n_traj`` runs
the repeated recovery study with exactly the same implementation.  Results
are checkpointed after every path, so a one-path run can later be extended.
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
from .estimation import estimate_complete_model, estimate_model
from .models import (
    PARAM_NAMES,
    constraints_valid,
    diffusion_variance,
    extract_free_params,
    free_names,
    parameter_row,
)
from .simulation import simulate_trajectory


def _q_recovery_metrics(params_hat, reference, trajectory):
    """Compare fitted and true q(x,v) along the latent simulated path."""
    x, v = trajectory[:-1].T
    q_true = diffusion_variance(x, v, reference)
    q_hat = diffusion_variance(x, v, params_hat)
    rmse = float(np.sqrt(np.mean((q_hat - q_true)**2)))
    return {
        "q_path_rmse": rmse,
        "q_path_relative_rmse": rmse / float(np.mean(q_true)),
        "q_path_min": float(np.min(q_hat)),
    }


def run_recovery_study(model_name, reference, n_traj, n_obs, output_path=None,
                       h_obs=config.H_OBS, h_sim=config.H_SIM,
                       init_state=(1.5, 0.0), n_starts=1, seed=20260611,
                       initial_params=None, start_at_truth=False,
                       resume=True, verbose=True):
    """
    Run or extend a same-model complete/partial parameter recovery study.

    Formal estimation starts from simulation-study initial values distinct
    from both the truth and the real-data initial values.  ``start_at_truth``
    is useful only for local likelihood debugging.
    """
    reference = np.asarray(reference, dtype=float)
    free_reference = extract_free_params(reference, model_name)
    initial_params = (
        config.RECOVERY_INIT_PARAMS_BY_MODEL[model_name]
        if initial_params is None else np.asarray(initial_params, dtype=float)
    )
    free_initial = extract_free_params(initial_params, model_name)
    if not constraints_valid(free_reference, model_name):
        raise ValueError(f"Reference parameters violate {model_name} constraints")
    if not constraints_valid(free_initial, model_name):
        raise ValueError(f"Initial parameters violate {model_name} constraints")
    if n_traj < 1 or n_obs < 3:
        raise ValueError("n_traj must be positive and n_obs must be at least three")

    output_path = Path(output_path) if output_path else None
    if output_path:
        context = checkpoint_context(
            "recovery_study", model_name, reference, np.asarray(init_state),
            n_obs=n_obs, h_obs=h_obs, h_sim=h_sim, init_state=list(init_state),
            n_starts=n_starts, seed=seed, start_at_truth=start_at_truth,
            initial_params=initial_params.tolist(),
        )
        # n_traj is intentionally omitted: increasing it extends the same run.
        prepare_checkpoint(output_path, context, resume=resume)

    table = load_table(output_path) if output_path and resume else pd.DataFrame()
    if len(table):
        table = table.drop_duplicates(["traj", "observation"], keep="last")
        counts = table.groupby("traj")["observation"].nunique()
        completed = set(counts[counts == 2].index.astype(int))
        completed_observations = set(zip(
            table["traj"].astype(int), table["observation"].astype(str),
        ))
        rows = table.to_dict("records")
    else:
        completed, completed_observations, rows = set(), set(), []

    start = free_reference if start_at_truth else free_initial
    for traj in range(n_traj):
        if traj in completed:
            continue
        path_seed = seed + traj
        trajectory = simulate_trajectory(
            reference, n_obs, h_obs, h_sim, init_state, path_seed,
        )
        partial_data = build_partial_data(trajectory[:, 0], h_obs)

        estimators = (
            ("complete", estimate_complete_model, trajectory),
            ("partial", estimate_model, partial_data),
        )
        for offset, (observation, estimator, data) in enumerate(estimators):
            if (traj, observation) in completed_observations:
                continue
            t0 = time.perf_counter()
            row = {
                "traj": traj,
                "seed": path_seed,
                "observation": observation,
                "success": False,
                "convergence": 1,
                "error": "",
            }
            try:
                kwargs = {
                    "start": start,
                    "n_starts": n_starts,
                    "seed": seed + 2*traj + offset,
                    "verbose": False,
                }
                if observation == "complete":
                    params_hat, nll, conv = estimator(model_name, data, h_obs, **kwargs)
                else:
                    params_hat, nll, conv = estimator(model_name, data, h=h_obs, **kwargs)
                row.update(parameter_row(model_name, params_hat, nll))
                row.update(_q_recovery_metrics(params_hat, reference, trajectory))
                row["convergence"] = conv
                row["success"] = conv == 0
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["time_sec"] = time.perf_counter() - t0
            rows.append(row)
            completed_observations.add((traj, observation))
            table = pd.DataFrame(rows).drop_duplicates(
                ["traj", "observation"], keep="last",
            ).sort_values(["traj", "observation"]).reset_index(drop=True)
            if output_path:
                save_table(table, output_path)
            if verbose:
                print(
                    f"Recovery {model_name}: traj {traj + 1}/{n_traj} | "
                    f"{observation}={bool(row['success'])} | "
                    f"NLL={row.get('nll', np.nan):.6f}",
                    flush=True,
                )
    return table


def summarize_recovery(table, reference, model_name):
    """Return parameter-level bias, RMSE, and empirical recovery intervals."""
    reference = np.asarray(reference, dtype=float)
    truth = dict(zip(PARAM_NAMES, reference))
    rows = []
    valid = table[table["success"].astype(bool)].copy()
    for observation, group in valid.groupby("observation", sort=False):
        total = int((table["observation"] == observation).sum())
        for name in free_names(model_name):
            values = group[name].to_numpy(dtype=float)
            errors = values - truth[name]
            rows.append({
                "observation": observation,
                "parameter": name,
                "truth": truth[name],
                "n_total": total,
                "n_success": len(values),
                "success_rate": len(values)/total if total else np.nan,
                "mean": float(np.mean(values)),
                "bias": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "q025": float(np.quantile(values, 0.025)),
                "median": float(np.median(values)),
                "q975": float(np.quantile(values, 0.975)),
            })
    return pd.DataFrame(rows)


def summarize_recovery_diagnostics(table):
    """Return estimator-level convergence, q-recovery, NLL, and runtime summaries."""
    rows = []
    for observation, group in table.groupby("observation", sort=False):
        valid = group[group["success"].astype(bool)]
        rows.append({
            "observation": observation,
            "n_traj": len(group),
            "n_success": len(valid),
            "success_rate": len(valid)/len(group),
            "median_nll": float(valid["nll"].median()) if len(valid) else np.nan,
            "median_q_relative_rmse": (
                float(valid["q_path_relative_rmse"].median()) if len(valid) else np.nan
            ),
            "median_time_sec": float(group["time_sec"].median()),
        })
    return pd.DataFrame(rows)


def diffusion_recovery_surfaces(table, reference, trajectory, traj=0, n_grid=100):
    """Return common-grid q surfaces for truth and successful fits of one path."""
    trajectory = np.asarray(trajectory, dtype=float)
    x_limits = np.quantile(trajectory[:, 0], [0.01, 0.99])
    v_limits = np.quantile(trajectory[:, 1], [0.01, 0.99])
    x_grid = np.linspace(*x_limits, n_grid)
    v_grid = np.linspace(*v_limits, n_grid)
    X, V = np.meshgrid(x_grid, v_grid)
    surfaces = {"truth": diffusion_variance(X, V, reference)}

    selected = table[(table["traj"] == traj) & table["success"].astype(bool)]
    for _, row in selected.iterrows():
        params = row[PARAM_NAMES].to_numpy(dtype=float)
        surfaces[row["observation"]] = diffusion_variance(X, V, params)
    return X, V, surfaces
