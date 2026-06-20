"""
discrimination.py - Partial-observation M3 versus M4 simulation study

The study estimates both M3 and M4 after simulation under

    M3 truth,
    weak, moderate, or strong M4 truth.

The likelihood contrast is

    contrast = 2 * [NLL(M3) - NLL(M4)].

Positive values favour M4.  The empirical distributions under M3 and M4
truth measure false-positive behaviour and power without a chi-square claim.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loading import (
    checkpoint_context,
    load_table,
    prepare_checkpoint,
    save_table,
)
from .estimation import estimate_model
from .models import (
    PARAM_NAMES,
    constraints_valid,
    extract_free_params,
)
from .simulation import simulate_partial_data


M4_EFFECT_SCALES = {
    "weak": 1.0,
    "moderate": 10.0,
    "strong": 25.0,
}


def discrimination_truth(truth):
    """Return generating model and parameters for one discrimination scenario."""
    if truth == "M3":
        return "M3", config.REFERENCE_PARAMS_BY_MODEL["M3"].copy()
    if truth not in M4_EFFECT_SCALES:
        raise KeyError(f"Unknown truth scenario {truth!r}")

    m3 = config.REFERENCE_PARAMS_BY_MODEL["M3"]
    m4 = config.REFERENCE_PARAMS_BY_MODEL["M4"]
    params = m3 + M4_EFFECT_SCALES[truth]*(m4 - m3)
    if not constraints_valid(extract_free_params(params, "M4"), "M4"):
        raise ValueError(f"{truth} M4 parameters violate constraints")
    return "M4", params


def run_discrimination_study(truth, n_traj, n_obs, output_path=None,
                             h_obs=config.H_OBS, h_sim=config.H_SIM,
                             init_state=(1.5, 0.0), n_starts=1, seed=20260611,
                             resume=True, verbose=True):
    """Run or extend one partial-observation M3-versus-M4 scenario."""
    generating_model, true_params = discrimination_truth(truth)
    output_path = Path(output_path) if output_path else None
    if output_path:
        context = checkpoint_context(
            "discrimination_study", f"{truth}:M3_vs_M4", true_params,
            np.asarray(init_state), n_obs=n_obs, h_obs=h_obs, h_sim=h_sim,
            init_state=list(init_state), n_starts=n_starts, seed=seed,
        )
        # n_traj is omitted so an existing pilot can be extended.
        prepare_checkpoint(output_path, context, resume=resume)

    table = load_table(output_path, deduplicate="traj") if output_path and resume else pd.DataFrame()
    completed = set(table["traj"].astype(int)) if len(table) else set()
    rows = table.to_dict("records") if len(table) else []

    for traj in range(n_traj):
        if traj in completed:
            continue
        row = {
            "traj": traj,
            "seed": seed + traj,
            "truth": truth,
            "generating_model": generating_model,
            "success": False,
            "error": "",
        }
        t0 = time.perf_counter()
        try:
            data = simulate_partial_data(
                true_params, n_obs, h_obs, h_sim, init_state, seed + traj,
            )
            fitted = {}
            for offset, model_name in enumerate(("M3", "M4")):
                if model_name == "M4":
                    # M4 contains M3.  Starting from the fitted M3 boundary
                    # guarantees that M4 has access to the M3 objective value.
                    start = extract_free_params(fitted["M3"][0], "M4")
                else:
                    start = extract_free_params(
                        config.RECOVERY_INIT_PARAMS_BY_MODEL[model_name], model_name,
                    )
                params, nll, conv = estimate_model(
                    model_name, data, h_obs, start=start, n_starts=n_starts,
                    seed=seed + 2*traj + offset, verbose=False,
                )
                fitted[model_name] = (params, nll, conv)
                row[f"nll_{model_name.lower()}"] = nll
                row[f"convergence_{model_name.lower()}"] = conv
                row.update({
                    f"{model_name.lower()}_{name}": value
                    for name, value in zip(PARAM_NAMES, params)
                })
            row["contrast"] = 2.0*(fitted["M3"][1] - fitted["M4"][1])
            row["success"] = fitted["M3"][2] == 0 and fitted["M4"][2] == 0
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["time_sec"] = time.perf_counter() - t0
        rows.append(row)
        table = pd.DataFrame(rows).drop_duplicates("traj", keep="last").sort_values("traj")
        if output_path:
            save_table(table, output_path)
        if verbose:
            print(
                f"Discrimination {truth}: traj {traj + 1}/{n_traj} | "
                f"success={row['success']} | contrast={row.get('contrast', np.nan):.4f}"
            )
    return table.reset_index(drop=True)


def summarize_discrimination(table):
    """Return convergence and empirical contrast summaries by truth scenario."""
    rows = []
    for truth, group in table.groupby("truth", sort=False):
        valid = group[group["success"].astype(bool)]
        values = valid["contrast"].to_numpy(dtype=float)
        rows.append({
            "truth": truth,
            "n_traj": len(group),
            "n_success": len(valid),
            "success_rate": len(valid)/len(group),
            "contrast_mean": float(np.mean(values)) if len(values) else np.nan,
            "contrast_median": float(np.median(values)) if len(values) else np.nan,
            "contrast_q025": float(np.quantile(values, 0.025)) if len(values) else np.nan,
            "contrast_q975": float(np.quantile(values, 0.975)) if len(values) else np.nan,
            "m4_win_rate": float(np.mean(values > 0.0)) if len(values) else np.nan,
            "median_time_sec": float(group["time_sec"].median()),
        })
    return pd.DataFrame(rows)
