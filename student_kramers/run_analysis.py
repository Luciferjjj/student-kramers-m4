"""
run_analysis.py - Compute numerical diagnostics for one fitted model.

This script deliberately does not draw figures. It saves compact diagnostic
tables that can be inspected and plotted directly in ``notebooks/final.ipynb``.
"""
import argparse

import numpy as np
import pandas as pd

from . import config
from .data_loading import load_model_fits, load_real_data, result_path, save_table
from .models import MODELS, PARAM_NAMES
from .simulation import (
    compute_derived_params,
    extract_waiting_times,
    observed_summary,
    path_summary_matrix,
    simulate_first_passage,
    simulate_model_check,
    simulate_waiting_times,
    summarize_waiting_times,
)


def main():
    """Save cheap summaries and optionally run simulation-based diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="M2")
    parser.add_argument("--run-name", default=config.DEFAULT_RUN_NAME)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--n-rep", type=int, default=config.N_MODEL_CHECK)
    parser.add_argument("--n-first-passage", type=int, default=config.N_FIRST_PASSAGE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, _, x, data = load_real_data()
    fits = load_model_fits(args.run_name, required=True, data=data)
    selected = fits.loc[fits["model"] == args.model]
    if not len(selected):
        raise ValueError(f"{args.model} has not been fitted in run {args.run_name!r}")

    derived = []
    for _, row in fits.iterrows():
        values = compute_derived_params(row[PARAM_NAMES].to_numpy(dtype=float))
        derived.append({"model": row["model"], **values})
    save_table(
        pd.DataFrame(derived),
        result_path("derived_parameters", run_name=args.run_name),
    )

    params = selected.iloc[0][PARAM_NAMES].to_numpy(dtype=float)
    derived_model = compute_derived_params(params)
    barrier = derived_model["barrier"] if np.isfinite(derived_model["barrier"]) else 0.0

    observed_waits = extract_waiting_times(x, barrier=barrier)
    save_table(
        summarize_waiting_times(observed_waits, "observed"),
        result_path("observed_waiting_times", run_name=args.run_name),
    )
    if not args.simulate:
        return

    init_state = data[0]
    X_paths, Vhat_paths = simulate_model_check(
        params, len(x), args.n_rep, init_state, seed=args.seed,
    )
    save_table(
        path_summary_matrix(X_paths, Vhat_paths),
        result_path("model_check", args.model, run_name=args.run_name),
    )
    save_table(
        pd.DataFrame([observed_summary(x[:-1], data[:, 1])]),
        result_path("observed_summary", args.model, run_name=args.run_name),
    )

    simulated_waits = simulate_waiting_times(
        params, len(x), args.n_rep, init_state, seed=args.seed, barrier=barrier,
    )
    save_table(
        summarize_waiting_times(simulated_waits, args.model),
        result_path("waiting_times", args.model, run_name=args.run_name),
    )

    if np.isfinite(derived_model["lower_well"]) and args.n_first_passage > 0:
        lower = simulate_first_passage(
            params, derived_model["lower_well"], barrier=barrier,
            n_paths=args.n_first_passage, seed=args.seed,
        )
        upper = simulate_first_passage(
            params, derived_model["upper_well"], barrier=barrier,
            n_paths=args.n_first_passage, seed=args.seed + 1,
        )
        save_table(
            pd.DataFrame({"lower_to_barrier": lower, "upper_to_barrier": upper}),
            result_path("first_passage", args.model, run_name=args.run_name),
        )


if __name__ == "__main__":
    main()
