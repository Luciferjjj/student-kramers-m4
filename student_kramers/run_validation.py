"""
run_validation.py - Complete- and partial-observation simulation validation

The generating model and fitted models are command-line choices.  This makes
the same recovery workflow usable for M1-M4 and for cross-model experiments.
"""
import argparse

import pandas as pd

from . import config
from .data_loading import build_partial_data, result_path, save_table
from .estimation import estimate_complete_model, estimate_model
from .models import MODELS, extract_free_params, parameter_row
from .simulation import simulate_trajectory


def main():
    """Run one known-parameter recovery or cross-model fitting experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-model", choices=list(MODELS), default="M4")
    parser.add_argument("--fit-models", nargs="+", choices=list(MODELS))
    parser.add_argument("--run-name", default=config.DEFAULT_RUN_NAME)
    parser.add_argument("--n-starts", type=int, default=config.N_RANDOM_STARTS)
    parser.add_argument("--n-obs", type=int, default=int(50.0/config.H_OBS) + 1)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    fit_models = args.fit_models or [args.generate_model]
    reference = config.REFERENCE_PARAMS_BY_MODEL[args.generate_model]
    traj = simulate_trajectory(
        reference,
        args.n_obs,
        h_obs=config.H_OBS,
        h_sim=config.H_SIM_VALIDATION,
        init_state=(1.5, 0.0),
        seed=args.seed,
    )
    partial_data = build_partial_data(traj[:, 0])

    reference_row = parameter_row(args.generate_model, reference)
    reference_row.update({
        "result_type": "reference",
        "generating_model": args.generate_model,
        "fitted_model": "",
        "convergence": 0,
    })
    rows = [reference_row]

    for index, model_name in enumerate(fit_models):
        start = (
            extract_free_params(reference, model_name)
            if model_name == args.generate_model
            else None
        )
        complete_hat, complete_nll, complete_conv = estimate_complete_model(
            model_name, traj, config.H_OBS, start=start, verbose=True,
            n_starts=args.n_starts, seed=args.seed + 2*index,
        )
        partial_hat, partial_nll, partial_conv = estimate_model(
            model_name, partial_data, start=start, verbose=True,
            n_starts=args.n_starts, seed=args.seed + 2*index + 1,
        )

        complete_row = parameter_row(model_name, complete_hat, complete_nll)
        complete_row.update({
            "result_type": "complete",
            "generating_model": args.generate_model,
            "fitted_model": model_name,
            "convergence": complete_conv,
        })
        partial_row = parameter_row(model_name, partial_hat, partial_nll)
        partial_row.update({
            "result_type": "partial",
            "generating_model": args.generate_model,
            "fitted_model": model_name,
            "convergence": partial_conv,
        })
        rows.extend([complete_row, partial_row])

    table = pd.DataFrame(rows)
    save_table(table, result_path("simulation_validation", run_name=args.run_name))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
