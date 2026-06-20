"""
run_bootstrap.py — IOS, parametric bootstrap, or nested-model contrast

Examples
--------
python3 -m student_kramers.run_bootstrap --mode ios-pilot --model M4 --fit-run m4_real_data_final
python3 -m student_kramers.run_bootstrap --mode ios --model M2 --fit-run m4_real_data_final
python3 -m student_kramers.run_bootstrap --mode parametric --model M2 --n-boot 100 --ios
python3 -m student_kramers.run_bootstrap --mode contrast --model M1 --contrast-alt M2 --n-boot 100
"""
import argparse

import numpy as np
import pandas as pd

from . import config
from .bootstrap import (
    influential_transitions,
    run_contrast_bootstrap,
    run_exact_ios,
    run_parametric_bootstrap,
    select_ios_pilot_transitions,
    summarize_ios,
)
from .estimation import make_random_starts
from .data_loading import load_model_fits, load_real_data, result_path, save_table
from .models import MODELS, PARAM_NAMES, extract_free_params


def _params(fits, model_name):
    """Extract one fitted full parameter vector."""
    return fits.loc[
        fits["model"] == model_name, PARAM_NAMES
    ].iloc[0].to_numpy(dtype=float)


def _pilot_start_candidates(model_name, fits, data, n_starts, seed):
    """Return the additional starts used to audit sampled M4 leave-one-out fits."""
    if model_name != "M4" or n_starts <= 1:
        return None

    m4_free = extract_free_params(_params(fits, "M4"), "M4")
    m3_boundary = extract_free_params(_params(fits, "M3"), "M4")
    starts = make_random_starts(
        "M4", m4_free, n_starts, seed=seed, data=data,
        extra_starts=[m3_boundary],
    )
    labelled = []
    for index, start in enumerate(starts[1:], start=1):
        label = "M3 boundary" if np.allclose(start, m3_boundary) else f"global interior {index}"
        labelled.append((label, start))
    return labelled


def main():
    """Parse one long-run workflow and save resumable results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["ios", "ios-pilot", "parametric", "contrast"],
        default="parametric",
    )
    parser.add_argument("--model", choices=list(MODELS), default="M2")
    parser.add_argument("--n-boot", type=int, default=config.N_BOOTSTRAP)
    parser.add_argument("--ios", action="store_true")
    parser.add_argument("--contrast-alt", choices=list(MODELS))
    parser.add_argument("--run-name", default=config.DEFAULT_RUN_NAME)
    parser.add_argument("--fit-run")
    parser.add_argument("--pilot-size", type=int, default=48)
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--maxiter", type=int, default=config.IOS_MAXITER)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=config.IOS_SAVE_EVERY)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    _, _, _, data = load_real_data()
    fit_run = args.fit_run or args.run_name
    fits = load_model_fits(fit_run, required=True, data=data)
    row = fits.loc[fits["model"] == args.model].iloc[0]
    params = row[PARAM_NAMES].to_numpy(dtype=float)

    if args.mode in {"ios", "ios-pilot"}:
        indices = args.indices
        start_candidates = None
        result_name = "ios"
        if args.mode == "ios-pilot":
            pilot = select_ios_pilot_transitions(fits, data, args.pilot_size)
            save_table(
                pilot,
                result_path("ios_pilot_indices", run_name=args.run_name),
            )
            indices = pilot["k"].astype(int).tolist()
            start_candidates = _pilot_start_candidates(
                args.model, fits, data, args.n_starts, args.seed,
            )
            result_name = "ios_pilot"

        table = run_exact_ios(
            args.model, data, params,
            result_path(result_name, args.model, run_name=args.run_name),
            maxiter=args.maxiter, indices=indices,
            save_every=1 if args.mode == "ios-pilot" else args.save_every,
            resume=not args.no_resume,
            start_candidates=start_candidates,
        )
        summary = summarize_ios(table, expected_n_transitions=len(data) - 1)
        summary.update({
            "model": args.model,
            "mode": args.mode,
            "fit_run": fit_run,
            "maxiter": args.maxiter,
            "n_requested_starts": (
                1 + len(start_candidates) if start_candidates else 1
            ),
        })
        save_table(
            pd.DataFrame([summary]),
            result_path(f"{result_name}_summary", args.model, run_name=args.run_name),
        )
        save_table(
            influential_transitions(table, data),
            result_path(f"{result_name}_influential", args.model, run_name=args.run_name),
        )
    elif args.mode == "contrast":
        if not args.contrast_alt:
            parser.error("--mode contrast requires --contrast-alt")
        table = run_contrast_bootstrap(
            args.model,
            args.contrast_alt,
            params,
            data,
            result_path(
                f"contrast_to_{args.contrast_alt.lower()}",
                args.model,
                run_name=args.run_name,
            ),
            n_boot=args.n_boot, resume=not args.no_resume,
        )
    else:
        table = run_parametric_bootstrap(
            args.model,
            params,
            data,
            result_path("parametric_bootstrap", args.model, run_name=args.run_name),
            n_boot=args.n_boot,
            compute_ios=args.ios,
            resume=not args.no_resume,
        )
    if args.mode in {"ios", "ios-pilot"}:
        print(pd.DataFrame([summary]).to_string(index=False))
    else:
        print(f"{table['success'].astype(bool).sum()} / {args.n_boot} successful")


if __name__ == "__main__":
    main()
