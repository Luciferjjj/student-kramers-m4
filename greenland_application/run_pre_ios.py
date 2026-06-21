"""Greenland command-line entry point for diagnostics required before IOS."""
import argparse

import pandas as pd

from . import config
from .data_loading import (
    load_model_fits,
    load_real_data,
    result_path,
    save_table,
)
from student_kramers.models import PARAM_NAMES
from .pre_ios import (
    audit_diffusion_minima,
    nested_bootstrap_cumulative,
    run_nested_m3_m4_bootstrap,
    run_optimization_stability,
    run_predictive_checks_checkpointed,
    run_tolerance_stability,
    summarize_nested_bootstrap,
    transition_improvement_table,
)


def _params(fits, model):
    """Extract one fitted full parameter vector."""
    return fits.loc[fits["model"] == model, PARAM_NAMES].iloc[0].to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True,
        choices=[
            "q-audit", "transition", "stability", "tolerance", "predictive", "nested",
        ],
    )
    parser.add_argument("--fit-run", default="m4_real_data_cholesky")
    parser.add_argument("--run-name", default="pre_ios_validation")
    parser.add_argument("--n-rep", type=int, default=100)
    parser.add_argument("--n-starts", type=int, default=12)
    parser.add_argument("--n-starts-m3", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 314, 2026])
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config.make_result_dirs(args.run_name)
    _, age, _, data = load_real_data()
    fits = load_model_fits(args.fit_run, required=True, data=data)
    m3_params, m4_params = _params(fits, "M3"), _params(fits, "M4")

    if args.mode == "q-audit":
        table = audit_diffusion_minima(fits, data)
        save_table(table, result_path("q_min_audit", run_name=args.run_name))
        print(table.to_string(index=False))
    elif args.mode == "transition":
        table = transition_improvement_table(fits, data, age)
        save_table(
            table, result_path("transition_improvement", run_name=args.run_name),
        )
        print(table["gain_m4_over_m3"].describe().to_string())
    elif args.mode == "stability":
        name = (
            "m4_optimization_stability_no_warm"
            if args.no_warm_start else "m4_optimization_stability"
        )
        table = run_optimization_stability(
            data, m3_params, m4_params,
            result_path(name, run_name=args.run_name),
            seeds=args.seeds, n_starts=args.n_starts,
            include_warm=not args.no_warm_start,
            resume=not args.no_resume,
        )
        print(
            table.groupby(["seed", "start_kind"])["nll"]
            .agg(["count", "min", "median"]).to_string()
        )
    elif args.mode == "tolerance":
        table = run_tolerance_stability(
            data, m4_params,
            result_path("m4_tolerance_stability", run_name=args.run_name),
            resume=not args.no_resume,
        )
        print(table.to_string(index=False))
    elif args.mode == "predictive":
        paths = {
            name: result_path(f"predictive_rep_{name}", run_name=args.run_name)
            for name in ("summary", "behavior", "waiting", "density_rep")
        }
        tables = run_predictive_checks_checkpointed(
            fits, data, paths, n_rep=args.n_rep, seed=args.seed,
            resume=not args.no_resume,
        )
        for name, table in tables.items():
            save_table(table, result_path(name, run_name=args.run_name))
        print(
            tables["predictive_behavior"]
            .groupby(["model", "source"])["n_switches"]
            .agg(["count", "mean", "median"]).to_string()
        )
    else:
        table = run_nested_m3_m4_bootstrap(
            m3_params, m4_params, data,
            result_path("m3_m4_nested_bootstrap", run_name=args.run_name),
            n_boot=args.n_rep, seed=args.seed,
            n_starts_m3=args.n_starts_m3, n_starts_m4=args.n_starts,
            resume=not args.no_resume,
        )
        observed = 2.0*(
            float(fits.loc[fits["model"] == "M3", "nll"].iloc[0])
            - float(fits.loc[fits["model"] == "M4", "nll"].iloc[0])
        )
        summary = summarize_nested_bootstrap(table, observed)
        save_table(summary, result_path("m3_m4_nested_summary", run_name=args.run_name))
        save_table(
            nested_bootstrap_cumulative(table, observed),
            result_path("m3_m4_nested_cumulative", run_name=args.run_name),
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
