"""Load saved Greenland results and reproduce the current pre-IOS figures."""
import argparse
from pathlib import Path

import pandas as pd

from . import config
from .data_loading import load_model_fits, load_real_data, load_result
from student_kramers.discrimination import M4_EFFECT_SCALES
from .figures import (
    plot_discrimination,
    plot_m4_diffusion_audit,
    plot_nested_bootstrap,
    plot_optimization_stability,
    plot_predictive_densities,
    plot_predictive_percentiles,
    plot_real_data_mechanisms,
    plot_real_data_state_space,
    plot_recovery_study,
    plot_transition_improvement,
    plot_waiting_time_comparison,
)


def _save_path(directory, name):
    """Return a PNG path inside one figure directory."""
    return directory / f"{name}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-run", default="m4_real_data_cholesky")
    parser.add_argument("--run-name", default="pre_ios_validation")
    args = parser.parse_args()

    directory = config.run_dir(args.run_name) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    _, age, x, data = load_real_data()
    fits = load_model_fits(args.fit_run, required=True, data=data)
    observed_contrast = 2.0*(
        float(fits.loc[fits["model"] == "M3", "nll"].iloc[0])
        - float(fits.loc[fits["model"] == "M4", "nll"].iloc[0])
    )

    plot_real_data_state_space(
        age, x, data, fits, _save_path(directory, "real_data_state_space"),
    )
    plot_real_data_mechanisms(
        fits, data, _save_path(directory, "real_data_mechanisms"),
    )
    transitions = load_result("transition_improvement", run_name=args.run_name)
    if len(transitions):
        plot_transition_improvement(
            transitions, _save_path(directory, "transition_improvement"),
        )

    q_audit = load_result("q_min_audit", run_name=args.run_name)
    if len(q_audit):
        plot_m4_diffusion_audit(
            fits, data, q_audit, _save_path(directory, "m4_diffusion_audit"),
        )

    stability = load_result("m4_optimization_stability", run_name=args.run_name)
    independent = load_result(
        "m4_optimization_stability_no_warm", run_name=args.run_name,
    )
    if len(independent):
        stability = pd.concat([stability, independent], ignore_index=True)
    tolerance = load_result("m4_tolerance_stability", run_name=args.run_name)
    if len(stability) and len(tolerance):
        plot_optimization_stability(
            stability, tolerance, fits, _save_path(directory, "m4_optimization_stability"),
        )

    density = load_result("predictive_density", run_name=args.run_name)
    summary = load_result("predictive_summary", run_name=args.run_name)
    behavior = load_result("predictive_behavior", run_name=args.run_name)
    waiting = load_result("predictive_waiting", run_name=args.run_name)
    if len(density):
        plot_predictive_densities(
            density, _save_path(directory, "predictive_density_bands"),
        )
    if len(summary) and len(behavior):
        plot_predictive_percentiles(
            summary, behavior, _save_path(directory, "predictive_check_map"),
        )
    if len(waiting):
        plot_waiting_time_comparison(
            waiting, _save_path(directory, "predictive_waiting_times"),
        )

    recovery_tables = []
    for model in ("M2", "M3", "M4"):
        table = load_result("recovery_study", model, run_name=args.run_name)
        if len(table):
            table["model"] = model
            recovery_tables.append(table)
    if len(recovery_tables) == 3:
        plot_recovery_study(
            pd.concat(recovery_tables, ignore_index=True),
            config.REFERENCE_PARAMS_BY_MODEL,
            _save_path(directory, "repeated_parameter_recovery"),
        )

    discrimination_tables = []
    for truth in ("M3", *M4_EFFECT_SCALES):
        table = load_result(f"discrimination_{truth.lower()}", run_name=args.run_name)
        if len(table):
            discrimination_tables.append(table)
    if len(discrimination_tables) == 4:
        plot_discrimination(
            pd.concat(discrimination_tables, ignore_index=True),
            observed_contrast, _save_path(directory, "m3_m4_discrimination"),
        )

    nested = load_result("m3_m4_nested_bootstrap", run_name=args.run_name)
    if len(nested):
        plot_nested_bootstrap(
            nested, observed_contrast, _save_path(directory, "m3_m4_nested_bootstrap"),
        )

    print(f"Figures saved in {directory}")


if __name__ == "__main__":
    main()
