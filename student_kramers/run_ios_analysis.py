"""Build saved IOS comparison tables and figures from completed checkpoints."""
import argparse

from . import config
from .data_loading import (
    load_model_fits,
    load_real_data,
    load_result,
    result_path,
    save_table,
)
from .figures import (
    plot_ios_numerical_diagnostics,
    plot_ios_overview,
    plot_ios_phase_space,
)
from .ios_analysis import (
    build_ios_pairwise_comparison,
    build_ios_parameter_shift_table,
    build_ios_regime_summary,
    build_ios_summary,
    build_ios_transition_table,
    validate_ios_tables,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-run", default="m4_real_data_cholesky")
    parser.add_argument("--run-name", default="ios_observed")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    _, age, _, data = load_real_data()
    fits = load_model_fits(args.fit_run, required=True, data=data)
    result_name = "ios_pilot" if args.pilot else "ios"
    tables = {
        model: load_result(result_name, model, run_name=args.run_name)
        for model in ("M2", "M3", "M4")
    }
    missing = [model for model, table in tables.items() if not len(table)]
    if missing:
        raise FileNotFoundError(
            f"Missing {result_name} checkpoints for {missing} in run {args.run_name!r}"
        )

    if args.pilot:
        pilot = load_result("ios_pilot_indices", run_name=args.run_name)
        expected_indices = pilot["k"].astype(int).tolist()
    else:
        expected_indices = list(range(len(data) - 1))

    validation = validate_ios_tables(tables, expected_indices)
    if not validation["complete_and_valid"].all():
        raise RuntimeError(
            "IOS tables are incomplete or invalid:\n" + validation.to_string(index=False)
        )

    summary = build_ios_summary(tables, len(data) - 1, fits=fits)
    transitions = build_ios_transition_table(tables, data, age)
    parameter_shifts = build_ios_parameter_shift_table(tables, fits)
    pairwise = build_ios_pairwise_comparison(transitions)
    regime = build_ios_regime_summary(transitions)

    save_table(validation, result_path(f"{result_name}_validation", run_name=args.run_name))
    save_table(summary, result_path(f"{result_name}_comparison", run_name=args.run_name))
    save_table(
        transitions,
        result_path(f"{result_name}_transitions", run_name=args.run_name),
    )
    save_table(
        parameter_shifts,
        result_path(f"{result_name}_parameter_shifts", run_name=args.run_name),
    )
    save_table(
        pairwise,
        result_path(f"{result_name}_pairwise_comparison", run_name=args.run_name),
    )
    save_table(
        regime,
        result_path(f"{result_name}_regime_summary", run_name=args.run_name),
    )

    figure_dir = config.run_dir(args.run_name) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = "ios_pilot" if args.pilot else "observed_exact_ios"
    plot_ios_overview(summary, transitions, figure_dir / f"{prefix}_overview.png")
    plot_ios_phase_space(transitions, figure_dir / f"{prefix}_phase_space.png")
    plot_ios_numerical_diagnostics(
        parameter_shifts, transitions,
        figure_dir / f"{prefix}_numerical_diagnostics.png",
    )

    print(validation.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print(f"\nTables and figures saved in {config.run_dir(args.run_name)}")


if __name__ == "__main__":
    main()
