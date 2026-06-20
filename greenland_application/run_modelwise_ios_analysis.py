"""Analyze a completed or partial Greenland model-wise IOS bootstrap run."""
import argparse

from . import config
from .bootstrap_analysis import (
    build_modelwise_ios_bootstrap_summary,
    build_modelwise_ios_cumulative,
)
from .data_loading import load_result, result_path, save_table
from .figures import plot_modelwise_ios_bootstrap


def main():
    """Build model-wise IOS bootstrap tables and figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="m4_modelwise_ios_bootstrap")
    parser.add_argument("--observed-run", default="ios_observed")
    parser.add_argument("--model", choices=["M4"], default="M4")
    args = parser.parse_args()

    table = load_result(
        "modelwise_ios_bootstrap", args.model,
        run_name=args.run_name, deduplicate="rep",
    )
    if not len(table):
        raise FileNotFoundError(
            f"No model-wise IOS bootstrap table found in {args.run_name!r}"
        )
    observed = load_result("ios_comparison", run_name=args.observed_run)
    if not len(observed):
        raise FileNotFoundError(
            f"No observed IOS comparison table found in {args.observed_run!r}"
        )
    observed_T_N = float(
        observed.loc[observed["model"] == args.model, "T_N"].iloc[0]
    )

    summary = build_modelwise_ios_bootstrap_summary(
        table, observed_T_N, model_name=args.model,
    )
    cumulative = build_modelwise_ios_cumulative(table, observed_T_N)

    save_table(
        summary,
        result_path("modelwise_ios_bootstrap_summary", args.model, run_name=args.run_name),
    )
    save_table(
        cumulative,
        result_path("modelwise_ios_bootstrap_cumulative", args.model, run_name=args.run_name),
    )

    figure_dir = config.run_dir(args.run_name) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_modelwise_ios_bootstrap(
        summary, cumulative, table,
        figure_dir / "m4_modelwise_ios_bootstrap.png",
    )

    print(summary.to_string(index=False))
    print(f"\nTables and figures saved in {config.run_dir(args.run_name)}")


if __name__ == "__main__":
    main()
