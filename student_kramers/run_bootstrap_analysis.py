"""Build saved tables and figures for parametric bootstrap runs."""
import argparse

from .data_loading import (
    load_model_fits,
    load_real_data,
    load_result,
    result_path,
    save_table,
)
from .bootstrap_analysis import (
    build_diffusion_bootstrap_diagnostics,
    build_parameter_bootstrap_summary,
    build_parametric_bootstrap_overview,
    build_path_diffusion_band,
    summarize_diffusion_bootstrap,
)
from .figures import (
    plot_m4_parametric_bootstrap_diffusion,
    plot_m4_parametric_bootstrap_parameters,
)
from .models import PARAM_NAMES


def main():
    """Analyze one completed parametric bootstrap run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-run", default="m4_real_data_cholesky")
    parser.add_argument("--run-name", default="m4_parametric_bootstrap")
    parser.add_argument("--model", default="M4", choices=["M4"])
    args = parser.parse_args()

    _, age, _, data = load_real_data()
    fits = load_model_fits(args.fit_run, required=True, data=data)
    params = fits.loc[
        fits["model"] == args.model, PARAM_NAMES
    ].iloc[0].to_numpy(dtype=float)
    table = load_result(
        "parametric_bootstrap", args.model, run_name=args.run_name,
        deduplicate="rep",
    )
    if not len(table):
        raise FileNotFoundError(f"No parametric bootstrap table in {args.run_name!r}")

    overview = build_parametric_bootstrap_overview(table)
    parameters = build_parameter_bootstrap_summary(table, params)
    diffusion = build_diffusion_bootstrap_diagnostics(table, params, data)
    diffusion_summary = summarize_diffusion_bootstrap(diffusion, params, data)
    path_band = build_path_diffusion_band(table, params, data, age)

    save_table(
        overview,
        result_path("parametric_bootstrap_overview", args.model, run_name=args.run_name),
    )
    save_table(
        parameters,
        result_path("parametric_bootstrap_parameters", args.model, run_name=args.run_name),
    )
    save_table(
        diffusion,
        result_path("parametric_bootstrap_diffusion", args.model, run_name=args.run_name),
    )
    save_table(
        diffusion_summary,
        result_path(
            "parametric_bootstrap_diffusion_summary",
            args.model,
            run_name=args.run_name,
        ),
    )
    save_table(
        path_band,
        result_path("parametric_bootstrap_path_band", args.model, run_name=args.run_name),
    )

    figure_dir = table_path = result_path(
        "parametric_bootstrap_overview", args.model, run_name=args.run_name,
    ).parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_m4_parametric_bootstrap_parameters(
        parameters,
        figure_dir / "m4_parametric_bootstrap_parameters.png",
    )
    plot_m4_parametric_bootstrap_diffusion(
        overview,
        diffusion,
        diffusion_summary,
        path_band,
        figure_dir / "m4_parametric_bootstrap_diffusion.png",
    )

    print(overview.to_string(index=False))
    print()
    print(diffusion_summary.to_string(index=False))
    print(f"\nTables and figures saved in {table_path.parent}")


if __name__ == "__main__":
    main()
