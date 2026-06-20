"""
run_application.py — Fit all registered models to the Greenland Ca2+ data

Execution order:
  1. Load and preprocess the official Ca2+ series.
  2. Fit every model registered in models.MODELS.
  3. Save the common model-comparison table for this named experiment.
"""
import argparse

from . import config
from .data_loading import load_model_fits, load_real_data, save_model_fits
from .estimation import estimate_models
from .models import MODELS, PARAM_NAMES


def main():
    """Run the complete real-data model-comparison application."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--run-name", default=config.DEFAULT_RUN_NAME)
    parser.add_argument("--n-starts", type=int, default=config.N_RANDOM_STARTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warm-start-run",
        help="Use saved best parameters from this run as additional starts.",
    )
    args = parser.parse_args()

    config.make_result_dirs(args.run_name)
    _, _, _, data = load_real_data()
    warm_starts = {}
    if args.warm_start_run:
        prior = load_model_fits(args.warm_start_run, required=True)
        warm_starts = {
            row["model"]: row[PARAM_NAMES].to_numpy(dtype=float)
            for _, row in prior.iterrows()
            if row["model"] in args.models
        }
    fits = estimate_models(
        args.models, data, n_starts=args.n_starts, seed=args.seed,
        warm_starts=warm_starts,
    )
    save_model_fits(
        fits, data, args.run_name, n_starts=args.n_starts, seed=args.seed,
    )
    print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
