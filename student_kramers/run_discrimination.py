"""Command-line entry point for the M3-versus-M4 discrimination study."""
import argparse

from . import config
from .data_loading import result_path
from .discrimination import (
    M4_EFFECT_SCALES,
    run_discrimination_study,
    summarize_discrimination,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", choices=["M3", *M4_EFFECT_SCALES], required=True)
    parser.add_argument("--run-name", default=config.DEFAULT_RUN_NAME)
    parser.add_argument("--n-traj", type=int, default=1)
    parser.add_argument("--n-obs", type=int, default=int(50.0/config.H_OBS) + 1)
    parser.add_argument("--n-starts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output_path = result_path(
        f"discrimination_{args.truth.lower()}", run_name=args.run_name,
    )
    table = run_discrimination_study(
        args.truth, args.n_traj, args.n_obs, output_path,
        n_starts=args.n_starts, seed=args.seed, resume=not args.no_resume,
    )
    print(summarize_discrimination(table).to_string(index=False))


if __name__ == "__main__":
    main()
