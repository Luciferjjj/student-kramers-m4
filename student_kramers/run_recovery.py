"""Command-line entry point for extendable complete/partial recovery studies."""
import argparse

from . import config
from .data_loading import result_path
from .models import MODELS
from .recovery import run_recovery_study, summarize_recovery_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="M4")
    parser.add_argument("--run-name", default=config.DEFAULT_RUN_NAME)
    parser.add_argument("--n-traj", "--n-paths", dest="n_traj", type=int, default=1)
    parser.add_argument("--n-obs", type=int, default=int(50.0/config.H_OBS) + 1)
    parser.add_argument("--n-starts", type=int, default=config.N_RANDOM_STARTS)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--start-at-truth", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    reference = config.REFERENCE_PARAMS_BY_MODEL[args.model]
    output_path = result_path(
        "recovery_study", model_name=args.model, run_name=args.run_name,
    )
    table = run_recovery_study(
        args.model, reference, args.n_traj, args.n_obs, output_path,
        n_starts=args.n_starts, seed=args.seed,
        start_at_truth=args.start_at_truth, resume=not args.no_resume,
    )
    print(summarize_recovery_diagnostics(table).to_string(index=False))


if __name__ == "__main__":
    main()
