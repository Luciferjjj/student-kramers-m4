"""
run_single.py — Quick environment and likelihood check

This is the cheapest executable.  It loads one saved fit, recomputes its
partial-observation NLL, and checks agreement with the saved value.
"""
import argparse

from . import config
from .data_loading import load_model_fits, load_real_data
from student_kramers.likelihoods import partial_neg_log_lik
from student_kramers.models import MODELS, PARAM_NAMES, extract_free_params


def main():
    """Evaluate one registered model on the real pseudo-state data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="M2")
    parser.add_argument("--run-name", default=config.DEFAULT_RUN_NAME)
    args = parser.parse_args()

    _, _, _, data = load_real_data()
    fits = load_model_fits(args.run_name, required=True, data=data)
    row = fits.loc[fits["model"] == args.model].iloc[0]
    params = row[PARAM_NAMES].to_numpy(dtype=float)
    free = extract_free_params(params, args.model)
    nll = partial_neg_log_lik(free, data, args.model)

    print(f"{data.shape[0]} pseudo-states | {data.shape[0] - 1} transitions")
    print(f"\n=== {args.model}: {row['description']} ===")
    for name, value in zip(PARAM_NAMES, params):
        print(f"  {name:7s}: {value:12.6f}")
    print(f"  {'NLL':7s}: {nll:12.6f}")
    print(f"  {'saved NLL':7s}: {float(row['nll']):12.6f}")


if __name__ == "__main__":
    main()
