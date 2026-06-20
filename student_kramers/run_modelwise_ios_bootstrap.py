"""
run_modelwise_ios_bootstrap.py - parallel model-wise IOS bootstrap

One replication performs the strict finite-sample IOS calibration step:

    simulate from fitted M4 -> refit M4 -> compute exact IOS on that sample.

Each replication writes two resumable artifacts:

    m4_ios_details/rep_XXXX.csv
        transition-level leave-one-out IOS checkpoint;
    m4_modelwise_ios_rows/rep_XXXX.csv
        one-row replication summary.

The final table ``m4_modelwise_ios_bootstrap.csv`` is rebuilt from the one-row
files whenever a worker finishes, so the run can be extended from 100 to 300 or
500 without changing the experiment identity.
"""
import argparse
import concurrent.futures as futures
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .bootstrap import run_exact_ios, summarize_ios
from .bootstrap_analysis import safe_diffusion_rectangle_minimum
from .data_loading import (
    checkpoint_context,
    load_model_fits,
    load_real_data,
    load_table,
    prepare_checkpoint,
    result_path,
    save_table,
)
from .estimation import estimate_model
from .models import (
    PARAM_NAMES,
    constraints_valid,
    diffusion_minimum,
    diffusion_variance,
    extract_free_params,
    parameter_row,
)
from .simulation import simulate_partial_data


def _row_files(row_dir):
    """Return saved one-row bootstrap summaries in replication order."""
    row_dir = Path(row_dir)
    return sorted(row_dir.glob("rep_*.csv")) if row_dir.exists() else []


def load_replication_rows(row_dir, output_path=None):
    """Load row-level checkpoints and optional aggregate table."""
    frames = []
    if output_path:
        table = load_table(output_path, deduplicate="rep")
        if len(table):
            frames.append(table)
    for path in _row_files(row_dir):
        if path.stat().st_size:
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    return (
        table.drop_duplicates(subset=["rep"], keep="last")
        .sort_values("rep")
        .reset_index(drop=True)
    )


def write_aggregate(row_dir, output_path):
    """Rebuild the sorted aggregate CSV from per-replication row files."""
    table = load_replication_rows(row_dir, output_path=output_path)
    save_table(table, output_path)
    return table


def completed_replications(row_dir, output_path=None, retry_failed=False):
    """Return replication indices that should not be scheduled again."""
    table = load_replication_rows(row_dir, output_path)
    if not len(table):
        return set()
    if retry_failed and "success" in table:
        table = table.loc[table["success"].astype(bool)]
    return set(table["rep"].astype(int))


def _run_one_replication(task):
    """Worker function for one model-wise IOS bootstrap replication."""
    (
        model_name, params_hat, data, run_dir, rep, seed,
        bootstrap_maxiter, ios_maxiter, resume,
    ) = task
    run_dir = Path(run_dir)
    row_dir = run_dir / f"{model_name.lower()}_modelwise_ios_rows"
    details_dir = run_dir / f"{model_name.lower()}_ios_details"
    row_path = row_dir / f"rep_{rep:04d}.csv"
    detail_path = details_dir / f"rep_{rep:04d}.csv"
    row_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    row = {
        "rep": int(rep),
        "seed": int(seed + rep),
        "model": model_name,
        "success": False,
        "fit_success": False,
        "ios_success": False,
        "convergence": np.nan,
        "error": "",
    }
    try:
        init_state = np.asarray(data[0], dtype=float)
        boot_data = simulate_partial_data(
            params_hat, data.shape[0] + 1,
            init_state=init_state, seed=seed + rep,
        )
        free_start = extract_free_params(params_hat, model_name)
        params_star, nll, conv = estimate_model(
            model_name, boot_data, start=free_start,
            maxiter=bootstrap_maxiter, verbose=False,
        )
        row.update(parameter_row(model_name, params_star, nll))
        row["convergence"] = int(conv)
        row["fit_success"] = bool(
            np.isfinite(nll)
            and nll < config.PENALTY
            and constraints_valid(
                extract_free_params(params_star, model_name),
                model_name,
                boot_data,
            )
        )

        _, q_min_global = diffusion_minimum(params_star)
        _, q_min_observed = safe_diffusion_rectangle_minimum(params_star, boot_data)
        q_path = diffusion_variance(boot_data[:, 0], boot_data[:, 1], params_star)
        row.update({
            "q_min_global": float(q_min_global),
            "q_min_observed": float(q_min_observed),
            "q_min_path": float(np.min(q_path)),
            "q_median_path": float(np.median(q_path)),
        })

        if row["fit_success"]:
            ios_table = run_exact_ios(
                model_name, boot_data, params_star, detail_path,
                maxiter=ios_maxiter, save_every=1,
                resume=resume, verbose=False,
            )
            ios_summary = summarize_ios(
                ios_table, expected_n_transitions=len(boot_data) - 1,
            )
            row.update({f"ios_{key}": value for key, value in ios_summary.items()})
            row["ios_success"] = bool(ios_summary["formal_T_N_complete"])
            row["success"] = bool(row["fit_success"] and row["ios_success"])
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

    row["time_sec"] = time.perf_counter() - t0
    save_table(pd.DataFrame([row]), row_path)
    return row


def main():
    """Run or resume a parallel model-wise IOS bootstrap."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-run", default="m4_real_data_cholesky")
    parser.add_argument("--run-name", default="m4_modelwise_ios_bootstrap")
    parser.add_argument("--model", choices=["M4"], default="M4")
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=config.BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-maxiter", type=int, default=config.BOOTSTRAP_MAXITER)
    parser.add_argument("--ios-maxiter", type=int, default=config.IOS_MAXITER)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    _, _, _, data = load_real_data()
    fits = load_model_fits(args.fit_run, required=True, data=data)
    params = fits.loc[
        fits["model"] == args.model, PARAM_NAMES
    ].iloc[0].to_numpy(dtype=float)

    output_path = result_path(
        "modelwise_ios_bootstrap", args.model, run_name=args.run_name,
    )
    context = checkpoint_context(
        "modelwise_ios_bootstrap", args.model, params, data,
        seed=args.seed, bootstrap_maxiter=args.bootstrap_maxiter,
        ios_maxiter=args.ios_maxiter,
    )
    prepare_checkpoint(output_path, context, resume=not args.no_resume)

    run_dir = config.run_dir(args.run_name)
    row_dir = run_dir / f"{args.model.lower()}_modelwise_ios_rows"
    done = completed_replications(
        row_dir, output_path=output_path, retry_failed=args.retry_failed,
    )
    todo = [rep for rep in range(args.n_boot) if rep not in done]
    if not todo:
        table = write_aggregate(row_dir, output_path)
        print(f"{int(table['success'].astype(bool).sum())} / {args.n_boot} successful")
        return

    tasks = [
        (
            args.model, params, data, run_dir, rep, args.seed,
            args.bootstrap_maxiter, args.ios_maxiter, not args.no_resume,
        )
        for rep in todo
    ]
    print(
        f"Scheduling {len(tasks)} remaining {args.model} model-wise IOS "
        f"bootstrap replications with {args.n_workers} workers."
    )
    completed = 0
    with futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        future_map = {
            executor.submit(_run_one_replication, task): task[4]
            for task in tasks
        }
        for future in futures.as_completed(future_map):
            rep = future_map[future]
            completed += 1
            row = future.result()
            table = write_aggregate(row_dir, output_path)
            n_success = int(table["success"].astype(bool).sum())
            t_n = row.get("ios_T_N", np.nan)
            print(
                f"{completed}/{len(tasks)} finished | rep={rep:04d} | "
                f"success={bool(row.get('success', False))} | "
                f"T_N={t_n:.6g} | total_success={n_success}"
            )


if __name__ == "__main__":
    main()
