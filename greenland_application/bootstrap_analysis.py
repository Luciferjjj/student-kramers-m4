"""Analysis helpers for saved Greenland parametric bootstrap runs."""
import numpy as np
import pandas as pd

from . import config
from student_kramers.models import (
    PARAM_NAMES,
    diffusion_augmented_matrix,
    diffusion_minimum,
    diffusion_quadratic_matrix,
    diffusion_rectangle_bounds,
    diffusion_rectangle_minimum,
    diffusion_variance,
)


def successful_bootstrap_rows(table):
    """Return successful bootstrap rows only."""
    if "success" not in table:
        return table.iloc[0:0].copy()
    return table.loc[table["success"].astype(bool)].copy()


def build_parametric_bootstrap_overview(table):
    """Summarize convergence rate, runtime, and fitted NLL distribution."""
    success = successful_bootstrap_rows(table)
    n_total = int(len(table))
    n_success = int(len(success))
    return pd.DataFrame([{
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_total - n_success,
        "success_rate": n_success/max(n_total, 1),
        "median_time_sec": float(table["time_sec"].median()),
        "total_time_sec": float(table["time_sec"].sum()),
        "nll_mean": float(success["nll"].mean()),
        "nll_sd": float(success["nll"].std(ddof=1)),
        "nll_q025": float(success["nll"].quantile(0.025)),
        "nll_q50": float(success["nll"].quantile(0.5)),
        "nll_q975": float(success["nll"].quantile(0.975)),
    }])


def build_parameter_bootstrap_summary(table, observed_params):
    """Return bootstrap intervals for every model parameter."""
    success = successful_bootstrap_rows(table)
    observed_params = np.asarray(observed_params, dtype=float)
    rows = []
    for name, observed in zip(PARAM_NAMES, observed_params):
        values = success[name].to_numpy(dtype=float)
        rows.append({
            "parameter": name,
            "observed": float(observed),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "q025": float(np.quantile(values, 0.025)),
            "q50": float(np.quantile(values, 0.5)),
            "q975": float(np.quantile(values, 0.975)),
            "bias": float(np.mean(values) - observed),
            "relative_sd": float(np.std(values, ddof=1)/max(abs(observed), 1.0)),
        })
    return pd.DataFrame(rows)


def safe_diffusion_rectangle_minimum(params, data, margin=0.0):
    """Return rectangle minimum of q, tolerating nearly singular Q matrices."""
    params = np.asarray(params, dtype=float)
    lo, hi = diffusion_rectangle_bounds(data, margin)
    x_lo, v_lo = lo
    x_hi, v_hi = hi
    alpha, beta, _, delta, epsilon, zeta = params[5:11]
    candidates = [
        (x_lo, v_lo), (x_lo, v_hi), (x_hi, v_lo), (x_hi, v_hi),
    ]
    if alpha > 0.0:
        for x in (x_lo, x_hi):
            v_star = -(beta + epsilon*x)/(2.0*alpha)
            if v_lo <= v_star <= v_hi:
                candidates.append((x, v_star))
    if delta > 0.0:
        for v in (v_lo, v_hi):
            x_star = -(zeta + epsilon*v)/(2.0*delta)
            if x_lo <= x_star <= x_hi:
                candidates.append((x_star, v))
    Q = diffusion_quadratic_matrix(params)
    linear = np.array([zeta, beta], dtype=float)
    try:
        x_star, v_star = -0.5*np.linalg.solve(Q, linear)
    except np.linalg.LinAlgError:
        pass
    else:
        if x_lo <= x_star <= x_hi and v_lo <= v_star <= v_hi:
            candidates.append((x_star, v_star))

    values = np.array([
        diffusion_variance(x, v, params) for x, v in candidates
    ], dtype=float)
    index = int(np.argmin(values))
    return np.asarray(candidates[index], dtype=float), float(values[index])


def build_diffusion_bootstrap_diagnostics(table, observed_params, data):
    """Compute diffusion-shape diagnostics for each successful bootstrap fit."""
    success = successful_bootstrap_rows(table)
    data = np.asarray(data, dtype=float)
    rows = []
    for _, row in success.iterrows():
        params = row[PARAM_NAMES].to_numpy(dtype=float)
        _, q_min_global = diffusion_minimum(params)
        _, q_min_observed = safe_diffusion_rectangle_minimum(params, data)
        _, q_min_protected = safe_diffusion_rectangle_minimum(
            params, data, margin=config.M4_DIAGNOSTIC_MARGIN,
        )
        q_path = diffusion_variance(data[:, 0], data[:, 1], params)
        eig_min = float(np.min(np.linalg.eigvalsh(
            diffusion_augmented_matrix(params, floor=0.0),
        )))
        rows.append({
            "rep": int(row["rep"]),
            "q_min_global": float(q_min_global),
            "q_min_observed": float(q_min_observed),
            "q_min_protected": float(q_min_protected),
            "q_path_min": float(np.min(q_path)),
            "q_path_median": float(np.median(q_path)),
            "q_path_q05": float(np.quantile(q_path, 0.05)),
            "q_path_q95": float(np.quantile(q_path, 0.95)),
            "augmented_eig_min": eig_min,
            "tail_margin": float(2.0*params[0] - params[5]),
            "nll": float(row["nll"]),
        "time_sec": float(row["time_sec"]),
        })
    return pd.DataFrame(rows)


def summarize_diffusion_bootstrap(diagnostics, observed_params, data):
    """Summarize bootstrap diffusion diagnostics against the observed fit."""
    observed_params = np.asarray(observed_params, dtype=float)
    data = np.asarray(data, dtype=float)
    _, observed_global = diffusion_minimum(observed_params)
    _, observed_rectangle = diffusion_rectangle_minimum(observed_params, data)
    _, observed_protected = diffusion_rectangle_minimum(
        observed_params, data, margin=config.M4_DIAGNOSTIC_MARGIN,
    )
    q_observed_path = diffusion_variance(data[:, 0], data[:, 1], observed_params)
    metrics = {
        "q_min_global": observed_global,
        "q_min_observed": observed_rectangle,
        "q_min_protected": observed_protected,
        "q_path_min": float(np.min(q_observed_path)),
        "q_path_median": float(np.median(q_observed_path)),
    }
    rows = []
    for name, observed in metrics.items():
        values = diagnostics[name].to_numpy(dtype=float)
        rows.append({
            "metric": name,
            "observed": float(observed),
            "q025": float(np.quantile(values, 0.025)),
            "q50": float(np.quantile(values, 0.5)),
            "q975": float(np.quantile(values, 0.975)),
            "observed_percentile": float(np.mean(values <= observed)),
        })
    return pd.DataFrame(rows)


def build_path_diffusion_band(table, observed_params, data, age):
    """Compute bootstrap bands for q(x_k, v_k) along the observed pseudo-path."""
    success = successful_bootstrap_rows(table)
    data = np.asarray(data, dtype=float)
    age = np.asarray(age, dtype=float)[:len(data)]
    observed_params = np.asarray(observed_params, dtype=float)
    q_observed = diffusion_variance(data[:, 0], data[:, 1], observed_params)
    q_values = np.vstack([
        diffusion_variance(
            data[:, 0], data[:, 1],
            row[PARAM_NAMES].to_numpy(dtype=float),
        )
        for _, row in success.iterrows()
    ])
    q_ratio = q_values/q_observed
    return pd.DataFrame({
        "k": np.arange(len(data)),
        "age_ka": age,
        "X": data[:, 0],
        "Vhat": data[:, 1],
        "q_observed_fit": q_observed,
        "q_boot_q025": np.quantile(q_values, 0.025, axis=0),
        "q_boot_q50": np.quantile(q_values, 0.5, axis=0),
        "q_boot_q975": np.quantile(q_values, 0.975, axis=0),
        "q_ratio_q025": np.quantile(q_ratio, 0.025, axis=0),
        "q_ratio_q50": np.quantile(q_ratio, 0.5, axis=0),
        "q_ratio_q975": np.quantile(q_ratio, 0.975, axis=0),
    })


def successful_modelwise_ios_rows(table):
    """Return bootstrap replications with complete refit and exact IOS."""
    if not len(table) or "success" not in table or "ios_T_N" not in table:
        return table.iloc[0:0].copy()
    return table.loc[
        table["success"].astype(bool) & table["ios_T_N"].notna()
    ].copy()


def build_modelwise_ios_bootstrap_summary(table, observed_T_N, model_name="M4"):
    """Summarize finite-sample IOS calibration from model-wise bootstrap rows."""
    success = successful_modelwise_ios_rows(table)
    values = success["ios_T_N"].to_numpy(dtype=float)
    n_total = int(len(table))
    n_success = int(len(success))
    if n_success == 0:
        return pd.DataFrame([{
            "model": model_name,
            "observed_T_N": float(observed_T_N),
            "n_total": n_total,
            "n_success": 0,
            "n_failed": n_total,
            "success_rate": 0.0,
            "mean": np.nan,
            "sd": np.nan,
            "q025": np.nan,
            "q50": np.nan,
            "q975": np.nan,
            "p_upper": np.nan,
            "p_lower": np.nan,
            "observed_percentile": np.nan,
            "median_time_sec": np.nan,
            "total_time_sec": float(table["time_sec"].sum()) if "time_sec" in table else 0.0,
            "median_ios_seconds": np.nan,
        }])
    return pd.DataFrame([{
        "model": model_name,
        "observed_T_N": float(observed_T_N),
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_total - n_success,
        "success_rate": n_success/max(n_total, 1),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if n_success > 1 else 0.0,
        "q025": float(np.quantile(values, 0.025)),
        "q50": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
        "p_upper": float((np.sum(values >= observed_T_N) + 1)/(n_success + 1)),
        "p_lower": float((np.sum(values <= observed_T_N) + 1)/(n_success + 1)),
        "observed_percentile": float(np.mean(values <= observed_T_N)),
        "median_time_sec": float(success["time_sec"].median()),
        "total_time_sec": float(table["time_sec"].sum()),
        "median_ios_seconds": float(success["ios_total_seconds"].median()),
    }])


def build_modelwise_ios_cumulative(table, observed_T_N):
    """Track bootstrap IOS tail probabilities as completed replications accrue."""
    success = successful_modelwise_ios_rows(table).sort_values("rep")
    values = success["ios_T_N"].to_numpy(dtype=float)
    reps = success["rep"].to_numpy(dtype=int)
    rows = []
    for index in range(1, len(values) + 1):
        prefix = values[:index]
        rows.append({
            "rep": int(reps[index - 1]),
            "n_success": index,
            "p_upper": float((np.sum(prefix >= observed_T_N) + 1)/(index + 1)),
            "p_lower": float((np.sum(prefix <= observed_T_N) + 1)/(index + 1)),
            "q50": float(np.quantile(prefix, 0.5)),
            "q95": float(np.quantile(prefix, 0.95)),
        })
    return pd.DataFrame(rows)
