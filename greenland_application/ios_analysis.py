"""
ios_analysis.py - Saved-table analysis for observed Greenland exact IOS

The numerical IOS computation remains in ``bootstrap.py``.  This module reads
the transition-level checkpoints and constructs model-comparison tables for
the observed data.  A sampled pilot can use the same functions, but its sum is
explicitly labelled diagnostic rather than the formal statistic.
"""
import numpy as np
import pandas as pd

from student_kramers.bootstrap import summarize_ios
from student_kramers.models import PARAM_NAMES, diffusion_augmented_matrix, free_names


IOS_REFERENCE = {
    "M2": 3.0 + 9.0/4.0*3.0,
    "M3": 5.0 + 9.0/4.0*3.0,
    "M4": 5.0 + 9.0/4.0*6.0,
}


def build_ios_summary(tables, expected_n_transitions, fits=None):
    """Return one comparable diagnostic row per model."""
    fit_index = fits.set_index("model") if fits is not None else None
    rows = []
    for model_name, table in tables.items():
        summary = summarize_ios(table, expected_n_transitions)
        valid = table.loc[table["loo_valid"].astype(bool)].copy()
        reference = IOS_REFERENCE[model_name]
        formal = bool(summary["formal_T_N_complete"])
        positive = np.sort(np.maximum(
            valid["ios_contribution"].to_numpy(dtype=float), 0.0,
        ))[::-1]
        positive_total = float(np.sum(positive))
        cumulative = np.cumsum(positive)/max(positive_total, 1e-12)
        n_for_80 = int(np.searchsorted(cumulative, 0.8) + 1)
        m4_constraint_eigenvalue = np.nan
        reference_applicable = True
        if model_name == "M4" and fit_index is not None:
            params = fit_index.loc[model_name, PARAM_NAMES].to_numpy(dtype=float)
            m4_constraint_eigenvalue = float(np.min(np.linalg.eigvalsh(
                diffusion_augmented_matrix(params),
            )))
            reference_applicable = m4_constraint_eigenvalue > 1e-6
        rows.append({
            "model": model_name,
            **summary,
            "asymptotic_reference": reference,
            "reference_applicable": reference_applicable,
            "m4_constraint_eigenvalue_min": m4_constraint_eigenvalue,
            "T_N_over_reference": (
                summary["T_N"]/reference
                if formal and reference_applicable else np.nan
            ),
            "sampled_sum_over_reference": summary["sampled_ios_sum"]/reference,
            "q95_ios_contribution": float(valid["ios_contribution"].quantile(0.95)),
            "q99_ios_contribution": float(valid["ios_contribution"].quantile(0.99)),
            "positive_ios_sum": positive_total,
            "negative_ios_sum": float(np.minimum(
                valid["ios_contribution"].to_numpy(dtype=float), 0.0,
            ).sum()),
            "n_negative_contributions": int(
                (valid["ios_contribution"] < 0.0).sum()
            ),
            "top_1_positive_share": float(positive[0]/max(positive_total, 1e-12)),
            "top_10_positive_share": float(
                positive[:10].sum()/max(positive_total, 1e-12)
            ),
            "fraction_transitions_for_80pct_positive": n_for_80/len(valid),
            "max_nit": int(valid["nit"].max()),
            "median_nit": float(valid["nit"].median()),
            "max_parameter_shift_relative": float(
                valid["parameter_shift_relative"].max()
            ),
            "median_parameter_shift_relative": float(
                valid["parameter_shift_relative"].median()
            ),
            "n_non_warm_selected": int(
                (valid["start_kind"] != "full fit warm start").sum()
            ),
            "max_warm_start_excess_nll": float(
                valid["warm_start_excess_nll"].max()
            ),
            "median_seconds": float(valid["seconds"].median()),
            "max_seconds": float(valid["seconds"].max()),
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def build_ios_pairwise_comparison(transitions):
    """Summarize whether pairs of models identify the same influential transitions."""
    rows = []
    for left, right in (("M2", "M3"), ("M2", "M4"), ("M3", "M4")):
        x = transitions[f"{left.lower()}_ios_contribution"]
        y = transitions[f"{right.lower()}_ios_contribution"]
        valid = x.notna() & y.notna()
        x, y = x[valid], y[valid]
        n_top = min(20, len(x))
        top_x = set(x.nlargest(n_top).index)
        top_y = set(y.nlargest(n_top).index)
        rows.append({
            "left_model": left,
            "right_model": right,
            "n_transitions": int(valid.sum()),
            "pearson_correlation": float(x.corr(y, method="pearson")),
            "spearman_correlation": float(x.corr(y, method="spearman")),
            "right_minus_left_sum": float((y - x).sum()),
            "right_greater_fraction": float((y > x).mean()),
            "top_20_overlap": len(top_x & top_y),
        })
    return pd.DataFrame(rows)


def build_ios_regime_summary(transitions):
    """Split exact IOS by observed sign-switch and non-switch transitions."""
    rows = []
    for model_name in ("M2", "M3", "M4"):
        column = f"{model_name.lower()}_ios_contribution"
        for regime_switch, group in transitions.groupby("regime_switch"):
            values = group[column].dropna().to_numpy(dtype=float)
            rows.append({
                "model": model_name,
                "regime_switch": bool(regime_switch),
                "n_transitions": len(values),
                "ios_sum": float(values.sum()),
                "mean_ios": float(values.mean()),
                "median_ios": float(np.median(values)),
                "max_ios": float(values.max()),
            })
    return pd.DataFrame(rows)


def build_ios_transition_table(tables, data, age=None):
    """Join model IOS contributions to the observed transition states."""
    data = np.asarray(data, dtype=float)
    n_transitions = len(data) - 1
    age_values = (
        np.arange(n_transitions, dtype=float)
        if age is None else np.asarray(age, dtype=float)[:n_transitions]
    )
    out = pd.DataFrame({
        "k": np.arange(n_transitions),
        "age_ka": age_values,
        "X_old": data[:-1, 0],
        "X_new": data[1:, 0],
        "Vhat_old": data[:-1, 1],
        "Vhat_new": data[1:, 1],
    })
    out["regime_switch"] = out["X_old"]*out["X_new"] <= 0.0

    for model_name, table in tables.items():
        selected = table[[
            "k", "ios_contribution", "loo_valid", "nit", "seconds",
            "parameter_shift_relative", "start_kind", "warm_start_excess_nll",
        ]].copy()
        selected = selected.rename(columns={
            column: f"{model_name.lower()}_{column}"
            for column in selected.columns if column != "k"
        })
        out = out.merge(selected, on="k", how="left")
    return out


def build_ios_parameter_shift_table(tables, fits):
    """Summarize leave-one-out parameter movement by model and parameter."""
    fit_index = fits.set_index("model")
    rows = []
    for model_name, table in tables.items():
        valid = table.loc[table["loo_valid"].astype(bool)].copy()
        for name in free_names(model_name):
            full_value = float(fit_index.loc[model_name, name])
            shift = valid[name].to_numpy(dtype=float) - full_value
            relative = shift/max(abs(full_value), 1.0)
            rows.append({
                "model": model_name,
                "parameter": name,
                "full_value": full_value,
                "median_shift": float(np.median(shift)),
                "q95_abs_shift": float(np.quantile(np.abs(shift), 0.95)),
                "max_abs_shift": float(np.max(np.abs(shift))),
                "q95_abs_relative_shift": float(
                    np.quantile(np.abs(relative), 0.95)
                ),
                "max_abs_relative_shift": float(np.max(np.abs(relative))),
            })
    return pd.DataFrame(rows)


def validate_ios_tables(tables, expected_indices):
    """Return explicit completeness and validity checks for saved IOS tables."""
    expected = set(map(int, expected_indices))
    rows = []
    for model_name, table in tables.items():
        found = set(table["k"].astype(int))
        valid = set(table.loc[table["loo_valid"].astype(bool), "k"].astype(int))
        rows.append({
            "model": model_name,
            "n_expected": len(expected),
            "n_rows": len(table),
            "n_valid": len(valid),
            "n_missing": len(expected - found),
            "n_invalid": len(found - valid),
            "unexpected_rows": len(found - expected),
            "complete_and_valid": expected == valid,
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
