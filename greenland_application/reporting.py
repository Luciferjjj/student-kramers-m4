"""
Build the versioned result snapshot and figures used by the research report.

The report distinguishes two evidence classes:

``current``
    Results produced with the globally feasible Cholesky M4 optimizer.

``development``
    Earlier recovery, discrimination, and M3-null bootstrap experiments run
    with the direct-coefficient M4 optimizer. These results document the
    research path but are not formal evidence for the current M4 fit.

The compact snapshot under ``docs/results`` is tracked by Git so collaborators
can inspect the numerical evidence without downloading local checkpoints.
"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import config
from .data_loading import load_real_data
from .figures import (
    plot_discrimination,
    plot_ios_numerical_diagnostics,
    plot_ios_overview,
    plot_ios_phase_space,
    plot_m4_diffusion_audit,
    plot_m4_parameter_distributions,
    plot_m4_parametric_bootstrap_diffusion,
    plot_m4_parametric_bootstrap_parameters,
    plot_modelwise_ios_bootstrap,
    plot_nested_bootstrap,
    plot_predictive_densities,
    plot_predictive_percentiles,
    plot_real_data_mechanisms,
    plot_real_data_state_space,
    plot_recovery_study,
    plot_transition_improvement,
    plot_waiting_time_comparison,
)
from student_kramers.discrimination import M4_EFFECT_SCALES
from student_kramers.models import PARAM_NAMES
from student_kramers.simulation import compute_derived_params


DOCS_DIR = config.PROJECT_DIR / "docs"
SNAPSHOT_DIR = DOCS_DIR / "results"
CURRENT_DIR = SNAPSHOT_DIR / "current"
DEVELOPMENT_DIR = SNAPSHOT_DIR / "development"
FIGURE_DIR = DOCS_DIR / "figures"
ARCHIVE_RUN = (
    config.PROJECT_DIR
    / "_local_archive"
    / "results_history"
    / "runs"
    / "pre_ios_validation"
)


CURRENT_SOURCES = {
    "model_fits.csv": "results/runs/m4_real_data_cholesky/model_fits.csv",
    "model_fits.csv.meta.json": (
        "results/runs/m4_real_data_cholesky/model_fits.csv.meta.json"
    ),
    "q_min_audit.csv": "results/runs/m4_report_current/q_min_audit.csv",
    "transition_improvement.csv": (
        "results/runs/m4_report_current/transition_improvement.csv"
    ),
    "predictive_summary.csv": "results/runs/m4_report_current/predictive_summary.csv",
    "predictive_behavior.csv": "results/runs/m4_report_current/predictive_behavior.csv",
    "predictive_density.csv": "results/runs/m4_report_current/predictive_density.csv",
    "predictive_waiting.csv": "results/runs/m4_report_current/predictive_waiting.csv",
    "m4_parametric_bootstrap.csv": (
        "results/runs/m4_parametric_bootstrap/m4_parametric_bootstrap.csv"
    ),
    "m4_parametric_bootstrap_overview.csv": (
        "results/runs/m4_parametric_bootstrap/m4_parametric_bootstrap_overview.csv"
    ),
    "m4_parametric_bootstrap_parameters.csv": (
        "results/runs/m4_parametric_bootstrap/m4_parametric_bootstrap_parameters.csv"
    ),
    "m4_parametric_bootstrap_diffusion.csv": (
        "results/runs/m4_parametric_bootstrap/m4_parametric_bootstrap_diffusion.csv"
    ),
    "m4_parametric_bootstrap_diffusion_summary.csv": (
        "results/runs/m4_parametric_bootstrap/"
        "m4_parametric_bootstrap_diffusion_summary.csv"
    ),
    "m4_parametric_bootstrap_path_band.csv": (
        "results/runs/m4_parametric_bootstrap/m4_parametric_bootstrap_path_band.csv"
    ),
    "ios_comparison.csv": "results/runs/ios_observed/ios_comparison.csv",
    "ios_pairwise_comparison.csv": (
        "results/runs/ios_observed/ios_pairwise_comparison.csv"
    ),
    "ios_regime_summary.csv": "results/runs/ios_observed/ios_regime_summary.csv",
    "ios_transitions.csv": "results/runs/ios_observed/ios_transitions.csv",
    "ios_parameter_shifts.csv": (
        "results/runs/ios_observed/ios_parameter_shifts.csv"
    ),
    "ios_validation.csv": "results/runs/ios_observed/ios_validation.csv",
    "m4_modelwise_ios_bootstrap.csv": (
        "results/runs/m4_modelwise_ios_bootstrap/m4_modelwise_ios_bootstrap.csv"
    ),
    "m4_modelwise_ios_bootstrap_summary.csv": (
        "results/runs/m4_modelwise_ios_bootstrap/"
        "m4_modelwise_ios_bootstrap_summary.csv"
    ),
    "m4_modelwise_ios_bootstrap_cumulative.csv": (
        "results/runs/m4_modelwise_ios_bootstrap/"
        "m4_modelwise_ios_bootstrap_cumulative.csv"
    ),
    "m4_optimization_stability.csv": (
        "results/runs/ios_cholesky_audit_v2/m4_optimization_stability.csv"
    ),
}


DEVELOPMENT_SOURCES = {
    "m2_recovery_study.csv": "m2_recovery_study.csv",
    "m2_recovery_study.csv.meta.json": "m2_recovery_study.csv.meta.json",
    "m3_recovery_study.csv": "m3_recovery_study.csv",
    "m3_recovery_study.csv.meta.json": "m3_recovery_study.csv.meta.json",
    "m4_recovery_study.csv": "m4_recovery_study.csv",
    "m4_recovery_study.csv.meta.json": "m4_recovery_study.csv.meta.json",
    "discrimination_m3.csv": "discrimination_m3.csv",
    "discrimination_m3.csv.meta.json": "discrimination_m3.csv.meta.json",
    "discrimination_weak.csv": "discrimination_weak.csv",
    "discrimination_weak.csv.meta.json": "discrimination_weak.csv.meta.json",
    "discrimination_moderate.csv": "discrimination_moderate.csv",
    "discrimination_moderate.csv.meta.json": "discrimination_moderate.csv.meta.json",
    "discrimination_strong.csv": "discrimination_strong.csv",
    "discrimination_strong.csv.meta.json": "discrimination_strong.csv.meta.json",
    "m3_m4_nested_bootstrap.csv": "m3_m4_nested_bootstrap.csv",
    "m3_m4_nested_bootstrap.csv.meta.json": "m3_m4_nested_bootstrap.csv.meta.json",
    "m3_m4_nested_summary.csv": "m3_m4_nested_summary.csv",
}


def _copy_required(source, target):
    """Copy one required report input and fail with a useful path."""
    if not source.exists():
        raise FileNotFoundError(f"Missing report input: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _predictive_comparison(summary, behavior):
    """Return observed values and simulation intervals for selected checks."""
    combined = summary.merge(
        behavior, on=["model", "source", "rep"], how="outer",
    )
    metrics = (
        "X_sd",
        "Vhat_sd",
        "lower_occupancy",
        "n_switches",
        "lower_wait_median",
        "upper_wait_median",
    )
    rows = []
    for model in ("M2", "M3", "M4"):
        selected = combined[combined["model"] == model]
        observed = selected[selected["source"] == "observed"].iloc[0]
        simulated = selected[selected["source"] == "simulated"]
        for metric in metrics:
            values = simulated[metric].dropna().to_numpy(dtype=float)
            rows.append({
                "model": model,
                "metric": metric,
                "observed": float(observed[metric]),
                "simulated_q025": float(np.quantile(values, 0.025)),
                "simulated_median": float(np.median(values)),
                "simulated_q975": float(np.quantile(values, 0.975)),
                "observed_percentile": float(np.mean(values <= observed[metric])),
            })
    return pd.DataFrame(rows)


def _derived_bootstrap_summary(table, observed_params):
    """Summarize interpretable potential and diffusion quantities."""
    successful = table[table["success"].astype(bool)]
    metrics = (
        "lower_well",
        "barrier",
        "upper_well",
        "Delta_U_lower",
        "Delta_U_upper",
        "q_min",
        "q_min_x",
        "q_min_v",
        "nu_proxy",
    )
    observed = compute_derived_params(observed_params)
    values = {
        metric: [] for metric in metrics
    }
    for _, row in successful.iterrows():
        derived = compute_derived_params(row[PARAM_NAMES].to_numpy(dtype=float))
        for metric in metrics:
            values[metric].append(derived[metric])

    rows = []
    for metric in metrics:
        array = np.asarray(values[metric], dtype=float)
        array = array[np.isfinite(array)]
        rows.append({
            "metric": metric,
            "observed": float(observed[metric]),
            "n_finite": int(len(array)),
            "q025": float(np.quantile(array, 0.025)) if len(array) else np.nan,
            "median": float(np.median(array)) if len(array) else np.nan,
            "q975": float(np.quantile(array, 0.975)) if len(array) else np.nan,
        })
    return pd.DataFrame(rows)


def refresh_report_snapshot():
    """Refresh the Git-trackable report tables from local result checkpoints."""
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    DEVELOPMENT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for target_name, relative_source in CURRENT_SOURCES.items():
        source = config.PROJECT_DIR / relative_source
        target = CURRENT_DIR / target_name
        _copy_required(source, target)
        manifest.append({
            "evidence_class": "current",
            "status": "current Cholesky optimizer",
            "file": f"current/{target_name}",
            "source": relative_source,
        })

    for target_name, source_name in DEVELOPMENT_SOURCES.items():
        source = ARCHIVE_RUN / source_name
        target = DEVELOPMENT_DIR / target_name
        _copy_required(source, target)
        manifest.append({
            "evidence_class": "development",
            "status": "pre-Cholesky optimizer; rerun required for formal use",
            "file": f"development/{target_name}",
            "source": str(source.relative_to(config.PROJECT_DIR)),
        })

    fits = pd.read_csv(CURRENT_DIR / "model_fits.csv")
    m4_params = fits.loc[
        fits["model"] == "M4", PARAM_NAMES
    ].iloc[0].to_numpy(dtype=float)
    bootstrap = pd.read_csv(CURRENT_DIR / "m4_parametric_bootstrap.csv")
    _derived_bootstrap_summary(bootstrap, m4_params).to_csv(
        CURRENT_DIR / "m4_bootstrap_derived_summary.csv", index=False,
    )

    predictive_summary = pd.read_csv(CURRENT_DIR / "predictive_summary.csv")
    predictive_behavior = pd.read_csv(CURRENT_DIR / "predictive_behavior.csv")
    _predictive_comparison(
        predictive_summary, predictive_behavior,
    ).to_csv(CURRENT_DIR / "predictive_comparison_summary.csv", index=False)

    pd.DataFrame(manifest).to_csv(SNAPSHOT_DIR / "manifest.csv", index=False)


def build_report_figures():
    """Rebuild all figures referenced by the research report."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf"):
        for path in FIGURE_DIR.glob(pattern):
            path.unlink()
    _, age, x, data = load_real_data()
    fits = pd.read_csv(CURRENT_DIR / "model_fits.csv")
    m4_params = fits.loc[
        fits["model"] == "M4", PARAM_NAMES
    ].iloc[0].to_numpy(dtype=float)

    plot_real_data_state_space(
        age, x, data, fits, FIGURE_DIR / "data_and_state_space.png",
    )
    plot_real_data_mechanisms(
        fits, data, FIGURE_DIR / "real_data_mechanisms.png",
    )
    plot_transition_improvement(
        pd.read_csv(CURRENT_DIR / "transition_improvement.csv"),
        FIGURE_DIR / "transition_improvement.png",
    )
    plot_m4_diffusion_audit(
        fits,
        data,
        pd.read_csv(CURRENT_DIR / "q_min_audit.csv"),
        FIGURE_DIR / "diffusion_domain_audit.png",
    )

    predictive_summary = pd.read_csv(CURRENT_DIR / "predictive_summary.csv")
    predictive_behavior = pd.read_csv(CURRENT_DIR / "predictive_behavior.csv")
    plot_predictive_densities(
        pd.read_csv(CURRENT_DIR / "predictive_density.csv"),
        FIGURE_DIR / "predictive_density_bands.png",
    )
    plot_predictive_percentiles(
        predictive_summary,
        predictive_behavior,
        FIGURE_DIR / "predictive_check_map.png",
    )
    plot_waiting_time_comparison(
        pd.read_csv(CURRENT_DIR / "predictive_waiting.csv"),
        FIGURE_DIR / "predictive_waiting_times.png",
    )

    recovery = []
    for model in ("M2", "M3", "M4"):
        table = pd.read_csv(DEVELOPMENT_DIR / f"{model.lower()}_recovery_study.csv")
        table["model"] = model
        recovery.append(table)
    plot_recovery_study(
        pd.concat(recovery, ignore_index=True),
        config.REFERENCE_PARAMS_BY_MODEL,
        FIGURE_DIR / "development_recovery.png",
        title="Development recovery study (pre-Cholesky M4 optimizer)",
    )

    discrimination = []
    for truth in ("M3", *M4_EFFECT_SCALES):
        discrimination.append(
            pd.read_csv(DEVELOPMENT_DIR / f"discrimination_{truth.lower()}.csv")
        )
    old_nested_summary = pd.read_csv(
        DEVELOPMENT_DIR / "m3_m4_nested_summary.csv"
    ).iloc[0]
    old_contrast = float(old_nested_summary["observed_contrast"])
    plot_discrimination(
        pd.concat(discrimination, ignore_index=True),
        old_contrast,
        FIGURE_DIR / "development_discrimination.png",
        title="Development discrimination pilot (pre-Cholesky M4 optimizer)",
    )
    plot_nested_bootstrap(
        pd.read_csv(DEVELOPMENT_DIR / "m3_m4_nested_bootstrap.csv"),
        old_contrast,
        FIGURE_DIR / "development_nested_bootstrap.png",
        title="Development M3-null bootstrap (pre-Cholesky M4 optimizer)",
        note="Historical diagnostic only; not valid for the current M4 fit",
    )

    parameter_table = pd.read_csv(
        CURRENT_DIR / "m4_parametric_bootstrap_parameters.csv"
    )
    bootstrap = pd.read_csv(CURRENT_DIR / "m4_parametric_bootstrap.csv")
    plot_m4_parameter_distributions(
        bootstrap,
        m4_params,
        FIGURE_DIR / "m4_parameter_distributions.png",
    )
    plot_m4_parametric_bootstrap_parameters(
        parameter_table,
        FIGURE_DIR / "m4_parameter_intervals.png",
    )
    plot_m4_parametric_bootstrap_diffusion(
        pd.read_csv(CURRENT_DIR / "m4_parametric_bootstrap_overview.csv"),
        pd.read_csv(CURRENT_DIR / "m4_parametric_bootstrap_diffusion.csv"),
        pd.read_csv(
            CURRENT_DIR / "m4_parametric_bootstrap_diffusion_summary.csv"
        ),
        pd.read_csv(CURRENT_DIR / "m4_parametric_bootstrap_path_band.csv"),
        FIGURE_DIR / "m4_diffusion_bootstrap.png",
    )

    ios_summary = pd.read_csv(CURRENT_DIR / "ios_comparison.csv")
    ios_transitions = pd.read_csv(CURRENT_DIR / "ios_transitions.csv")
    plot_ios_overview(
        ios_summary, ios_transitions, FIGURE_DIR / "observed_exact_ios.png",
    )
    plot_ios_phase_space(
        ios_transitions, FIGURE_DIR / "observed_ios_phase_space.png",
    )
    plot_ios_numerical_diagnostics(
        pd.read_csv(CURRENT_DIR / "ios_parameter_shifts.csv"),
        ios_transitions,
        FIGURE_DIR / "ios_numerical_diagnostics.png",
    )
    plot_modelwise_ios_bootstrap(
        pd.read_csv(CURRENT_DIR / "m4_modelwise_ios_bootstrap_summary.csv"),
        pd.read_csv(CURRENT_DIR / "m4_modelwise_ios_bootstrap_cumulative.csv"),
        pd.read_csv(CURRENT_DIR / "m4_modelwise_ios_bootstrap.csv"),
        FIGURE_DIR / "m4_modelwise_ios_bootstrap.png",
    )
    for path in FIGURE_DIR.glob("*.pdf"):
        path.unlink()
