"""
figures.py - Research figures for the unified M1-M4 Student Kramers workflow

The figures answer model-comparison questions rather than merely displaying
saved numbers.  Plotting stays separate from numerical experiments so the
notebook can load tables and call short, reusable functions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .models import (
    PARAM_LABELS,
    PARAM_NAMES,
    diffusion_minimum,
    diffusion_variance,
    potential,
)
from .simulation import potential_extrema


COLORS = {"M1": "#7A7A7A", "M2": "#D55E00", "M3": "#0072B2", "M4": "#009E73"}
LINESTYLES = {"M1": "--", "M2": "-", "M3": "-.", "M4": "-"}


def _style(ax):
    """Apply the same readable style to one axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#E5E5E5", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def _finish(fig, save_path=None, tight=True):
    """Save one figure as PNG and PDF, then return it."""
    if tight:
        fig.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def _fit_params(fits):
    """Return a model-to-parameter dictionary from a fit table."""
    return {
        str(row["model"]): row[PARAM_NAMES].to_numpy(dtype=float)
        for _, row in fits.iterrows()
    }


def plot_real_data_state_space(age, x_series, data, fits, save_path=None):
    """Show the fitted data, reconstructed velocity, and M4 diffusion change."""
    params = _fit_params(fits)
    age = np.asarray(age, dtype=float)
    x_series = np.asarray(x_series, dtype=float)
    data = np.asarray(data, dtype=float)
    q_ratio = (
        diffusion_variance(data[:, 0], data[:, 1], params["M4"])
        / diffusion_variance(data[:, 0], data[:, 1], params["M3"])
    )

    log_q_ratio = np.log10(q_ratio)

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(3, 2, width_ratios=[2.2, 1.0], hspace=0.24, wspace=0.25)
    ax_x = fig.add_subplot(grid[0, 0])
    ax_v = fig.add_subplot(grid[1, 0], sharex=ax_x)
    ax_q = fig.add_subplot(grid[2, 0], sharex=ax_x)
    ax_phase = fig.add_subplot(grid[:, 1])

    ax_x.plot(age, x_series, color="black", linewidth=0.9)
    ax_x.set_ylabel("Centered $-\\log(\\mathrm{Ca}^{2+})$")
    ax_x.set_title("(A) Observed climate coordinate", fontweight="bold")
    ax_x.tick_params(axis="x", labelbottom=False)
    _style(ax_x)

    ax_v.plot(age[:-1], data[:, 1], color="#4D4D4D", linewidth=0.75)
    ax_v.set_xlabel("")
    ax_v.set_ylabel(r"Reconstructed velocity $\widehat V$")
    ax_v.set_title("(B) Velocity used by the partial likelihood", fontweight="bold")
    ax_v.tick_params(axis="x", labelbottom=False)
    _style(ax_v)

    smooth_q_ratio = (
        pd.Series(log_q_ratio).rolling(51, center=True, min_periods=1).median()
        .to_numpy()
    )
    ax_q.plot(
        age[:-1], log_q_ratio, color="#8C8C8C", linewidth=0.35, alpha=0.28,
        label="transition-level adjustment",
    )
    ax_q.fill_between(
        age[:-1], 0.0, smooth_q_ratio, where=smooth_q_ratio >= 0.0,
        color="#D55E00", alpha=0.30, label="M4 diffusion larger than M3",
    )
    ax_q.fill_between(
        age[:-1], 0.0, smooth_q_ratio, where=smooth_q_ratio < 0.0,
        color="#0072B2", alpha=0.30, label="M4 diffusion smaller than M3",
    )
    ax_q.plot(age[:-1], smooth_q_ratio, color="#333333", linewidth=1.3)
    ax_q.axhline(0.0, color="black", linewidth=1.0)
    ax_q.set_xlabel("Age (ka before 2000 AD)")
    ax_q.set_ylabel(r"$\log_{10}\{q_{M4}/q_{M3}\}$")
    ax_q.set_title("(C) M4 diffusion adjustment along the observed path", fontweight="bold")
    ax_q.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax_q)

    limit = np.nanquantile(np.abs(log_q_ratio), 0.995)
    scatter = ax_phase.scatter(
        data[:, 0], data[:, 1], c=log_q_ratio, s=11, alpha=0.65,
        cmap="RdBu_r", vmin=-limit, vmax=limit,
    )
    fig.colorbar(scatter, ax=ax_phase, label=r"$\log_{10}\{q_{M4}/q_{M3}\}$")
    ax_phase.set_xlabel("Position $X$")
    ax_phase.set_ylabel(r"Reconstructed velocity $\widehat V$")
    ax_phase.set_title("(D) Observed phase space and M4 diffusion change", fontweight="bold")
    _style(ax_phase)

    fig.suptitle("Real data used by the M2/M3/M4 partial-observation likelihood",
                 fontweight="bold", fontsize=16, y=0.98)
    return _finish(fig, save_path)


def plot_real_data_mechanisms(fits, data, save_path=None):
    """Compare objective values, potential, and diffusion mechanisms."""
    params = _fit_params(fits)
    models = [model for model in ("M2", "M3", "M4") if model in params]
    x = data[:, 0]
    v = data[:, 1]
    x_grid = np.linspace(np.quantile(x, 0.005), np.quantile(x, 0.995), 400)
    v_grid = np.linspace(np.quantile(v, 0.005), np.quantile(v, 0.995), 400)
    x_slices = np.quantile(x, [0.1, 0.5, 0.9])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    metrics = ["nll", "aic", "bic"]
    offsets = [-0.22, 0.0, 0.22]
    for metric, offset in zip(metrics, offsets):
        values = fits.set_index("model").loc[models, metric]
        relative = values - values.min()
        ax.scatter(relative, np.arange(len(models)) + offset, s=70, label=metric.upper())
        for y, value in zip(np.arange(len(models)) + offset, relative):
            ax.text(value, y + 0.06, f"{value:.1f}", fontsize=8, ha="center")
    ax.set_yticks(np.arange(len(models)), models)
    ax.set_xlabel("Difference from the best value; smaller is better")
    ax.set_title("(A) Descriptive fit comparison", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    _style(ax)

    ax = axes[0, 1]
    for model in models:
        U = potential(x_grid, params[model])
        U = U - np.min(U)
        ax.plot(
            x_grid, U, color=COLORS[model], linestyle=LINESTYLES[model],
            linewidth=2.3, label=model,
        )
        for point in potential_extrema(params[model]):
            if np.isfinite(point) and x_grid.min() <= point <= x_grid.max():
                ax.axvline(point, color=COLORS[model], alpha=0.18, linewidth=1)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel(r"$U(x)-\min U$")
    ax.set_title("(B) Fitted potential and equilibria", fontweight="bold")
    ax.legend(frameon=False, ncol=len(models))
    _style(ax)

    ax = axes[1, 0]
    q_m3 = diffusion_variance(0.0, v_grid, params["M3"])
    for x_value, alpha in zip(x_slices, [0.65, 0.85, 1.0]):
        ax.plot(
            v_grid, diffusion_variance(x_value, v_grid, params["M4"])/q_m3,
            color=COLORS["M4"], linewidth=2.0, alpha=alpha,
            label=f"M4 at x={x_value:.2f}",
        )
    ax.axhline(1.0, color=COLORS["M3"], linestyle="--", linewidth=2.0, label="M3 reference")
    ax.set_xlabel(r"Reconstructed velocity $\widehat v$")
    ax.set_ylabel(r"Diffusion ratio $q_{M4}(x,\widehat v)/q_{M3}(\widehat v)$")
    ax.set_title("(C) Size of the M4 diffusion change", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[1, 1]
    xg = np.linspace(np.min(x), np.max(x), 180)
    vg = np.linspace(np.quantile(v, 0.002), np.quantile(v, 0.998), 180)
    X, V = np.meshgrid(xg, vg)
    ratio = diffusion_variance(X, V, params["M4"])/diffusion_variance(X, V, params["M3"])
    vmax = np.nanquantile(np.abs(np.log10(ratio)), 0.99)
    image = ax.pcolormesh(
        X, V, np.log10(ratio), shading="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
    )
    selected = np.linspace(0, len(data) - 1, 250, dtype=int)
    ax.scatter(x[selected], v[selected], s=7, c="black", alpha=0.35, label="observed states")
    fig.colorbar(image, ax=ax, label=r"$\log_{10}\{q_{M4}/q_{M3}\}$")
    ax.set_xlabel("Position $x$")
    ax.set_ylabel(r"Reconstructed velocity $\widehat v$")
    ax.set_title("(D) Where M4 changes diffusion on observed phase space", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    _style(ax)

    fig.suptitle("What changes when the real data are fitted with M2, M3, and M4",
                 fontweight="bold", fontsize=16, y=1.02)
    return _finish(fig, save_path)


def plot_transition_improvement(transitions, save_path=None):
    """Show which real-data transitions create the M4 likelihood improvement."""
    table = transitions.copy()
    gain = table["gain_m4_over_m3"].to_numpy(dtype=float)
    limit = float(np.quantile(np.abs(gain), 0.995))
    positive_share = float(np.mean(gain > 0.0))
    total_gain = float(np.sum(gain))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.plot(table["age_ka"], table["cumulative_gain"], color=COLORS["M4"], linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Age (ka before 2000 AD)")
    ax.set_ylabel(r"Cumulative $\mathrm{NLL}_{M3}-\mathrm{NLL}_{M4}$")
    ax.set_title("(A) Where the total M4 likelihood gain accumulates", fontweight="bold")
    ax.invert_xaxis()
    _style(ax)

    ax = axes[0, 1]
    selected = np.argsort(np.abs(gain))[-20:]
    colors = np.where(gain[selected] >= 0.0, COLORS["M4"], COLORS["M3"])
    ax.barh(np.arange(len(selected)), gain[selected], color=colors, alpha=0.8)
    ax.set_yticks(
        np.arange(len(selected)),
        [f"{age:.2f} ka" for age in table.iloc[selected]["age_ka"]],
        fontsize=8,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel(r"Transition gain $\mathrm{NLL}_{M3}-\mathrm{NLL}_{M4}$")
    ax.set_title("(B) Twenty transitions with largest absolute effect", fontweight="bold")
    _style(ax)

    ax = axes[1, 0]
    scatter = ax.scatter(
        table["x"], table["vhat"], c=gain, cmap="RdBu_r",
        vmin=-limit, vmax=limit, s=13, alpha=0.65,
    )
    fig.colorbar(scatter, ax=ax, label=r"$\mathrm{NLL}_{M3}-\mathrm{NLL}_{M4}$")
    ax.set_xlabel("Position $X$")
    ax.set_ylabel(r"Reconstructed velocity $\widehat V$")
    ax.set_title("(C) Transition-level gain in observed phase space", fontweight="bold")
    _style(ax)

    ax = axes[1, 1]
    ax.scatter(
        table["log10_q_ratio"], gain, s=12, alpha=0.40, color="#4D4D4D",
    )
    smooth = (
        table[["log10_q_ratio", "gain_m4_over_m3"]].sort_values("log10_q_ratio")
        .rolling(101, center=True, min_periods=25).mean()
    )
    ax.plot(
        smooth["log10_q_ratio"], smooth["gain_m4_over_m3"],
        color=COLORS["M4"], linewidth=2.2, label="rolling mean",
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(r"$\log_{10}\{q_{M4}/q_{M3}\}$")
    ax.set_ylabel(r"Transition gain $\mathrm{NLL}_{M3}-\mathrm{NLL}_{M4}$")
    ax.set_title("(D) Does changing diffusion improve transition fit?", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    fig.suptitle(
        f"Real-data transition audit: total M4 gain = {total_gain:.2f}; "
        f"M4 improves {100*positive_share:.1f}% of transitions",
        fontweight="bold", fontsize=16, y=1.02,
    )
    return _finish(fig, save_path)


def plot_m4_diffusion_audit(fits, data, q_audit, save_path=None):
    """Show why the small global M4 minimum is remote from observed states."""
    params = _fit_params(fits)
    m4 = params["M4"]
    point, q_min = diffusion_minimum(m4)
    x, v = data[:, 0], data[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    ax = axes[0]
    xg = np.linspace(np.min(x), np.max(x), 180)
    vg = np.linspace(np.quantile(v, 0.002), np.quantile(v, 0.998), 180)
    X, V = np.meshgrid(xg, vg)
    Q = diffusion_variance(X, V, m4)
    image = ax.contourf(X, V, np.log10(Q), levels=20, cmap="viridis")
    selected = np.linspace(0, len(data) - 1, 300, dtype=int)
    ax.scatter(x[selected], v[selected], s=7, color="white", alpha=0.55)
    fig.colorbar(image, ax=ax, label=r"$\log_{10} q_{M4}$")
    ax.set_title("(A) Diffusion in the observed region", fontweight="bold")
    ax.set_xlabel("Position $x$")
    ax.set_ylabel(r"Reconstructed velocity $\widehat v$")
    _style(ax)

    ax = axes[1]
    xg = np.linspace(point[0] - 5.0, np.max(x) + 2.0, 220)
    vg = np.linspace(point[1] - 8.0, np.max(v) + 5.0, 220)
    X, V = np.meshgrid(xg, vg)
    Q = diffusion_variance(X, V, m4)
    image = ax.contourf(X, V, np.log10(Q), levels=22, cmap="viridis")
    fig.colorbar(image, ax=ax, label=r"$\log_{10} q_{M4}$")
    ax.scatter(*point, s=80, marker="*", color="#D55E00", label=f"global minimum: {q_min:.1f}")
    ax.add_patch(plt.Rectangle(
        (np.min(x), np.min(v)), np.ptp(x), np.ptp(v),
        fill=False, edgecolor="black", linewidth=2, label="observed rectangle",
    ))
    ax.set_title("(B) Global minimum is far outside the data", fontweight="bold")
    ax.set_xlabel("Position $x$")
    ax.set_ylabel(r"Reconstructed velocity $\widehat v$")
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[2]
    audit = q_audit.set_index("model").loc[["M2", "M3", "M4"]]
    y = np.arange(3)
    for index, model in enumerate(("M2", "M3", "M4")):
        global_min = audit.loc[model, "q_min_global"]
        path_min = audit.loc[model, "q_path_min"]
        ax.plot([global_min, path_min], [index, index], color=COLORS[model], linewidth=3)
        ax.scatter(global_min, index, marker="*", s=140, color=COLORS[model],
                   label="global minimum" if index == 0 else None, zorder=3)
        ax.scatter(path_min, index, marker="o", s=70, facecolor="white",
                   edgecolor=COLORS[model], linewidth=2,
                   label="minimum on observed path" if index == 0 else None, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, ["M2", "M3", "M4"])
    ax.set_xlabel(r"Minimum squared diffusion $q$")
    ax.set_ylabel("")
    ax.set_title("(C) Global minimum versus data-supported minimum", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    m4_row = q_audit.loc[q_audit["model"] == "M4"].iloc[0]
    fig.suptitle(
        "M4 global minimum audit: "
        f"minimum is {m4_row['q_min_standardized_distance']:.1f} standardized units "
        "from the data center",
        fontweight="bold", fontsize=15, y=1.02,
    )
    return _finish(fig, save_path)


def plot_optimization_stability(stability, tolerance, fits, save_path=None):
    """Show whether the final M4 fit depends on one starting value."""
    m3_nll = float(fits.loc[fits["model"] == "M3", "nll"].iloc[0])
    m4_nll = float(fits.loc[fits["model"] == "M4", "nll"].iloc[0])
    kinds = ["M3 boundary", "global interior", "final M4 warm start"]
    palette = {
        "M3 boundary": COLORS["M3"],
        "global interior": "#CC79A7",
        "final M4 warm start": COLORS["M4"],
    }
    valid = stability[np.isfinite(stability["nll"])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    sns.stripplot(
        data=valid, x="start_kind", y="nll", hue="seed", order=kinds,
        dodge=True, jitter=0.12, size=6, ax=ax,
    )
    ax.axhline(m3_nll, color=COLORS["M3"], linestyle="--", label="M3 boundary NLL")
    ax.axhline(m4_nll, color=COLORS["M4"], linestyle="-", label="final M4 NLL")
    ax.set_xlabel("")
    ax.set_ylabel("Final NLL")
    ax.set_title("(A) Result from every M4 start", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[0, 1]
    for kind in kinds:
        group = valid[valid["start_kind"] == kind]
        ax.scatter(
            group["q_min_global"], group["nll"], s=45, alpha=0.75,
            color=palette[kind], label=kind,
        )
    ax.set_xscale("log")
    ax.axhline(m4_nll, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Global minimum $q$")
    ax.set_ylabel("Final NLL")
    ax.set_title("(B) Fit quality versus global diffusion margin", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    new_params = ["delta", "epsilon", "zeta"]
    long = valid.melt(
        id_vars=["start_kind"], value_vars=new_params,
        var_name="parameter", value_name="estimate",
    )
    ax = axes[1, 0]
    sns.stripplot(
        data=long, x="parameter", y="estimate", hue="start_kind",
        palette=palette, dodge=True, jitter=0.12, size=5, ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(new_params)), [PARAM_LABELS[name] for name in new_params])
    ax.set_xlabel("")
    ax.set_ylabel("Final estimate")
    ax.set_title("(C) New M4 coefficients across starts", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1, 1]
    ax.axis("off")
    max_nll_change = float(np.max(np.abs(tolerance["nll"] - m4_nll)))
    max_param_distance = float(np.max(tolerance["distance_to_final_m4"]))
    q_range = float(np.ptp(tolerance["q_min_global"]))
    ax.text(
        0.5, 0.76, f"{len(tolerance)}/{len(tolerance)} settings reproduced the final M4 fit",
        ha="center", va="center", fontsize=15, fontweight="bold",
        transform=ax.transAxes,
    )
    audit_lines = [
        rf"maximum $|\Delta\,\mathrm{{NLL}}|$ = {max_nll_change:.2g}",
        rf"maximum parameter distance = {max_param_distance:.2g}",
        rf"range of global minimum $q$ = {q_range:.2g}",
        "tolerances: " + ", ".join(f"{value:.0e}" for value in tolerance["tol"]),
    ]
    for index, line in enumerate(audit_lines):
        ax.text(
            0.5, 0.57 - 0.13*index, line, ha="center", va="center",
            fontsize=12, transform=ax.transAxes,
        )
    ax.set_title("(D) Sensitivity to optimizer tolerance", fontweight="bold")

    fig.suptitle("M4 optimization stability", fontweight="bold", fontsize=16, y=1.02)
    return _finish(fig, save_path)


def plot_predictive_densities(density, save_path=None):
    """Compare observed marginal densities with simulated prediction bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, variable, label in zip(
        axes, ["X", "Vhat"], ["Position $X$", r"Reconstructed velocity $\widehat V$"],
    ):
        selected = density[density["variable"] == variable]
        first = selected[selected["model"] == "M2"]
        ax.plot(first["value"], first["observed_density"], color="black", linewidth=2.8,
                label="observed")
        for model in ("M2", "M3", "M4"):
            group = selected[selected["model"] == model]
            ax.fill_between(
                group["value"], group["simulated_q025"], group["simulated_q975"],
                color=COLORS[model], alpha=0.10,
            )
            ax.plot(
                group["value"], group["simulated_median"],
                color=COLORS[model], linestyle=LINESTYLES[model],
                linewidth=2.0, label=f"{model} simulated median",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"({'A' if variable == 'X' else 'B'}) Marginal density of {label}",
                     fontweight="bold")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Observed data against model-based density prediction bands",
                 fontweight="bold", fontsize=15, y=1.02)
    return _finish(fig, save_path)


def plot_predictive_percentiles(summary, behavior, save_path=None):
    """Show the percentile of every observed statistic under each fitted model."""
    metrics = [
        "X_mean", "X_sd", "X_q05", "X_q95", "Vhat_sd",
        "lower_occupancy", "n_switches", "lower_wait_median", "upper_wait_median",
    ]
    labels = {
        "X_mean": "mean X", "X_sd": "SD X", "X_q05": "5% X", "X_q95": "95% X",
        "Vhat_sd": "SD Vhat", "lower_occupancy": "lower occupancy",
        "n_switches": "number of switches", "lower_wait_median": "lower median wait",
        "upper_wait_median": "upper median wait",
    }
    combined = summary.merge(
        behavior, on=["model", "source", "rep"], how="outer",
    )
    rows = []
    for model in ("M2", "M3", "M4"):
        group = combined[combined["model"] == model]
        for metric in metrics:
            observed = float(group.loc[group["source"] == "observed", metric].iloc[0])
            simulated = group.loc[group["source"] == "simulated", metric].dropna().to_numpy()
            rows.append({
                "model": model,
                "metric": labels[metric],
                "percentile": 100.0*np.mean(simulated <= observed),
            })
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(0, 2.5, color="#D55E00", alpha=0.12)
    ax.axvspan(97.5, 100, color="#D55E00", alpha=0.12)
    ax.axvspan(25, 75, color="#009E73", alpha=0.07)
    sns.scatterplot(
        data=table, x="percentile", y="metric", hue="model", style="model",
        palette=COLORS, s=90, ax=ax,
    )
    ax.axvline(50, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(-1, 101)
    ax.set_xlabel("Observed statistic percentile under fitted-model simulations")
    ax.set_ylabel("")
    ax.set_title(
        "Predictive-check map: values near 50% are well centered; tails indicate mismatch",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=3)
    _style(ax)
    return _finish(fig, save_path)


def plot_waiting_time_comparison(waiting, save_path=None):
    """Compare observed and simulated waiting-time distributions by regime."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, regime in zip(axes, ("lower", "upper")):
        selected = waiting[waiting["regime"] == regime]
        observed = selected[
            (selected["source"] == "observed") & (selected["model"] == "M2")
        ]["waiting_time_years"]
        ax.hist(observed, bins=14, density=True, color="black", alpha=0.25,
                label="observed")
        for model in ("M2", "M3", "M4"):
            values = selected[
                (selected["source"] == "simulated") & (selected["model"] == model)
            ]["waiting_time_years"]
            if len(values) > 1:
                sns.kdeplot(
                    values, color=COLORS[model], linestyle=LINESTYLES[model],
                    linewidth=2.2, label=model, ax=ax,
                )
        ax.set_xlim(left=0)
        ax.set_xlabel("Waiting time (years)")
        ax.set_ylabel("Density")
        ax.set_title(f"{regime.capitalize()}-regime occupancy time", fontweight="bold")
        ax.legend(frameon=False)
        _style(ax)
    fig.suptitle("Can the fitted models reproduce regime persistence?",
                 fontweight="bold", fontsize=15, y=1.02)
    return _finish(fig, save_path)


def plot_discrimination(discrimination, observed_contrast, save_path=None):
    """Show M3/M4 likelihood contrasts under each generating scenario."""
    valid = discrimination[discrimination["success"].astype(bool)].copy()
    order = ["M3", "weak", "moderate", "strong"]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.violinplot(
        data=valid, x="truth", y="contrast", order=order, inner=None,
        color="#B8D7E8", cut=0, ax=ax,
    )
    sns.stripplot(
        data=valid, x="truth", y="contrast", order=order,
        color="black", alpha=0.55, jitter=0.18, size=5, ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="equal NLL")
    ax.axhline(
        observed_contrast, color=COLORS["M4"], linewidth=2,
        label=f"real-data contrast = {observed_contrast:.1f}",
    )
    ax.set_xlabel("Generating scenario")
    ax.set_ylabel(r"$2\{\mathrm{NLL}_{M3}-\mathrm{NLL}_{M4}\}$")
    ax.set_title("M3-to-M4 likelihood contrast under each simulated truth",
                 fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)
    return _finish(fig, save_path)


def plot_nested_bootstrap(nested, observed_contrast, save_path=None):
    """Compare the real-data M3/M4 contrast with its M3-null bootstrap."""
    values = nested.loc[nested["success"].astype(bool), "contrast"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(values, bins=max(10, int(np.sqrt(len(values)))), density=True,
            color=COLORS["M3"], alpha=0.55, label="M3-null bootstrap")
    ax.axvline(
        observed_contrast, color=COLORS["M4"], linewidth=2.5, linestyle="--",
        label=f"observed = {observed_contrast:.1f}",
    )
    if len(values):
        q95 = float(np.quantile(values, 0.95))
        exceed = int(np.sum(values >= observed_contrast))
        p_upper = (exceed + 1)/(len(values) + 1)
        ax.axvline(q95, color="black", linewidth=1.5,
                   label=f"bootstrap 95% quantile = {q95:.1f}")
        ax.text(
            0.98, 0.96,
            f"valid bootstrap samples: {len(values)}\n"
            f"at least as large as observed: {exceed}\n"
            f"finite-sample upper-tail p = {p_upper:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
        )
    ax.set_xlabel(r"$2\{\mathrm{NLL}_{M3}-\mathrm{NLL}_{M4}\}$")
    ax.set_ylabel("Density")
    ax.set_title("Formal nested-model comparison under fitted M3",
                 fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)
    return _finish(fig, save_path)


def plot_recovery_study(recovery, references, save_path=None):
    """Summarize complete- and partial-observation parameter recovery."""
    valid = recovery[recovery["success"].astype(bool)].copy()
    error_rows = []
    for _, row in valid.iterrows():
        model = row["model"]
        reference = references[model]
        for index, name in enumerate(PARAM_NAMES):
            if name not in row or np.isclose(reference[index], 0.0):
                continue
            error_rows.append({
                "model": model,
                "observation": row["observation"],
                "parameter": name,
                "relative_error": (float(row[name]) - reference[index])/abs(reference[index]),
            })
    errors = pd.DataFrame(error_rows)
    rmse = (
        errors.assign(squared=lambda frame: frame["relative_error"]**2)
        .groupby(["parameter", "model", "observation"])["squared"]
        .mean().pow(0.5).unstack(["model", "observation"])
    )
    column_order = [
        (model, observation)
        for model in ("M2", "M3", "M4")
        for observation in ("complete", "partial")
        if (model, observation) in rmse.columns
    ]
    rmse = rmse.reindex(columns=column_order)
    rmse.columns = [f"{model}\n{observation}" for model, observation in rmse.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    sns.heatmap(
        rmse, cmap="mako_r", annot=True, fmt=".2f", linewidths=0.5,
        cbar_kws={"label": "Relative RMSE"}, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("(A) Parameter recovery error", fontweight="bold")

    ax = axes[0, 1]
    sns.boxplot(
        data=valid, x="model", y="q_path_relative_rmse", hue="observation",
        showfliers=False, ax=ax,
    )
    sns.stripplot(
        data=valid, x="model", y="q_path_relative_rmse", hue="observation",
        dodge=True, color="black", alpha=0.35, size=3, ax=ax, legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Relative RMSE of $q$ on latent path")
    ax.set_title("(B) Recovery of the diffusion surface", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[1, 0]
    rates = (
        recovery.groupby(["model", "observation"])["success"].mean()
        .reset_index(name="success_rate")
    )
    sns.barplot(data=rates, x="model", y="success_rate", hue="observation", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Successful fit fraction")
    ax.set_title("(C) Numerical success rate", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[1, 1]
    m4 = errors[
        (errors["model"] == "M4")
        & errors["parameter"].isin(["delta", "epsilon", "zeta"])
    ]
    sns.boxplot(
        data=m4, x="parameter", y="relative_error", hue="observation",
        showfliers=False, ax=ax,
    )
    sns.stripplot(
        data=m4, x="parameter", y="relative_error", hue="observation",
        dodge=True, color="black", alpha=0.4, size=4, ax=ax, legend=False,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(
        np.arange(3), [PARAM_LABELS[name] for name in ("delta", "epsilon", "zeta")],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Relative estimation error")
    ax.set_title("(D) Recovery of the three new M4 parameters", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    fig.suptitle("Repeated parameter-recovery study", fontweight="bold", fontsize=16, y=1.02)
    return _finish(fig, save_path)


def plot_ios_overview(summary, transitions, save_path=None):
    """Compare IOS magnitude, timing, concentration, and M3/M4 agreement."""
    summary = summary.copy().sort_values("model")
    models = summary["model"].tolist()
    formal = bool(summary["formal_T_N_complete"].all())
    value_column = "T_N" if formal else "sampled_ios_sum"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    values = summary[value_column].to_numpy(dtype=float)
    references = summary["asymptotic_reference"].to_numpy(dtype=float)
    applicable = summary["reference_applicable"].astype(bool).to_numpy()
    positions = np.arange(len(models))
    ax.bar(positions, values, color=[COLORS[model] for model in models], alpha=0.82)
    ax.scatter(positions[applicable], references[applicable], marker="_", s=550,
               linewidth=3, color="black",
               label=r"asymptotic reference $r+\frac{9}{4}s$")
    if np.any(~applicable):
        ax.scatter(
            positions[~applicable], references[~applicable], marker="x", s=80,
            linewidth=1.8, color="#D55E00", label="reference not regular at boundary",
        )
    for x, value, reference in zip(positions, values, references):
        ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        if formal and applicable[x]:
            ax.text(x, value/2, f"{value/reference:.2f}x", ha="center", va="center",
                    color="white", fontweight="bold")
    ax.set_xticks(positions, models)
    ax.set_ylabel("Observed exact $T_N$" if formal else "Sampled contribution sum")
    ax.set_title(
        "(A) Formal observed IOS against its reference"
        if formal else "(A) Pilot magnitude only; not a formal statistic",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[0, 1]
    for model in models:
        column = f"{model.lower()}_ios_contribution"
        valid = transitions[column].notna()
        x = transitions.loc[valid, "age_ka"]
        y = transitions.loc[valid, column]
        ax.scatter(x, y, s=7, alpha=0.18, color=COLORS[model])
        if len(y) >= 100:
            smooth = y.rolling(101, center=True, min_periods=25).mean()
            ax.plot(x, smooth, linewidth=1.8, color=COLORS[model], label=model)
        else:
            ax.plot(x, y, linewidth=1.1, color=COLORS[model], label=model)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Age (ka before 2000 AD)")
    ax.set_ylabel(r"Transition IOS contribution $\mathrm{IOS}_k$")
    ax.set_title("(B) When influential transitions occur", fontweight="bold")
    ax.legend(frameon=False, ncol=len(models))
    ax.invert_xaxis()
    _style(ax)

    ax = axes[1, 0]
    for model in models:
        ranked = np.sort(np.maximum(
            transitions[f"{model.lower()}_ios_contribution"].dropna().to_numpy(float),
            0.0,
        ))[::-1]
        cumulative = np.cumsum(ranked)/max(np.sum(ranked), 1e-12)
        ax.plot(
            np.arange(1, len(ranked) + 1)/len(ranked), cumulative,
            color=COLORS[model], linewidth=2.2, label=model,
        )
    ax.axhline(0.8, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Fraction of transitions, ranked by positive IOS")
    ax.set_ylabel("Cumulative share of positive IOS")
    ax.set_title("(C) Is lack of fit diffuse or concentrated?", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[1, 1]
    x = transitions["m3_ios_contribution"]
    y = transitions["m4_ios_contribution"]
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    limit = float(np.nanquantile(np.abs(np.r_[x, y]), 0.995))
    ax.scatter(
        x, y,
        c=transitions.loc[valid, "regime_switch"].map(
            {False: "#8C8C8C", True: "#D55E00"},
        ),
        s=13, alpha=0.48,
    )
    ax.plot([-limit, limit], [-limit, limit], color="black", linestyle="--",
            linewidth=1, label="equal influence")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"M3 $\mathrm{IOS}_k$")
    ax.set_ylabel(r"M4 $\mathrm{IOS}_k$")
    ax.set_title("(D) Does M4 remove the same influential transitions?", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    title = (
        "Observed exact information-omission sensitivity"
        if formal else "Sampled IOS pilot: optimizer and influence diagnostics"
    )
    fig.suptitle(title, fontweight="bold", fontsize=16, y=1.02)
    return _finish(fig, save_path)


def plot_ios_phase_space(transitions, save_path=None):
    """Map each model's IOS contributions and the M4-minus-M3 change."""
    models = ("M2", "M3", "M4")
    all_values = np.concatenate([
        transitions[f"{model.lower()}_ios_contribution"].dropna().to_numpy(float)
        for model in models
    ])
    nonzero = np.abs(all_values[np.abs(all_values) > 0.0])
    scale = float(np.quantile(nonzero, 0.5)) if len(nonzero) else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.08, top=0.90,
                        hspace=0.20, wspace=0.16)
    transformed = {
        model: np.arcsinh(
            transitions[f"{model.lower()}_ios_contribution"].to_numpy(float)/scale
        )
        for model in models
    }
    difference = (
        transitions["m4_ios_contribution"] - transitions["m3_ios_contribution"]
    ).to_numpy(float)
    transformed["M4 - M3"] = np.arcsinh(difference/scale)
    vmax = float(np.nanquantile(np.abs(np.concatenate([
        values[np.isfinite(values)] for values in transformed.values()
    ])), 0.99))

    for panel, (ax, label) in enumerate(zip(axes.flat, (*models, "M4 - M3"))):
        values = transformed[label]
        valid = np.isfinite(values)
        scatter = ax.scatter(
            transitions.loc[valid, "X_old"], transitions.loc[valid, "Vhat_old"],
            c=values[valid], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            s=13, alpha=0.68, linewidths=0,
        )
        raw = (
            difference if label == "M4 - M3"
            else transitions[f"{label.lower()}_ios_contribution"].to_numpy(float)
        )
        finite = np.flatnonzero(np.isfinite(raw))
        top = finite[np.argsort(np.abs(raw[finite]))[-5:]]
        ax.scatter(
            transitions.loc[top, "X_old"], transitions.loc[top, "Vhat_old"],
            s=45, facecolors="none", edgecolors="black", linewidths=0.8,
        )
        ax.set_title(f"({chr(65 + panel)}) {label}", fontweight="bold")
        ax.set_xlabel("Position $X$")
        ax.set_ylabel(r"Reconstructed velocity $\widehat V$")
        _style(ax)
    colorbar_axis = fig.add_axes([0.90, 0.15, 0.018, 0.68])
    fig.colorbar(
        scatter, cax=colorbar_axis,
        label=r"$\operatorname{asinh}(\mathrm{IOS}_k/\mathrm{median}|\mathrm{IOS}|)$",
    )
    fig.suptitle(
        "Where each model is sensitive to omission of one real-data transition",
        fontweight="bold", fontsize=16, y=0.98,
    )
    return _finish(fig, save_path, tight=False)


def plot_ios_numerical_diagnostics(parameter_shifts, transitions, save_path=None):
    """Show parameter stability, optimizer iterations, and runtime."""
    shifts = parameter_shifts.copy()
    order = [name for name in PARAM_NAMES if name in set(shifts["parameter"])]
    q95 = shifts.pivot(index="model", columns="parameter", values="q95_abs_relative_shift")
    maximum = shifts.pivot(index="model", columns="parameter", values="max_abs_relative_shift")
    q95, maximum = q95.reindex(columns=order), maximum.reindex(columns=order)

    long_rows = []
    for model in ("M2", "M3", "M4"):
        for metric in ("nit", "seconds"):
            column = f"{model.lower()}_{metric}"
            for value in transitions[column].dropna():
                long_rows.append({"model": model, "metric": metric, "value": value})
    diagnostics = pd.DataFrame(long_rows)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.heatmap(
        q95, cmap="mako_r", annot=True, fmt=".3g", linewidths=0.5,
        cbar_kws={"label": "95th percentile absolute relative shift"}, ax=axes[0, 0],
    )
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")
    axes[0, 0].set_title("(A) Typical large leave-one-out parameter shifts", fontweight="bold")

    sns.heatmap(
        maximum, cmap="rocket_r", annot=True, fmt=".3g", linewidths=0.5,
        cbar_kws={"label": "Maximum absolute relative shift"}, ax=axes[0, 1],
    )
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")
    axes[0, 1].set_title("(B) Worst observed parameter shifts", fontweight="bold")

    ax = axes[1, 0]
    sns.boxplot(
        data=diagnostics[diagnostics["metric"] == "nit"],
        x="model", y="value", hue="model", palette=COLORS,
        showfliers=False, legend=False, ax=ax,
    )
    sns.stripplot(
        data=diagnostics[diagnostics["metric"] == "nit"],
        x="model", y="value", color="black", alpha=0.25, size=2.5, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("L-BFGS-B iterations")
    ax.set_title("(C) Leave-one-out optimizer effort", fontweight="bold")
    _style(ax)

    ax = axes[1, 1]
    sns.boxplot(
        data=diagnostics[diagnostics["metric"] == "seconds"],
        x="model", y="value", hue="model", palette=COLORS,
        showfliers=False, legend=False, ax=ax,
    )
    sns.stripplot(
        data=diagnostics[diagnostics["metric"] == "seconds"],
        x="model", y="value", color="black", alpha=0.25, size=2.5, ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Seconds per transition (log scale)")
    ax.set_title("(D) Computational cost of exact IOS", fontweight="bold")
    _style(ax)

    fig.suptitle(
        "Numerical stability diagnostics for leave-one-out IOS fits",
        fontweight="bold", fontsize=16, y=1.02,
    )
    return _finish(fig, save_path)


def plot_m4_parametric_bootstrap_parameters(parameter_summary, save_path=None):
    """Show M4 parametric-bootstrap uncertainty for each fitted coefficient."""
    summary = parameter_summary.copy()
    summary["label"] = summary["parameter"].map(PARAM_LABELS)
    summary["scale"] = np.maximum(summary["observed"].abs(), 1.0)
    for column in ("q025", "q50", "q975", "mean"):
        summary[f"{column}_relative"] = (
            summary[column] - summary["observed"]
        )/summary["scale"]

    order = PARAM_NAMES[::-1]
    summary["parameter"] = pd.Categorical(summary["parameter"], order, ordered=True)
    summary = summary.sort_values("parameter")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), width_ratios=[1.35, 1.0])
    ax = axes[0]
    y = np.arange(len(summary))
    ax.hlines(
        y, summary["q025_relative"], summary["q975_relative"],
        color="#8C8C8C", linewidth=2.5, label="95% bootstrap interval",
    )
    ax.scatter(summary["q50_relative"], y, color=COLORS["M4"], s=42,
               label="bootstrap median", zorder=3)
    ax.scatter(summary["mean_relative"], y, color="black", s=25, marker="x",
               label="bootstrap mean", zorder=4)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, [PARAM_LABELS[name] for name in summary["parameter"]])
    ax.set_xlabel(r"Bootstrap estimate minus observed estimate, scaled by $\max(|\hat\theta|,1)$")
    ax.set_title("(A) Parameter uncertainty around the observed M4 fit", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    _style(ax)

    ax = axes[1]
    new_terms = summary[summary["parameter"].astype(str).isin(
        ["delta", "epsilon", "zeta"],
    )].copy()
    labels = [PARAM_LABELS[name] for name in new_terms["parameter"]]
    positions = np.arange(len(new_terms))
    ax.errorbar(
        new_terms["q50"], positions,
        xerr=[
            new_terms["q50"] - new_terms["q025"],
            new_terms["q975"] - new_terms["q50"],
        ],
        fmt="o", color=COLORS["M4"], ecolor="#666666",
        elinewidth=2, capsize=4,
    )
    ax.scatter(new_terms["observed"], positions, color="black", marker="|",
               s=220, linewidth=2.2, label="observed estimate")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Direct M4 diffusion coefficient")
    ax.set_title("(B) New position-dependent diffusion terms", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    fig.suptitle(
        "M4 parametric bootstrap: parameter uncertainty from successful simulated refits",
        fontweight="bold", fontsize=16, y=1.02,
    )
    return _finish(fig, save_path)


def plot_m4_parametric_bootstrap_diffusion(
    overview, diffusion, diffusion_summary, path_band, save_path=None,
):
    """Show M4 bootstrap convergence, diffusion minima, and pathwise bands."""
    overview = overview.iloc[0] if isinstance(overview, pd.DataFrame) else overview
    diffusion = diffusion.copy()
    diffusion_summary = diffusion_summary.copy()
    path_band = path_band.copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.hist(diffusion["nll"], bins=28, color=COLORS["M4"], alpha=0.78)
    ax.axvline(diffusion["nll"].median(), color="black", linestyle="--",
               label="bootstrap median")
    ax.set_xlabel("Refitted NLL on simulated data")
    ax.set_ylabel("Bootstrap replications")
    ax.set_title(
        f"(A) Refit distribution; {int(overview['n_success'])}/{int(overview['n_total'])} succeeded",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[0, 1]
    minima = diffusion[["q_min_global", "q_min_observed", "q_path_min"]].copy()
    minima = minima.rename(columns={
        "q_min_global": "global",
        "q_min_observed": "observed rectangle",
        "q_path_min": "observed path",
    })
    long = minima.melt(var_name="region", value_name="minimum_q")
    sns.boxplot(
        data=long, x="region", y="minimum_q", color="#B7E1CD",
        showfliers=False, ax=ax,
    )
    sns.stripplot(
        data=long.sample(min(len(long), 900), random_state=20260616),
        x="region", y="minimum_q", color="black", alpha=0.18, size=2.2, ax=ax,
    )
    observed_lookup = dict(zip(
        diffusion_summary["metric"], diffusion_summary["observed"],
    ))
    observed_values = [
        observed_lookup["q_min_global"],
        observed_lookup["q_min_observed"],
        observed_lookup["q_path_min"],
    ]
    ax.scatter(np.arange(3), observed_values, marker="D", s=45,
               color="#D55E00", label="observed fit")
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Minimum squared diffusion $q$ (log scale)")
    ax.set_title("(B) Does simulated refitting stay away from zero diffusion?", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[1, 0]
    ax.fill_between(
        path_band["age_ka"],
        np.log10(path_band["q_ratio_q025"]),
        np.log10(path_band["q_ratio_q975"]),
        color=COLORS["M4"], alpha=0.20, label="95% bootstrap band",
    )
    ax.plot(
        path_band["age_ka"], np.log10(path_band["q_ratio_q50"]),
        color=COLORS["M4"], linewidth=1.5, label="bootstrap median",
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1,
               label="observed fit")
    ax.invert_xaxis()
    ax.set_xlabel("Age (ka before 2000 AD)")
    ax.set_ylabel(r"$\log_{10}\{q^*(X_k,\widehat V_k)/q_{\mathrm{obs}}(X_k,\widehat V_k)\}$")
    ax.set_title("(C) Uncertainty in diffusion along the observed path", fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    _style(ax)

    ax = axes[1, 1]
    ax.scatter(
        diffusion["q_min_observed"], diffusion["q_path_median"],
        c=diffusion["tail_margin"], cmap="viridis", s=24, alpha=0.75,
    )
    ax.scatter(
        observed_lookup["q_min_observed"],
        observed_lookup["q_path_median"],
        marker="D", s=55, color="#D55E00", label="observed fit",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Minimum q on observed rectangle")
    ax.set_ylabel("Median q on observed path")
    ax.set_title("(D) Bootstrap diffusion scale and tail margin", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    fig.suptitle(
        "M4 parametric bootstrap: refit stability and diffusion uncertainty",
        fontweight="bold", fontsize=16, y=1.02,
    )
    return _finish(fig, save_path)


def plot_modelwise_ios_bootstrap(summary, cumulative, table, save_path=None):
    """Show finite-sample IOS calibration from a model-wise bootstrap."""
    summary = summary.iloc[0] if isinstance(summary, pd.DataFrame) else summary
    cumulative = cumulative.copy()
    table = table.copy()
    success = table.loc[
        table["success"].astype(bool) & table["ios_T_N"].notna()
    ].copy()
    observed = float(summary["observed_T_N"])
    color = COLORS.get(str(summary.get("model", "M4")), COLORS["M4"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.hist(success["ios_T_N"], bins=24, color=color, alpha=0.75)
    ax.axvline(observed, color="#D55E00", linewidth=2.4,
               label=f"observed $T_N$ = {observed:.2f}")
    ax.axvline(summary["q025"], color="black", linestyle="--", linewidth=1.0)
    ax.axvline(summary["q975"], color="black", linestyle="--", linewidth=1.0,
               label="bootstrap 95% interval")
    ax.set_xlabel(r"Bootstrap exact IOS statistic $T_N^*$")
    ax.set_ylabel("Bootstrap replications")
    ax.set_title("(A) Model-wise finite-sample IOS reference distribution",
                 fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[0, 1]
    if len(cumulative):
        ax.plot(
            cumulative["n_success"], cumulative["p_upper"],
            color=color, linewidth=2.0, label=r"$P(T_N^* \geq T_N^{obs})$",
        )
        ax.plot(
            cumulative["n_success"], cumulative["p_lower"],
            color="#0072B2", linewidth=2.0, label=r"$P(T_N^* \leq T_N^{obs})$",
        )
    ax.axhline(0.05, color="black", linestyle=":", linewidth=1.2,
               label="5% reference")
    ax.set_xlabel("Successful bootstrap replications")
    ax.set_ylabel("Finite-sample tail probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(B) Does the bootstrap p-value stabilize?", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[1, 0]
    scatter = ax.scatter(
        success["nll"], success["ios_T_N"],
        c=success.get("q_min_path", success["ios_T_N"]),
        cmap="viridis", s=35, alpha=0.75,
    )
    fig.colorbar(scatter, ax=ax, label=r"Minimum $q$ on simulated path")
    ax.axhline(observed, color="#D55E00", linewidth=1.8)
    ax.set_xlabel("Refitted NLL on simulated data")
    ax.set_ylabel(r"Bootstrap exact IOS $T_N^*$")
    ax.set_title("(C) IOS calibration is not just refit NLL", fontweight="bold")
    _style(ax)

    ax = axes[1, 1]
    runtime = success[["rep", "time_sec", "ios_total_seconds"]].copy()
    ax.scatter(
        runtime["rep"], runtime["ios_total_seconds"]/60.0,
        color=color, alpha=0.75, s=35, label="exact IOS time",
    )
    ax.axhline(
        runtime["ios_total_seconds"].median()/60.0,
        color="black", linestyle="--", linewidth=1.1, label="median",
    )
    ax.set_xlabel("Bootstrap replication")
    ax.set_ylabel("Minutes")
    ax.set_title("(D) Computational cost per model-wise IOS replication",
                 fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)

    fig.suptitle(
        (
            "M4 model-wise IOS bootstrap: "
            f"{int(summary['n_success'])}/{int(summary['n_total'])} complete, "
            f"upper-tail p={summary['p_upper']:.3f}"
        ),
        fontweight="bold", fontsize=16, y=1.02,
    )
    return _finish(fig, save_path)
