"""Build the executed research-index notebook from short reproducible cells."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "greenland_m4_analysis.ipynb"


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


cells = [
    markdown(
        r"""
# Greenland Student Kramers M4 research workflow

This notebook is the interactive index of the M4 study. It follows the same
order as the previous project: simulation validation, real-data fitting,
predictive diagnostics, bootstrap uncertainty, and IOS goodness-of-fit.

Long calculations are run by command modules and saved as checkpoints. The
cells below load the versioned report snapshot and call reusable plotting
functions.
"""
    ),
    markdown(
        r"""
## Evidence status

- **Current evidence** uses the globally feasible Cholesky M4 optimizer.
- **Development evidence** records earlier recovery, discrimination, and
  M3-null bootstrap runs made before the Cholesky parameterization.

Development results explain what was tried and what must be rerun. They are
not combined with the current M4 likelihood contrast.
"""
    ),
    code(
        """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from greenland_application.data_loading import load_real_data
from greenland_application.figures import (
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
from student_kramers import config
from student_kramers.models import PARAM_NAMES

CURRENT = ROOT / "docs" / "results" / "current"
DEVELOPMENT = ROOT / "docs" / "results" / "development"
"""
    ),
    code(
        """
_, age, x, data = load_real_data()
fits = pd.read_csv(CURRENT / "model_fits.csv")
m4_params = fits.loc[fits["model"] == "M4", PARAM_NAMES].iloc[0].to_numpy(float)
fits[["model", "nll", "aic", "bic", "q_min_global", "q_min_observed"]]
"""
    ),
    markdown(
        r"""
## 1. Data and observation scheme

Only the climate coordinate \(X\) is observed. The partial likelihood uses

$$
\widehat V_k = \frac{X_{k+1}-X_k}{h}, \qquad h=0.02\ \mathrm{kyr}.
$$

The following figure shows the time series, reconstructed velocity, observed
phase space, and where M4 changes diffusion relative to M3.
"""
    ),
    code(
        """
plot_real_data_state_space(age, x, data, fits)
plt.show()
"""
    ),
    markdown(
        r"""
## 2. Simulation validation

The original simulation programme contained:

1. complete-data recovery using latent \((X,V)\);
2. partial-data recovery after discarding \(V\) and reconstructing
   \(\widehat V\);
3. repeated M2, M3, and M4 recovery;
4. M3/M4 discrimination under M3 and weak, moderate, and strong M4 truth.

The saved studies below used the pre-Cholesky M4 optimizer. They are retained
as development evidence and must be rerun before they support a final
methodological claim.
"""
    ),
    code(
        """
recovery = []
for model in ("M2", "M3", "M4"):
    table = pd.read_csv(DEVELOPMENT / f"{model.lower()}_recovery_study.csv")
    table["model"] = model
    recovery.append(table)
recovery = pd.concat(recovery, ignore_index=True)
recovery.groupby(["model", "observation"]).agg(
    n=("success", "size"),
    success_rate=("success", "mean"),
    median_q_relative_rmse=("q_path_relative_rmse", "median"),
    median_seconds=("time_sec", "median"),
)
"""
    ),
    code(
        """
plot_recovery_study(
    recovery,
    config.REFERENCE_PARAMS_BY_MODEL,
    title="Development recovery study (pre-Cholesky M4 optimizer)",
)
plt.show()
"""
    ),
    code(
        """
scenarios = ("m3", "weak", "moderate", "strong")
discrimination = pd.concat(
    [pd.read_csv(DEVELOPMENT / f"discrimination_{name}.csv") for name in scenarios],
    ignore_index=True,
)
discrimination.groupby("truth").agg(
    n=("success", "size"),
    success_rate=("success", "mean"),
    median_contrast=("contrast", "median"),
    m4_win_rate=("contrast", lambda values: np.mean(values > 0)),
)
"""
    ),
    code(
        """
old_nested = pd.read_csv(DEVELOPMENT / "m3_m4_nested_summary.csv").iloc[0]
plot_discrimination(
    discrimination,
    float(old_nested["observed_contrast"]),
    title="Development discrimination pilot (pre-Cholesky M4 optimizer)",
)
plt.show()
"""
    ),
    markdown(
        r"""
## 3. Real-data fitting and functional comparison

M2, M3, and M4 use the same data and corrected partial-observation
pseudo-likelihood. M4 adds position-dependent diffusion:

$$
q_{M4}(x,v)
=\alpha v^2+\beta v+\gamma+\delta x^2+\epsilon xv+\zeta x.
$$
"""
    ),
    code(
        """
fits[[
    "model", "n_free", "nll", "aic", "bic",
    "q_min_global", "q_min_observed", "time_sec",
]].round(4)
"""
    ),
    code(
        """
plot_real_data_mechanisms(fits, data)
plt.show()
"""
    ),
    markdown(
        r"""
## 4. Transition-level improvement and diffusion domain

The total M3-to-M4 improvement is decomposed into individual transition
contributions. A separate domain audit distinguishes the global minimum of
\(q\) from values supported by the observed state region.
"""
    ),
    code(
        """
transition = pd.read_csv(CURRENT / "transition_improvement.csv")
pd.Series({
    "NLL(M3)-NLL(M4)": transition["gain_m4_over_m3"].sum(),
    "fraction favouring M4": np.mean(transition["gain_m4_over_m3"] > 0),
    "largest positive gain": transition["gain_m4_over_m3"].max(),
    "largest negative gain": transition["gain_m4_over_m3"].min(),
})
"""
    ),
    code(
        """
plot_transition_improvement(transition)
plt.show()
"""
    ),
    code(
        """
q_audit = pd.read_csv(CURRENT / "q_min_audit.csv")
display(q_audit[[
    "model", "q_min_global", "q_min_observed_rectangle",
    "q_path_min", "q_min_standardized_distance",
]].round(4))
plot_m4_diffusion_audit(fits, data, q_audit)
plt.show()
"""
    ),
    markdown(
        r"""
## 5. Predictive and regime diagnostics

For each fitted model, 100 current-fit trajectories were simulated. Their
latent velocities were discarded and reconstructed from simulated \(X\), so
the comparison uses the same observation scheme as the real data.
"""
    ),
    code(
        """
predictive = pd.read_csv(CURRENT / "predictive_comparison_summary.csv")
predictive.round(3)
"""
    ),
    code(
        """
plot_predictive_densities(pd.read_csv(CURRENT / "predictive_density.csv"))
plt.show()
"""
    ),
    code(
        """
plot_predictive_percentiles(
    pd.read_csv(CURRENT / "predictive_summary.csv"),
    pd.read_csv(CURRENT / "predictive_behavior.csv"),
)
plt.show()
"""
    ),
    code(
        """
plot_waiting_time_comparison(pd.read_csv(CURRENT / "predictive_waiting.csv"))
plt.show()
"""
    ),
    markdown(
        r"""
## 6. Three bootstrap questions

The project uses three different bootstrap designs.

1. **M3-null nested bootstrap:** can M4 obtain the observed likelihood gain
   when M3 generates the data?
2. **M4 parametric bootstrap:** how uncertain are the M4 parameters and fitted
   diffusion function?
3. **M4 model-wise IOS bootstrap:** is observed leave-one-out sensitivity
   unusually large under fitted M4?

Only designs 2 and 3 use the current Cholesky optimizer. Design 1 remains a
historical diagnostic until rerun.
"""
    ),
    code(
        """
bootstrap_overview = pd.read_csv(
    CURRENT / "m4_parametric_bootstrap_overview.csv"
)
bootstrap_overview.round(4)
"""
    ),
    code(
        """
m4_bootstrap = pd.read_csv(CURRENT / "m4_parametric_bootstrap.csv")
plot_m4_parameter_distributions(m4_bootstrap, m4_params)
plt.show()
"""
    ),
    code(
        """
parameter_summary = pd.read_csv(
    CURRENT / "m4_parametric_bootstrap_parameters.csv"
)
plot_m4_parametric_bootstrap_parameters(parameter_summary)
plt.show()
"""
    ),
    code(
        """
plot_m4_parametric_bootstrap_diffusion(
    bootstrap_overview,
    pd.read_csv(CURRENT / "m4_parametric_bootstrap_diffusion.csv"),
    pd.read_csv(CURRENT / "m4_parametric_bootstrap_diffusion_summary.csv"),
    pd.read_csv(CURRENT / "m4_parametric_bootstrap_path_band.csv"),
)
plt.show()
"""
    ),
    code(
        """
pd.read_csv(CURRENT / "m4_bootstrap_derived_summary.csv").round(4)
"""
    ),
    markdown(
        r"""
### Historical M3-null bootstrap

The figure below used the earlier direct-coefficient M4 optimizer and the old
observed contrast. It documents the previous calculation only. The current
contrast cannot be inserted into this old null distribution.
"""
    ),
    code(
        """
old_nested_table = pd.read_csv(DEVELOPMENT / "m3_m4_nested_bootstrap.csv")
plot_nested_bootstrap(
    old_nested_table,
    float(old_nested["observed_contrast"]),
    title="Development M3-null bootstrap (pre-Cholesky M4 optimizer)",
    note="Historical diagnostic only; not valid for the current M4 fit",
)
plt.show()
"""
    ),
    markdown(
        r"""
## 7. Exact information-omission sensitivity

For transition \(k\),

$$
\operatorname{IOS}_k
=\ell_k(\widehat\theta_{-k})-\ell_k(\widehat\theta),
\qquad
T_N=\sum_k\operatorname{IOS}_k.
$$

All 2499 transitions were recomputed for M2, M3, and M4.
"""
    ),
    code(
        """
ios_summary = pd.read_csv(CURRENT / "ios_comparison.csv")
ios_summary[[
    "model", "T_N", "n_valid", "success_rate",
    "fraction_transitions_for_80pct_positive", "total_seconds",
]].round(4)
"""
    ),
    code(
        """
ios_transitions = pd.read_csv(CURRENT / "ios_transitions.csv")
plot_ios_overview(ios_summary, ios_transitions)
plt.show()
"""
    ),
    code(
        """
plot_ios_phase_space(ios_transitions)
plt.show()
"""
    ),
    code(
        """
plot_ios_numerical_diagnostics(
    pd.read_csv(CURRENT / "ios_parameter_shifts.csv"),
    ios_transitions,
)
plt.show()
"""
    ),
    markdown(
        r"""
## 8. Model-wise M4 IOS bootstrap

Each replication simulates from fitted M4, refits M4, and recomputes all 2499
leave-one-out fits. This calibrates the IOS statistic under the full current
estimation pipeline.
"""
    ),
    code(
        """
modelwise_summary = pd.read_csv(
    CURRENT / "m4_modelwise_ios_bootstrap_summary.csv"
)
modelwise_summary.round(4)
"""
    ),
    code(
        """
plot_modelwise_ios_bootstrap(
    modelwise_summary,
    pd.read_csv(CURRENT / "m4_modelwise_ios_bootstrap_cumulative.csv"),
    pd.read_csv(CURRENT / "m4_modelwise_ios_bootstrap.csv"),
)
plt.show()
"""
    ),
    markdown(
        r"""
## 9. Current conclusion

The current evidence shows that M4 is numerically viable, improves the
real-data objective, changes the fitted diffusion over observed phase space,
and does not have unusually large IOS under its own bootstrap calibration.

The predictive simulations still switch more often than the observed path,
and the position distribution remains too narrow. M4 therefore improves the
local likelihood without solving every model-check discrepancy.

M4 has not yet been formally selected over M3. The next required experiment
is the M3-null nested bootstrap using the current Cholesky optimizer. Repeated
recovery and discrimination should then be rerun with the same optimizer.
"""
    ),
    markdown(
        r"""
## 10. Reproducibility commands

```bash
python3 -m greenland_application.run_report_assets --refresh-snapshot
quarto render docs/M4_GREENLAND_RESEARCH_REPORT.md --to html
quarto render docs/M4_GREENLAND_RESEARCH_REPORT.md --to pdf
python3 -m unittest discover tests -v
```
"""
    ),
]

for index, cell in enumerate(cells, start=1):
    cell["id"] = f"m4-research-{index:03d}"


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.9",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT}")
