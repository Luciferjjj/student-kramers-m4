"""
config.py — Shared paths and numerical experiment settings

Parameter order is fixed throughout the project:

    [eta, a, b, c, d, alpha, beta, gamma, delta, epsilon, zeta].

``H_OBS`` is the observation spacing used by the likelihood.
``H_SIM`` is the finer Euler--Maruyama spacing used for bootstrap and
diagnostic simulation.  Expensive run lengths are kept here so a paper-scale
run can be configured without editing mathematical modules.
"""
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
DEFAULT_RUN_NAME = "development"

DATA_FILE = DATA_DIR / "official_ice_data.xlsx"

DATA_URL = (
    "https://www.iceandclimate.nbi.ku.dk/data/"
    "GICC05modelext_GRIP_and_GISP2_and_resampled_data_series_"
    "Seierstad_et_al._2014_version_10Dec2014-2.xlsx"
)
AGE_PREFILTER = (17000.0, 90000.0)
AGE_WINDOW = (30000.0, 80000.0)

H_OBS = 0.02
H_SIM = 0.001
H_SIM_VALIDATION = 0.0001
CORRECTION_FACTOR = 2.0 / 3.0
PENALTY = 1e15
EPS = 1e-10
# The observed state rectangle is enlarged by this amount for M4 diagnostic
# plots.  Formal estimation still requires q(x, v) >= 0 globally.
M4_DIAGNOSTIC_MARGIN = 1.0

LBFGS_MAXITER = 800
LBFGS_TOL = 1e-8
N_RANDOM_STARTS = 8
M4_CHOLESKY_JITTER = 1e-10
M4_CHOLESKY_REL_STEP = 1e-5
M4_CHOLESKY_FTOL = 1e-12
M4_CHOLESKY_GTOL = 1e-7

N_BOOTSTRAP = 100
BOOTSTRAP_SEED = 20260603
BOOTSTRAP_MAXITER = 200
IOS_MAXITER = 120
IOS_SAVE_EVERY = 25

N_MODEL_CHECK = 100
N_FIRST_PASSAGE = 1000

REFERENCE_PARAMS_BY_MODEL = {
    "M1": np.array([
        30.0, -125.0, 0.0, 150.0, 0.0, 0.0, 0.0, 1280.8,
        0.0, 0.0, 0.0,
    ]),
    "M2": np.array([
        30.0, -125.0, 0.0, 150.0, 0.0, 20.0, -8.0, 1280.8,
        0.0, 0.0, 0.0,
    ]),
    "M3": np.array([
        30.0, -125.0, 40.0, 150.0, -20.0, 20.0, -8.0, 1280.8,
        0.0, 0.0, 0.0,
    ]),
    "M4": np.array([
        30.0, -125.0, 40.0, 150.0, -20.0, 20.0, -8.0, 1280.8,
        4.0, 2.0, 1.0,
    ]),
}

# Simulation-study initial values follow the reference repository's
# ``INIT_PARAMS`` convention.  They are deliberately distinct from both the
# true simulation parameters and the real-data initial values in ``MODELS``.
RECOVERY_INIT_PARAMS_BY_MODEL = {
    "M1": np.array([
        50.0, -200.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1000.0,
        0.0, 0.0, 0.0,
    ]),
    "M2": np.array([
        50.0, -200.0, 0.0, 100.0, 0.0, 30.0, -5.0, 1000.0,
        0.0, 0.0, 0.0,
    ]),
    "M3": np.array([
        50.0, -200.0, 10.0, 100.0, 10.0, 30.0, -5.0, 1000.0,
        0.0, 0.0, 0.0,
    ]),
    "M4": np.array([
        50.0, -200.0, 10.0, 100.0, 10.0, 30.0, -5.0, 1000.0,
        5.0, 1.0, 1.0,
    ]),
}

# Backward-compatible alias for scripts that previously validated M3 only.
REFERENCE_PARAMS = REFERENCE_PARAMS_BY_MODEL["M3"]

def run_dir(run_name=DEFAULT_RUN_NAME):
    """Return the isolated output directory for one named experiment."""
    if not run_name or any(part in run_name for part in ("/", "\\", "..")):
        raise ValueError("run_name must be a non-empty directory-safe name")
    return RUNS_DIR / run_name


def make_result_dirs(run_name=DEFAULT_RUN_NAME):
    """Create the output folder for one named experiment."""
    run_dir(run_name).mkdir(parents=True, exist_ok=True)
