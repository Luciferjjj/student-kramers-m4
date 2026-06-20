"""
data_loading.py — Greenland Ca2+ preprocessing and project file I/O

Raw Ca2+ observations are transformed exactly as in the completed paper:

    Z_t = -log(Ca2_t),             X_t = Z_t - mean(Z)
    Vhat_t = (X_{t+h} - X_t) / h

The partial-observation likelihood receives an array ``data`` of shape
``(N, 2)`` with columns ``[X_t, Vhat_t]``. Each new experiment writes to an
isolated named directory so unfinished or changed models cannot overwrite one
another.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config


MODEL_FIT_IMPLEMENTATION_FILES = (
    "config.py",
    "models.py",
    "data_loading.py",
    "likelihoods.py",
    "estimation.py",
)


def ensure_official_excel(path=config.DATA_FILE):
    """Download the official workbook when it is not already available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(config.DATA_URL, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def load_raw_excel(path=config.DATA_FILE):
    """Read age, d18O, and Ca2+ columns from the official workbook."""
    path = ensure_official_excel(path)
    raw = pd.read_excel(path, sheet_name=2, skiprows=49, usecols=[0, 7, 8])
    raw.columns = ["age", "d18O", "Ca2"]
    return raw.iloc[1:].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")


def preprocess_ca2(raw):
    """
    Apply the final paper preprocessing pipeline.

    1. Restrict to 17--90 ka and interpolate missing Ca2+ values.
    2. Average duplicate ages and transform Ca2+ by ``-log``.
    3. Restrict to 30--80 ka and center within this final window.
    """
    lo, hi = config.AGE_PREFILTER
    df = raw[(raw["age"] >= lo) & (raw["age"] <= hi)].copy()
    df["Ca2"] = df["Ca2"].interpolate(method="linear")
    df = df.groupby("age", as_index=False)["Ca2"].mean()
    df["age_ka"] = df["age"] / 1000.0
    df["X"] = -np.log(df["Ca2"])

    lo, hi = config.AGE_WINDOW
    df = df[(df["age"] >= lo) & (df["age"] <= hi)].copy()
    df["X"] = df["X"] - df["X"].mean()
    return df[["age_ka", "X"]].reset_index(drop=True)


def build_partial_data(x, h=config.H_OBS):
    """Build pseudo-states ``[X_t, Vhat_t]``, where ``Vhat_t = dX_t/h``."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x[:-1], np.diff(x)/h])


def load_real_data(path=config.DATA_FILE, h=config.H_OBS):
    """Return processed frame, oldest-to-youngest X, and pseudo-state data."""
    df = preprocess_ca2(load_raw_excel(path))
    x = df["X"].to_numpy(dtype=float)[::-1]
    age = df["age_ka"].to_numpy(dtype=float)[::-1]
    data = build_partial_data(x, h)
    return df, age, x, data


def result_path(name, model_name=None, suffix="csv", run_name=config.DEFAULT_RUN_NAME):
    """Return ``results/runs/{run_name}/{model_}{name}.{suffix}``."""
    prefix = f"{model_name.lower()}_" if model_name else ""
    return config.run_dir(run_name) / f"{prefix}{name}.{suffix}"


def save_table(table, path):
    """Save one result table and create its parent folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_table(path, deduplicate=None):
    """Load one CSV, optionally keeping the last row for each checkpoint key."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if deduplicate and deduplicate in table.columns:
        table = table.drop_duplicates(subset=[deduplicate], keep="last")
        table = table.sort_values(deduplicate).reset_index(drop=True)
    return table


def load_result(name, model_name=None, run_name=config.DEFAULT_RUN_NAME,
                deduplicate=None):
    """Load one result from a named experiment; missing results return an empty table."""
    return load_table(
        result_path(name, model_name, run_name=run_name),
        deduplicate=deduplicate,
    )


def load_model_fits(run_name=config.DEFAULT_RUN_NAME, required=False, data=None):
    """Load model fits and verify provenance when data are supplied."""
    path = result_path("model_fits", run_name=run_name)
    table = load_table(path)
    if required and not len(table):
        raise FileNotFoundError(
            f"No model fits for run {run_name!r}. Run "
            f"`python3 -m student_kramers.run_application --run-name {run_name}` first."
        )
    if len(table):
        from .models import PARAM_NAMES

        # Pre-M4 fit tables contain the first eight parameters only.  Padding
        # the new M4 coefficients lets provenance report a clear stale-result
        # error instead of failing first with a missing-column KeyError.
        new_names = ("delta", "epsilon", "zeta")
        missing = [name for name in PARAM_NAMES if name not in table]
        if missing and set(missing).issubset(new_names):
            for name in missing:
                table[name] = 0.0
        elif missing:
            raise RuntimeError(f"Model fits in {path} are missing columns: {missing}")

    if len(table) and data is not None:
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        if not metadata_path.exists():
            raise RuntimeError(
                f"Model fits in {path} have no provenance. Refit using the current code."
            )
        saved = json.loads(metadata_path.read_text())
        current = {
            "parameter_hash": _hash_array(table[PARAM_NAMES].to_numpy(dtype=float)),
            "data_hash": _hash_array(data),
            "code_hash": code_fingerprint(MODEL_FIT_IMPLEMENTATION_FILES),
        }
        mismatches = [key for key, value in current.items() if saved.get(key) != value]
        if mismatches:
            raise RuntimeError(
                f"Model fits in {path} are stale ({', '.join(mismatches)} changed). "
                "Use a new run name and refit."
            )
    return table


def save_model_fits(table, data, run_name=config.DEFAULT_RUN_NAME, **settings):
    """Save fitted models together with data, parameter, code, and setting provenance."""
    from .models import PARAM_NAMES

    path = result_path("model_fits", run_name=run_name)
    # Hash the CSV round-trip rather than the pre-serialization float array.
    # This prevents harmless decimal formatting changes from making a newly
    # saved fit table fail its own provenance check.
    save_table(table, path)
    saved_table = load_table(path)
    model_names = ",".join(saved_table["model"].astype(str))
    params = saved_table[PARAM_NAMES].to_numpy(dtype=float)
    context = checkpoint_context(
        "model_fits", model_names, params, data,
        code_files=MODEL_FIT_IMPLEMENTATION_FILES,
        **settings,
    )
    prepare_checkpoint(path, context, resume=False)
    return path


def _hash_array(array):
    """Return a stable SHA-256 hash for one numerical array."""
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def code_fingerprint(implementation_files=None):
    """Hash the Python implementation used by a resumable computation."""
    digest = hashlib.sha256()
    implementation_files = implementation_files or (
        "config.py", "models.py", "data_loading.py", "likelihoods.py",
        "estimation.py", "simulation.py", "recovery.py", "discrimination.py",
        "bootstrap.py",
    )
    for name in implementation_files:
        path = config.PROJECT_DIR / "student_kramers" / name
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_context(workflow, model_name, params, data, code_files=None, **settings):
    """Build provenance stored beside every resumable long-run CSV."""
    return {
        "workflow": workflow,
        "model": model_name,
        "parameter_hash": _hash_array(params),
        "data_hash": _hash_array(data),
        "code_hash": code_fingerprint(code_files),
        "h_obs": config.H_OBS,
        "h_sim": config.H_SIM,
        "settings": settings,
    }


def prepare_checkpoint(path, context, resume=True):
    """Validate or create the JSON provenance beside a checkpoint CSV."""
    path = Path(path)
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if path.exists() and resume:
        if not metadata_path.exists():
            raise RuntimeError(
                f"Cannot resume {path}: provenance file is missing. "
                "Use a new --run-name or rerun without resume."
            )
        saved = json.loads(metadata_path.read_text())
        if saved != context:
            raise RuntimeError(
                f"Cannot resume {path}: model, data, settings, or code changed. "
                "Use a new --run-name."
            )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
    return path
