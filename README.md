# Student Kramers M4 research code

This repository extends the partially observed Student Kramers model used for
the Greenland ice-core calcium application. M4 adds position-dependent terms
to the diffusion variance,

$$
q(x,v)=
\alpha v^2+\beta v+\gamma+
\delta x^2+\epsilon xv+\zeta x.
$$

The code is organized for continued collaboration and numerical experiments.
The current repository is a research snapshot, not a final claim that M4 has
replaced M3.

## Repository structure

```text
student_kramers/
    General M1-M4 model definitions, Strang likelihoods, estimation,
    simulation, recovery, discrimination, IOS, and bootstrap algorithms.

greenland_application/
    Greenland data access, real-data runners, application diagnostics,
    IOS summaries, bootstrap summaries, and figures.

notebooks/
    Main interactive analysis notebook.

docs/
    English research report, code reference, selected figures, and a
    Git-trackable numerical result snapshot.

tests/
    Numerical, regression, checkpoint, and workflow tests.

data/
    Local copy of the official workbook. The workbook is not tracked by Git.

results/runs/
    Local generated results and resumable checkpoints. These are not tracked.

docs/results/
    Compact tables used by the report. Current Cholesky results and
    pre-Cholesky development results are separated explicitly.
```

The dependency direction is
`greenland_application -> student_kramers`. The mathematical core does not
depend on the application package.

## Main files

- [Research report](docs/M4_GREENLAND_RESEARCH_REPORT.md)
- [Rendered research report](docs/M4_GREENLAND_RESEARCH_REPORT.pdf)
- [Code reference](docs/CODE_REFERENCE.md)
- [Report result snapshot](docs/results/README.md)
- [Executed analysis notebook](notebooks/greenland_m4_analysis.ipynb)

## Current numerical status

The current formal real-data run is `m4_real_data_cholesky`.

| Model | NLL | AIC | BIC |
|---|---:|---:|---:|
| M2 | 8524.348 | 17060.695 | 17095.637 |
| M3 | 8524.059 | 17064.118 | 17110.707 |
| M4 | 8499.312 | 17020.623 | 17084.683 |

Completed calculations:

- M1-M4 nesting and moment-equation tests;
- Cholesky-constrained M4 real-data fit;
- shared sampled IOS pilot;
- exact observed IOS for all 2499 transitions in M2, M3, and M4;
- 500-replication M4 parametric bootstrap, with 496 successful fits;
- 200-replication model-wise M4 IOS bootstrap, with 200 complete results.
- 100 current-fit predictive simulations for each of M2, M3, and M4;
- current transition-level M3/M4 likelihood decomposition and diffusion audit.

The model-wise IOS upper-tail probability is 0.965. This does not find
unusually large leave-one-out sensitivity under fitted M4. The lower-tail
probability is 0.040 and is retained as a diagnostic.

The current M3 versus M4 likelihood contrast has not yet been calibrated with
a nested bootstrap using the Cholesky optimizer. The older nested bootstrap
used an earlier M4 optimization strategy. It is retained under
`docs/results/development/` as research history, not as a formal
model-selection result for the current fit.

## Installation

Run the following commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

The project requires Python 3.9 or later and installs NumPy, pandas, SciPy,
Matplotlib, Seaborn, Requests, OpenPyXL, and JupyterLab.

## Tests

```bash
python3 -m unittest discover tests -v
```

The current suite contains 44 tests.

## Notebook

```bash
jupyter lab notebooks/greenland_m4_analysis.ipynb
```

The notebook is the main research interface. It loads saved results, displays
tables, calls plotting functions, and records the current interpretation.
Expensive fits are run from command modules so they can resume from CSV
checkpoints.

## Main commands

Fit the real data:

```bash
python3 -m greenland_application.run_application \
    --run-name m4_trial \
    --models M1 M2 M3 M4 \
    --n-starts 8
```

Run one simulation validation:

```bash
python3 -m student_kramers.run_validation \
    --generate-model M4 \
    --fit-models M3 M4 \
    --run-name m4_validation
```

Run or resume exact IOS:

```bash
env VECLIB_MAXIMUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    python3 -m greenland_application.run_bootstrap \
    --mode ios \
    --fit-run m4_real_data_cholesky \
    --run-name ios_observed \
    --model M4 \
    --maxiter 120
```

Build the IOS summaries and figures:

```bash
python3 -m greenland_application.run_ios_analysis \
    --fit-run m4_real_data_cholesky \
    --run-name ios_observed
```

Refresh the report tables and rebuild every report figure:

```bash
python3 -m greenland_application.run_report_assets --refresh-snapshot
```

Render the report:

```bash
quarto render docs/M4_GREENLAND_RESEARCH_REPORT.md --to html
quarto render docs/M4_GREENLAND_RESEARCH_REPORT.md --to pdf
```

Run a new M3-null nested bootstrap with the current optimizer:

```bash
python3 -m greenland_application.run_pre_ios \
    --mode nested \
    --fit-run m4_real_data_cholesky \
    --run-name m3_m4_nested_cholesky \
    --n-rep 100
```

Use a new run name for any calculation performed after changing the numerical
implementation.

## Results and provenance

Generated results are saved under:

```text
results/runs/<run_name>/
```

Long calculations write CSV checkpoints and a neighboring `.meta.json` file.
The metadata records hashes of the data, fitted parameters, and implementation,
plus the numerical settings. A checkpoint cannot resume when these values
change.

The full local bootstrap checkpoints are intentionally excluded from Git.
Selected tables and provenance records are stored in `docs/results/`, while
report figures are stored in `docs/figures/`. Development tables are labeled
as pre-Cholesky evidence so they cannot be mistaken for current formal runs.

## Data

The official Greenland workbook is downloaded on first use from the University
of Copenhagen ice and climate data site and cached locally as
`data/official_ice_data.xlsx`. The workbook is excluded from Git.

## Collaboration status

The next required analysis is the M3-null nested bootstrap using the current
Cholesky optimizer. Repeated M4 recovery and M3/M4 discrimination should also
be rerun with the current optimizer before the method is treated as final.
