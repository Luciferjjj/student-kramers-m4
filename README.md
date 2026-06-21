# Student Kramers M4: position-dependent diffusion in the Greenland application

This repository studies an M4 extension of the partially observed Student
Kramers model used for the Greenland ice-core calcium record. M4 keeps the M3
drift and allows the diffusion variance to depend on both position and
velocity:

$$
q_{M4}(x,v)=\alpha v^2+\beta v+\gamma
+\delta x^2+\epsilon xv+\zeta x.
$$

The project builds on the
[Strang splitting estimator repository](https://github.com/PredragPilipovic/nonlinear_multivariate_pearson_diffusions)
by Predrag Pilipović. That repository contains the original Student Kramers
simulation study, Greenland application, and reproduction scripts associated
with the
[supplementary software release](https://doi.org/10.5281/zenodo.19632952).
The present repository is a research extension, not a replacement for the
reference implementation.

![Current M2, M3, and M4 real-data comparison](docs/figures/real_data_mechanisms.png)

> **Supervisor: start here.** The concise
> [current M4 research update](docs/M4_RESEARCH_UPDATE_FOR_PREDRAG.md) contains
> the main result, five discussion questions, five selected figures, and the
> proposed next steps. GitHub renders its equations directly in the browser.
> The [full PDF report](docs/M4_GREENLAND_RESEARCH_REPORT.pdf) is available as
> a fixed-layout alternative.

## Research status

The current M4 implementation uses a globally feasible Cholesky
parameterization. Write

$$
q(x,v)-q_{\mathrm{floor}}=
\begin{pmatrix}x&v&1\end{pmatrix}
H
\begin{pmatrix}x\\v\\1\end{pmatrix},
\qquad
H=LL^\top.
$$

The optimizer also uses

$$
\alpha=2\eta\,\mathrm{logistic}(\rho),
$$

so every M4 proposal satisfies global diffusion positivity and
$0<\alpha<2\eta$.

The current formal real-data run is `m4_real_data_cholesky`.

| Model | Free parameters | NLL | AIC | BIC |
|---|---:|---:|---:|---:|
| M2 | 6 | 8524.348 | 17060.695 | 17095.637 |
| M3 | 8 | 8524.059 | 17064.118 | 17110.707 |
| M4 | 11 | 8499.312 | 17020.623 | 17084.683 |

The current M3 versus M4 contrast is 49.495. The current M3-null nested
bootstrap has also been completed with the same optimizer. Thirteen of 100
null contrasts were at least as large as the observed value, giving a
corrected upper-tail probability of 0.1386. The null 95% quantile is 78.848.
The descriptive improvement therefore does not provide finite-sample evidence
at the 5% level for selecting M4 over M3.

### Completed current-optimizer calculations

- M1 to M4 registry, nesting, likelihood, simulation, and moment-equation
  tests;
- Cholesky M4 real-data fit and optimization audit;
- 100 predictive simulations for each of M2, M3, and M4;
- transition-level decomposition of the M3 to M4 likelihood gain;
- exact observed IOS for all 2499 transitions in M2, M3, and M4;
- 500-replication M4 parametric bootstrap, with 496 successful refits;
- 200-replication model-wise M4 IOS bootstrap, with 200 complete exact-IOS
  results;
- 100-replication current-optimizer M3-null nested bootstrap, with 100 valid
  M3 and Cholesky M4 refits;
- current-optimizer recovery with 100 M3 and 100 M4 paths under both complete
  and partial observation;
- current-optimizer M3/M4 discrimination with 100 paths in each of four
  generating scenarios.

The model-wise IOS bootstrap gives upper-tail $p=0.965$. The observed path is
not unusually sensitive to deleting one transition under fitted M4. Its
lower-tail probability is 0.040, so the unusually low observed percentile is
retained as a diagnostic.

### Historical evidence

The earlier direct-coefficient M4 fit and the first recovery, discrimination,
and nested-bootstrap experiments used the pre-Cholesky optimizer. They remain
under `docs/results/development/` as research history. Current replacements
are stored under `docs/results/current/`.

## Documentation

| Document | Intended use |
|---|---|
| [Short research update](docs/M4_RESEARCH_UPDATE_FOR_PREDRAG.md) | Five-minute supervisor briefing, five selected figures, decisions requested, and next steps |
| [M4 implementation note](docs/M4_IMPLEMENTATION_NOTE_FOR_SUPERVISOR.md) | Mathematical design, Cholesky coordinates, likelihood integration, and validation |
| [Full research report](docs/M4_GREENLAND_RESEARCH_REPORT.md) | Complete technical record, including predictive checks, bootstrap, and IOS |
| [Rendered full report](docs/M4_GREENLAND_RESEARCH_REPORT.pdf) | PDF version of the full report |
| [Code reference](docs/CODE_REFERENCE.md) | Function and module lookup dictionary |
| [Result snapshot](docs/results/README.md) | Git-tracked current and historical evidence tables |
| [Analysis notebook](notebooks/greenland_m4_analysis.ipynb) | Interactive inspection of saved results and figures |

The private local archive also contains a detailed Chinese Python guide and a
chronological Chinese research audit. They are intentionally excluded from
GitHub.

## Repository layout

```text
student-kramers-m4/
├── student_kramers/          reusable mathematical and statistical code
├── greenland_application/    Greenland data, runners, summaries, and figures
├── notebooks/                interactive research notebook
├── docs/                     collaborator-facing documentation and evidence
│   ├── figures/              Git-tracked report figures
│   └── results/              compact current and development result tables
├── tests/                    numerical and workflow regression tests
├── data/                     local cached workbook, excluded from Git
└── results/runs/             local checkpoints and long-run outputs
```

The dependency direction is:

```text
greenland_application -> student_kramers
```

The core package does not import the application package.

## Installation

The project requires Python 3.9 or later.

```bash
git clone https://github.com/Luciferjjj/student-kramers-m4.git
cd student-kramers-m4

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

The first data-loading run downloads the official workbook from the
[University of Copenhagen Ice and Climate data archive](https://www.iceandclimate.nbi.ku.dk/data/)
and caches it as `data/official_ice_data.xlsx`.

## Verify the installation

Run the complete test suite:

```bash
python3 -m unittest discover tests -v
```

The current suite contains 44 tests. It checks model nesting, diffusion
constraints, likelihood regression values, moment propagation, recovery and
discrimination workflows, checkpoint behavior, IOS, and bootstrap summaries.

## Fit the Greenland data

Run a fresh M1 to M4 comparison under a new run name:

```bash
python3 -m greenland_application.run_application \
    --run-name m4_trial \
    --models M1 M2 M3 M4 \
    --n-starts 8
```

Results are written to:

```text
results/runs/m4_trial/
```

Do not overwrite `m4_real_data_cholesky` when testing a modified numerical
implementation. A new run name preserves the provenance of the formal fit.

## Main analysis commands

### Quick saved-fit check

This command recomputes one likelihood from a local saved formal run:

```bash
python3 -m greenland_application.run_single \
    --run-name m4_real_data_cholesky \
    --model M4
```

### Predictive and pre-IOS diagnostics

```bash
python3 -m greenland_application.run_pre_ios \
    --mode predictive \
    --fit-run m4_real_data_cholesky \
    --run-name m4_report_current \
    --n-rep 100
```

### Exact observed IOS

Small matrix exponentials are faster with one BLAS thread on the tested macOS
setup:

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

Build IOS summaries and figures:

```bash
python3 -m greenland_application.run_ios_analysis \
    --fit-run m4_real_data_cholesky \
    --run-name ios_observed
```

### Current M3-null nested bootstrap

The completed calculation can be reproduced or extended from its checkpoint:

```bash
python3 -m greenland_application.run_pre_ios \
    --mode nested \
    --fit-run m4_real_data_cholesky \
    --run-name m3_m4_nested_cholesky_v2 \
    --n-rep 100 \
    --n-starts-m3 8 \
    --n-starts 12
```

The target count is omitted from checkpoint identity, so `--n-rep 300` would
extend the same run without repeating completed indices. Extension is not
required for the current decision because the 95% interval for the underlying
exceedance probability is $[0.071,0.212]$, which excludes 0.05.

## Reports and figures

Refresh the compact result snapshot and rebuild the existing report figures:

```bash
python3 -m greenland_application.run_report_assets --refresh-snapshot
```

Render the collaborator-facing documents:

```bash
quarto render docs/M4_RESEARCH_UPDATE_FOR_PREDRAG.md --to html
quarto render docs/M4_IMPLEMENTATION_NOTE_FOR_SUPERVISOR.md --to html
quarto render docs/M4_GREENLAND_RESEARCH_REPORT.md --to html
quarto render docs/M4_GREENLAND_RESEARCH_REPORT.md --to pdf
```

The reporting command reads saved tables. It does not rerun the expensive
bootstrap or exact-IOS calculations.

## Results and provenance

Long calculations write replication-level or transition-level CSV
checkpoints. Each resumable table has a neighboring `.meta.json` file with the
model, parameter, data, code, and numerical-setting hashes. A run cannot
resume if its saved identity no longer matches the current context.

The full checkpoints under `results/runs/` are excluded from Git because they
are large and repetitive. Selected evidence is copied to:

```text
docs/results/current/       current Cholesky M4 results
docs/results/development/   pre-Cholesky historical results
```

See [the result manifest](docs/results/manifest.csv) for the source and status
of each tracked table.

## Data

The Greenland workbook is downloaded from:

[GICC05modelext GRIP, GISP2, and resampled data series, Seierstad et al.
2014](https://www.iceandclimate.nbi.ku.dk/data/GICC05modelext_GRIP_and_GISP2_and_resampled_data_series_Seierstad_et_al._2014_version_10Dec2014-2.xlsx)

The analysis uses the 30 to 80 kyr BP interval after a 17 to 90 kyr BP
prefilter. Calcium is negative-log transformed, centered inside the final
window, and ordered from oldest to youngest. The observation spacing is
$h=0.02$ kyr.

## Current limitations and next steps

This repository currently supports a narrower conclusion than "M4 is the
selected final model."

- M4 is numerically feasible under the Cholesky parameterization.
- M4 improves the corrected partial pseudo-likelihood descriptively.
- Its diffusion is stable over the observed state region despite a remote
  near-boundary global minimum.
- The finite-sample IOS upper-tail check does not reject fitted M4.
- The current nested bootstrap does not select M4 over M3 at the 5% level.
- The null distribution is strongly right-skewed. A few M4 refits use very
  large position terms and approach the diffusion floor on simulated data.
- Current recovery confirms that the diffusion function is more stable than
  the three new coefficients.
- The existing weak, moderate, and strong discrimination scenarios remain
  below the calibrated nested threshold after 100 paths per scenario. This is
  not a calibrated power study because effect size was defined by coefficient
  multipliers rather than functional separation.
- The next scientific work should examine near-floor null fits and redesign
  discrimination scenarios using functional separation in $q(x,v)$.
