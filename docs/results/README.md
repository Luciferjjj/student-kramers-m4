# Research result snapshot

This directory contains the compact numerical evidence used by
`M4_GREENLAND_RESEARCH_REPORT.md`. It is tracked by Git so a collaborator can
inspect the reported values without downloading local bootstrap checkpoints.

## Evidence classes

`current/` contains results produced with the globally feasible Cholesky M4
optimizer. These tables support the current real-data, predictive, parameter
bootstrap, exact IOS, and model-wise IOS conclusions.

`development/` contains the earlier recovery, discrimination, and M3-null
bootstrap experiments. Those runs used the direct-coefficient M4 optimizer
before the Cholesky parameterization was introduced. They document the
research process and the design of the experiments, but they are not formal
evidence for the current fitted M4 model.

The file `manifest.csv` records the local source and evidence status of every
copied table.

## What is not included

The snapshot does not contain:

- the official Greenland Excel workbook;
- every leave-one-out IOS checkpoint;
- every simulated trajectory;
- intermediate caches or failed temporary runs.

These files are either downloadable, reproducible from the command modules,
or too large and repetitive for source control.

## Refresh command

From the repository root:

```bash
python3 -m greenland_application.run_report_assets --refresh-snapshot
```

This command first refreshes the selected tables from local runs and then
rebuilds all report figures.
