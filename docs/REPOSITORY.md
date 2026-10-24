# Repository Layout and Sync Policy

| Location | Purpose |
|---|---|
| `models/` | Model implementations and shared watermark codebooks |
| `experiments/architectures/` | Main comparisons and saved-model inspection |
| `experiments/audio/` | Audio preparation and Stable Audio runners |
| `experiments/baselines/` | Matched baseline comparisons |
| `experiments/endpoint/` | Translation-loop construction, training and sample transfer |
| `experiments/payload/` | Payload limits, queries and embedding steps |
| `experiments/quality/` | Sample quality and paired image figures |
| `experiments/robustness/` | Fine-tuning, pruning and quantization |
| `experiments/security/` | Carrier attacks, transfer and trigger pilots |
| `experiments/verification/` | Recovery and false-positive evaluations |
| `experiments/reporting/` | Aggregation and paper tables |
| `tools/` | Compact evidence export and artifact utilities |
| `results/` | Reviewed compact results, including negative results |
| `table_runs/` | Full local logs, checkpoints and samples; not synced |
| `paper/` | Manuscript sources and referenced figures |

Focused tests live beside their task's runners. Use the single launcher:

```bash
python run.py --list
python run.py experiments/endpoint/run_endpoint_loop_training.py --help
python run.py run_main_architectures.py --help
python run.py test
python run.py test test_endpoint_loop_distillation test_watermark_security
```

Use the launcher rather than running nested scripts directly. It establishes
the project working directory and module paths, including for child processes.
Historical source basenames remain accepted by the launcher;
[source_migration.json](source_migration.json) maps relocated root files.
Dataset and watermark experiment paths are unchanged.

## FTSS Studies

FTSS code, analyses, tests and requirements moved to
`../ftss-memorization/studies/flow_finetuning/` and
`../ftss-memorization/requirements-finetuning.txt`:

```bash
cd ../ftss-memorization
python run_study.py --list
python run_study.py run_ftss_cifar_study.py --help
python run_study.py test
```

Local FTSS run folders and logs moved to that repository's `table_runs/`.
Compatibility symlinks at their old locations preserve recorded checkpoint
paths on this machine. Those symlinks are not remote content. FTSS reuses this
repository's model helpers, data and public base checkpoints; watermark tasks
do not depend on the FTSS repository.

## Reproducibility

Pre-relocation source versions are preserved in watermark commit `d07f9ea`.
Path edits change runner fingerprints; historical manifests and measurements
have not been rewritten to hide that change. A frozen run may reject resume
after relocation. Use its historical source checkout to resume the exact run,
or a new output directory for the organized runner. Do not disable the checks.

## Retention and Sync

Keep implementations, settings, seeds, controls and compact per-run evidence
for successful and negative experiments. A log alone is insufficient.
`python tools/export_run_summaries.py` refreshes watermark snapshots.
FTSS has its own `export_study_summaries.py` command.

Sync source, tests, requirements, documentation, manuscript sources, referenced
figures and reviewed compact summaries. Do not sync datasets, model weights,
optimizer states, owner-key files, credentials, environments or full run
folders. Full evidence remains local; this reorganization deleted no scientific
results.
