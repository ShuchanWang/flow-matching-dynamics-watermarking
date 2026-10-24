# Repository Layout and Sync Policy

## Navigation

| Location | Purpose | Git remote? |
|---|---|---|
| Root Python runners and tests | Reproducible implementation | Yes |
| `docs/` | Setup, experiment commands, organization policy | Yes |
| `baseline_protocol.md` | Frozen current study design | Yes |
| `paper/` sources and referenced figures | Rebuildable manuscript | Yes |
| `results/` | Compact exported experimental evidence | Yes, review first |
| `table_runs/` | Full local experiment artifacts and logs | No |
| `data/`, `hf_flow_unet_*` | Datasets, downloaded/generated model artifacts | No |
| `.env-stable-audio/`, `.venv-audio/` | Local Python environments | No |
| Compiled paper, caches, IDE files, credentials | Local machine state | No |

The detailed command reference is [EXPERIMENTS.md](EXPERIMENTS.md). FTSS files
form a separate exploratory study: `run_ftss_*`, `ftss_batched.py`,
`run_unmarked_memorization.py`, `run_matched_ftss_mlp.py`, their analysis scripts
and tests. They are not current watermark-paper results.

Do not move the root sources while the baseline sequence runs: its manifests
freeze source paths/hashes, and confirmation imports these modules by name.
Source migration into packages should be a separate tested change after the run,
not mixed with deleting old evidence. No root runner has been deleted here.

## Negative Results

Do not retain only a log. Keep the implementation, settings, training seeds,
per-model recovery/bit metrics, controls, attack costs and a short conclusion.
Logs alone often omit the information needed to reproduce or audit a failure.
Apply the same retention policy to successful and unsuccessful experiments.

`tools/export_run_summaries.py` exports selected completed watermark studies,
including the secret-carrier and tail-gate pilots. It keeps compact per-trial
outcomes but omits decoded messages and large score vectors. Full unmodified
records remain in `table_runs/`; snapshots contain their SHA-256 fingerprints.
An active study is not exported as a completed result.

Full checkpoints, optimizer states, feature tensors and duplicate sample grids
belong in a separate artifact archive, not ordinary Git. Before deleting any,
verify the archive and preserve any checkpoint needed by paper metrics or
follow-up experiments. Duplicate smoke-test outputs are the first candidates
for local cleanup; scientific negative results are not disposable scratch work.
The explicitly authorized cleanup removed disposable baseline/carrier/gate smoke
outputs, Python caches, stale LaTeX auxiliaries and unused figure duplicates.
Scientific study directories, negative pilots and the active baseline run remain.

## What to Sync

1. Reviewed source and focused tests, including negative-result runners.
2. Requirements files, the frozen protocol, README and documentation.
3. `paper/main.tex`, `paper/ref.bib`, `paper/performance_results.tex`.
4. The figures currently included by `main.tex`:
   `sd35_clean_watermarked_trained_v2.pdf` and `payload_query_scaling.pdf`.
5. Reviewed compact `results/` snapshots and their export manifest.

The two obsolete `paper/outputs/*.png` files and unused SD figure variants were
removed after checking that the current paper does not reference them. If the
paper adds a new figure, deliberately add its path to the figure allowlist in
`.gitignore`.

Do not sync owner-key `.pt` files, private carrier state, access tokens, downloaded
weights, datasets, environments, optimizer checkpoints, or all of `table_runs/`.
Publish experimental keys separately only after an explicit disclosure decision.
The export tool removes selected credential fields and local path prefixes; it
is not a substitute for reviewing files before making a public commit.

## Before a Commit

```bash
python tools/export_run_summaries.py
git status --short
git diff --check
git diff --stat
```

Review untracked files too: `git diff` does not show their contents. Stage explicit
paths or logical groups, inspect `git diff --cached`, and then commit/push. Avoid
`git add -f table_runs` or broad `rsync` of the working directory to a code remote.
Staging, committing and pushing require an explicit user request. The cleanup
and sync were subsequently authorized; no datasets or full run directories are
included in that sync.
