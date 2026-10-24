# Dynamics-Level Watermarking of Flow Matching Models

Code and manuscript for dynamics-level watermarking through velocity queries.

## Start Here

- [Repository layout and remote-sync policy](docs/REPOSITORY.md)
- [Detailed experiment commands](docs/EXPERIMENTS.md)
- [Frozen baseline-first protocol](baseline_protocol.md)
- [Current saved 32-bit screen](docs/BASELINE_SCREEN.md)
- [Endpoint-preserving construction and standalone training test](docs/ENDPOINT_LOOP.md)
- [Compact result snapshots](results/README.md)
- [Paper source](paper/main.tex)

Run commands from the repository root. Downloaded data, model checkpoints,
environments and full experiment outputs are local artifacts, not Git content.

## Setup

Install the core dependencies in a Python environment:

```bash
pip install -r requirements.txt
```

The public RF-UNet checkpoint uses the UNet implementation from
`keishihara/flow-matching`, not the unrelated PyPI package of the same name.
See the [detailed setup](docs/EXPERIMENTS.md#setup) and the optional
`requirements-audio.txt`, `requirements-stable-audio.txt` and
`requirements-ftss.txt` files for their respective studies.

## Main Entry Points

| Purpose | Entry Point |
|---|---|
| MNIST MLP | `flow_watermark_mnist_mlp.py` |
| Local UNet | `flow_watermark_unet_lora.py` |
| Pretrained RF-UNet | `watermark_hf_flow_unet.py` |
| SD3.5 | `watermark_sd35.py` |
| Audio flow models | `watermark_audio_flow.py` |
| Main architecture comparison | `run_main_architectures.py` |
| Whole-model embedding comparison | `run_embedding_strategy_robustness.py` |
| Removal and transfer | `run_watermark_security.py` |
| Baseline-first sequence | `run_baseline_first_sequence.py` |

Historical diagnostics, pilot runners and their tests remain available.
Negative results are retained as experimental evidence, not removed.

## Current Comparison

```bash
python run_baseline_screen.py
```

Five-bit tuning and the confirmation sequence were stopped at the user's request.
This screen reuses the saved, perfectly decoded 32-bit candidate with the lowest
quality cost per method. It tests whole-model modifications and same/smaller-
architecture transfer with paired controls and 5,000-image FID/KID. It is a
one-key screening comparison, not five-key confirmation. Use the identical
command to resume. Progress is in `table_runs/baseline_screen32.log` and
`table_runs/baseline_screen32/results.json`. The original frozen protocol and
all failed development results remain available; do not restart the old sequence
unless five-bit work and confirmation are intentionally requested again.

## Result Snapshots

```bash
python tools/export_run_summaries.py
```

This exports selected completed watermark studies from `table_runs/` into
`results/`, without checkpoints, owner keys, full score vectors or logs.
It does not change the source runs. The snapshots are compact evidence, not
a replacement for an external archive of full research artifacts.

## Tests

```bash
python -m unittest discover -p 'test_*.py'
```

Some tests and integrations need optional dependencies. Focused tests can be
run by module, for example `python -m unittest test_baseline_first`.

## Paper

Keep `paper/main.tex`, `paper/ref.bib`, `paper/performance_results.tex`, and
the figures referenced by the manuscript together. Compiled PDFs, LaTeX
auxiliaries and unused figure variants are not part of the default Git sync.
