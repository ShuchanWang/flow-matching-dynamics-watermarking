# Dynamics-Level Watermarking of Flow Matching Models

This repository contains the experiment code for the paper:

**Dynamics-Level Watermarking of Flow Matching Models with Random Codes**

Paper: [arXiv:2605.16239](https://arxiv.org/abs/2605.16239)

The method embeds a keyed, multi-bit watermark directly into the learned
velocity field of a flow-matching generative model. A secret projection matrix
and codebook define a time-modulated perturbation during training, and the
message is recovered later from black-box velocity queries by synchronous
demodulation.

## Files

- `flow_watermark_mnist_mlp.py`  
  MNIST MLP experiments with clean and watermarked flow-matching models,
  detection metrics, signature statistics, FID-style sample-quality analysis,
  and visualizations.

- `flow_watermark_unet_lora.py`  
  MNIST/CIFAR-10 UNet experiments with checkpoint resume support and LoRA
  watermark fine-tuning.

- `watermark_hf_flow_unet.py`  
  Remote-run experiments that download public Hugging Face flow-matching
  UNet checkpoints, freeze the base model, and train only lightweight LoRA
  watermark adapters. This is intended for filling the extended evaluation
  tables without retraining a clean base model from scratch.

- `watermark_sd35.py`  
  Stable Diffusion 3.5 Medium LoRA watermark experiments for the main
  image-generation rows and SD ablations.

- `run_table_experiments.py`  
  Generates the remote command grid for 5-bit SD3.5, cross-detection, ablations,
  and public Hugging Face flow-UNet experiments. It writes a manifest and can
  optionally execute the jobs on a GPU machine.

- `collect_results.py`  
  Aggregates per-message experiment outputs and reports mean/std across
  watermark messages for table-ready summaries.

- `paper/`  
  LaTeX source, bibliography, and figures for the arXiv paper.

## Setup

Create an environment with Python 3.10+ and install the dependencies:

```bash
pip install -r requirements.txt
```

For the Hugging Face checkpoint runner, make sure the model-code dependency
comes from `keishihara/flow-matching`. If a different PyPI package named
`flow-matching` is already installed, replace it with:

```bash
pip uninstall -y flow-matching
pip install "git+https://github.com/keishihara/flow-matching.git" torchdiffeq einops
```

The scripts download MNIST or CIFAR-10 through `torchvision` when needed.
Training is GPU-oriented and may be slow on CPU.

## Running Experiments

Run the MNIST MLP experiment:

```bash
python flow_watermark_mnist_mlp.py
```

Run the UNet + LoRA experiment:

```bash
python flow_watermark_unet_lora.py
```

In `flow_watermark_unet_lora.py`, set:

```python
DATASET = "mnist"    # or "cifar10"
```

before running. Checkpoints are written under `checkpoints/` or
`checkpointsCIFAR/`, and figures are written under `outputs/`.

Run a Hugging Face checkpoint-based flow-matching evaluation:

```bash
python watermark_hf_flow_unet.py --dataset cifar10 --wm_message 10101
```

Useful table-oriented variants:

```bash
python watermark_hf_flow_unet.py --dataset mnist --wm_message 10101
python watermark_hf_flow_unet.py --dataset cifar10 --wm_message 10101 --sweep queries
python watermark_hf_flow_unet.py --dataset cifar10 --wm_message 10101 --wm_eps 0.5
python watermark_hf_flow_unet.py --dataset celeba64 --data_source base_samples --base_sample_pool cache/celeba64_base_samples.pt
```

The script writes `results.json`, `results.csv`, LoRA adapters, and the
watermark key under `hf_flow_unet_wm/`.

Generate the broader table experiment grid without running it:

```bash
python run_table_experiments.py --out_root table_runs --manifest table_runs/manifest.json
```

Generate only SD3.5 ablations:

```bash
python run_table_experiments.py --only sd35-query,sd35-epsilon,sd35-rank,sd35-steps --slurm table_runs/run_sd35_grid.sh
```

On a GPU node, add `--run` to execute directly. The HF flow-UNet rows are
supporting evidence only; include them in the paper if the resulting
watermark accuracy and separation are strong enough.

After runs finish, aggregate across watermark messages:

```bash
python collect_results.py --manifest table_runs/manifest.json --out_dir table_runs/summary
```

Use `table_runs/summary/aggregate.csv` for table entries; it reports
mean, standard deviation, minimum, and maximum for detection, false-positive
rate, separation, score margin, endpoint drift, and FID ratio.

### Fixed-protocol payload capacity

The payload-capacity sweep varies message length while keeping the CIFAR-10
checkpoint, projection dimension (`K=32`), detection budget (`N=4096`),
LoRA training, and FID protocol fixed. It uses normalized random codebooks
at every length (5, 8, 12, 16, 18, and 20 bits), with five sampled messages
and 40 detection trials per message. Report exact-message accuracy, the
true-versus-nearest-competitor score margin, and FID ratio. If accuracy
remains perfect at the largest tested length, report a tested lower bound
rather than an estimated maximum capacity.

Run the sweep from the repository root on the GPU machine. The runner
creates `table_runs/` itself, and `--skip_existing` resumes after an
interruption without repeating completed messages:

```bash
nohup python -u run_table_experiments.py \
  --only flow-unet-payload-capacity --run --skip_existing \
  --manifest table_runs/manifest_payload_capacity.json \
  > payload_capacity.log 2>&1 &
```

After it finishes, aggregate the five messages at each payload size:

```bash
python collect_results.py --manifest table_runs/manifest_payload_capacity.json \
  --out_dir table_runs/summary_payload_capacity
```

## Main Architecture Experiments

The main grid uses five-bit orthogonal codes and 32-bit implicit hypercube
codes, both at `K=32`, with five messages per configuration. The implicit
implementation never allocates the `2**32` possible codewords. A separate
five-bit hypercube grid provides the matched-payload encoding ablation.

Generate the image experiment manifest (without starting training):

```bash
python run_main_architectures.py --families mlp unet sd35 \
  --schemes orthogonal5 hypercube32
```

Add `--run --skip_existing` to execute or resume. This covers HF flow UNets
on MNIST, CIFAR-10, and CelebA-64, and SD3.5's DiT. New output directories
are separate from earlier experiments; existing results in the older
directories are not automatically reused. Use `--schemes hypercube32` to
run only the new 32-bit configurations. Add `hypercube5` for the matched
encoding ablation. SD3.5 retains its existing 256-query protocol, while the
UNets and MLP use 4,096 queries; these are not controlled architecture comparisons
at a common query budget.

Add `--reuse_legacy` to reuse configuration-compatible completed results
from the older `hf_flow_unet`, `flow_unet_payload_hypercube`, `sd35_msg_*`,
and `classic_mlp` directories. The manifest points to the original outputs
without copying or relabeling measurements. A larger saved detection trial
count is accepted and must retain its original count in reporting; different
training budgets or codebooks are not reused. `--continue_on_error` attempts
later jobs after a failure and returns nonzero if any jobs failed.

The original MNIST MLP is included with full-weight scratch training for
5,000 updates per clean/watermarked model and its existing MNIST
feature-space quality diagnostic. Its budget and quality metric differ
from the pretrained image models and must be labeled accordingly.

### Audio Extension

Install `requirements-audio.txt` in a suitable environment before preparing
audio. The latent-space MLP and Transformer use a frozen Music2Latent codec.
Stable Audio LoRA uses its own frozen codec and requires a checkpoint whose
objective is `rectified_flow`; incompatible objectives are rejected.
Music2Latent is a codec here, not a third generative architecture.

Prepare local recordings using their official train/test splits:

FMA-small audio and metadata can be downloaded from the official host with
`python download_audio_datasets.py`. Downloads resume from partial files,
are checked against the publisher's SHA-1 hashes, and are extracted into
`data/audio/`. Reserve at least 20 GiB for the archives and extraction.
Use `data/audio/fma_small` as the FMA root and
`data/audio/fma_metadata/tracks.csv` as its metadata path. Downloading does
not automatically launch codec preparation or GPU training.

```bash
python prepare_audio_latents.py --dataset maestro \
  --root /path/to/maestro-v3.0.0 \
  --metadata /path/to/maestro-v3.0.0/maestro-v3.0.0.csv \
  --output table_runs/cache/maestro_music2latent.pt

python prepare_audio_latents.py --dataset fma \
  --root /path/to/fma_small --metadata /path/to/fma_metadata/tracks.csv \
  --output table_runs/cache/fma_music2latent.pt
```

Defaults select up to 2,000 training clips and 500 held-out clips, taking
the first six seconds of each eligible recording. These are subset studies,
not whole-dataset training. Short recordings are excluded; validation is
not used for training or testing. Increase the preparation limits for a
larger study. The clean MLP/Transformer baseline receives 5,000 optimizer
updates and is cached for reuse across messages and codebooks; watermark
training receives 1,000 updates.

Run both main configurations on the two audio flow architectures:

```bash
python run_main_architectures.py --families audio \
  --audio-pools maestro=table_runs/cache/maestro_music2latent.pt \
    fma=table_runs/cache/fma_music2latent.pt \
  --schemes orthogonal5 hypercube32 \
  --manifest table_runs/manifest_main_audio.json --run --skip_existing
```

For Stable Audio, prepare a separate cache with `--codec stable` and
`--stable-model MODEL_ID`, then pass it as `--stable-pools DATASET=PATH`
instead. Verify that the chosen checkpoint uses rectified flow before a
full run; pretrained model access and dependencies may require setup.

Audio detection uses held-out interpolations, 4,096 queries, and 20 trials
per message, with clean-model and independent wrong-key controls. Quality
uses paired clean/watermarked generation and 500 held-out reference clips,
reported as **FD-CLAP**, not image FID or VGGish-based FAD. Do not pool audio
and image quality metrics. `--skip_quality` is only for smoke tests and
marks results incomplete.

Aggregate each manifest after its runs finish:

```bash
python collect_results.py --manifest table_runs/manifest_main_architectures.json \
  --out_dir table_runs/summary_main_architectures
python collect_results.py --manifest table_runs/manifest_main_audio.json \
  --out_dir table_runs/summary_main_audio
```

The collector keeps architectures, datasets, codebooks, and payloads
separate. These runners do not insert unmeasured values into `paper/main.tex`.
Run implementation checks with `python -m unittest test_main_architectures`.

## Reproducibility

The scripts set random seeds for Python, NumPy, and PyTorch. Results may still
vary slightly across hardware and CUDA/cuDNN versions.

Downloaded datasets, checkpoints, and generated outputs are not committed to
the repository; they are created when running the scripts.

## Citation

If this code is useful for your work, please cite the accompanying paper:

```bibtex
@article{wang2026dynamicswatermark,
  title  = {Dynamics-Level Watermarking of Flow Matching Models with Random Codes},
  author = {Wang, Shuchan},
  year   = {2026},
  eprint = {2605.16239},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url    = {https://arxiv.org/abs/2605.16239}
}
```
