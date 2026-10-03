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
