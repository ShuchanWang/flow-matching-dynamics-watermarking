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

- `paper/`  
  LaTeX source, bibliography, and figures for the arXiv paper.

## Setup

Create an environment with Python 3.10+ and install the dependencies:

```bash
pip install -r requirements.txt
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

## Reproducibility Notes

Both scripts fix random seeds for Python, NumPy, and PyTorch. Exact results can
still vary across hardware, CUDA/cuDNN versions, and stochastic training order.

The repository intentionally does not track downloaded datasets, generated
outputs, or model checkpoints. These artifacts are reproducible from the
scripts and can be large.

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
