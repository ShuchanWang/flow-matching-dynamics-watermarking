"""Hugging Face flow-matching UNet watermark experiments.

This script evaluates dynamics-level watermarking on public pretrained
flow-matching checkpoints instead of training a clean base model locally.
It downloads a pretrained velocity model from Hugging Face, freezes it,
adds small LoRA adapters, fine-tunes only those adapters for a watermark,
and writes table-ready JSON/CSV metrics.

Default checkpoints are from:
  WayBob/FlowMatching-Unet-Celeb-64x64

Supported default rows:
  - MNIST flow-matching UNet
  - CIFAR-10 flow-matching UNet
  - CelebA-64 flow-matching UNet

The script intentionally does not cover SD 1.4/SDXL, because those are
diffusion denoisers rather than flow/rectified-flow velocity models.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Subset
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm


@dataclass(frozen=True)
class DatasetConfig:
    filename: str
    dim: tuple[int, int, int]
    num_channels: int
    num_res_blocks: int
    num_classes: int
    class_cond: bool
    real_dataset: str | None
    params_label: str


DEFAULT_CONFIGS = {
    "mnist": DatasetConfig(
        filename="mnist/ckpt.pth",
        dim=(1, 28, 28),
        num_channels=64,
        num_res_blocks=2,
        num_classes=10,
        class_cond=True,
        real_dataset="mnist",
        params_label="6.2M",
    ),
    "cifar10": DatasetConfig(
        filename="cifar10/ckpt.pth",
        dim=(3, 32, 32),
        num_channels=64,
        num_res_blocks=2,
        num_classes=10,
        class_cond=True,
        real_dataset="cifar10",
        params_label="9.0M",
    ),
    "celeba64": DatasetConfig(
        filename="celeba64/ckpt.pth",
        dim=(3, 64, 64),
        num_channels=128,
        num_res_blocks=2,
        num_classes=0,
        class_cond=False,
        real_dataset=None,
        params_label="83.0M",
    ),
}


class LoRALinear(nn.Module):
    def __init__(self, layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.layer = layer
        self.scale = alpha / max(rank, 1)
        self.lora_A = nn.Linear(layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.layer(x) + self.scale * self.lora_B(self.lora_A(x))


class LoRAConv2d(nn.Module):
    def __init__(self, layer: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        self.layer = layer
        self.scale = alpha / max(rank, 1)
        self.lora_A = nn.Conv2d(layer.in_channels, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, layer.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.layer(x) + self.scale * self.lora_B(self.lora_A(x))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="WayBob/FlowMatching-Unet-Celeb-64x64")
    parser.add_argument("--dataset", choices=sorted(DEFAULT_CONFIGS), default="cifar10")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="hf_flow_unet_wm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wm_message", default="10101")
    parser.add_argument("--wm_K", type=int, default=32)
    parser.add_argument("--wm_eps", type=float, default=0.2)
    parser.add_argument("--wm_lambda", type=float, default=0.01)
    parser.add_argument("--n_queries", type=int, default=4096)
    parser.add_argument("--n_detect_trials", type=int, default=20)
    parser.add_argument("--n_stat_trials", type=int, default=30)

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_targets", choices=["conv", "linear", "both"], default="conv")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_every", type=int, default=100)

    parser.add_argument("--n_train_samples", type=int, default=20000)
    parser.add_argument(
        "--data_source",
        choices=["auto", "real", "base_samples"],
        default="auto",
        help=(
            "Training data for LoRA watermarking. `auto` uses torchvision real "
            "data when available, otherwise samples from the downloaded base model."
        ),
    )
    parser.add_argument("--base_sample_pool", default=None)
    parser.add_argument("--base_sample_batch_size", type=int, default=128)
    parser.add_argument("--n_fid_samples", type=int, default=500)
    parser.add_argument("--n_sample_steps", type=int, default=100)
    parser.add_argument("--eval_class", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--sweep",
        choices=["none", "queries", "epsilon", "bits", "finetune"],
        default="none",
        help="Optional table-oriented sweep. Uses the same downloaded base checkpoint.",
    )
    parser.add_argument("--query_values", default="16,32,64,128,256,512,1024,4096")
    parser.add_argument("--epsilon_values", default="0.1,0.5,1.0,1.5,3.0,5.0")
    parser.add_argument("--bit_values", default="1,3,5,8,12")
    parser.add_argument("--finetune_values", default="0,100,500,1000")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_unet_model():
    install_hint = (
        "pip uninstall -y flow-matching && "
        "pip install 'git+https://github.com/keishihara/flow-matching.git' "
        "torchdiffeq einops"
    )
    try:
        module = importlib.import_module("flow_matching.models")
        return module.UNetModel
    except Exception:
        pass

    try:
        module = importlib.import_module("flow_matching.models.unet")
        return module.UNetModelWrapper
    except Exception as exc:
        raise RuntimeError(
            "Could not import the UNet implementation used by the Hugging Face "
            "checkpoint. These checkpoints follow keishihara/flow-matching, not "
            "the unrelated PyPI `flow-matching` package. On the remote machine, "
            f"run:\n\n  {install_hint}\n"
        ) from exc


def download_checkpoint(repo_id: str, filename: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("Install huggingface_hub on the remote machine.") from exc
    return hf_hub_download(repo_id=repo_id, filename=filename)


def build_model(cfg: DatasetConfig, device: torch.device):
    UNetModel = import_unet_model()
    return UNetModel(
        dim=cfg.dim,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
        num_classes=cfg.num_classes,
        class_cond=cfg.class_cond,
    ).to(device)


def load_state_dict_flexible(model: nn.Module, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("model", "ema", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    cleaned = {}
    for key, val in ckpt.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = val
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)}")


def add_lora(module: nn.Module, rank: int, alpha: float, targets: str, prefix: str = ""):
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and targets in ("conv", "both"):
            if child.kernel_size == (1, 1) or child.stride != (1, 1):
                continue
            setattr(module, name, LoRAConv2d(child, rank, alpha))
        elif isinstance(child, nn.Linear) and targets in ("linear", "both"):
            setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            add_lora(child, rank, alpha, targets, full_name)


def freeze_non_lora(model: nn.Module):
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No LoRA parameters were added. Try --lora_targets both.")
    return params


def make_codebook(D: int, n_bits: int, K: int, device: torch.device):
    n_messages = 2 ** n_bits
    with torch.no_grad():
        P_raw = torch.randn(D, K, device=device, dtype=torch.float32)
        Q_p, _ = torch.linalg.qr(P_raw)
        P = Q_p[:, :K]
        codes_raw = torch.randn(n_messages, K, device=device, dtype=torch.float32)
        Q_c, _ = torch.linalg.qr(codes_raw.T)
        codes = Q_c.T
        codes = codes / codes.norm(dim=1, keepdim=True)
    codebook = {}
    for idx in range(n_messages):
        bits = tuple((idx >> i) & 1 for i in range(n_bits))
        codebook[bits] = codes[idx]
    return P, codes, codebook


def load_real_data(cfg: DatasetConfig, args, device: torch.device):
    if cfg.real_dataset is None:
        return None, None
    c, h, w = cfg.dim
    if cfg.real_dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    elif cfg.real_dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(cfg.real_dataset)

    n_train = min(args.n_train_samples, len(train))
    train_loader = DataLoader(
        Subset(train, range(n_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    real_loader = DataLoader(
        Subset(test, range(min(args.n_fid_samples, len(test)))),
        batch_size=min(args.n_fid_samples, len(test)),
        shuffle=False,
        num_workers=args.num_workers,
    )
    real_images = next(iter(real_loader))[0].to(device)
    if real_images.shape[1:] != (c, h, w):
        raise RuntimeError(f"Real data shape {real_images.shape[1:]} != {(c, h, w)}")
    return train_loader, real_images


@torch.no_grad()
def build_base_sample_loader(base_model, cfg, args, device: torch.device):
    pool_path = None if args.base_sample_pool is None else Path(args.base_sample_pool)
    if pool_path is not None and pool_path.exists():
        print(f"Loading base sample pool: {pool_path}")
        samples = torch.load(pool_path, map_location="cpu")
        if isinstance(samples, dict):
            samples = samples.get("samples", samples.get("x"))
        samples = samples[: args.n_train_samples]
    else:
        print(f"Generating {args.n_train_samples} base samples for LoRA pseudo-data...")
        base_model.eval()
        batches = []
        remaining = args.n_train_samples
        while remaining > 0:
            b = min(args.base_sample_batch_size, remaining)
            batch = sample(base_model, cfg, args, device, b).cpu()
            batches.append(batch)
            remaining -= b
        samples = torch.cat(batches, dim=0)
        if pool_path is not None:
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"samples": samples, "dataset": args.dataset}, pool_path)
            print(f"Saved base sample pool: {pool_path}")

    dataset = TensorDataset(samples)

    def collate(batch):
        x = torch.stack([item[0] for item in batch], dim=0)
        y = torch.full((x.shape[0],), args.eval_class, dtype=torch.long)
        return x, y

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )


def labels_for_batch(cfg: DatasetConfig, batch_size: int, args, device: torch.device):
    if not cfg.class_cond:
        return None
    return torch.full((batch_size,), args.eval_class, device=device, dtype=torch.long)


def model_velocity(model, x, t, y=None):
    try:
        return model(x=x, t=t, y=y)
    except TypeError:
        try:
            return model(x, t, y)
        except TypeError:
            return model(x, t)


@torch.no_grad()
def get_base_velocity(base, x, t, y):
    return model_velocity(base, x, t, y)


def train_lora(model, base_model, train_loader, cfg, P, wm_code, args, device, out_dir):
    c, h, w = cfg.dim
    opt_params = freeze_non_lora(model)
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(args.steps, 1))
    model.train()
    base_model.eval()

    data_iter = iter(train_loader) if train_loader is not None else None
    history = []
    pbar = tqdm(range(args.steps), desc="LoRA watermark")
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_vel = 0.0
        total_corr = 0.0
        for _ in range(args.grad_accum):
            if train_loader is None:
                x1 = torch.randn(args.batch_size, c, h, w, device=device)
                y = labels_for_batch(cfg, args.batch_size, args, device)
            else:
                try:
                    x1, y_data = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x1, y_data = next(data_iter)
                x1 = x1.to(device)
                y = y_data.to(device) if cfg.class_cond else None

            b = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(b, device=device)
            t_view = t.view(b, 1, 1, 1)
            x_t = (1 - t_view) * x0 + t_view * x1
            u_true = x1 - x0
            carrier = torch.sin(2 * math.pi * t).view(b, 1)
            wm_flat = args.wm_eps * carrier * (P @ wm_code).view(1, -1)
            wm = wm_flat.view(b, c, h, w)

            pred = model_velocity(model, x_t, t, y)
            with torch.no_grad():
                base_pred = get_base_velocity(base_model, x_t, t, y)
            residual = pred - base_pred
            loss_vel = F.mse_loss(residual.float(), wm.float())

            proj = residual.reshape(b, -1).float() @ P
            wm_corr = (carrier * proj * wm_code.view(1, -1)).sum(dim=1).mean()
            loss = loss_vel - args.wm_lambda * wm_corr
            loss = loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()
            total_vel += loss_vel.item()
            total_corr += wm_corr.item()

        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        opt.step()
        sched.step()
        row = {
            "step": step + 1,
            "loss": total_loss,
            "residual_mse": total_vel / args.grad_accum,
            "wm_corr": total_corr / args.grad_accum,
        }
        history.append(row)
        if step % 10 == 0:
            pbar.set_postfix(mse=f"{row['residual_mse']:.5f}", corr=f"{row['wm_corr']:.4f}")
        if args.save_every and (step + 1) % args.save_every == 0:
            save_adapter(model, out_dir / f"adapter_step_{step + 1}.pt")
    return history


def save_adapter(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(state, path)


@torch.no_grad()
def detect(model, cfg, P, codes, codebook, args, device, n_queries=None):
    n_queries = n_queries or args.n_queries
    c, h, w = cfg.dim
    signature = torch.zeros(args.wm_K, device=device)
    all_codes = codes.to(device)
    keys = list(codebook.keys())
    for start in range(0, n_queries, args.batch_size):
        b = min(args.batch_size, n_queries - start)
        x = torch.randn(b, c, h, w, device=device)
        t = torch.rand(b, device=device)
        y = labels_for_batch(cfg, b, args, device)
        v = model_velocity(model, x, t, y)
        carrier = torch.sin(2 * math.pi * t).view(b, 1)
        signature += (carrier * (v.reshape(b, -1).float() @ P)).sum(dim=0)
    signature /= n_queries
    scores = signature.view(1, -1) @ all_codes.T
    best_idx = int(scores.argmax(dim=1).item())
    return keys[best_idx], scores.squeeze(0).detach().cpu().numpy(), signature.detach().cpu().numpy()


@torch.no_grad()
def sample(model, cfg, args, device, n_samples, use_classes=True):
    c, h, w = cfg.dim
    x = torch.randn(n_samples, c, h, w, device=device)
    y = labels_for_batch(cfg, n_samples, args, device) if use_classes else None
    time_grid = torch.linspace(0, 1, args.n_sample_steps + 1, device=device)
    for i in range(args.n_sample_steps):
        t0 = time_grid[i]
        t1 = time_grid[i + 1]
        t = torch.full((n_samples,), float(t0), device=device)
        v = model_velocity(model, x, t, y)
        x = x + (t1 - t0) * v
        x = torch.clamp(x, -5, 5)
    return torch.clamp(x, -1, 1)


def compute_fid(real, gen):
    real = real.detach().cpu().reshape(real.shape[0], -1).numpy()
    gen = gen.detach().cpu().reshape(gen.shape[0], -1).numpy()
    mu_r, sigma_r = real.mean(0), np.cov(real, rowvar=False)
    mu_g, sigma_g = gen.mean(0), np.cov(gen, rowvar=False)
    eps = 1e-6
    sigma_r += eps * np.eye(sigma_r.shape[0])
    sigma_g += eps * np.eye(sigma_g.shape[0])
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))


def evaluate(model, base_model, cfg, P, codes, codebook, true_msg, args, device, real_images=None):
    true_bits = tuple(int(b) for b in true_msg)
    true_idx = list(codebook.keys()).index(true_bits)

    wm_hits = 0
    clean_hits = 0
    wm_scores = []
    clean_scores = []
    for _ in range(args.n_detect_trials):
        decoded, scores, _ = detect(model, cfg, P, codes, codebook, args, device)
        wm_hits += int(decoded == true_bits)
        wm_scores.append(float(scores[true_idx]))
        decoded_clean, clean_scores_arr, _ = detect(base_model, cfg, P, codes, codebook, args, device)
        clean_hits += int(decoded_clean == true_bits)
        clean_scores.append(float(clean_scores_arr[true_idx]))

    wm_mean = float(np.mean(wm_scores))
    clean_mean = float(np.mean(clean_scores))
    wm_std = float(np.std(wm_scores))
    clean_std = float(np.std(clean_scores))
    sep = (wm_mean - clean_mean) / max(wm_std, clean_std, 1e-8)

    metrics = {
        "wm_acc": 100.0 * wm_hits / args.n_detect_trials,
        "clean_fp": 100.0 * clean_hits / args.n_detect_trials,
        "sep_sigma": float(sep),
        "wm_score_mean": wm_mean,
        "clean_score_mean": clean_mean,
        "wm_score_std": wm_std,
        "clean_score_std": clean_std,
    }

    if real_images is not None and args.n_fid_samples > 1:
        n = min(args.n_fid_samples, real_images.shape[0])
        clean_samples = sample(base_model, cfg, args, device, n)
        wm_samples = sample(model, cfg, args, device, n)
        fid_clean = compute_fid(real_images[:n], clean_samples)
        fid_wm = compute_fid(real_images[:n], wm_samples)
        metrics.update({
            "fid_clean": fid_clean,
            "fid_wm": fid_wm,
            "fid_ratio": fid_wm / max(fid_clean, 1e-8),
        })
    return metrics


def run_query_sweep(model, base_model, cfg, P, codes, codebook, true_msg, args, device):
    rows = []
    saved_n = args.n_queries
    for n in [int(x) for x in args.query_values.split(",") if x]:
        args.n_queries = n
        rows.append({"sweep": "queries", "value": n, **evaluate(
            model, base_model, cfg, P, codes, codebook, true_msg, args, device, None
        )})
    args.n_queries = saved_n
    return rows


def write_outputs(out_dir: Path, config: dict, metrics: dict, history: list[dict], sweep_rows: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": config, "metrics": metrics, "history": history, "sweeps": sweep_rows}, f, indent=2)
    flat = [{"kind": "main", **metrics}] + sweep_rows
    if flat:
        fieldnames = sorted({k for row in flat for k in row.keys()})
        with open(out_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat)


def main():
    args = parse_args()
    if args.quick:
        args.steps = min(args.steps, 20)
        args.n_train_samples = min(args.n_train_samples, 512)
        args.n_fid_samples = min(args.n_fid_samples, 32)
        args.n_queries = min(args.n_queries, 256)
        args.n_detect_trials = min(args.n_detect_trials, 5)
        args.n_sample_steps = min(args.n_sample_steps, 20)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = DEFAULT_CONFIGS[args.dataset]
    out_dir = Path(args.output_dir) / args.dataset / args.wm_message
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset config: {asdict(cfg)}")
    print(f"Device: {device}")

    ckpt_file = args.checkpoint or cfg.filename
    ckpt_path = download_checkpoint(args.repo_id, ckpt_file)
    print(f"Checkpoint: {ckpt_path}")

    base_model = build_model(cfg, device)
    load_state_dict_flexible(base_model, ckpt_path, device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    model = build_model(cfg, device)
    load_state_dict_flexible(model, ckpt_path, device)
    add_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)

    D = int(np.prod(cfg.dim))
    wm_bits = tuple(int(b) for b in args.wm_message)
    P, codes, codebook = make_codebook(D, len(wm_bits), args.wm_K, device)
    wm_code = codebook[wm_bits]

    real_images = None
    if args.data_source == "real" or (args.data_source == "auto" and cfg.real_dataset is not None):
        train_loader, real_images = load_real_data(cfg, args, device)
    elif args.data_source == "auto" or args.data_source == "base_samples":
        train_loader = build_base_sample_loader(base_model, cfg, args, device)
    else:
        raise ValueError(args.data_source)
    t0 = time.time()
    history = train_lora(model, base_model, train_loader, cfg, P, wm_code, args, device, out_dir)
    elapsed_min = (time.time() - t0) / 60
    save_adapter(model, out_dir / "adapter_final.pt")
    torch.save(
        {"P": P.cpu(), "codes": codes.cpu(), "message": args.wm_message, "config": vars(args)},
        out_dir / "watermark_key.pt",
    )

    metrics = evaluate(model, base_model, cfg, P, codes, codebook, args.wm_message, args, device, real_images)
    metrics.update({
        "dataset": args.dataset,
        "model": "HF FlowMatching UNet",
        "repo_id": args.repo_id,
        "checkpoint": ckpt_file,
        "params": cfg.params_label,
        "train": "LoRA",
        "bits": len(wm_bits),
        "N": args.n_queries,
        "steps": args.steps,
        "elapsed_min": elapsed_min,
    })

    sweep_rows = []
    if args.sweep == "queries":
        sweep_rows = run_query_sweep(model, base_model, cfg, P, codes, codebook, args.wm_message, args, device)
    elif args.sweep != "none":
        print(f"[note] Sweep {args.sweep!r} needs one run per value to retrain LoRA fairly.")
        print("       Use multiple invocations with --wm_eps, --wm_message, or --steps.")

    write_outputs(out_dir, vars(args), metrics, history, sweep_rows)

    print("\nTable-ready summary")
    print(f"  Dataset:    {args.dataset}")
    print(f"  WM acc:     {metrics['wm_acc']:.1f}%")
    print(f"  Clean FP:   {metrics['clean_fp']:.1f}%")
    print(f"  Sep:        {metrics['sep_sigma']:.2f}")
    if "fid_ratio" in metrics:
        print(f"  FID ratio:  {metrics['fid_ratio']:.3f}")
    print(f"  Outputs:    {out_dir}")


if __name__ == "__main__":
    main()
