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
        self.lora_A.to(device=layer.weight.device, dtype=layer.weight.dtype)
        self.lora_B.to(device=layer.weight.device, dtype=layer.weight.dtype)
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
        self.lora_A.to(device=layer.weight.device, dtype=layer.weight.dtype)
        self.lora_B.to(device=layer.weight.device, dtype=layer.weight.dtype)
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
    parser.add_argument("--wm_eps", type=float, default=0.5)
    parser.add_argument("--wm_lambda", type=float, default=0.2)
    parser.add_argument("--wm_proj_weight", type=float, default=1.0)
    parser.add_argument("--wm_tanh_scale", type=float, default=1.0)
    parser.add_argument(
        "--objective",
        choices=["full", "residual"],
        default="full",
        help=(
            "`full` continues flow-matching training toward u_true + watermark. "
            "`residual` matches only model-base residual to the watermark."
        ),
    )
    parser.add_argument(
        "--detect_distribution",
        choices=["train_interp", "noise"],
        default="train_interp",
        help="Query distribution for detection. `train_interp` matches training x_t.",
    )
    parser.add_argument("--n_queries", type=int, default=4096)
    parser.add_argument("--n_detect_trials", type=int, default=20)
    parser.add_argument("--n_stat_trials", type=int, default=30)

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_targets", choices=["conv", "linear", "both"], default="both")
    parser.add_argument(
        "--train_extra",
        default="time_embed,out.",
        help=(
            "Comma-separated name fragments for non-LoRA parameters to train. "
            "Use `out.` rather than `out`, otherwise modules such as "
            "`output_blocks` may be unintentionally unfrozen."
        ),
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--post_ft_steps",
        type=int,
        default=0,
        help=(
            "After watermark training, continue clean flow-matching fine-tuning "
            "for this many steps before evaluation. This probes robustness to "
            "ordinary downstream fine-tuning."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--adapter_path", default=None)

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
    parser.add_argument("--fid_batch_size", type=int, default=16)
    parser.add_argument("--fid_feature", choices=["inception", "pixel"], default="inception")
    parser.add_argument(
        "--fid_reference",
        choices=["auto", "real", "base_samples", "none"],
        default="auto",
        help=(
            "`real` uses torchvision/ImageFolder real data, `base_samples` uses "
            "held-out base-model samples as the reference, and `auto` uses real "
            "data when available otherwise base samples."
        ),
    )
    parser.add_argument(
        "--real_data_dir",
        default=None,
        help="Optional ImageFolder root for real FID reference, useful for CelebA64.",
    )
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


def freeze_non_lora(model: nn.Module, train_extra: str = ""):
    extra_tokens = [tok for tok in train_extra.split(",") if tok]
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name or any(tok in name for tok in extra_tokens)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No LoRA parameters were added. Try --lora_targets both.")
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_train:,} / {n_total:,} ({100*n_train/n_total:.3f}%)")
    if n_train / max(n_total, 1) > 0.25:
        print(
            "[warning] More than 25% of the model is trainable. "
            "Check --train_extra; a broad token such as `out` may match `output_blocks`."
        )
    return params


def make_codebook(D: int, n_bits: int, K: int, device: torch.device):
    n_messages = 2 ** n_bits
    if K > D:
        raise ValueError(
            f"wm_K={K} exceeds flattened data dimension D={D}. "
            "Choose a smaller --wm_K or a larger latent/image dimension."
        )
    with torch.no_grad():
        P_raw = torch.randn(D, K, device=device, dtype=torch.float32)
        Q_p, _ = torch.linalg.qr(P_raw)
        P = Q_p[:, :K]
        codes_raw = torch.randn(n_messages, K, device=device, dtype=torch.float32)
        if n_messages <= K:
            Q_c, _ = torch.linalg.qr(codes_raw.T)
            codes = Q_c.T
        else:
            print(
                f"[codebook] Using overcomplete random codebook: "
                f"{n_messages} messages in K={K} dimensions.",
                flush=True,
            )
            codes = codes_raw
        codes = codes / codes.norm(dim=1, keepdim=True)
    codebook = {}
    for idx in range(n_messages):
        bits = tuple((idx >> i) & 1 for i in range(n_bits))
        codebook[bits] = codes[idx]
    return P, codes, codebook


def load_real_data(cfg: DatasetConfig, args, device: torch.device):
    if cfg.real_dataset is None and args.real_data_dir is None:
        return None, None
    c, h, w = cfg.dim
    if args.real_data_dir is not None:
        transform = transforms.Compose([
            transforms.Resize(max(h, w)),
            transforms.CenterCrop((h, w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * c, (0.5,) * c),
        ])
        train = datasets.ImageFolder(args.real_data_dir, transform=transform)
        test = train
    elif cfg.real_dataset == "mnist":
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
    real_images = None
    if args.n_fid_samples > 1:
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
def build_fid_reference(base_model, cfg, args, device):
    if args.n_fid_samples <= 1 or args.fid_reference == "none":
        return None
    if args.fid_reference == "base_samples" or (args.fid_reference == "auto" and cfg.real_dataset is None and args.real_data_dir is None):
        print(f"Generating {args.n_fid_samples} base-model FID reference samples...")
        return sample_for_fid(base_model, cfg, args, device, args.n_fid_samples)
    return None


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


def next_train_batch(train_loader, data_iter, cfg, args, device):
    c, h, w = cfg.dim
    if train_loader is None:
        x1 = torch.randn(args.batch_size, c, h, w, device=device)
        y = labels_for_batch(cfg, args.batch_size, args, device)
        return x1, y, data_iter
    try:
        x1, y_data = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x1, y_data = next(data_iter)
    x1 = x1.to(device)
    y = y_data.to(device) if cfg.class_cond else None
    return x1, y, data_iter


def train_lora(model, base_model, train_loader, cfg, P, wm_code, args, device, out_dir):
    c, h, w = cfg.dim
    opt_params = freeze_non_lora(model, args.train_extra)
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
        total_data = 0.0
        total_proj = 0.0
        total_corr = 0.0
        for _ in range(args.grad_accum):
            x1, y, data_iter = next_train_batch(train_loader, data_iter, cfg, args, device)
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
            proj = residual.reshape(b, -1).float() @ P
            target_proj = args.wm_eps * carrier * wm_code.view(1, -1)

            loss_vel = F.mse_loss(residual.float(), wm.float())
            loss_data = F.mse_loss(pred.float(), (u_true + wm).float())
            loss_proj = F.mse_loss(proj, target_proj.float())
            wm_corr = (carrier * (proj * wm_code.view(1, -1)).sum(dim=1, keepdim=True)).mean()
            wm_corr_norm = wm_corr / (0.5 * args.wm_eps + 1e-8)
            loss_wm = -torch.tanh(wm_corr_norm * args.wm_tanh_scale)
            if args.objective == "full":
                loss = loss_data + args.wm_proj_weight * loss_proj + args.wm_lambda * loss_wm
            else:
                loss = loss_vel + args.wm_proj_weight * loss_proj + args.wm_lambda * loss_wm
            loss = loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()
            total_vel += loss_vel.item()
            total_data += loss_data.item()
            total_proj += loss_proj.item()
            total_corr += wm_corr.item()

        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        opt.step()
        sched.step()
        row = {
            "step": step + 1,
            "loss": total_loss,
            "residual_mse": total_vel / args.grad_accum,
            "data_mse": total_data / args.grad_accum,
            "proj_mse": total_proj / args.grad_accum,
            "wm_corr": total_corr / args.grad_accum,
            "wm_corr_norm": (total_corr / args.grad_accum) / (0.5 * args.wm_eps + 1e-8),
        }
        history.append(row)
        if step % 10 == 0:
            pbar.set_postfix(
                data=f"{row['data_mse']:.4f}",
                proj=f"{row['proj_mse']:.4f}",
                corr=f"{row['wm_corr_norm']:.3f}",
            )
        if args.save_every and (step + 1) % args.save_every == 0:
            save_adapter(model, out_dir / f"adapter_step_{step + 1}.pt")
    return history


def clean_finetune(model, train_loader, cfg, args, device):
    if args.post_ft_steps <= 0:
        return []
    c, h, w = cfg.dim
    opt_params = [p for p in model.parameters() if p.requires_grad]
    if not opt_params:
        opt_params = freeze_non_lora(model, args.train_extra)
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(args.post_ft_steps, 1))
    model.train()
    data_iter = iter(train_loader) if train_loader is not None else None
    history = []
    pbar = tqdm(range(args.post_ft_steps), desc="Clean fine-tune")
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x1, y, data_iter = next_train_batch(train_loader, data_iter, cfg, args, device)
            b = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(b, device=device)
            x_t = (1 - t.view(b, 1, 1, 1)) * x0 + t.view(b, 1, 1, 1) * x1
            u_true = x1 - x0
            pred = model_velocity(model, x_t, t, y)
            loss = F.mse_loss(pred.float(), u_true.float()) / args.grad_accum
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        opt.step()
        sched.step()
        row = {"post_ft_step": step + 1, "clean_finetune_mse": total_loss}
        history.append(row)
        if step % 10 == 0:
            pbar.set_postfix(clean=f"{total_loss:.4f}")
    return history


def save_adapter(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if "lora_" in k or k in trainable_names
    }
    torch.save(state, path)


def load_adapter(model: nn.Module, path: Path, device: torch.device):
    state = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded adapter/eval state from {path}")
    if unexpected:
        print(f"[adapter] unexpected keys: {len(unexpected)}")
    # Missing keys are expected because adapter checkpoints contain only trainable parameters.
    if missing:
        print(f"[adapter] missing base keys: {len(missing)}")


@torch.no_grad()
def sample_detection_batch(model, cfg, args, device, n_queries, train_loader=None):
    c, h, w = cfg.dim
    if args.detect_distribution == "noise" or train_loader is None:
        x = torch.randn(n_queries, c, h, w, device=device)
        y = labels_for_batch(cfg, n_queries, args, device)
        t = torch.rand(n_queries, device=device)
        return x, t, y

    xs = []
    ys = []
    data_iter = iter(train_loader)
    remaining = n_queries
    while remaining > 0:
        x1, y, data_iter = next_train_batch(train_loader, data_iter, cfg, args, device)
        b = min(x1.shape[0], remaining)
        x1 = x1[:b]
        if y is not None:
            y = y[:b]
        x0 = torch.randn_like(x1)
        t = torch.rand(b, device=device)
        x_t = (1 - t.view(b, 1, 1, 1)) * x0 + t.view(b, 1, 1, 1) * x1
        xs.append((x_t, t))
        if y is not None:
            ys.append(y)
        remaining -= b
    x = torch.cat([item[0] for item in xs], dim=0)
    t = torch.cat([item[1] for item in xs], dim=0)
    y = torch.cat(ys, dim=0) if ys else None
    return x, t, y


@torch.no_grad()
def detect(model, cfg, P, codes, codebook, args, device, n_queries=None, train_loader=None):
    n_queries = n_queries or args.n_queries
    signature = torch.zeros(args.wm_K, device=device)
    all_codes = codes.to(device)
    keys = list(codebook.keys())
    for start in range(0, n_queries, args.batch_size):
        b = min(args.batch_size, n_queries - start)
        x, t, y = sample_detection_batch(model, cfg, args, device, b, train_loader)
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


@torch.no_grad()
def sample_for_fid(model, cfg, args, device, n_samples):
    batches = []
    for start in range(0, n_samples, args.fid_batch_size):
        b = min(args.fid_batch_size, n_samples - start)
        batch = sample(model, cfg, args, device, b).cpu()
        batches.append(batch)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(batches, dim=0)


def compute_fid_pixel(real, gen):
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


def compute_fid_from_features(features_a, features_b):
    f_a = features_a.detach().cpu().numpy()
    f_b = features_b.detach().cpu().numpy()
    mu_a, mu_b = f_a.mean(0), f_b.mean(0)
    sig_a, sig_b = np.cov(f_a, rowvar=False), np.cov(f_b, rowvar=False)
    eps = 1e-6
    sig_a += eps * np.eye(sig_a.shape[0])
    sig_b += eps * np.eye(sig_b.shape[0])
    diff = mu_a - mu_b
    covmean = sqrtm(sig_a @ sig_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_a + sig_b - 2 * covmean))


@torch.no_grad()
def inception_features(images, device, batch_size=16):
    from torchvision.models import Inception_V3_Weights, inception_v3

    inception = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    ).to(device).eval()
    inception.fc = torch.nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = (batch.clamp(-1, 1) + 1) / 2
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        feats.append(inception(batch).cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del inception
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)


def compute_fid(real, gen, args, device):
    if args.fid_feature == "pixel":
        return compute_fid_pixel(real, gen)
    real_feats = inception_features(real, device, batch_size=args.fid_batch_size)
    gen_feats = inception_features(gen, device, batch_size=args.fid_batch_size)
    return compute_fid_from_features(real_feats, gen_feats)


def evaluate(model, base_model, cfg, P, codes, codebook, true_msg, args, device, real_images=None, train_loader=None):
    true_bits = tuple(int(b) for b in true_msg)
    true_idx = list(codebook.keys()).index(true_bits)

    wm_hits = 0
    clean_hits = 0
    wm_scores = []
    clean_scores = []
    for _ in range(args.n_detect_trials):
        decoded, scores, _ = detect(model, cfg, P, codes, codebook, args, device, train_loader=train_loader)
        wm_hits += int(decoded == true_bits)
        wm_scores.append(float(scores[true_idx]))
        decoded_clean, clean_scores_arr, _ = detect(base_model, cfg, P, codes, codebook, args, device, train_loader=train_loader)
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
        clean_samples = sample_for_fid(base_model, cfg, args, device, n)
        wm_samples = sample_for_fid(model, cfg, args, device, n)
        fid_clean = compute_fid(real_images[:n], clean_samples, args, device)
        fid_wm = compute_fid(real_images[:n], wm_samples, args, device)
        metrics.update({
            "fid_clean": fid_clean,
            "fid_wm": fid_wm,
            "fid_ratio": fid_wm / max(fid_clean, 1e-8),
        })
    return metrics


def run_query_sweep(model, base_model, cfg, P, codes, codebook, true_msg, args, device, train_loader=None):
    rows = []
    saved_n = args.n_queries
    for n in [int(x) for x in args.query_values.split(",") if x]:
        args.n_queries = n
        rows.append({"sweep": "queries", "value": n, **evaluate(
            model, base_model, cfg, P, codes, codebook, true_msg, args, device, None, train_loader
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
    model.to(device)

    D = int(np.prod(cfg.dim))
    wm_bits = tuple(int(b) for b in args.wm_message)
    P, codes, codebook = make_codebook(D, len(wm_bits), args.wm_K, device)
    wm_code = codebook[wm_bits]

    real_images = None
    if args.data_source == "real" or (
        args.data_source == "auto" and (cfg.real_dataset is not None or args.real_data_dir is not None)
    ):
        train_loader, real_images = load_real_data(cfg, args, device)
    elif args.data_source == "auto" or args.data_source == "base_samples":
        train_loader = build_base_sample_loader(base_model, cfg, args, device)
    else:
        raise ValueError(args.data_source)

    if real_images is None:
        real_images = build_fid_reference(base_model, cfg, args, device)
    t0 = time.time()
    if args.eval_only:
        adapter_path = Path(args.adapter_path) if args.adapter_path else out_dir / "adapter_final.pt"
        load_adapter(model, adapter_path, device)
        history = []
    else:
        history = train_lora(model, base_model, train_loader, cfg, P, wm_code, args, device, out_dir)
        save_adapter(model, out_dir / "adapter_final.pt")
        post_history = clean_finetune(model, train_loader, cfg, args, device)
        if post_history:
            history.extend(post_history)
            save_adapter(model, out_dir / f"adapter_post_ft_{args.post_ft_steps}.pt")
    elapsed_min = (time.time() - t0) / 60
    torch.save(
        {"P": P.cpu(), "codes": codes.cpu(), "message": args.wm_message, "config": vars(args)},
        out_dir / "watermark_key.pt",
    )

    metrics = evaluate(
        model, base_model, cfg, P, codes, codebook, args.wm_message,
        args, device, real_images, train_loader
    )
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
        "post_ft_steps": args.post_ft_steps,
        "objective": args.objective,
        "detect_distribution": args.detect_distribution,
        "elapsed_min": elapsed_min,
    })

    sweep_rows = []
    if args.sweep == "queries":
        sweep_rows = run_query_sweep(
            model, base_model, cfg, P, codes, codebook, args.wm_message,
            args, device, train_loader
        )
    elif args.sweep != "none":
        print(f"[note] Sweep {args.sweep!r} needs one run per value to retrain LoRA fairly.")
        print("       Use multiple invocations with --wm_eps, --wm_message, or --steps.")

    write_outputs(out_dir, vars(args), metrics, history, sweep_rows)

    print("\nTable-ready summary")
    print(f"  Dataset:    {args.dataset}")
    print(f"  WM acc:     {metrics['wm_acc']:.1f}%")
    print(f"  Clean FP:   {metrics['clean_fp']:.1f}%")
    print(f"  Sep:        {metrics['sep_sigma']:.2f}")
    print(f"  WM score:   {metrics['wm_score_mean']:.4f} ± {metrics['wm_score_std']:.4f}")
    print(f"  Clean score:{metrics['clean_score_mean']:.4f} ± {metrics['clean_score_std']:.4f}")
    if "fid_ratio" in metrics:
        print(f"  FID ratio:  {metrics['fid_ratio']:.3f}")
    print(f"  Outputs:    {out_dir}")


if __name__ == "__main__":
    main()
