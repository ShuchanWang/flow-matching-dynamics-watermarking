"""Batch the paper's independent trajectory branches without changing its probes."""

import numpy as np
import torch


@torch.no_grad()
def estimate_ftss_batched(model, device, n_samples=15, n_steps=200, dimension=784):
    model.eval()
    times = np.linspace(0, 0.95, 20)
    starts = [int(t * n_steps) for t in times]
    # Preserve the original CUDA random-call shapes and trajectory/time order.
    noises, directions = [], []
    for _ in range(n_samples):
        noises.append(torch.randn(1, dimension, device=device))
        directions.append(torch.cat([torch.randn(1, dimension, device=device) for _ in times]))
    x = torch.cat(noises)
    trajectory = [x]
    for step in range(n_steps):
        t = torch.full((n_samples, 1), step / n_steps, device=device)
        x = (x + model(x, t).clamp(-10, 10) / n_steps).clamp(-5, 5)
        trajectory.append(x)
    terminal = trajectory[-1]
    directions = torch.stack(directions).transpose(0, 1)
    unit = directions / directions.norm(dim=-1, keepdim=True)
    states = torch.stack([trajectory[step] for step in starts])
    delta = unit * 0.1 * states.norm(dim=-1, keepdim=True)
    branches = states + delta
    active_from = torch.tensor(starts, device=device)[:, None, None]
    for step in range(n_steps):
        flat = branches.reshape(-1, dimension)
        t = torch.full((len(flat), 1), step / n_steps, device=device)
        velocity = model(flat, t).reshape_as(branches).clamp(-10, 10)
        updated = (branches + velocity / n_steps).clamp(-5, 5)
        branches = torch.where(active_from <= step, updated, branches)
    gains = (branches - terminal[None]).norm(dim=-1) / (delta.norm(dim=-1) + 1e-8)
    gains = gains.double().cpu().numpy()
    curve = np.sqrt(np.mean(gains ** 2, axis=1))
    return {"mean_curve": curve, "test_times": times, "early": np.mean(curve[:5]),
            "late": np.mean(curve[-5:]), "min": np.min(curve), "overall": np.mean(curve)}
