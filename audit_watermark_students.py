"""Reload saved velocity students and independently audit fresh watermark queries."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

import watermark_hf_flow_unet as flow
from run_watermark_security import digest, write_json


ROOT = Path(__file__).resolve().parent


def decode(signature, key, mode):
    projection = key["P"].to(signature.device, dtype=torch.float64)
    codes = key["codes"].to(signature.device, dtype=torch.float64)
    scores = signature.double() @ projection @ codes.T
    truth = list(map(int, key["message"]))
    if mode == "hypercube":
        decoded = (scores > 0).long().tolist()
        signs = scores.new_tensor([2*b-1 for b in truth])
        margin = float((scores*signs).min())
    else:
        best = int(scores.argmax())
        decoded = [(best >> i) & 1 for i in range(len(truth))]
        true_index = sum(b << i for i, b in enumerate(truth))
        competing = scores.clone()
        competing[true_index] = -torch.inf
        margin = float(scores[true_index]-competing.max())
    return dict(decoded=decoded, exact=decoded == truth,
                bit_accuracy=float(np.mean(np.asarray(decoded) == truth)),
                decision_margin=margin)


@torch.no_grad()
def audit(args):
    from huggingface_hub import hf_hub_download
    device = torch.device(args.device)
    torch.set_num_threads(4)
    config = json.loads((args.study/"config.json").read_text())
    cfg = flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    base = flow.build_model(cfg, device)
    flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval()
    base_state = {k: v.cpu() for k,v in base.state_dict().items()}
    dataset = CIFAR10(str(ROOT/"data"), train=True, download=False)
    pool = torch.from_numpy(dataset.data[:20000]).permute(0,3,1,2).to(device).float()/127.5-1
    labels = torch.tensor(dataset.targets[:20000], device=device)
    report = dict(state="running", queries=args.queries, trials=args.trials, seed=args.seed,
                  source_sha256=digest(Path(__file__)), runs=[])
    destination = args.study/"student_reload_audit.json"
    for job in config["plan"]:
        c = job["config"]
        folder = args.study/job["scheme"]/c["wm_message"]
        path = folder/"velocity.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["step"] != config["arguments"]["steps"]:
            raise ValueError("Student training did not finish")
        student = flow.build_model(cfg, device)
        student.load_state_dict(saved["model"], strict=True)
        student.eval()
        difference = sum(float((saved["model"][k].double()-v.double()).square().sum()) for k,v in base_state.items())
        if difference == 0:
            raise ValueError("Saved student equals its clean initialization")
        teacher = flow.build_model(cfg, device)
        flow.load_state_dict_flexible(teacher, str(checkpoint), device)
        flow.add_lora(teacher, c["lora_rank"], c["lora_alpha"], c["lora_targets"])
        flow.load_adapter(teacher, Path(job["run_dir"])/"adapter_final.pt", device)
        teacher.eval()
        key_path = Path(job["run_dir"])/"watermark_key.pt"
        if digest(key_path) != job["hashes"]["watermark_key.pt"]:
            raise ValueError("Original owner key changed")
        key = torch.load(key_path, map_location="cpu", weights_only=False)
        if key["message"] != c["wm_message"]:
            raise ValueError("Original owner message mismatch")
        rng = torch.Generator(device=device).manual_seed(args.seed-1)
        probe = torch.randn(128,3,32,32, generator=rng, device=device)
        times = torch.rand(128, generator=rng, device=device)
        classes = torch.arange(128, device=device)%10
        a = flow.model_velocity(student, probe, times, classes)
        b = flow.model_velocity(teacher, probe, times, classes)
        nrms = float(((a-b).square().mean()/b.square().mean()).sqrt())
        del teacher, saved
        trials = []
        calls = {"queries": 0}
        def count_queries(module, inputs, kwargs, output):
            calls["queries"] += len(kwargs["x"] if "x" in kwargs else inputs[0])
        hook = student.register_forward_hook(count_queries, with_kwargs=True)
        for trial in range(args.trials):
            rng = torch.Generator(device=device).manual_seed(args.seed+trial)
            signature = torch.zeros(3072, dtype=torch.float64, device=device)
            for start in range(0, args.queries, args.batch):
                n = min(args.batch, args.queries-start)
                indices = torch.randint(len(pool), (n,), generator=rng, device=device)
                noise = torch.randn(n,3,32,32, generator=rng, device=device)
                t = torch.rand(n, generator=rng, device=device)
                x = (1-t[:,None,None,None])*noise+t[:,None,None,None]*pool[indices]
                v = flow.model_velocity(student, x, t, labels[indices])
                signature += (torch.sin(2*math.pi*t)[:,None]*v.flatten(1)).double().sum(0)
            trials.append(decode(signature/args.queries, key, c.get("codebook_mode", "auto")))
        hook.remove()
        if calls["queries"] != args.trials*args.queries:
            raise ValueError("Student query count mismatch")
        with np.load(folder/"marked.npz") as marked, np.load(folder/"velocity.npz") as velocity:
            samples_identical = np.array_equal(marked["samples"], velocity["samples"])
            sample_rmse = float(np.sqrt(np.mean((marked["samples"]-velocity["samples"])**2)))
        row = dict(scheme=job["scheme"], message=c["wm_message"], student_checkpoint_sha256=digest(path),
                   saved_training_steps=config["arguments"]["steps"], squared_parameter_change=difference,
                   fresh_noise_relative_velocity_rms_error=nrms, student_forward_queries=calls["queries"],
                   teacher_student_saved_images_identical=samples_identical,
                   teacher_student_paired_image_rmse=sample_rmse,
                   exact_recovery_percent=100*np.mean([t["exact"] for t in trials]),
                   bit_accuracy_percent=100*np.mean([t["bit_accuracy"] for t in trials]), trials=trials)
        report["runs"].append(row)
        write_json(destination, report)
        print(f"AUDIT {job['scheme']} {c['wm_message']}: exact={row['exact_recovery_percent']:.1f}% "
              f"bits={row['bit_accuracy_percent']:.1f}% velocity_NRMSE={nrms:.6f} "
              f"student_queries={calls['queries']}", flush=True)
        del student
    report["state"] = "complete"
    write_json(destination, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--study", type=Path, default=ROOT/"table_runs/watermark_security")
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=1040042)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if min(args.queries, args.trials, args.batch) < 1:
        p.error("Positive query/trial/batch counts required")
    audit(args)
