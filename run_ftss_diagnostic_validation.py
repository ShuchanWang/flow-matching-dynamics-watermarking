"""Prospective held-out validation of an FTSS warning rule against near-copy generation."""

import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from evaluate_generative_generalization import DigitEvaluator, features, prdc, training_preference
from ftss_batched import estimate_ftss_batched
from run_ftss_finetuning import digest, load_ftss, write_json
from run_matched_ftss_mlp import ROOT, numeric
from run_unmarked_memorization import CHECKPOINTS, load_model, nearest, membership_auc


ARMS = ("uniform", "repeat", "specialize")


class ClampAudit(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.velocity_exceedances = torch.zeros((), dtype=torch.float64, device=device)
        self.state_boundary = torch.zeros_like(self.velocity_exceedances)
        self.coordinates = 0

    def forward(self, x, t):
        v = self.model(x,t)
        self.velocity_exceedances += (v.abs()>10).sum()
        self.state_boundary += (x.abs()>=4.999).sum()
        self.coordinates += x.numel()
        return v


def build_pools(labels, seed, train_size=400, audit_size=400, digit=1, calibration_size=200):
    if train_size % 10 or audit_size % 10 or audit_size > train_size:
        raise ValueError("Balanced sizes must be multiples of ten, audit <= training")
    rng = np.random.default_rng(seed)
    members, reference, calibration, targets = [], [], [], []
    for label in range(10):
        order = rng.permutation(np.flatnonzero(np.asarray(labels) == label))
        n, a, c = train_size//10, audit_size//10, calibration_size//10
        if n+a+c > len(order):
            raise ValueError("Insufficient images for disjoint broad pools")
        members.extend(order[:n].tolist())
        reference.extend(order[n:n+a].tolist())
        calibration.extend(order[n+a:n+a+c].tolist())
        targets.append(int(order[0]))
    audit = np.asarray(members).reshape(10, -1)[:, :audit_size//10].reshape(-1).tolist()
    broad = dict(train=members, audit=audit, reference=reference, calibration=calibration, targets=targets)
    order = rng.permutation(np.flatnonzero(np.asarray(labels) == digit))
    if train_size+audit_size+calibration_size > len(order):
        raise ValueError("Insufficient distinct specialization images; do not duplicate to fill the pool")
    narrow = dict(train=order[:train_size].tolist(), audit=order[:audit_size].tolist(),
                  reference=order[train_size:train_size+audit_size].tolist(),
                  calibration=order[train_size+audit_size:train_size+audit_size+calibration_size].tolist(), targets=[])
    return dict(uniform=copy.deepcopy(broad), repeat=copy.deepcopy(broad), specialize=narrow)


def exposure_weights(split, repeat_mass=.8):
    if not 0 < repeat_mass < 1:
        raise ValueError("Repeat mass must lie in (0,1)")
    target = np.isin(split["train"], split["targets"])
    if not target.any() or target.all():
        raise ValueError("Exposure arm needs targets and other examples")
    return np.where(target, repeat_mass/target.sum(), (1-repeat_mass)/(~target).sum())


def confusion(truth, predictions):
    y, p = np.asarray(truth, bool), np.asarray(predictions, bool)
    tp, fn = int((y&p).sum()), int((y&~p).sum())
    fp, tn = int((~y&p).sum()), int((~y&~p).sum())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn,
                sensitivity=tp/(tp+fn) if tp+fn else None,
                false_alarm_rate=fp/(fp+tn) if fp+tn else None,
                specificity=tn/(fp+tn) if fp+tn else None)


def replication_metrics(training_distance, reference_distance, tolerance):
    train, reference = np.asarray(training_distance), np.asarray(reference_distance)
    scale = 2*np.sqrt(784)
    copies = (train/scale <= tolerance)&(train < reference)
    reference_copies = (reference/scale <= tolerance)&(reference < train)
    return dict(copy_rate=float(copies.mean()), reference_copy_rate=float(reference_copies.mean()),
                copy_excess=float(copies.mean()-reference_copies.mean()),
                copy_count=int(copies.sum()), reference_copy_count=int(reference_copies.sum())), copies


def fit_rule(ratios, labels):
    x, y = np.asarray(ratios, float), np.asarray(labels, bool)
    if not y.any() or y.all():
        raise ValueError("Calibration requires both observed replication classes")
    values = np.unique(np.r_[0., x[(x > 0)&(x < 1)], 1.])
    candidates = np.unique(np.r_[values, (values[:-1]+values[1:])/2])
    scored = []
    for tau in candidates:
        c = confusion(y, x <= tau)
        balanced = (c["sensitivity"]+c["specificity"])/2
        scored.append((balanced, -c["false_alarm_rate"], -tau, tau, c))
    best = max(scored, key=lambda r: r[:3])
    return dict(threshold=float(best[3]), calibration_balanced_accuracy=float(best[0]), confusion=best[4],
                decision="Flag if paired FTSS minimum ratio <= threshold; threshold constrained to [0,1]",
                selection="Maximize calibration balanced accuracy; ties favor lower false alarms, then smaller threshold")


def metrics_summary(rows, tau, rate, final_step=None):
    result = {}
    for model in sorted({r["model"] for r in rows}):
        result[model] = {}
        for arm in ("all", *ARMS):
            selected = [r for r in rows if r["model"] == model and r["step"] > 0
                        and (final_step is None or r["step"] == final_step)
                        and (arm == "all" or r["arm"] == arm)]
            c = confusion([r["metrics"]["copy_excess"] >= rate for r in selected],
                          [r["ftss_ratio"] <= tau for r in selected])
            c.update(checkpoints=len(selected), seed_blocks=len({r["seed"] for r in selected}))
            result[model][arm] = c
    return result


class Measurement:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.dataset = MNIST(str(ROOT/"data"), train=False, download=False)
        self.images = self.dataset.data.flatten(1).float()/127.5-1
        self.labels = self.dataset.targets.numpy()
        self.helpers = load_ftss(args.repo)
        self.classifier = DigitEvaluator().to(device)
        self.classifier.load_state_dict(torch.load(args.classifier, map_location="cpu", weights_only=True)["model"])
        self.classifier.eval().requires_grad_(False)
        self.real = features(self.classifier, self.dataset.data[:,None].float()/255, device)["spatial"]

    @torch.no_grad()
    def ftss(self, model, seed):
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            monitored = ClampAudit(model,self.device)
            result = numeric(estimate_ftss_batched(monitored, self.device, self.args.probes, self.args.ode_steps))
            result["velocity_exceedance_fraction"] = float(monitored.velocity_exceedances/monitored.coordinates)
            result["near_state_clamp_fraction"] = float(monitored.state_boundary/monitored.coordinates)
            return result

    @torch.no_grad()
    def __call__(self, model, split, seed, name):
        model.eval()
        m = self.ftss(model, seed)
        rng = torch.Generator(device=self.device).manual_seed(seed+1)
        noise = torch.randn(self.args.samples, 784, generator=rng, device=self.device)
        samples = self.helpers.generate(model, noise, self.args.ode_steps, self.args.eval_batch)
        visible = samples.clamp(-1, 1)
        member, reference = self.images[split["audit"]], self.images[split["reference"]]
        train_d, train_i = nearest(visible, member)
        ref_d, _ = nearest(visible, reference)
        rmse = train_d/(2*np.sqrt(784))
        replication, copies = replication_metrics(train_d,ref_d,self.args.copy_rmse)
        m.update(replication)
        m.update(pixel_preference=float((train_d < ref_d).mean()),
                 copied_member_fraction=float(len(np.unique(train_i[copies]))/len(member)))
        if split["targets"]:
            assigned = np.asarray(split["audit"])[train_i]
            m["repeated_target_copy_rate"] = float((copies&np.isin(assigned, split["targets"])).mean())
        generated = features(self.classifier, visible.reshape(-1,1,28,28).add(1).div(2), self.device)
        m.update(training_preference(generated["spatial"], self.real[split["audit"]], self.real[split["reference"]]))
        m.update(prdc(self.real[split["calibration"]], generated["spatial"], 3))
        m["specialization_digit_fraction"] = float((generated["probabilities"].argmax(1) == self.args.digit).mean())
        # Common noise/time draws and equal, class-matched member/reference audit sizes.
        losses = [torch.zeros(len(member), device=self.device), torch.zeros(len(reference), device=self.device)]
        generator = torch.Generator(device=self.device).manual_seed(seed+2)
        for _ in range(8):
            noise = torch.randn(member.shape, generator=generator, device=self.device)
            t = torch.rand(len(member),1, generator=generator, device=self.device)
            for j, target in enumerate((member, reference)):
                target = target.to(self.device)
                for start in range(0, len(target), self.args.eval_batch):
                    sl = slice(start, start+self.args.eval_batch)
                    v = model((1-t[sl])*noise[sl]+t[sl]*target[sl], t[sl])
                    losses[j][sl] += (v-(target[sl]-noise[sl])).square().mean(1)/8
        a, b = [v.cpu().numpy() for v in losses]
        m.update(loss_gap=float(b.mean()-a.mean()), membership_auc=membership_auc(a,b))
        np.savez_compressed(self.args.output/f"{name}_audit.npz", samples=samples.numpy(),
                            training_distance=train_d, reference_distance=ref_d, rmse=rmse,
                            nearest_training_indices=np.asarray(split["audit"])[train_i],
                            copies=copies, member_losses=a, reference_losses=b, **generated)
        save_image(visible[:100].reshape(-1,1,28,28).add(1).div(2), self.args.output/f"{name}_samples.png", nrow=10)
        print(f"MEASURE {name}: FTSS={m['min']:.4f} copies={m['copy_count']}/{len(samples)} "
              f"preference={m['training_preference']:.3f}", flush=True)
        return m


def calibrate(args, measurement, device):
    path = args.output/"frozen_rule.json"
    if path.exists():
        return json.loads(path.read_text())
    source = json.loads((args.calibration/"results.json").read_text())
    records_path = args.output/"calibration_remeasurements.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    for old in [r for r in source["measurements"] if r["model"] == "unet" and r["step"] == 3000]:
        if any((r["run"],r["size"]) == (old["run"],old["size"]) for r in records):
            continue
        seed = 121000+old["run"]
        model = load_model("unet", device)
        before = measurement.ftss(model, seed)["min"]
        checkpoint = args.calibration/f"unet_n{old['size']}_run{old['run']}_latest.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["step"] != 3000:
            raise ValueError("Calibration checkpoint is not the expected final model")
        model.load_state_dict(saved["model"])
        after = measurement.ftss(model, seed)["min"]
        with np.load(args.calibration/f"unet_n{old['size']}_run{old['run']}_step3000_audit.npz") as audit:
            replication,_ = replication_metrics(audit["visible_train_distances"],audit["visible_reference_distances"],args.copy_rmse)
        records.append(dict(run=old["run"], size=old["size"], ftss_ratio=after/before,
                            **replication, positive=bool(replication["copy_excess"] >= args.copy_rate),
                            checkpoint_sha256=digest(checkpoint)))
        write_json(records_path, records)
        print(f"CALIBRATE n={old['size']} run={old['run']} ratio={after/before:.4f} copy_excess={replication['copy_excess']:.3f}", flush=True)
        del saved, model
    rule = fit_rule([r["ftss_ratio"] for r in records], [r["positive"] for r in records])
    rule.update(copy_rmse=args.copy_rmse, positive_copy_excess=args.copy_rate, calibration=records,
                provenance="Retrospective calibration on previously inspected UNet runs; freeze before new validation outcomes",
                source_sha256=digest(records_path))
    write_json(path, rule)
    return rule


def save_report(args, report, rule=None):
    write_json(args.output/"results.json", report)
    lines = ["# Held-Out FTSS Diagnostic Validation", "", f"State: {report['state']}", "",
             f"Primary outcome: generated images within fixed visible-pixel RMSE <= {args.copy_rmse} of a prespecified "
             "audited training image and closer to it than to an equal-size, class-matched reference pool. "
             "The same proximity rule is applied symmetrically to untouched reference examples. "
             f"A checkpoint is outcome-positive if training-minus-reference replication exceeds or equals {100*args.copy_rate:g} percentage point(s). "
             "These fixed operational tolerances define this study, not a universal definition of memorization.", "",
             "The FTSS warning threshold is fitted only on old UNet endpoint runs and frozen before new "
             "validation measurements. Calibration FTSS is remeasured using the same original-paper batched "
             f"estimator as validation ({args.probes} trajectories, 20 times through 0.95, {args.ode_steps} Euler steps). "
             "Old generation audits are reused for calibration labels; their sample and anchor counts differ "
             "from validation and this is a distribution-shift test. Calibration data were previously viewed "
             "and are not described as blinded. New seeds are held out; MLP is an architecture-transfer test.", "",
             f"All three arms have {args.train_size} distinct training images and the same update schedule. Uniform and "
             f"repeat arms share the exact data; the repeat arm assigns {100*args.repeat_mass:g}% of draws to ten prespecified images. "
             f"Specialization uses {args.train_size} distinct digit-{args.digit} images. Each arm audits {args.audit_size} members, "
             f"{args.audit_size} untouched same-domain reference images, and {args.calibration_size} quality-reference images. Specialization is not automatically "
             "labeled negative: its actual replication outcome determines the label. There is no watermark.", ""]
    if rule:
        summary = metrics_summary(report["rows"], rule["threshold"], args.copy_rate, args.steps)
        write_json(args.output/"summary.json", dict(state=report["state"], rule=rule, metrics=summary,
                   secondary_all_checkpoints=metrics_summary(report["rows"], rule["threshold"], args.copy_rate)))
        lines += [f"Frozen FTSS ratio threshold: {rule['threshold']:.6f}.", "",
                  "| Model | Arm | Checkpoints | TP | FN | FP | TN | Sensitivity | False alarm rate |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model, groups in summary.items():
            for arm, m in groups.items():
                def show(v):
                    return "not estimable" if v is None else f"{v:.3f}"
                lines.append(f"| {model} | {arm} | {m['checkpoints']} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} | "
                             f"{show(m['sensitivity'])} | {show(m['false_alarm_rate'])} |")
        lines += ["", f"Primary confusion counts use only the final {args.steps}-step checkpoint. "
                  "Earlier checkpoints are secondary monitoring measurements. All draws use the same fixed "
                  f"{args.samples}-sample generation budget per checkpoint. Repeated checkpoints share pretrained bases and tuning realizations. Report counts by "
                  "seed/arm, not independence-based significance tests. Absence of positive validation cases "
                  "makes sensitivity unidentifiable, not perfect. The pixel reproduction definition is aligned-MNIST "
                  "specific and does not establish natural-image perceptual copying.", "",
                  "| Model | Seed | Arm | Step | FTSS ratio | Training copies | Reference copies | Copy excess | Flagged |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---|"]
        for r in report["rows"]:
            lines.append(f"| {r['model']} | {r['seed']} | {r['arm']} | {r['step']} | {r['ftss_ratio']:.4f} | "
                         f"{r['metrics']['copy_rate']:.4f} | {r['metrics']['reference_copy_rate']:.4f} | "
                         f"{r['metrics']['copy_excess']:.4f} | {r['ftss_ratio'] <= rule['threshold']} |")
    (args.output/"report.md").write_text("\n".join(lines)+"\n")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    source_paths = [Path(__file__), ROOT/"ftss_batched.py", ROOT/"run_unmarked_memorization.py",
                    ROOT/"evaluate_generative_generalization.py", ROOT/"repair_mlp_watermarks.py",
                    ROOT/"flow_watermark_unet_lora.py",
                    args.repo/"finetune_memorization_ftss.py", args.classifier, *[CHECKPOINTS[m] for m in args.models],
                    args.calibration/"results.json"]
    source_paths += sorted(args.calibration.glob("unet_*_latest.pt"))
    source_paths += sorted(args.calibration.glob("unet_*_step3000_audit.npz"))
    config = dict(arguments={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  hashes={str(p): digest(p) for p in source_paths}, validation_partition="New tuning/probe seeds; original base checkpoints")
    path = args.output/"config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Resume requires unchanged protocol and sources")
    write_json(path, config)
    path = args.output/"results.json"
    report = json.loads(path.read_text()) if path.exists() else dict(state="running", rows=[])
    rule = None
    try:
        report["state"] = "calibrating"
        save_report(args, report)
        measurement = Measurement(args, device)
        rule = calibrate(args, measurement, device)
        report["frozen_rule_sha256"] = digest(args.output/"frozen_rule.json")
        report["state"] = "running"
        save_report(args, report, rule)
        for name in args.models:
            for seed in args.seeds:
                pools = build_pools(measurement.labels, seed, args.train_size, args.audit_size, args.digit, args.calibration_size)
                for arm in ARMS:
                    prefix = f"{name}_seed{seed}_{arm}"
                    rows = [r for r in report["rows"] if (r["model"],r["seed"],r["arm"]) == (name,seed,arm)]
                    if {r["step"] for r in rows}.issuperset(args.check_steps):
                        continue
                    split = pools[arm]
                    write_json(args.output/f"{prefix}_indices.json", split)
                    torch.manual_seed(seed+1000)
                    model = load_model(name, device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
                    generator = torch.Generator(device=device).manual_seed(seed+2000)
                    train = measurement.images[split["train"]].to(device)
                    weights = (torch.tensor(exposure_weights(split, args.repeat_mass), device=device, dtype=torch.float32)
                               if arm == "repeat" else torch.full((len(train),), 1/len(train), device=device))
                    latest = args.output/f"{prefix}_latest.pt"
                    step = 0
                    if latest.exists():
                        saved = torch.load(latest, map_location="cpu", weights_only=True)
                        model.load_state_dict(saved["model"])
                        optimizer.load_state_dict(saved["optimizer"])
                        generator.set_state(saved["generator"])
                        torch.set_rng_state(saved["cpu_rng"])
                        if device.type == "cuda":
                            torch.cuda.set_rng_state(saved["cuda_rng"], device)
                        step = saved["step"]
                        del saved

                    def measure(current):
                        if any(r["step"] == current for r in rows):
                            return
                        m = measurement(model, split, seed+3000, f"{prefix}_step{current}")
                        before = next((r["metrics"]["min"] for r in rows if r["step"] == 0), m["min"])
                        row = dict(model=name, seed=seed, arm=arm, step=current, metrics=m, ftss_ratio=m["min"]/before)
                        rows.append(row)
                        report["rows"].append(row)
                        save_report(args, report, rule)

                    measure(step)
                    started = time.monotonic()
                    for current in range(step+1,args.steps+1):
                        model.train()
                        indices = torch.multinomial(weights, args.batch_size, replacement=True, generator=generator)
                        target = train[indices]
                        noise = torch.randn(target.shape, generator=generator, device=device)
                        t = torch.rand(len(target),1, generator=generator, device=device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.mse_loss(model((1-t)*noise+t*target,t), target-noise)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite loss {prefix}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                        if current % 500 == 0:
                            print(f"TRAIN {prefix} {current}/{args.steps} loss={float(loss):.5f} elapsed={time.monotonic()-started:.0f}s", flush=True)
                        if current in args.check_steps:
                            saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         generator=generator.get_state(), cpu_rng=torch.get_rng_state(), step=current)
                            if device.type == "cuda":
                                saved["cuda_rng"] = torch.cuda.get_rng_state(device)
                            temp = latest.with_suffix(".tmp")
                            torch.save(saved,temp)
                            temp.replace(latest)
                            del saved
                            measure(current)
                    del model, optimizer, train
                    torch.cuda.empty_cache() if device.type == "cuda" else None
        if digest(args.output/"frozen_rule.json") != report["frozen_rule_sha256"]:
            raise ValueError("Frozen diagnostic rule changed during validation")
        plot_results(args.output,report,rule,args.copy_rate)
        report["state"] = "complete"
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        raise
    finally:
        save_report(args, report, rule)


def plot_results(output,report,rule,copy_rate):
    from matplotlib import pyplot as plt
    models = sorted({r["model"] for r in report["rows"]})
    fig,axes = plt.subplots(1,len(models),figsize=(6*len(models),3.8),squeeze=False)
    for ax,name in zip(axes[0],models):
        for arm in ARMS:
            rows = [r for r in report["rows"] if r["model"]==name and r["arm"]==arm and r["step"]>0]
            ax.scatter([r["ftss_ratio"] for r in rows],[r["metrics"]["copy_excess"] for r in rows],label=arm,alpha=.75)
        ax.axvline(rule["threshold"],color="gray",lw=1)
        ax.axhline(copy_rate,color="gray",lw=1)
        ax.set_xlabel("FTSS / before")
        ax.set_ylabel("Training-minus-reference near-copy rate")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(output/"diagnostic_validation.png",dpi=180)
    fig.savefig(output/"diagnostic_validation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"table_runs/ftss_diagnostic_validation")
    p.add_argument("--calibration",type=Path,default=ROOT/"table_runs/unmarked_memorization_multimodel")
    p.add_argument("--repo",type=Path,default=ROOT.parent/"ftss-memorization")
    p.add_argument("--classifier",type=Path,default=ROOT/"table_runs/generative_generalization/feature_evaluator.pt")
    p.add_argument("--models",nargs="+",choices=list(CHECKPOINTS),default=["unet","mlp"])
    p.add_argument("--seeds",type=int,nargs="+",default=[23101,23102,23103])
    p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--check-steps",type=int,nargs="+",default=[0,1000,3000])
    p.add_argument("--train-size",type=int,default=400)
    p.add_argument("--audit-size",type=int,default=400)
    p.add_argument("--calibration-size",type=int,default=200)
    p.add_argument("--digit",type=int,default=1)
    p.add_argument("--repeat-mass",type=float,default=.8)
    p.add_argument("--copy-rmse",type=float,default=.05)
    p.add_argument("--copy-rate",type=float,default=.01)
    p.add_argument("--samples",type=int,default=1000)
    p.add_argument("--probes",type=int,default=15)
    p.add_argument("--ode-steps",type=int,default=200)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--eval-batch",type=int,default=128)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--device",default="cuda")
    args = p.parse_args()
    if (min(args.steps,args.samples,args.probes,args.ode_steps,args.batch_size,args.eval_batch) < 1
            or not 0<args.repeat_mass<1 or not 0<args.copy_rate<1 or not 0<args.copy_rmse<1
            or not 0<=args.digit<10 or args.lr<=0 or any(s<0 or s>args.steps for s in args.check_steps)
            or args.audit_size<10 or args.audit_size%10 or args.train_size%10 or args.train_size!=args.audit_size
            or args.calibration_size<10 or args.calibration_size%10):
        p.error("Invalid validation protocol")
    args.check_steps = sorted(set(args.check_steps)|{0,args.steps})
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"study.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
