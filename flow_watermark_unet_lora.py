"""UNet and LoRA experiments for dynamics-level flow-matching watermarking.

This script trains or resumes a clean UNet flow-matching model, embeds
message-specific watermarks with LoRA fine-tuning, and evaluates black-box
message recovery and sample quality on MNIST or CIFAR-10.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
from scipy.linalg import sqrtm
import math
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATASET = "cifar10"  # "mnist" or "cifar10"
CHECKPOINT_DIR = "checkpointsCIFAR"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

if DATASET == "mnist":
    IMG_SIZE, IN_CHANNELS, D = 28, 1, 784
    N_TRAIN, N_TEST = 50000, 5000
    BASE_CH, BATCH_SIZE, LR = 64, 256, 1e-3
elif DATASET == "cifar10":
    IMG_SIZE, IN_CHANNELS, D = 32, 3, 3072
    N_TRAIN, N_TEST = 20000, 2000
    BASE_CH, BATCH_SIZE, LR = 64, 128, 3e-4

TARGET_STEPS = 5000
SAVE_EVERY = 500

# Watermark
K, N_BITS = 32, 5
EPSILON, WM_LOSS_WEIGHT = 0.2, 0.01
N_QUERIES = 4096
N_CLEAN, N_WM, TEST_MESSAGES = 1, 1, 3

# LoRA
LORA_RANK, LORA_ALPHA = 4, 1.0
LORA_STEPS, LORA_LR = 500, 5e-4

print(f"\n{'='*70}")
print(f"UNet WATERMARK: {DATASET.upper()} (Resume to {TARGET_STEPS} steps)")
print(f"{'='*70}")

# ============================================================
# DATA
# ============================================================
if DATASET == "mnist":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
elif DATASET == "cifar10":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_subset = Subset(train_ds, range(N_TRAIN))
test_subset = Subset(test_ds, range(N_TEST))
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
real_loader = DataLoader(Subset(test_ds, range(min(1000, N_TEST))), batch_size=1000, shuffle=False)
real_images_all = next(iter(real_loader))[0].view(-1, D).to(device)

print("Preparing flow matching data...")
t0 = time.time()
all_x1 = []
for x, _ in train_loader:
    all_x1.append(x.view(x.size(0), -1))
all_x1 = torch.cat(all_x1, dim=0)
print(f"Samples: {all_x1.size(0)}")
x1_pool = all_x1
del all_x1
print(f"Data prep: {time.time() - t0:.1f}s")

# ============================================================
# SECRET KEY
# ============================================================
torch.manual_seed(12345)
with torch.no_grad():
    P_raw = torch.randn(D, K, device=device)
    Q_p, _ = torch.linalg.qr(P_raw)
    P = Q_p[:, :K]
    codes_raw = torch.randn(2**N_BITS, K, device=device)
    Q_c, _ = torch.linalg.qr(codes_raw.T)
    codes = Q_c.T
    codebook = {}
    for k in range(2**N_BITS):
        bits = tuple((k >> i) & 1 for i in range(N_BITS))
        codebook[bits] = codes[k] / codes[k].norm()

all_messages = list(codebook.keys())
test_messages = random.sample(all_messages, TEST_MESSAGES)
print(f"Test messages: {test_messages}")

# ============================================================
# MODEL
# ============================================================
class SimpleUNet(nn.Module):
    def __init__(self, img_size, in_channels, base_ch=64):
        super().__init__()
        self.img_size, self.in_channels = img_size, in_channels
        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4
        
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, ch1, 3, padding=1), nn.InstanceNorm2d(ch1, affine=True), nn.SiLU())
        self.down1 = nn.Conv2d(ch1, ch1, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.Conv2d(ch1, ch2, 3, padding=1), nn.InstanceNorm2d(ch2, affine=True), nn.SiLU())
        self.down2 = nn.Conv2d(ch2, ch2, 4, stride=2, padding=1)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch2, ch3, 3, padding=1), nn.InstanceNorm2d(ch3, affine=True), nn.SiLU(),
            nn.Conv2d(ch3, ch3, 3, padding=1), nn.InstanceNorm2d(ch3, affine=True), nn.SiLU(),
        )
        
        self.up1 = nn.ConvTranspose2d(ch3, ch2, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(ch2 + ch2, ch2, 3, padding=1), nn.InstanceNorm2d(ch2, affine=True), nn.SiLU())
        self.up2 = nn.ConvTranspose2d(ch2, ch1, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(ch1 + ch1, ch1, 3, padding=1), nn.InstanceNorm2d(ch1, affine=True), nn.SiLU())
        self.final = nn.Conv2d(ch1, in_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(nn.Linear(1, ch3), nn.SiLU(), nn.Linear(ch3, ch3))
    
    def forward(self, x, t):
        B = x.shape[0]
        x_img = x.view(B, self.in_channels, self.img_size, self.img_size)
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        
        h1 = self.enc1(x_img)
        h1_d = self.down1(h1)
        h2 = self.enc2(h1_d)
        h2_d = self.down2(h2)
        
        h = self.bottleneck[0](h2_d)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h)
        h = h + t_emb
        h = self.bottleneck[3](h)
        h = self.bottleneck[4](h)
        
        h = self.up1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec1[0](h); h = self.dec1[1](h); h = self.dec1[2](h)
        
        h = self.up2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec2[0](h); h = self.dec2[1](h); h = self.dec2[2](h)
        
        out = self.final(h)
        return out.reshape(B, -1)

# ============================================================
# LORA ADAPTERS
# ============================================================
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=8, alpha=1.0):
        super().__init__()
        self.conv = conv
        self.rank, self.alpha = rank, alpha
        out_ch, in_ch, _, _ = conv.weight.shape
        self.lora_A = nn.Conv2d(in_ch, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_ch, 1, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        for p in self.conv.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        return self.conv(x) + self.alpha * self.lora_B(self.lora_A(x))

def add_lora(model, rank=8, alpha=1.0):
    """Add LoRA adapters to non-strided Conv2d layers."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1) and module.stride == (1, 1):
            setattr(model, name, LoRAConv2d(module, rank, alpha))
        elif isinstance(module, (nn.Sequential, SimpleUNet)):
            add_lora(module, rank, alpha)
    return model

# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def ckpt_path(name): return f"{CHECKPOINT_DIR}/{name}_{DATASET}.pt"

def save_ckpt(model, opt, sched, step, name):
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                'sched': sched.state_dict(), 'step': step}, ckpt_path(name))

def load_ckpt(model, opt, sched, name):
    path = ckpt_path(name)
    if os.path.exists(path):
        ckpt = torch.load(path)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['opt'])
            sched.load_state_dict(ckpt['sched'])
            return ckpt['step'] + 1
        else:
            model.load_state_dict(ckpt)
            return 0
    return 0

# ============================================================
# TRAINING WITH RESUME
# ============================================================
def train_with_resume(use_wm=False, wm_code=None, target_steps=TARGET_STEPS, name="clean"):
    model = SimpleUNet(IMG_SIZE, IN_CHANNELS, BASE_CH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, target_steps)
    torch.backends.cudnn.benchmark = True
    n_batches = len(x1_pool) // BATCH_SIZE
    
    start_step = load_ckpt(model, opt, sched, name)
    
    if start_step == 0:
        print(f"  Starting from scratch -> target {target_steps} steps")
    else:
        print(f"  Resumed from step {start_step} -> target {target_steps}")
    
    for step in range(start_step, target_steps):
        total_vel = 0; total_wm = 0
        idx = torch.randperm(len(x1_pool))[:BATCH_SIZE * n_batches]
        x1_batch = x1_pool[idx].to(device)
        
        for i in range(n_batches):
            x1 = x1_batch[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            x0 = torch.randn(BATCH_SIZE, D, device=device)
            t_val = torch.rand(BATCH_SIZE, 1, device=device)
            x_t = (1 - t_val) * x0 + t_val * x1
            u_true = x1 - x0
            
            if use_wm and wm_code is not None:
                wm = EPSILON * torch.sin(2 * math.pi * t_val) * (P @ wm_code).unsqueeze(0)
                u_target = u_true + wm
            else:
                u_target = u_true
            
            pred = model(x_t, t_val)
            vel_loss = F.mse_loss(pred, u_target)
            loss = vel_loss
            
            if use_wm and wm_code is not None:
                proj = pred @ P
                demod = torch.sin(2 * math.pi * t_val) * proj
                wm_corr = (demod * wm_code.unsqueeze(0)).sum(dim=1).mean()
                loss = vel_loss + WM_LOSS_WEIGHT * (-wm_corr)
                total_wm += wm_corr.item()
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_vel += vel_loss.item()
        sched.step()
        
        if step % SAVE_EVERY == 0 and step > start_step:
            save_ckpt(model, opt, sched, step, name)
            n = n_batches
            if use_wm:
                print(f"    Step {step}: vel={total_vel/n:.4f}, wm={total_wm/n:.4f} [saved]")
            else:
                print(f"    Step {step}: loss={total_vel/n:.4f} [saved]")
    
    save_ckpt(model, opt, sched, target_steps - 1, name)
    return model

# ============================================================
# LoRA FINE-TUNING WITH RESUME
# ============================================================
def lora_finetune_resume(pretrained_model, wm_code, steps=LORA_STEPS, lr=LORA_LR, msg_name="wm"):
    model = pretrained_model
    model = add_lora(model, LORA_RANK, LORA_ALPHA)
    model = model.to(device)
    
    trainable = []
    for name, p in model.named_parameters():
        if 'lora' in name or 'final' in name or 'time_mlp' in name:
            p.requires_grad = True
            trainable.append(p)
        else:
            p.requires_grad = False
    
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n_batches = len(x1_pool) // BATCH_SIZE
    
    name = f"lora_{msg_name}"
    start_step = load_ckpt(model, opt, sched, name)
    
    if start_step == 0:
        print(f"    LoRA from scratch -> {steps} steps")
    else:
        print(f"    LoRA resumed from step {start_step}")
    
    for step in range(start_step, steps):
        total_vel = 0; total_wm = 0
        idx = torch.randperm(len(x1_pool))[:BATCH_SIZE * n_batches]
        x1_batch = x1_pool[idx].to(device)
        
        for i in range(n_batches):
            x1 = x1_batch[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            x0 = torch.randn(BATCH_SIZE, D, device=device)
            t_val = torch.rand(BATCH_SIZE, 1, device=device)
            x_t = (1 - t_val) * x0 + t_val * x1
            wm = EPSILON * torch.sin(2 * math.pi * t_val) * (P @ wm_code).unsqueeze(0)
            u_target = (x1 - x0) + wm
            
            pred = model(x_t, t_val)
            vel_loss = F.mse_loss(pred, u_target)
            proj = pred @ P
            demod = torch.sin(2 * math.pi * t_val) * proj
            wm_corr = (demod * wm_code.unsqueeze(0)).sum(dim=1).mean()
            loss = vel_loss + WM_LOSS_WEIGHT * (-wm_corr)
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total_vel += vel_loss.item()
            total_wm += wm_corr.item()
        sched.step()
        
        if step % 100 == 0 and step > start_step:
            save_ckpt(model, opt, sched, step, name)
            print(f"    LoRA step {step}: vel={total_vel/n_batches:.4f}, wm={total_wm/n_batches:.4f} [saved]")
    
    save_ckpt(model, opt, sched, steps - 1, name)
    return model

# ============================================================
# DETECTION & GENERATION
# ============================================================
@torch.no_grad()
def decode_watermark(model, P, codebook, n_queries=N_QUERIES):
    x_query = torch.randn(n_queries, D, device=device)
    t_query = torch.rand(n_queries, 1, device=device)
    v = model(x_query, t_query)
    proj = v @ P
    carrier = torch.sin(2 * math.pi * t_query)
    signature = (carrier * proj).mean(dim=0)
    all_codes = torch.stack(list(codebook.values()))
    all_scores = (signature.unsqueeze(0) * all_codes).sum(dim=1)
    best_idx = all_scores.argmax().item()
    return list(codebook.keys())[best_idx], all_scores[best_idx].item()

@torch.no_grad()
def compute_signature_stats(model, P, codebook, true_msg, n_trials=50):
    """Compute overall signature norm and score statistics."""
    code = codebook[true_msg]
    sig_norms = []
    true_scores = []
    other_scores = []
    
    for _ in range(n_trials):
        x_q = torch.randn(1024, D, device=device)
        t_q = torch.rand(1024, 1, device=device)
        v = model(x_q, t_q)
        proj = v @ P
        carrier = torch.sin(2 * math.pi * t_q)
        sig = (carrier * proj).mean(dim=0)
        
        sig_norms.append(sig.norm().item())
        
        all_codes = torch.stack(list(codebook.values()))
        scores = (sig.unsqueeze(0) * all_codes).sum(dim=1)
        true_idx = list(codebook.keys()).index(true_msg)
        true_scores.append(scores[true_idx].item())
        
        mask = torch.ones(len(scores), dtype=torch.bool)
        mask[true_idx] = False
        other_scores.append(scores[mask].max().item())

    return {
        'sig_norm_mean': np.mean(sig_norms), 'sig_norm_std': np.std(sig_norms),
        'true_mean': np.mean(true_scores), 'true_std': np.std(true_scores),
        'other_mean': np.mean(other_scores), 'other_std': np.std(other_scores),
        'margin': np.mean(true_scores) - np.mean(other_scores),
    }

@torch.no_grad()
def generate_samples(model, n_samples=64, n_steps=200):
    x = torch.randn(n_samples, D, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = torch.ones(n_samples, 1, device=device) * (i * dt)
        v = model(x, t_val)
        v = torch.clamp(v, -10, 10)
        x = x + v * dt
        x = torch.clamp(x, -5, 5)
    return torch.clamp(x, -1, 1)

def compute_fid(real, gen):
    real, gen = real.cpu().numpy(), gen.cpu().numpy()
    mu_r, sigma_r = np.mean(real, 0), np.cov(real, rowvar=False)
    mu_g, sigma_g = np.mean(gen, 0), np.cov(gen, rowvar=False)
    eps = 1e-6
    sigma_r += eps * np.eye(sigma_r.shape[0])
    sigma_g += eps * np.eye(sigma_g.shape[0])
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))

def save_fid_samples(real_data, generated_data, fid_val, filename="outputs/fid_samples.png"):
    """Save side-by-side real and generated samples with the FID value."""
    n = 8
    
    real_imgs = real_data[:n].reshape(n, IN_CHANNELS, IMG_SIZE, IMG_SIZE).cpu().numpy()
    gen_imgs = generated_data[:n].reshape(n, IN_CHANNELS, IMG_SIZE, IMG_SIZE).cpu().numpy()
    
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    
    for i in range(n):
        if IN_CHANNELS == 3:
            r = np.transpose(real_imgs[i], (1, 2, 0))
            g = np.transpose(gen_imgs[i], (1, 2, 0))
        else:
            r = real_imgs[i][0]
            g = gen_imgs[i][0]
        
        axes[0, i].imshow(r, cmap='gray' if IN_CHANNELS == 1 else None, vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[1, i].imshow(g, cmap='gray' if IN_CHANNELS == 1 else None, vmin=0, vmax=1)
        axes[1, i].axis('off')
    
    axes[0, 0].set_title('Real', fontsize=10, fontweight='bold')
    axes[1, 0].set_title(f'Generated (FID={fid_val:.1f})', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")

# ============================================================
# MAIN
# ============================================================
print(f"\n{'='*70}")
print(f"TRAINING CLEAN MODEL (target: {TARGET_STEPS} steps)")
print(f"{'='*70}")

t0 = time.time()
clean_model = train_with_resume(use_wm=False, target_steps=TARGET_STEPS, name="clean")
print(f"  Time: {(time.time()-t0)/60:.1f} min")

clean_samples = generate_samples(clean_model, n_samples=500, n_steps=100)
real_01 = (real_images_all + 1) / 2
clean_01 = (clean_samples + 1) / 2
clean_fid = compute_fid(real_01[:500], clean_01[:500])
print(f"  Clean FID: {clean_fid:.1f}")
save_fid_samples(real_01, clean_01, clean_fid, f"outputs/{DATASET}_clean_fid{clean_fid:.0f}.png")

clean_models = [clean_model]

# ============================================================
# LoRA FINE-TUNE
# ============================================================
print(f"\n{'='*70}")
print("LORA FINE-TUNING FOR WATERMARKS")
print(f"{'='*70}")

wm_models = {}
for msg_idx, msg in enumerate(test_messages):
    print(f"\n  Message {msg_idx+1}/{len(test_messages)}: {msg}")
    
    model = SimpleUNet(IMG_SIZE, IN_CHANNELS, BASE_CH).to(device)
    ckpt_data = torch.load(ckpt_path("clean"))
    model.load_state_dict(ckpt_data['model'] if isinstance(ckpt_data, dict) and 'model' in ckpt_data else ckpt_data)
    
    t0 = time.time()
    wm_model = lora_finetune_resume(model, codebook[msg], msg_name=str(msg))
    wm_models[msg] = [wm_model]
    print(f"    Time: {(time.time()-t0)/60:.1f} min")

# ============================================================
# EVALUATION
# ============================================================
print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")

N_TRIALS = 20

print("\nWatermarked models:")
wm_correct = 0
for msg in test_messages:
    for model in wm_models[msg]:
        correct = sum(decode_watermark(model, P, codebook)[0] == msg for _ in range(N_TRIALS))
        wm_correct += correct
        print(f"  {msg}: {correct}/{N_TRIALS}")

wm_acc = wm_correct / (len(test_messages) * N_WM * N_TRIALS) * 100
print(f"  Overall: {wm_acc:.1f}%")

print("\nClean model:")
clean_hits = {msg: 0 for msg in test_messages}
for model in clean_models:
    for _ in range(N_TRIALS):
        decoded, _ = decode_watermark(model, P, codebook)
        if decoded in test_messages: clean_hits[decoded] += 1
for msg in test_messages:
    print(f"  {msg}: {clean_hits[msg]/(N_CLEAN*N_TRIALS)*100:.1f}%")

# ============================================================
# STATISTICAL SEPARATION
# ============================================================
print("\nStatistical separation (Welch's t-test):")
from scipy import stats as scipy_stats

N_STAT_SAMPLES = 30

for msg in test_messages:
    wm_scores = []
    code = codebook[msg]
    for model in wm_models[msg]:
        for _ in range(N_STAT_SAMPLES):
            x_q = torch.randn(1024, D, device=device)
            t_q = torch.rand(1024, 1, device=device)
            v = model(x_q, t_q)
            s = (torch.sin(2*math.pi*t_q) * (v@P)).mean(0)
            wm_scores.append((s * code).sum().item())
    
    clean_scores = []
    for model in clean_models:
        for _ in range(N_STAT_SAMPLES):
            x_q = torch.randn(1024, D, device=device)
            t_q = torch.rand(1024, 1, device=device)
            v = model(x_q, t_q)
            s = (torch.sin(2*math.pi*t_q) * (v@P)).mean(0)
            clean_scores.append((s * code).sum().item())
    
    wm_mean, wm_std = np.mean(wm_scores), np.std(wm_scores)
    clean_mean, clean_std = np.mean(clean_scores), np.std(clean_scores)
    
    t_stat, p_value = scipy_stats.ttest_ind(wm_scores, clean_scores, equal_var=False)
    
    pooled_std = np.sqrt((np.var(wm_scores) + np.var(clean_scores)) / 2)
    cohens_d = (wm_mean - clean_mean) / (pooled_std + 1e-8)

    print(f"  {msg}:")
    print(f"    WM:           {wm_mean:.4f} +/- {wm_std:.4f}")
    print(f"    Clean:        {clean_mean:.4f} +/- {clean_std:.4f}")
    print(f"    Cohen's d:    {cohens_d:.1f}")
    print(f"    t = {t_stat:.2f}, p = {p_value:.2e}")

print("\nSample Quality:")
wm_fids = {}
for msg in test_messages:
    wm_samples = generate_samples(wm_models[msg][0], n_samples=500, n_steps=100)
    wm_01 = (wm_samples + 1) / 2
    wm_fids[msg] = compute_fid(real_01[:500], wm_01[:500])
    print(f"  WM {str(msg)[:12]}... FID: {wm_fids[msg]:.1f} (ratio: {wm_fids[msg]/clean_fid:.3f}x)")

n_rows = 2 + len(test_messages)
fig, axes = plt.subplots(
    n_rows,
    8,
    figsize=(12, 0.7 * n_rows),
    constrained_layout=True
)

real_01 = (real_images_all + 1) / 2

# ============================================================
# OVERALL SIGNATURE STATISTICS
# ============================================================
print("\nOverall signature statistics:")
for msg in test_messages:
    wm_stats = compute_signature_stats(wm_models[msg][0], P, codebook, msg)
    clean_stats = compute_signature_stats(clean_models[0], P, codebook, msg)
    print(f"  {msg}:")
    print(f"    WM:    sig_norm={wm_stats['sig_norm_mean']:.4f}, true={wm_stats['true_mean']:.4f}+/-{wm_stats['true_std']:.4f}, margin={wm_stats['margin']:.4f}")
    print(f"    Clean: sig_norm={clean_stats['sig_norm_mean']:.4f}, true={clean_stats['true_mean']:.4f}+/-{clean_stats['true_std']:.4f}")

for i in range(8):
    img_data = real_01[i].reshape(IN_CHANNELS, IMG_SIZE, IMG_SIZE).cpu().numpy()
    img = img_data.squeeze() if IN_CHANNELS == 1 else np.transpose(img_data, (1, 2, 0))
    axes[0, i].imshow(img, cmap='gray' if IN_CHANNELS == 1 else None, vmin=0, vmax=1)
    axes[0, i].axis('off')
axes[0, 0].set_title('Real', fontsize=10, fontweight='bold')

clean_imgs = generate_samples(clean_model, n_samples=8)
clean_imgs = ((clean_imgs + 1) / 2).clamp(0, 1)
for i in range(8):
    img_data = clean_imgs[i].reshape(IN_CHANNELS, IMG_SIZE, IMG_SIZE).cpu().numpy()
    img = img_data.squeeze() if IN_CHANNELS == 1 else np.transpose(img_data, (1, 2, 0))
    axes[1, i].imshow(img, cmap='gray' if IN_CHANNELS == 1 else None, vmin=0, vmax=1)
    axes[1, i].axis('off')
axes[1, 0].set_title(f'Clean (FID={clean_fid:.0f})', fontsize=10, fontweight='bold')

for row, msg in enumerate(test_messages):
    wm_imgs = generate_samples(wm_models[msg][0], n_samples=8)
    wm_imgs = ((wm_imgs + 1) / 2).clamp(0, 1)
    for i in range(8):
        img_data = wm_imgs[i].reshape(IN_CHANNELS, IMG_SIZE, IMG_SIZE).cpu().numpy()
        img = img_data.squeeze() if IN_CHANNELS == 1 else np.transpose(img_data, (1, 2, 0))
        axes[row + 2, i].imshow(img, cmap='gray' if IN_CHANNELS == 1 else None, vmin=0, vmax=1)
        axes[row + 2, i].axis('off')
    axes[row + 2, 0].set_title(f'WM (FID={wm_fids[msg]:.0f})', fontsize=10)

fig.suptitle(f'{DATASET.upper()}: LoRA Watermark', fontsize=12, y=0.98)
fig.tight_layout(
    rect=[0, 0, 1, 0.96],
    h_pad=0.3,
    w_pad=0.1,
    pad=0.5
)
fig.savefig(f"outputs/{DATASET}_lora_results.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"\n{'='*70}")
print(f"FINAL: WM acc={wm_acc:.1f}%, Clean FID={clean_fid:.1f}")
print(f"Resume: Change TARGET_STEPS and run again to continue training")
print(f"{'='*70}")
print("Done!")
