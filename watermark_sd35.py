"""
SD3.5 Medium LoRA watermark — bounded watermark loss + FID.

Key fixes:
1. LoRA B explicit zero-init
2. Both v_base and v_theta under same autocast
3. Bounded watermark reward via tanh
4. FID enabled
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys, math, json, argparse, random, copy, gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ============================================================================
# ARGS
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='stabilityai/stable-diffusion-3.5-medium')
parser.add_argument('--output_dir', type=str, default='wm_cond_sd35')
parser.add_argument('--prompt', type=str, default='a photo of a cat')

parser.add_argument('--lora_rank', type=int, default=16)
parser.add_argument('--lora_alpha', type=float, default=16.0)

parser.add_argument('--steps', type=str, default='100')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--grad_accum', type=int, default=8)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--save_every', type=int, default=10000)

parser.add_argument('--wm_message', type=str, default='10101')
parser.add_argument('--wm_K', type=int, default=32)
parser.add_argument('--codebook_mode', choices=['auto', 'orthogonal', 'hypercube'], default='auto')
parser.add_argument('--wm_eps', type=float, default=1.5)
parser.add_argument('--wm_lambda', type=float, default=0.5)
parser.add_argument('--wm_tanh_scale', type=float, default=1.0,
                    help='Scale for tanh in bounded watermark loss')
parser.add_argument('--resid_reg', type=float, default=0.0)

parser.add_argument('--n_eval_seeds', type=int, default=5)
parser.add_argument('--n_eval_steps', type=int, default=15)
parser.add_argument('--n_detect_seeds', type=int, default=10)
parser.add_argument('--n_detect_queries', type=int, default=256)
parser.add_argument('--n_fid_samples', type=int, default=30)
parser.add_argument('--n_fid_real', type=int, default=30)
parser.add_argument('--n_train_latents', type=int, default=100)
parser.add_argument('--latent_batch_size', type=int, default=4)
parser.add_argument('--gen_batch_size', type=int, default=2)

parser.add_argument('--offload', type=str, default='none', choices=['none', 'model'])
parser.add_argument('--attn_slicing', action='store_true')
parser.add_argument('--grad_ckpt', action='store_true', default=True)
parser.add_argument('--quick', action='store_true')
parser.add_argument('--lora_targets', type=str, default='mlp',
                    choices=['mlp', 'attn', 'both'])

args = parser.parse_args()
from watermark_codebooks import make_key, message_code, decode_signature, target_score
if not args.wm_message or any(bit not in '01' for bit in args.wm_message):
    parser.error('--wm_message must be a nonempty binary string')
if args.codebook_mode == 'orthogonal' and 2 ** len(args.wm_message) > args.wm_K:
    parser.error('Orthogonal encoding requires 2**bits <= wm_K')
if args.codebook_mode == 'hypercube' and len(args.wm_message) > args.wm_K:
    parser.error('Hypercube encoding requires bits <= wm_K')
if args.codebook_mode != 'hypercube' and len(args.wm_message) > 20:
    parser.error('Use --codebook_mode hypercube above 20 bits')

if args.quick:
    args.steps = '100'
    args.n_eval_seeds = 2
    args.n_eval_steps = 12
    args.n_detect_seeds = 5
    args.n_detect_queries = 64
    args.n_fid_samples = 15
    args.n_fid_real = 15
    args.n_train_latents = 40
    print("[QUICK MODE]")

os.makedirs(args.output_dir, exist_ok=True)
import psutil
print(f"RAM avail: {psutil.virtual_memory().available/1e9:.1f} GB", flush=True)
free, total = torch.cuda.mem_get_info()
print(f"VRAM free: {free/1e9:.1f} / {total/1e9:.1f} GB", flush=True)

STEPS_LIST = sorted(int(s) for s in args.steps.split(','))
WM_BITS = [int(b) for b in args.wm_message]
WM_N_BITS = len(WM_BITS)
WM_K = args.wm_K
WM_EPS = args.wm_eps
WM_LAMBDA = args.wm_lambda
WM_TANH_SCALE = args.wm_tanh_scale
RESID_REG = args.resid_reg
N_MESSAGES = 2 ** WM_N_BITS

print(f"Message: {args.wm_message}, k={WM_K}, ε={WM_EPS}, λ={WM_LAMBDA}, tanh_scale={WM_TANH_SCALE}")
print(f"LoRA targets: {args.lora_targets}")
print(f"Steps: {STEPS_LIST}")


# ============================================================================
# LOAD MODEL
# ============================================================================
print(f"\nLoading {args.model}...", flush=True)
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

pipe = StableDiffusion3Pipeline.from_pretrained(
    args.model, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map="balanced",
)
if args.attn_slicing:
    pipe.enable_attention_slicing()

transformer = pipe.transformer
vae = pipe.vae
vae.eval()

LATENT_C, LATENT_H, LATENT_W = 16, args.latent_size, args.latent_size


# ============================================================================
# LATENT POOL
# ============================================================================
pool_path = os.path.join(args.output_dir, f'latent_pool_{args.latent_size}_{args.n_train_latents}.pt')

if os.path.exists(pool_path):
    print(f"\nLoading cached latent pool from {pool_path}", flush=True)
    latent_pool = torch.load(pool_path)
else:
    print(f"\nGenerating {args.n_train_latents} latents...", flush=True)
    latent_pool = []
    n_batches = (args.n_train_latents + args.latent_batch_size - 1) // args.latent_batch_size
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc='Generating'):
            b = min(args.latent_batch_size, args.n_train_latents - i * args.latent_batch_size)
            gen = torch.Generator('cuda').manual_seed(i * args.latent_batch_size)
            out = pipe(
                [args.prompt] * b,
                num_inference_steps=20,
                output_type='latent',
                generator=gen,
                height=args.latent_size * 8,
                width=args.latent_size * 8,
            ).images
            latent_pool.append(out.cpu())
    latent_pool = torch.cat(latent_pool, dim=0).to(torch.bfloat16)
    torch.save(latent_pool, pool_path)

print(f"Latent pool: {latent_pool.shape}")


# ============================================================================
# PROMPT ENCODING
# ============================================================================
print(f"\nEncoding prompt: '{args.prompt}'", flush=True)
with torch.no_grad():
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt, prompt_2=args.prompt, prompt_3=args.prompt,
        device='cuda', num_images_per_prompt=1, do_classifier_free_guidance=False,
    )
prompt_embeds = prompt_embeds.to(torch.bfloat16)
pooled_prompt_embeds = pooled_prompt_embeds.to(torch.bfloat16)

print("Freeing text encoders...", flush=True)
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.text_encoder_3 = None
gc.collect(); torch.cuda.empty_cache()


# ============================================================================
# SECRET KEY
# ============================================================================
D_latent = 16 * args.latent_size * args.latent_size
if WM_K > D_latent:
    raise ValueError(
        f"wm_K={WM_K} exceeds flattened latent dimension D={D_latent}. "
        "Choose a smaller --wm_K or a larger --latent_size."
    )
torch.manual_seed(12345)
with torch.no_grad():
    P, codes = make_key(D_latent, WM_N_BITS, WM_K, 'cuda', args.codebook_mode)
WM_CODE = message_code(WM_BITS, codes, args.codebook_mode)
Pc_target = (P @ WM_CODE).float()

print(f"P: {P.shape}, codes: {codes.shape}")


# ============================================================================
# LoRA
# ============================================================================
def is_linear(name, module):
    return isinstance(module, torch.nn.Linear)

def is_attn(name):
    return any(k in name for k in ['to_q', 'to_k', 'to_v', 'to_out.0'])

def is_mlp(name):
    return ('ff' in name or 'mlp' in name.lower()) and not is_attn(name)

if args.lora_targets == 'mlp':
    target_modules = [n for n, m in transformer.named_modules()
                      if is_linear(n, m) and is_mlp(n)]
elif args.lora_targets == 'attn':
    target_modules = [n for n, m in transformer.named_modules()
                      if is_linear(n, m) and is_attn(n)]
else:
    target_modules = [n for n, m in transformer.named_modules()
                      if is_linear(n, m) and (is_attn(n) or is_mlp(n))]

target_modules = sorted(set(target_modules))
print(f"\nLoRA target modules ({len(target_modules)})")

lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                         target_modules=target_modules, lora_dropout=0.0, bias='none')
transformer = get_peft_model(transformer, lora_config)
transformer.print_trainable_parameters()

print("\nExplicitly zeroing LoRA B weights...", flush=True)
with torch.no_grad():
    for name, p in transformer.named_parameters():
        if 'lora_B' in name:
            p.zero_()

max_b = max((p.abs().max().item() for n, p in transformer.named_parameters()
             if 'lora_B' in n), default=0.0)
print(f"  max |lora_B| = {max_b:.8f}")

n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in transformer.parameters())
assert n_train < n_total * 0.01

if args.grad_ckpt:
    transformer.enable_gradient_checkpointing()

INIT_STATE = copy.deepcopy(transformer.state_dict())
def reset_lora():
    transformer.load_state_dict(INIT_STATE, strict=False)


# ============================================================================
# VELOCITY WRAPPERS
# ============================================================================
class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def get_velocity(x, t_scalar, use_lora=True, gradient=False):
    B = x.shape[0]
    ts = torch.full((B,), t_scalar * 1000.0, device=x.device, dtype=torch.bfloat16)
    if use_lora:
        ctx = _NullCtx() if gradient else torch.no_grad()
    else:
        ctx = transformer.disable_adapter()
    with ctx:
        out = transformer(hidden_states=x, timestep=ts,
                          encoder_hidden_states=prompt_embeds,
                          pooled_projections=pooled_prompt_embeds,
                          return_dict=False)[0]
    return out


def get_velocity_autocast(x, t_scalar, use_lora=True, gradient=False):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        return get_velocity(x, t_scalar, use_lora=use_lora, gradient=gradient)


def sample_x0(B):
    return torch.randn(B, LATENT_C, LATENT_H, LATENT_W, device='cuda', dtype=torch.bfloat16)

def sample_x1(B):
    idx = torch.randint(0, len(latent_pool), (B,))
    return latent_pool[idx].to('cuda', dtype=torch.bfloat16)


# ============================================================================
# VERIFY
# ============================================================================
print("\nVerifying v_base == v_theta at init (both under autocast)...", flush=True)
transformer.eval()
with torch.no_grad():
    x_test = torch.randn(1, LATENT_C, LATENT_H, LATENT_W, device='cuda', dtype=torch.bfloat16)
    for t_test in [0.1, 0.5, 0.9]:
        v_base_t = get_velocity_autocast(x_test, t_test, use_lora=False, gradient=False)
        v_theta_t = get_velocity_autocast(x_test, t_test, use_lora=True, gradient=False)
        diff = (v_base_t.float() - v_theta_t.float()).abs().max().item()
        print(f"  t={t_test:.1f}: max|v_base - v_theta| = {diff:.8f}")
        del v_base_t, v_theta_t
transformer.train()
gc.collect(); torch.cuda.empty_cache()


lora_params = [p for n, p in transformer.named_parameters()
               if 'lora' in n and p.requires_grad]

def make_optimizer(n_steps):
    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
    return opt, sched


# ============================================================================
# GENERATION
# ============================================================================
@torch.no_grad()
def generate_latent_batch(seeds, n_steps, use_lora):
    B = len(seeds)
    z = torch.stack([
        torch.randn(LATENT_C, LATENT_H, LATENT_W, device='cuda', dtype=torch.bfloat16,
                    generator=torch.Generator(device='cuda').manual_seed(s))
        for s in seeds
    ])
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device='cuda')
    x = z.clone()
    for i in range(n_steps):
        t = ts[i].item()
        dt = ts[i+1].item() - t
        v = get_velocity_autocast(x, t, use_lora=use_lora, gradient=False)
        x = x + v.to(torch.bfloat16) * dt
    return x

@torch.no_grad()
def generate_image_set(n_samples, use_lora, seed_offset=20000, gen_batch_size=2):
    if n_samples <= 0:
        return torch.empty(0)
    all_imgs = []
    for i in range(0, n_samples, gen_batch_size):
        b = min(gen_batch_size, n_samples - i)
        seeds = [seed_offset + i + j for j in range(b)]
        latents = generate_latent_batch(seeds, args.n_eval_steps, use_lora)
        with torch.no_grad():
            imgs = vae.decode((latents / vae.config.scaling_factor).to(torch.bfloat16)).sample
            imgs = ((imgs.clamp(-1, 1) + 1) / 2).to(torch.float32)
        for j in range(b):
            img = imgs[j].cpu()
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            all_imgs.append(img)
        del latents, imgs
        torch.cuda.empty_cache()
    return torch.stack(all_imgs)


# ============================================================================
# DETECTOR
# ============================================================================
@torch.no_grad()
def detect(n_queries=256, use_lora=True):
    signature = torch.zeros(WM_K, device='cuda', dtype=torch.float32)
    for _ in range(n_queries):
        x0 = sample_x0(1)
        x1 = sample_x1(1)
        t_val = random.random()
        t_b = torch.full((1, 1, 1, 1), t_val, device='cuda', dtype=torch.bfloat16)
        x_t = ((1 - t_b) * x0 + t_b * x1).to(torch.bfloat16)
        v = get_velocity_autocast(x_t, t_val, use_lora=use_lora, gradient=False)
        v_flat = v.reshape(1, -1).float()
        s = math.sin(2 * math.pi * t_val)
        signature += s * (v_flat @ P).squeeze(0)
    signature /= n_queries
    best_bits, scores = decode_signature(signature, codes, WM_N_BITS, args.codebook_mode)
    return best_bits, scores.cpu().numpy(), signature.cpu().numpy()


# ============================================================================
# FID
# ============================================================================
def compute_fid(images_a, images_b):
    from torchvision.models import inception_v3, Inception_V3_Weights
    from scipy.linalg import sqrtm
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                              transform_input=False).to('cuda').eval()
    inception.fc = torch.nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1)

    def get_feats(imgs):
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs), 16):
                batch = imgs[i:i+16].to('cuda')
                if batch.shape[1] == 1:
                    batch = batch.repeat(1, 3, 1, 1)
                batch = F.interpolate(batch, size=(299,299), mode='bilinear',
                                      align_corners=False)
                batch = (batch - mean) / std
                feats.append(inception(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)

    f_a = get_feats(images_a)
    f_b = get_feats(images_b)
    mu_a, mu_b = f_a.mean(0), f_b.mean(0)
    sig_a = np.cov(f_a, rowvar=False); sig_b = np.cov(f_b, rowvar=False)
    eps = 1e-6
    sig_a += eps * np.eye(sig_a.shape[0]); sig_b += eps * np.eye(sig_b.shape[0])
    diff = mu_a - mu_b
    covmean = sqrtm(sig_a @ sig_b)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff @ diff + np.trace(sig_a + sig_b - 2*covmean))


FID_ENABLED = (args.n_fid_samples > 1 and args.n_fid_real > 1)
if FID_ENABLED:
    print(f"\nBuilding FID reference ({args.n_fid_real} samples)...", flush=True)
    real_imgs = generate_image_set(args.n_fid_real, use_lora=False,
                                    seed_offset=100000, gen_batch_size=args.gen_batch_size)
    print(f"Pre-computing clean FID ({args.n_fid_samples} samples)...", flush=True)
    clean_imgs_fid = generate_image_set(args.n_fid_samples, use_lora=False,
                                         seed_offset=50000, gen_batch_size=args.gen_batch_size)
    fid_clean_precomputed = compute_fid(real_imgs, clean_imgs_fid)
    print(f"  FID(real, clean) = {fid_clean_precomputed:.3f}")
    del clean_imgs_fid
    gc.collect(); torch.cuda.empty_cache()
else:
    real_imgs = None
    fid_clean_precomputed = None
    print("\nFID disabled")

def evaluate_fid():
    if not FID_ENABLED or real_imgs is None:
        return {}
    wm_imgs = generate_image_set(args.n_fid_samples, use_lora=True,
                                  seed_offset=50000, gen_batch_size=args.gen_batch_size)
    fid_wm = compute_fid(real_imgs, wm_imgs)
    del wm_imgs
    gc.collect(); torch.cuda.empty_cache()
    return {'fid_real_clean': fid_clean_precomputed,
            'fid_real_wm': fid_wm,
            'fid_ratio': fid_wm / max(fid_clean_precomputed, 1e-6)}


# ============================================================================
# SWEEP LOOP
# ============================================================================
print(f"\n{'='*60}\nSWEEP\n{'='*60}")

sweep_results = []
prev_step = 0

for run_idx, target_steps in enumerate(STEPS_LIST):
    print(f"\n{'#'*60}")
    print(f"# RUN {run_idx+1}/{len(STEPS_LIST)}: target_steps={target_steps}")
    print(f"{'#'*60}")

    if run_idx == 0:
        reset_lora()
        prev_step = 0

    n_new = target_steps - prev_step
    if n_new <= 0:
        continue

    optimizer, scheduler = make_optimizer(n_new)
    transformer.train()
    vel_history = []
    resid_history = []

    pbar = tqdm(range(n_new), desc=f'Train to {target_steps}')
    for step in pbar:
        optimizer.zero_grad()
        accum_vel = 0.0
        accum_resid = 0.0
        first_diag = None

        for micro in range(args.grad_accum):
            B = args.batch_size
            x0 = sample_x0(B)
            x1 = sample_x1(B)
            t_val = random.random()
            t_b = torch.full((B, 1, 1, 1), t_val, device='cuda', dtype=torch.bfloat16)
            x_t = ((1 - t_b) * x0 + t_b * x1).to(dtype=torch.bfloat16)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    v_base = get_velocity(x_t, t_val, use_lora=False, gradient=False)

            s_t = math.sin(2 * math.pi * t_val)
            wm_flat = (WM_EPS * s_t) * Pc_target
            wm = wm_flat.to(dtype=torch.bfloat16).view(1, LATENT_C, LATENT_H, LATENT_W)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                v_theta = get_velocity(x_t, t_val, use_lora=True, gradient=True)
            v_theta = v_theta.to(dtype=torch.bfloat16)

            residual = v_theta - v_base
            loss_vel = F.mse_loss(residual.float(), wm.float())

            # Bounded watermark loss via tanh
            residual_flat = residual.reshape(B, -1).float()
            proj = residual_flat @ P
            wm_corr = (s_t * (proj * WM_CODE.unsqueeze(0)).sum(dim=1)).mean()
            wm_corr_norm = wm_corr / (WM_EPS * 0.5 + 1e-8)
            loss_wm = -torch.tanh(wm_corr_norm * WM_TANH_SCALE)

            loss = loss_vel + WM_LAMBDA * loss_wm
            if RESID_REG > 0:
                loss = loss + RESID_REG * (residual.float() ** 2).mean()

            loss = loss / args.grad_accum
            if loss.dtype != torch.float32:
                loss = loss.float()
            loss.backward()
            accum_vel += loss_vel.item()
            accum_resid += residual.norm().item() / math.sqrt(B * LATENT_C * LATENT_H * LATENT_W)

            if step == 0 and micro < 2:
                resid_val = residual.reshape(B, -1).norm(dim=1).mean().item()
                target_val = wm.reshape(1, -1).norm(dim=1).mean().item()
                first_diag = (resid_val, target_val, loss_vel.item())

        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()
        vel_history.append(accum_vel / args.grad_accum)
        resid_history.append(accum_resid / args.grad_accum)

        if step == 0 and first_diag is not None:
            print(f"  [step0] resid={first_diag[0]:.6f}  target={first_diag[1]:.6f}  "
                  f"loss_vel={first_diag[2]:.6f}")

        if step % 10 == 0:
            pbar.set_postfix({
                'vel': f'{vel_history[-1]:.6f}',
                'rms': f'{resid_history[-1]:.4f}',
            })

    prev_step = target_steps
    transformer.eval()
    optimizer = None; scheduler = None
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n  Evaluating at {target_steps} steps...")

    dists = []
    for s_ in range(args.n_eval_seeds):
        z = torch.randn(1, LATENT_C, LATENT_H, LATENT_W, device='cuda',
                        dtype=torch.bfloat16,
                        generator=torch.Generator('cuda').manual_seed(s_))
        ts = torch.linspace(0.0, 1.0, args.n_eval_steps + 1, device='cuda')
        x_c = z.clone(); x_w = z.clone()
        for i in range(args.n_eval_steps):
            t = ts[i].item()
            dt = ts[i+1].item() - t
            v_c = get_velocity_autocast(x_c, t, use_lora=False, gradient=False)
            v_w = get_velocity_autocast(x_w, t, use_lora=True, gradient=False)
            x_c = x_c + v_c.to(torch.bfloat16) * dt
            x_w = x_w + v_w.to(torch.bfloat16) * dt
        dists.append((x_w - x_c).norm().item() / (x_c.norm().item() + 1e-8))
    endpoint_mean = float(np.mean(dists))

    wm_correct = 0; wm_scores_list = []
    for _ in range(args.n_detect_seeds):
        decoded, scores, sig = detect(args.n_detect_queries, use_lora=True)
        if list(decoded) == WM_BITS:
            wm_correct += 1
        wm_scores_list.append(target_score(torch.from_numpy(scores), WM_BITS, args.codebook_mode))

    clean_correct = 0; clean_scores_list = []
    for _ in range(args.n_detect_seeds):
        decoded, scores, sig = detect(args.n_detect_queries, use_lora=False)
        if list(decoded) == WM_BITS:
            clean_correct += 1
        clean_scores_list.append(target_score(torch.from_numpy(scores), WM_BITS, args.codebook_mode))

    wm_acc = 100.0 * wm_correct / args.n_detect_seeds
    clean_acc = 100.0 * clean_correct / args.n_detect_seeds
    wm_mean = np.mean(wm_scores_list); wm_std = np.std(wm_scores_list)
    cl_mean = np.mean(clean_scores_list); cl_std = np.std(clean_scores_list)
    wm_score_var = float(np.var(wm_scores_list))
    clean_score_var = float(np.var(clean_scores_list))
    margin = float(wm_mean - cl_mean)
    sep = (wm_mean - cl_mean) / max(wm_std, cl_std, 1e-8)

    eval_metrics = {
        'steps': target_steps,
        'message': args.wm_message,
        'bits': WM_N_BITS,
        'codebook_mode': args.codebook_mode,
        'wm_K': WM_K,
        'n_detect_trials': args.n_detect_seeds,
        'n_detect_queries': args.n_detect_queries,
        'detection_accuracy_wm': wm_acc,
        'detection_accuracy_clean': clean_acc,
        'separation_sigma': float(sep),
        'score_margin': margin,
        'wm_score_mean': float(wm_mean),
        'wm_score_std': float(wm_std),
        'wm_score_var': wm_score_var,
        'clean_score_mean': float(cl_mean),
        'clean_score_std': float(cl_std),
        'clean_score_var': clean_score_var,
        'paired_endpoint_distance_mean': endpoint_mean,
        'paired_endpoint_distance_std': float(np.std(dists)),
        'final_resid_loss': float(np.mean(vel_history[-20:])),
        'final_resid_rms': float(np.mean(resid_history[-20:])),
    }

    is_final = (run_idx == len(STEPS_LIST) - 1)
    if FID_ENABLED and is_final:
        print(f"  FID (WM generation, final step only)...")
        try:
            eval_metrics.update(evaluate_fid())
        except Exception as e:
            print(f"  FID failed: {e}")
            eval_metrics['fid_ratio'] = None
    else:
        eval_metrics['fid_ratio'] = None

    sweep_results.append(eval_metrics)
    transformer.save_pretrained(os.path.join(args.output_dir, f'step_{target_steps}'))

    print(f"\n  Results at {target_steps}:")
    print(f"    WM acc:           {eval_metrics['detection_accuracy_wm']:.1f}%")
    print(f"    Clean acc:        {eval_metrics['detection_accuracy_clean']:.1f}%")
    print(f"    Separation:       {eval_metrics['separation_sigma']:.2f} σ")
    print(f"    Score margin:     {eval_metrics['score_margin']:.4f}")
    print(f"    WM score:         {eval_metrics['wm_score_mean']:.4f} ± {eval_metrics['wm_score_std']:.4f}")
    print(f"    Clean score:      {eval_metrics['clean_score_mean']:.4f} ± {eval_metrics['clean_score_std']:.4f}")
    print(f"    Endpoint dist:    {eval_metrics['paired_endpoint_distance_mean']:.4f}")
    print(f"    Resid loss:       {eval_metrics['final_resid_loss']:.6f}")
    print(f"    Resid RMS:        {eval_metrics['final_resid_rms']:.6f}")
    if eval_metrics.get('fid_ratio') is not None:
        print(f"    FID(real,clean):  {eval_metrics['fid_real_clean']:.3f}")
        print(f"    FID(real,wm):     {eval_metrics['fid_real_wm']:.3f}")
        print(f"    FID ratio:        {eval_metrics['fid_ratio']:.3f}")

transformer.save_pretrained(os.path.join(args.output_dir, 'final'))
torch.save({'P': P.cpu(), 'codes': codes.cpu(), 'WM_CODE': WM_CODE.cpu(),
            'message': args.wm_message, 'codebook_mode': args.codebook_mode},
           os.path.join(args.output_dir, 'codebook.pt'))


print(f"\n{'='*95}")
print(f"SWEEP SUMMARY")
print(f"{'='*95}")
print(f"  {'Steps':>8} {'WM Acc':>8} {'Clean':>8} {'Sep(σ)':>8} "
      f"{'Endpt':>8} {'ResRMS':>10} {'FID(cl)':>10} {'FID(wm)':>10} {'Ratio':>8}")
print(f"  {'-'*92}")
for r in sweep_results:
    fr = r.get('fid_ratio'); fc = r.get('fid_real_clean'); fw = r.get('fid_real_wm')
    frs = f"{fr:.3f}" if fr is not None else "n/a"
    fcs = f"{fc:.2f}" if fc is not None else "n/a"
    fws = f"{fw:.2f}" if fw is not None else "n/a"
    print(f"  {r['steps']:>8d} {r['detection_accuracy_wm']:>7.1f}% "
          f"{r['detection_accuracy_clean']:>7.1f}% "
          f"{r['separation_sigma']:>7.2f} "
          f"{r['paired_endpoint_distance_mean']:>8.4f} "
          f"{r['final_resid_rms']:>10.6f} "
          f"{fcs:>10} {fws:>10} {frs:>8}")

with open(os.path.join(args.output_dir, 'sweep_results.json'), 'w') as f:
    json.dump({'config': vars(args), 'results': sweep_results}, f, indent=2, default=str)

print(f"\nSaved to {args.output_dir}/")
print("Done!")
