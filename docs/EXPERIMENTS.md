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

### SD3.5 Hypercube Diagnostics

Exact recovery and pooled target-score separation measure different things:
a positive pooled score does not imply every hypercube bit has the correct
sign. Diagnose a saved SD3.5 adapter without retraining:

```bash
python diagnose_sd35_hypercube.py \
  --run-dir table_runs/main_architectures/sd35/hypercube32/MESSAGE \
  --output table_runs/sd35_diagnostics/MESSAGE.json
```

The report records bit accuracy, signed coordinate margins, and raw clean
and watermarked signatures. Paired base-subtracted recovery is a diagnostic
requiring access to the unwatermarked base, not a replacement for the main
black-box verifier. Four trials are a screening diagnostic, not a substitute
for the full experimental protocol.

`watermark_sd35.py` also accepts optional `--wm_proj_weight` and
`--wm_bit_weight` for coordinatewise residual fitting and raw-velocity
hypercube bit-margin training. Both default to zero to preserve the old
objective. These are experimental remedies, not demonstrated improvements;
evaluate recovery and FID in separate output directories before reporting
them. Changing LoRA rank alone does not establish an effective payload limit.

For a short continuation of a saved adapter, the diagnostic accepts
`--repair-steps 50 --repair-proj-weight 10 --repair-bit-weight 1`.
It saves a separate adapter beside the diagnostic JSON and does not modify
the source run. `--adapter-path PATH` evaluates or continues an alternative
adapter trained with the source run's original key. Repair output is
explicitly quality-incomplete: no FID is evaluated by this diagnostic.

Screening the first saved 32-bit SD3.5 message (four trials, 256 queries
each) found 78.125% raw bit accuracy and zero exact recovery. Paired base
subtraction raised bit accuracy to 93.75% but still failed exact recovery.
A 50-update coordinate-fitting/bit-margin continuation at unchanged rank
16 reached 100% base-subtracted recovery, but raw recovery remained zero.
Fifty more updates with stronger bit-margin weight reached 89.0625% raw bit
accuracy, still zero exact recovery, and 100% base-subtracted recovery.
The latter screening used batch size eight instead of four; these are
preliminary diagnostics, not matched full-protocol results or evidence of
quality preservation. The raw-verifier failure is not solved by these probes.

### Audio Extension

Install `requirements-audio.txt` in a suitable environment before preparing
audio. The latent-space MLP and Transformer use a frozen Music2Latent codec.
Stable Audio LoRA uses its own frozen codec and requires a checkpoint whose
objective is `rectified_flow`; incompatible objectives are rejected.
Music2Latent is a codec here, not a third generative architecture.

The local `.venv-audio` isolates audio dependencies from the image
environment. CLAP's published PyTorch checkpoint requires Torch >= 2.6
with the installed Transformers version; the tested CUDA pair is
Torch 2.6.0 / Torchvision 0.21.0 (CUDA 12.4). Music2Latent 0.1.8 and
librosa 0.11.0 retain compatibility with the existing SciPy 1.14.1 pin.
Do not bypass the checkpoint safety check to use older Torch.

The sequential FMA study checks encoding, both model architectures,
both payloads, decoding, and CLAP quality on four tiny smoke runs first.
These runs test execution only, not recovery efficacy. It then prepares
2,000 train / 500 test clips, runs the 20 main jobs, and aggregates them:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  nohup .venv-audio/bin/python -u run_fma_audio_study.py \
  > table_runs/run_fma_audio_study.log 2>&1 &
```

Follow `table_runs/run_fma_audio_study.log` and
`table_runs/audio_study_status.json`. A lock prevents concurrent launches;
complete main results are skipped on restart. Unreadable/short/nonfinite
recordings are recorded as exclusions; insufficient eligible split counts
fail preparation rather than silently reducing the study. Raw clean-model
message matches are retained and are not calibrated ownership decisions.
Stable Audio needs separate approved gated-model access; MAESTRO needs
its own downloaded recordings. Neither is included in this FMA launch.

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

Stable Audio 3 uses a separate Python 3.10 environment, `.env-stable-audio`,
with `requirements-stable-audio.txt`. The current official library requires
Torch/Torchaudio 2.7.1; install their CUDA 12.6 builds before the remaining
requirements. The pinned base checkpoint is
`stabilityai/stable-audio-3-small-music-base`, not the adversarially
post-trained denoiser. Its T5Gemma encoder requires access to
`stabilityai/stable-audio-3-small-music`. The cache preparer honors
`--seconds` (default six) and pads only to the codec input multiple;
it does not use the checkpoint's full two-minute sample size. Conditioning
uses a zero inpainting mask/input for generation without supplied audio.
The stable main grid uses training batch 1 and evaluation batch 4; these
differ from the scratch audio architectures and are not equal-budget
architecture comparisons.

Run the real-audio execution check before a full Stable Audio study:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  ENABLE_TORCH_COMPILE=0 .env-stable-audio/bin/python -u run_stable_audio_smoke.py
```

Logs, generated WAVs, and the smoke-only summary are under
`table_runs/stable_audio_smoke/`. This test uses four clips, five updates
per payload, 128 detection queries, two trials, and two quality samples;
it checks execution, not minimum embedding cost or reliable recovery.

Audio detection uses held-out interpolations, 4,096 queries, and 20 trials
per message, with clean-model and independent wrong-key controls. Quality
uses paired clean/watermarked generation and 500 held-out reference clips,
reported as **FD-CLAP** on normalized CLAP embeddings, not image FID or VGGish-based FAD. Do not pool audio
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

## Original-Key MLP Repair Probes

`repair_mlp_watermarks.py` continues saved MLP checkpoints without changing
their key or the raw pure-noise verification distribution. It penalizes
weak signed hypercube coordinates (or an orthogonal codeword's margin
against its strongest rival), while distilling the original velocity on
both pure-noise and interpolated queries. These are additional optimizer
updates, not a replacement for the original training budget.

```bash
python repair_mlp_watermarks.py \
  --run-dir table_runs/main_architectures/mlp/hypercube32 \
  --output-dir table_runs/repairs/mlp/hypercube32 --steps 200
```

The runner screens every 50 updates, then evaluates on separate held-out
queries (20 trials, 4,096 queries each by default). It also recomputes the
original MLP raw-pixel Frechet metric with 1,000 samples and 200 Euler steps;
this is not Inception FID. Original outputs remain untouched. Five-bit
hypercube and orthogonal checkpoints use the same command with their own
directories. `--steps 0` audits without training.

Raw decoding always returns a message. `raw_exact_recovery` therefore is
not an ownership test. The additional `ownership_acceptance` requires a
matching message and a score exceeding the maximum of 40 separate clean
calibration trials plus two score standard deviations. This empirical
gate uses one original clean model; it does not establish a population
false-positive bound. Do not replace the original paper metrics with gated
metrics without explicitly changing and validating the verification protocol.

For scratch MLP training, `--wm_query_bit_weight 10 --wm_bit_margin 0.15`
enables a matching-distribution hypercube margin loss. Its default is zero
to preserve earlier runs; successful continuation does not itself validate
this scratch-training setting. The original MLP `--steps` argument counts
epochs, each containing `n_train // batch_size` optimizer updates. Thus the
5,000-epoch grid used 115,000 updates per model at the default settings.

SD3.5 diagnostics accept `--repair-clean-calibration PATH` pointing to a
previous diagnostic JSON with the same clean model, projection and query pool
(the runner checks these before reuse). This
compensates its measured clean temporal signature in the training target;
verification remains raw, with no base subtraction. Optional
`--quality-samples 30` measures the original small-sample base-reference
image-quality protocol. These remain repair probes, not automatic main-table
replacements; test fresh queries and all messages before making broader claims.
The default `--quality-protocol legacy` reproduces the original forward-Euler
quality experiment. It is not the standard SD3 sampler. For actual SD3
generation, use `--quality-protocol standard --quality-samples 30`: this uses
the official Diffusers pipeline (descending noise schedule and VAE shift),
guidance scale one, the saved resolution and step count, and a base-generated
reference. It also saves a grid with clean images above repaired images.
Do not mix quality estimates from these two sampling protocols.

`run_sd35_repair_suite.py --calibration PATH --run` applies the same staged
schedule to all five original 32-bit messages: 50 projected-loss warm-up
updates, 100 clean-compensated updates, and 50 compensated updates at epsilon
3. Each final checkpoint gets 20 fresh 256-query trials and 30-sample quality
evaluation. The suite uses effective batch size four and writes stage logs
and a summary under `table_runs/repairs/sd35/hypercube32`. It does not overwrite
original results or automatically retry valid negative results. Omit `--run`
to inspect commands, or pass `--messages` to select a subset. `--skip-existing`
resumes matching completed stages; rebuilding a stage also rebuilds its
downstream stages.

After the repair suite, run `run_sd35_repair_suite.py --standard-quality-only
--run --skip-existing` to measure official-pipeline quality for the saved
adapters. If the first adapter was trained separately, provide its path with
`--reference-adapter`. This writes a separate `standard_quality` summary and
does not rerun training. Run SD jobs sequentially: a loading placement budget
does not limit temporary CUDA allocations during checkpoint loading.

## Stable Audio Pilot

Run `run_stable_audio_pilot.py` with `.env-stable-audio/bin/python` after the
Stable Audio smoke test. The pilot uses 128 FMA training clips and 32 held-out
clips, six seconds each, and one message per codebook (orthogonal5 and
hypercube32). Each model receives 500 continuous optimizer updates at the
existing learning rate, with adapter snapshots and original saved keys.
Recovery is screened every 100 updates using 512 queries and two trials;
screening preserves the training RNG state. Final recovery uses a separate
query-seed range with 1,024 queries and five trials, followed by preliminary
32-sample FD-CLAP evaluation with 50 Euler steps. These budgets do not replace
the full paper protocol. Reported training time includes checkpoint audits.

Logs, progress, snapshots, and results are under
`table_runs/stable_audio_pilot/`. The launcher locks against concurrent copies
and refuses to overwrite a completed pilot. Snapshots contain trainable
parameters only and require the original pretrained backbone and LoRA setup;
they are not standalone models or optimizer-resume checkpoints.

`run_stable_audio_validation.py` evaluates the original 100-step orthogonal
pilot adapter without retraining, then trains the other four orthogonal
messages and five hypercube messages for 100 updates each. Fresh runs use a
100-update cosine schedule; the reused checkpoint came from the first 100
updates of the pilot's 500-update schedule. Both retain their actual embedding
configuration. Recovery uses 4,096 queries, 20 trials, a fresh seed range,
and clean-model and wrong-key controls. Quality uses 32 held-out clips and
100 Euler steps and remains preliminary, not a 500-clip quality study.
Outputs are separate under `table_runs/stable_audio_validation/`, including
per-trial progress and recovery results saved before quality evaluation.
Completed matching results are skipped on a subsequent launch.

`run_stable_audio_more_steps.py` continues only the four H32 messages
with partial exact recovery in the 100-update validation. It adds 200
updates with the same adapters, original keys, LoRA rank, objective,
watermark strength, batch size, and initial learning rate. Because the
original snapshots did not save optimizer state, continuation starts a
fresh AdamW optimizer and 200-update cosine schedule. This is 100+200
updates, not an uninterrupted 300-update schedule. Snapshots are saved
every 50 added updates, labeled by cumulative updates; final validation
uses fresh seeds, 4,096 queries, 20 trials, and preliminary 32-clip quality.
The successful fifth message is not retrained. Results remain separate
under `table_runs/stable_audio_more_steps/`. New audio evaluations save
per-trial decoded bits and scores, enabling measured BER even when exact
recovery is partial; this does not retroactively fill missing old logs.

`run_stable_audio_margin_pilot.py` tests a separate repair on the first H32
message after its 300-update steps-only run. It adds 100 updates with the
same rank and nominal epsilon, training-only clean-bias compensation, and
a raw-velocity bit-margin penalty. The compensation coefficient is
`epsilon * code - 2 * clean_bias`, because the carrier's mean squared
value is one half for uniform query times. The saved effective coefficient
norm can differ from epsilon and must be reported alongside quality.
Calibration uses 4,096 training-pool queries, never held-out detection
trials. Raw decoding does not subtract the base velocity. Final testing
uses fresh seeds and the same 4,096-query/20-trial protocol, with 32-clip
preliminary quality. This is a changed embedding objective, not a steps-only
or controlled budget comparison. Results are isolated under
`table_runs/stable_audio_margin_pilot/`; success is not assumed.

`run_stable_audio_margin_suite.py` applies that successful pilot's stages
to the other four messages. Existing 300-update checkpoints are reused;
otherwise a training-only 200-update continuation first brings the saved
100-update adapter to 300 updates. All four then receive the identical
100-update compensated bit-margin repair and full recovery/quality
evaluation. Preparation skips redundant full detection and audio quality,
but saves screening and adapters. The original pilot is reused, not
retrained; its result is included in the final five-message summary.
Completed matching results are skipped on subsequent launches, including
negative results. Original and repaired results stay separate under
`table_runs/stable_audio_margin_suite/`. This matched-stage study does not
claim an uninterrupted 400-update learning-rate schedule or isolate the
individual effects of compensation, margin loss, and extra updates.

## Saved Metric Audit

`python audit_saved_metrics.py` exports existing run metrics to
`table_runs/metrics_audit/metrics.csv` and `metrics.json` without model loading
or changes to source results. It derives exact-recovery failure rates and
BER where saved bit accuracy is available, or where 100% exact recovery
implies zero BER. For partial recovery without decoded bits, it reports BER
bounds rather than an invented point estimate. Saved MLP clean-calibrated
ownership gates support empirical FPR/FNR/TPR. The separate
`exact_match_fpr_pct` and `exact_match_fnr_pct` fields use the fixed rule
"accept when the decoded message equals the predeclared target": clean
target matches give FPR and marked-model recovery failures give FNR.
Exploratory empirical ROC-AUC is calculated
only where per-trial marked and clean scores are saved (SD signed-bit margins
and Stable screening target scores). It is not a calibrated low-FPR claim.
Unsupported ownership rates and ROC metrics remain blank. Original, repaired, screening, and robustness rows must not be
pooled indiscriminately. Run again to include newly completed experiments.

The main paper now uses `paper/performance_results.tex`, generated by
`refresh_paper_performance.py`, to combine model, encoding, recovery,
measured clean-model FPR and relative sample quality in one table.
FPR measures acceptance of independent uniform full-codebook message claims.
MLP uses the completed fresh-query study. `complete_paper_fpr.py` completes
other generators using 20 clean decoded outputs under one saved key and
one million sampled claims per cohort. Audio/SD H32 reuse recorded clean
controls; UNet and SD O5/H5 collect new clean query trials. Those one million
claims are not one million model evaluations. Counts and bounds are retained
with the results and documented in the paper appendix. Analytical expected
FPR is not reported as a performance result. `sample_random_claim_fpr.py` provides a seeded
Monte Carlo check conditional on saved clean MLP decodes; it adds no new
velocity queries. The extended check uses 10 million claims per encoding:
O5 313,093 matches (3.13093%), H5 312,341 (3.12341%), H32 zero matches
(one-sided 95% upper bound 0.0000299573%). Zero 32-bit matches cannot
establish a zero population rate. Reproduce with
`python sample_random_claim_fpr.py --trials 10000000 --seed 830042 --output table_runs/random_claim_fpr/results_10m.json`.
The paper's FPR table instead uses genuine fresh-query results from
`evaluate_fresh_mlp_fpr.py --trials 10000`: independent query seeds,
random keys and claims, plus the original fixed-target controls, sharing
clean forward passes across encodings. `python refresh_paper_performance.py`
regenerates the main performance table from completed measurements. The fresh-key study
remains specific to MLP; the other cohorts use their saved random keys.
`generate_sd35_sample_figure.py --varied-subjects --steps 40 --output table_runs/sd35_sample_figure_varied --figure-name sd35_clean_watermarked_varied`
reproduces the paired high-resolution figure with all predefined samples.
`evaluate_mlp_image_fid.py` measures the final MLP checkpoints with the
same 500-sample Inception feature protocol as RF-UNet. The old pixel
distances remain supplementary diagnostics. `run_missing_image_quality.py`
waits for the existing audio job to release the GPU, evaluates SD3.5 O5/H5
with the standard sampler using `--quality-only`, and fills the remaining
quality entries. These quality evaluations do not train or calibrate models.

## Clean FTSS Outcome Study

`run_ftss_outcome_study.py` extends the matched FTSS MLP experiment using
its two saved clean 12,000-image base checkpoints. It fine-tunes on nested,
class-balanced sets of 50, 200, and 1,000 new images, with three subset
seeds per base, and evaluates EMA weights at 0, 1,000, and 3,000 steps.
No watermark is trained. The sibling `../ftss-memorization` repository is
read without modification.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u run_ftss_outcome_study.py
```

The original FTSS probe definition is evaluated with the validated batched
backend. Paired generation uses 1,000 samples per checkpoint. Outcomes include
equal-pool training-example preference, training/held-out flow-loss gap,
loss-based membership AUC, feature precision and coverage, and style diversity.
Results, per-sample audits, descriptive associations, and figures are saved in
`table_runs/ftss_outcome_study/`. Re-running the identical command resumes
completed measurements and saved optimizer states; changed protocols require a
new output directory. Shared base models and repeated checkpoints are recorded
as dependent observations, not independent model replications. This study does
not modify the watermark paper.

### Pretrained CIFAR-10 Extension

`python -u run_ftss_cifar_study.py` evaluates the public clean CIFAR-10
class-conditional RF-UNet during full-parameter fine-tuning. Defaults are
50/200/1,000-image nested test-split subsets, three subset seeds, and
0/300/1,000/3,000-step EMA evaluations. FTSS uses 20 balanced trajectories
and keeps the requested class fixed across each trajectory and its
perturbations. Generation uses 500 balanced class conditions and 100 Euler
steps. Quality is measured with Inception FID and PRDC, with disjoint
real-versus-real controls at matched sample counts. Training-example
preference is class-matched in feature and pixel spaces; flow-loss
membership and within-class diversity provide separate outcomes.

Results, optimizer snapshots, per-sample audits, and sample grids are in
`table_runs/ftss_cifar_study/`. The identical command resumes saved work.
The final report and association figures are generated automatically.
This uses one pretrained base, not independently trained model seeds, and
does not change either paper or the original checkpoint. Subset seeds
repeat fine-tuning, not base pretraining.

## Watermark Security Follow-Up

`python -u run_watermark_security.py` evaluates the saved five O5 and five
H32 CIFAR-10 watermarks without re-embedding them. The three attack families
are key-free carrier subtraction, sample-only versus velocity-supervised
student transfer, and full-backbone clean fine-tuning. Carrier estimation
uses 16,384 marked-model interpolation queries with prespecified subtraction
strengths 0.5/1/1.5, without querying a clean counterpart. Owner keys are used
only to evaluate recovery on independent queries.

Students are initialized from the public clean backbone and trained for
1,000 full-parameter updates; sample-only transfer uses 2,000 completed
teacher samples. Full-backbone fine-tuning starts from the marked model
after exactly merging its LoRA layers; its initialization is checked against
the teacher before any updates. Teacher-velocity errors are independently
audited, so failure to transfer is interpreted alongside imitation quality.

Each condition evaluates 20 trials of 4,096 queries plus balanced-class,
500-image Inception FID against a balanced CIFAR-10 test reference. These
new clean/teacher quality baselines differ from the older fixed-class
protocol and must not be mixed with those FID values. All original keys
and checkpoints are preserved. Per-trial decodes, images, student snapshots,
summaries and the quality/recovery figure are in
`table_runs/watermark_security/`. Re-running the identical command resumes
saved work; changed settings require a new output directory. The runner
does not modify the paper automatically.

### Held-Out FTSS Diagnostic Validation

`python -u run_ftss_diagnostic_validation.py` calibrates an FTSS ratio
threshold on the earlier UNet endpoint runs, then freezes it before
measuring new tuning seeds. Both clean MNIST MLP and UNet models are
tested. Each condition uses 400 distinct images: uniform sampling,
80% sampling mass on ten selected examples without changing the data,
and diverse digit-1 specialization. All 400 members are audited against
400 untouched same-domain examples; 200 more examples supply quality
references. A fixed pixel-RMSE near-copy rule is applied symmetrically
to members and nonmembers. The primary outcome is at least one percentage
point of training-minus-reference near-copy excess, not the condition's
name. All checkpoints generate 1,000 samples.

Primary sensitivity and false-alarm counts use the final 3,000-step
checkpoints; intermediate evaluations are secondary. Calibration FTSS is
remeasured with the same original-paper estimator used in validation.
Previously inspected calibration data are explicitly distinguished from
the new seeds, and the MLP results test transfer of the UNet-calibrated
rule. Checkpoints share one pretrained base per architecture, so seed
repeats are not independent base-model replications. Reports, the frozen
rule, source hashes, near-neighbor audits, numerical-clamp counters and
figures are saved in `table_runs/ftss_diagnostic_validation/`. The identical
command resumes work. This runner does not modify either paper.

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
# Audio ownership audit

`run_audio_ownership.py` evaluates the final FMA MLP, Transformer, and
Stable Audio checkpoints without training. It uses the MNIST MLP ownership
rule: exact target decoding and a target score strictly above the maximum
of 40 clean calibration scores plus twice their population standard deviation.
Calibration uses training-pool interpolations; 20 test trials use held-out
recordings, with 4096 queries per trial. Original keys and checkpoints are
preserved. Stable Audio H32 reuses saved per-trial test scores; scratch audio
and Stable Audio O5 receive fresh tests. This measures repeated-query behavior
against one clean counterpart and one fixed wrong key per message, not a
population false-positive guarantee.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
ENABLE_TORCH_COMPILE=0 .env-stable-audio/bin/python -u run_audio_ownership.py
.env-stable-audio/bin/python finalize_audio_ownership.py
```

The evaluator resumes completed calibration trials and per-message tests in
`table_runs/audio_ownership/`. The finalizer requires all 30 results, verifies
the metrics, generates `paper/audio_ownership_results.tex`, refreshes the metric
audit, and rebuilds the paper. `finalize_audio_ownership.py --wait` waits for an
already-running evaluator before performing those steps. It refuses to publish
partial or failed evaluations.
# Clean SD3.5 Fine-Tuning Sensitivity

`run_ftss_sd35_finetuning.py` tests whether ordinary, unwatermarked LoRA
fine-tuning changes SD3.5's remaining-flow sensitivity. Defaults use 10 and
200 real CIFAR-10 cat images, three seeds, and checkpoints at 0, 100, 300,
and 1,000 optimizer steps. The cached clean backbone is used without any
watermark adapter. Training uses AdamW at `5e-5`, rank-16 attention and
feed-forward LoRA, batch 8, and the standard noise-minus-data target.

This is a 256px, cat-conditioned, CFG=1 pilot, not a native-resolution or
full-parameter experiment. Euler integration runs from noise sigma 1 to
data sigma 0, using FP32 latent states and BF16 network evaluation. Paired
probes compare RMS finite-difference gains at relative perturbations 0.02
and 0.1. Reports include complete temporal curves and train/reference
flow losses; latent nearest-neighbor distances are not labeled as copies.

```bash
python -u run_ftss_sd35_finetuning.py
python -m unittest test_ftss_sd35_finetuning
```

Results, checkpoints, the report, and the final curve figure are written to
`table_runs/ftss_sd35_finetuning/`. Existing output directories are preserved;
use `--output` for an independent run.

## Embedding Strategy and Whole-Model Robustness

`run_embedding_strategy_robustness.py` compares LoRA and full-parameter
watermark embedding into the same pretrained CIFAR-10 RF-UNet. It reuses
the five original O5 keys/messages and five H32 keys/messages. Both arms
use identical objectives, independently paired minibatch/noise/time draws,
batch 64, AdamW at `1e-4`, and a 1,000-update cosine schedule. Full-parameter
embedding here means adapting the clean pretrained model, not training a
new generator from scratch. The LoRA arm retains the original additional
trainable time/output-selected layers.

All 20 marked baselines are evaluated before attacks. LoRA updates are
merged and equivalence checked first. Independent whole-model attacks are:
25/50% magnitude pruning; simulated 8/4/2-bit quantization; 1,000 clean
full-backbone updates at `1e-5` or `1e-4`; and strength-one sine/cosine
carrier regression/subtraction. Compression includes every floating-point
parameter, including bias and normalization; it is not integer-backend
inference. Actual zero counts are saved to expose threshold ties.

Evaluation uses 20 trials of 4,096 queries and balanced 500-image,
100-step FID. Results report exact recovery, bit accuracy, FID/clean, and
FID relative to each arm's own marked baseline. Initial performance must
be compared before interpreting attack differences; identical training
budgets do not guarantee identical initial recovery or quality.

```bash
python -u run_embedding_strategy_robustness.py
python -m unittest test_embedding_strategy_robustness
```

The default matrix has 200 evaluations, saved with checkpoints, source
hashes, `results.json`, `summary.json`, `summary.csv`, and `report.md` under
`table_runs/embedding_strategy_robustness/`. The identical command resumes
completed evaluations and interrupted training; changed sources/settings
require a separate `--output`. Existing paper tables are not overwritten.

## Secret-Carrier Pilot

`python run_secret_carrier_pilot.py` compares sine, secret multi-tone, and
smooth input-dependent carriers on one CIFAR UNet H32 message. Each has an exact
injected-field control and a separate 300-step LoRA embedding. A fresh shared
projection and private carrier seed are saved locally for reproducibility.
Attacks use marked-model responses and public CIFAR query inputs only: joint
Fourier regression, time-bin SVD at ranks 1 and 4, and independent nonlinear
random-feature regression. Owner decoding does not subtract clean-model responses.

Results and sample grids go to `table_runs/secret_carrier_pilot/`, including
`report.md`, per-trial scores, fitted attack parameters and the private owner keys.
The default 100-image FID is a paired pilot quality screen. An injected-field
success is not evidence that a trained model can embed the carrier, and failed
attacks are not a general security guarantee. Resume with the identical command;
finished evaluations and embeddings are reused, while interrupted short embeddings
restart deterministically. This pilot does not modify the manuscript.

## Rare-Region Gate Pilot

`python run_tail_gate_pilot.py` screens a smooth sigmoid gate on a private
standardized state projection. Thresholds 2.5/4 and slopes 2/4 give four cases.
The coordinate uses the time-dependent mean and variance of the interpolation
distribution, measured from public CIFAR training data. Owner queries shift an
ordinary interpolated state into the keyed region. Each model receives 300
mixed ordinary/trigger updates; its paired unwatermarked control receives the
same queries and losses with a zero watermark target. The trigger objective
preserves clean-model velocity plus the watermark, rather than assigning an
invalid ordinary flow target to an OOD state. Projection fitting also enforces
the small gated signal on ordinary inputs, penalizing an always-on shortcut.

Exact injected fields and trained models are reported separately. Attacks use
ordinary Fourier regression or random/gradient OOD probing followed by regression.
The OOD objective uses neither clean responses nor the owner gate; gate discovery
is measured afterward against matching probes of the unwatermarked control.
Checkpoint gradients are needed only for gradient probing. Verification uses
marked velocities alone; clean subtraction is an analysis-only leakage metric.
Models with 100% baseline recovery additionally receive 50% whole-model pruning,
4-bit simulated quantization and 300 full-backbone clean fine-tuning updates.
These are screening tests, not a comprehensive trigger-removal benchmark.

Results, private keys, attack fits, probes and sample grids are saved under
`table_runs/tail_gate_pilot/`. The pilot reuses the previous secret-carrier
pilot's real Inception features, and uses 100 generated images by default.
Resume with identical arguments/sources; interrupted short embedding jobs restart.
Neither pilot automatically changes the manuscript.

## Baseline-First Comparison

The frozen design is in `baseline_protocol.md`. Start development calibration:

```bash
python run_baseline_first.py --stage tune
```

This runs three prespecified strengths for each of four methods at 32 and five
bits (24 development trials), with paired unwatermarked controls, 1,000 full-
backbone updates and 5,000-image FID/KID. Methods are dynamics, Uchida-inspired
weight regularization, an explicitly labelled Adi-inspired generative trigger
adaptation, and a public-direction constant-carrier control. Every failed trial
is retained. Development strengths are selected without attack results.

After calibration completes, five independent-key confirmation runs per setting:

```bash
python run_baseline_first.py --stage confirm
```

The confirmation command verifies that sources, protocol, sampling, optimizer
and evaluation settings match development. Reuse the identical command to resume.
Use a separate `--output` for small smoke tests; they are not scientific results.
Artifacts are under `table_runs/baseline_first/`, including stage reports,
`selection.json`, raw feature arrays, checkpoints, costs and clean-reseed noise
floor. `baseline_gate.md` remains pending until the later comparative attacks
and transfer studies exist. This runner implements calibration and confirmation,
not the subsequent attack matrix or a premature claim of advantage.

`python run_baseline_first_sequence.py` runs both stages sequentially with the
frozen defaults. It records progress in `sequence.json` and stops on an error.
It does not start the attack matrix before the baseline acceptance audit.
