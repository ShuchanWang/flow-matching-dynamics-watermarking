# Baseline-First Protocol, Frozen Before Tuning

## Scope and Order

Use the public pretrained CIFAR-10 RF-UNet checkpoint and the first 20,000
public CIFAR training examples for every method. Freeze this file and hash it
in every run. Run tuning, then five-key confirmation, then whole-model attacks,
then transfer/removal, then the comparative report. Do not change the manuscript
or run new SD3.5/theory/ECC work before the comparative gate.

The previous tail-gate pilot is complete (40 evaluations). It is not a baseline
in this study. Historical five-message results are not independent-key evidence.

## Replication and Tuning

Primary payload: 32 information bits. Secondary: five bits, using the orthogonal
codebook for the dynamics method. All baselines carry the same information bits.
Use three prespecified strength trials per method/payload on one development key
and seed, disjoint from confirmation. Then freeze the selected strength and run
five fresh independent keys/training seeds per method/payload. Pair messages,
ordinary minibatches and generation noise across methods at each replication.
Save all failed trials. No attack result influences tuning or selection.

All embedding runs start from exactly the same pretrained clean checkpoint and
update the full backbone for 1,000 AdamW updates, lr 1e-4, weight decay .01,
cosine schedule, batch 64, clip norm 1. Trigger auxiliary batches contain 16
examples and their extra computation is recorded. Clean controls use the same
initialization, steps, optimizer and method-specific auxiliary regularizers,
with zero watermark targets. Trigger controls preserve the clean velocity on
the same auxiliary trigger bridge rather than learn the encoded target.

Strength grids (three independent restarts, not successive continuation):

| Method | Strengths |
|---|---|
| Dynamics | epsilon = .5, 1.5, 3; projection weight 10, correlation weight 1 |
| Weight | BCE regularizer weight = .1, 1, 10 |
| Trigger | auxiliary bridge loss weight = .1, 1, 10 |
| Naive | epsilon = .5, 1.5, 3; public directions, constant time carrier |

Select the lowest FID/control candidate among development trials with >=95%
exact recovery and FID/control <=1.05. If none qualifies, select the highest
recovery candidate (tie: lowest FID/control), label it unqualified, and still
report its confirmation results. Do not add trials without a recorded protocol
amendment and equal additional effort for every method.

## Baseline Definitions

- Dynamics: original projected residual/correlation objective, sine carrier,
  fresh private projection/codebook; H32 or O5 as appropriate.
- Weight: Uchida-inspired random-projection binary cross-entropy regularizer
  on the output-filter-averaged weights of a fixed convolution. Read bits from
  the same weights after merging, with no detachable verification head.
  Record the layer, tensor shape, projection and logit scale (10).
- Trigger: Adi-inspired generative adaptation, not a reproduction of the
  classification method. A private Gaussian initial noise and class map to a
  designated 32-tile image encoding the message. Train flow-matching bridges
  between jittered trigger noise and the designated image (jitter std .02).
  Verify generated endpoint tile means, with the private cell permutation.
- Naive: public fixed orthogonal output directions and constant carrier, with
  the same fitting/correlation objective. This controls private directions and
  temporal modulation jointly, not their separate contributions. A single
  direction is not a fair 32-bit matched-capacity baseline.

Primary sources: https://arxiv.org/abs/1701.04082 and
https://www.usenix.org/conference/usenixsecurity18/presentation/adi.

## Evaluation

Use 5,000 balanced generated images, 100 Euler steps, and 5,000 balanced CIFAR
test reference images, fixed paired seeds. Use the existing torchvision Inception
feature pipeline identically throughout. Report absolute FID, FID/control, KID
against real images and against the generated control. KID is an unbiased cubic
polynomial MMD estimate: 20 subsets of size 1,000, fixed subset seed. Do not use
a KID ratio because unbiased estimates can be negative. Save feature arrays.
Generate a second clean sample set with a different seed for the clean/clean
noise floor. Its pairwise FID/KID is not a quality degradation threshold.

Dynamics/naive verification: 20 trials of 4,096 ordinary interpolation queries,
no clean-model subtraction. Trigger: 20 independently jittered trigger initial
states, Euler 100, jitter std .01; report image-query and velocity-call costs.
Weight extraction is deterministic: one white-box read, not 20 independent
trials. Record exact recovery and bit accuracy per model and summarize over
five independent trained keys using sample standard deviation. Keep binomial
intervals for within-model query trials separate from key-level success intervals.

## Locked Attack Matrix (After Baseline Confirmation)

All conditions independently start from a merged, unchanged marked baseline.
Apply to the whole backbone and apply matched modifications to its clean control.
Pruning: .25, .5, .75, .9. Simulated quantization: 8,6,4,3,2 bits. Precision: bf16.
Actual int8 PTQ is a separate backend experiment; if unsupported, retain an
explicit not-run/unsupported cell, never substitute simulated rounding.

Ordinary fine-tuning: original embedding-trained tensor names, 1k/5k updates at
1e-5 and 1e-4; because this study embeds all parameters, these are whole-backbone
updates. The requested full-backbone 1k condition is therefore the same condition,
not a duplicate experiment. Distill with velocity or sample-only supervision
into the same architecture and a smaller UNet; record initialization, parameter
ratio, utility and the common state/time interface. Weight extraction on the
smaller architecture is N/A; same-architecture extraction is tested.

Removal: marked-response-only sine/full-harmonic regression for dynamics;
fine-tuning/removal curves for all methods; trigger search for the trigger
adaptation. Report attack compute/queries and post-attack utility, not recovery
alone. Do not interpret N/A as a win or hide baseline failures.

## Reporting and Gate

Write per-key artifacts and report.md under table_runs/baseline_first/.
Produce one comparative table, recovery-versus-quality compression curves, and
transfer comparison. baseline_gate.md distinguishes broad advantage, an
interface-specific advantage, and no demonstrated advantage. A venue decision
is not inferred mechanically from the number of table cells won.

Small smoke tests use separate output directories, are labelled non-scientific,
and are never pooled with 5,000-image results. The first launch performs tuning
only; confirmation and attacks follow the frozen selection and acceptance audit.
