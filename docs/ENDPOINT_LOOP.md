# Exact Endpoint-Preserving Translation Loop

Let u be the actual frozen clean checkpoint. Choose a fixed keyed unit direction
q, alpha(t) = sin(2*pi*t), beta(t) = (1-cos(2*pi*t))/(2*pi), and
delta(t) = epsilon*beta(t)*q, with the endpoints explicitly set to zero.

The continuous field is v(y,t) = u(y-delta(t),t) + delta'(t). If X is a clean
trajectory, Y(t) = X(t) + delta(t) solves this field. Under ODE uniqueness,
Y(1) = X(1) for every initial state. This statement does not require the clean
checkpoint to be a population-optimal flow-matching model.

## Correct Discrete Construction

For any deterministic clean step map C_k, including its clipping operation,
define the marked step as

    F_k(y) = delta(t_{k+1}) + C_k(y - delta(t_k)).

In real arithmetic, the translations telescope, so the final map equals the
clean discrete map. For Euler this uses the grid carrier
(beta(t_{k+1})-beta(t_k))/(t_{k+1}-t_k), not simply alpha(t_k).
Clipping must occur in the translated coordinates.

`run_endpoint_loop_pilot.py` tests two implementations independently against a
clean run from the same seeds:

- `grid_conjugate` stores marked states and calls the shifted velocity field.
  Its finite-precision inversion of the translation can incur roundoff.
- `canonical_exact` retains the canonical state internally, calls the original
  clean step, and represents the marked state as canonical + delta. At the
  endpoint delta is zero, so identical arithmetic gives bitwise-identical output.
  This is a structured sampler, not an arbitrary black-box ODE integration.

`continuous_euler` measures the discretization error of ordinary Euler on the
continuous conjugated field (with clipping handled in translated coordinates).
`additive_sine` is the earlier uncompensated construction and a negative control.
The existing clean sampler clips intermediate values to [-5,5]; final [-1,1]
image clipping is evaluated separately and cannot hide endpoint errors.

## Detection Is A Separate Test

The pilot decodes the mean on-policy demodulated velocity using marked responses
only. The statistic is divided by its actual grid carrier energy. Clean signatures
are recorded only to diagnose clean bias and the ideal signal difference; they
are never subtracted before decoding. Independent trajectories are the replication
units, not all time samples within one trajectory.

For an arbitrary checkpoint the clean demodulated mean need not vanish. Exact
endpoint preservation does not prove reference-free detectability. A toy test
explicitly checks this counterexample. No finite-query Gaussian guarantee is
assumed by the pilot.

## Scope

Run `python run_endpoint_loop_pilot.py` for the real CIFAR UNet H32 test, 160
independent seeds and 25/50/100 steps in FP32 without TF32. Configurations, source
hashes, full endpoint tensors, signatures and image grids go to
`table_runs/endpoint_loop_pilot/`. This is an analytic wrapper experiment, not a
new trained checkpoint, security result, or demonstration of robustness to removal.

A standalone network trained by frozen-teacher regression only approximates the
conjugated field. To retain exactness by construction, keep the coordinate change
in the model/sampler architecture; that structure and its key may be removable
by a weights holder. These two deployment options must not be conflated.

## Standalone Training Test

`run_endpoint_loop_training.py` fits the actual frozen checkpoint target:

    input  = x + epsilon * beta(t) * q
    target = u(x,t) + epsilon * alpha(t) * q
    error  = student(input,t) - target
    loss   = mean(error^2) + weight * mean((P^T error)^2)

Both terms have the same zero-error optimum. There is no correlation reward
and no subsequent change of the shift based on an estimated amplitude.
Epsilon is prescribed in both the shift and velocity target. An independently
measured effective epsilon quantifies whether the student achieved that target;
it is a diagnostic, not a replacement for the training epsilon.

The matched additive arm removes only the input shift. The unmarked control
fits u at the same shifted inputs. All use the same initialization, queries,
optimizer and update counts. The default parameterization is the existing
rank-16 LoRA plus selected time/output weights, with zero weight decay and
dropout disabled for deterministic teacher regression. Half the states are
data interpolants and half are cached teacher rollouts. Held-out diagnostics
use CIFAR test images and separately generated teacher trajectories.

Evaluation uses the standalone learned field and ordinary Euler, with no
key-dependent transformation of the sampler. Decoding uses suspect velocities
only. The teacher is used to measure target error, effective amplitude and
paired endpoint error, not to decode. Default checkpoints are 100, 300 and
1,000 updates; raw outputs are in `table_runs/endpoint_loop_training/`.

If r(y,t) is the learned field minus the ideal conjugated field and L(t) bounds
the ideal field's spatial Lipschitz constant, a continuous-ODE endpoint bound is

    ||Y_student(1) - X_clean(1)||
        <= integral_0^1 exp(integral_t^1 L(s) ds) * ||r(Y_student(t),t)|| dt.

This explains why measured target MSE alone is insufficient: the residual must
be controlled on the student's trajectory, and unstable dynamics can amplify
it. Numerical integration adds its own error. Paired endpoint measurements
therefore remain necessary even when the target construction is exact.

## Actual Trained Results

The CIFAR H32 screen trained standalone networks, not analytic wrappers. All
rows below use 1,000 updates, prescribed epsilon 1.5, one saved key, 160 paired
seeds and the ordinary 100-step Euler sampler. All four marked networks decoded
correctly in all four on-policy groups (40 independent trajectories per group).
The unmarked LoRA control remained bitwise identical to the clean checkpoint
throughout its 1,000 updates.

| Setting | Target | Measured epsilon | Endpoint RMSE | Paired PSNR | Training seconds |
|---|---|---:|---:|---:|---:|
| LoRA, projected weight 10, lr 1e-4 | Additive | 1.371 | 0.09406 | 26.57 dB | 184.5 |
| LoRA, projected weight 10, lr 1e-4 | Loop | 1.377 | 0.11888 | 24.54 dB | 199.4 |
| Full, projected weight 0.1, lr 1e-5 | Additive | 1.070 | 0.02154 | 39.37 dB | 205.3 |
| Full, projected weight 0.1, lr 1e-5 | Loop | 1.148 | 0.02324 | 38.71 dB | 192.3 |

Additive versus loop is matched within each setting. The setting change includes
both lower projection weighting and full-parameter updates. In the implemented
mean-squared losses, projected error is averaged over 32 coordinates while full
error is averaged over 3,072 pixels/channels. A coefficient of 10 therefore
adds 960 times the ordinary per-coordinate penalty in the projected subspace;
0.1 adds 9.6 times that penalty. The latter setting fits the whole field more
closely while still recovering the message.

The tested full-parameter recipe is:

```bash
python run_endpoint_loop_training.py \
  --output table_runs/endpoint_loop_training_balanced_full \
  --parameterization full --arms additive loop \
  --checkpoints 100 300 1000 --projected-weight 0.1 --lr 0.00001
```

The loop learns a detectable watermark with substantially better endpoint
fidelity than the first LoRA setting. Its fitted field still has measurable
approximation error and does not outperform the balanced additive arm in paired
endpoint fidelity. The exact theorem remains a statement about the constructed
field. For comparison, the earlier ideal continuous-field Euler test had endpoint
RMSE 4.70e-5 at 100 steps, much smaller than the learned loop's 0.02324. Fitting
error, rather than the ideal field's numerical integration error, dominates here.
These paired metrics measure closeness to the teacher's samples; distributional
FID and robustness are separate measurements.

Epsilon in training remains 1.5, not the measured 1.148. The measured value is a
held-out regression of student responses minus teacher responses at the paired
shifted coordinates. It cannot retroactively replace epsilon in the conjugacy
identity. To change the intended strength, change epsilon in both the shift and
the velocity target and fit that new whole target. A projection-error weight can
affect finite-capacity fitting, but does not reward amplitudes beyond the target.

`audit_endpoint_loop_training.py` additionally evaluates saved final models with
the existing independent-query decoder, using 20 trials of 4,096 CIFAR test
interpolants per model. Its source, model and key provenance are recorded in
`table_runs/endpoint_loop_detection_audit/`. This decoder also uses suspect
velocities only, with no clean subtraction.

The independent-query audit is complete: all four final marked networks
achieved 20/20 exact recoveries and 100% bit accuracy. Thus the existing
independent-query decoder works for the learned loop in this one-key CIFAR
screen; on-policy sampling is not needed for its demonstrated recovery.
This also confirms that the full trained checkpoint can be loaded and decoded
without querying the clean teacher. The numerical endpoint errors above remain
nonzero despite perfect message recovery.

## Sample-Only Student Test

`run_endpoint_loop_distillation.py` tests both final shifted checkpoints against
the clean teacher. Each teacher supplies 2,000 completed FP32 images from the
same noise seeds and class conditions. Each same-architecture student starts
from the public clean checkpoint and fits ordinary flow matching for 1,000
updates, batch 64, AdamW lr 1e-4, weight decay 0.01, cosine decay. Training
receives images and labels only; a dummy teacher raises if a velocity is queried.
No marked weights, intermediate states or owner key enter student fitting.

Three paired student-training seeds share the fixed teacher datasets. Decoding
uses 20 independent 4,096-query trials on held-out CIFAR interpolants, without
clean subtraction. A paired clean-trained student measures background agreement
with the same message. Raw bit accuracy, gain over this control, consensus
correct bits, and signed score changes are retained. The 20 query trials do not
create 20 independent messages. FID uses 1,000 generated images and a matched
balanced real reference for all students and teachers.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python run_endpoint_loop_distillation.py
```

Full checkpoints, teacher datasets and progress are saved under
`table_runs/endpoint_loop_distillation/`; completed compact results are exported
under `results/pilots/endpoint_loop_distillation/`. The question is whether the
finite learned fields transfer recoverable message information through samples,
not whether their measured endpoint drift is exactly zero.

The test completed all nine student fits. Mean +/- sample SD across the three
student-training seeds, for the single saved H32 key:

| Teacher | Ordinary bit accuracy (%) | Gain over clean (pp) | Student FID (1,000 images) |
|---|---:|---:|---:|
| Clean | 44.64 +/- 0.33 | - | 101.68 |
| Shifted, balanced full training | 49.43 +/- 0.33 | 4.79 | 102.90 |
| Shifted, LoRA | 81.72 +/- 0.27 | 37.08 | 111.29 |

All marked-student query trials had zero exact-message recovery. The LoRA
prototype nevertheless transfers many correct bits through completed images.
The balanced full-training prototype stays just below 50% ordinary bit accuracy
in all three seeds, with image quality close to its clean-student control.
Its positive gain over the control and signed score changes motivate a separate
diagnostic of weak structured differences.

`tools/analyze_endpoint_loop_distillation.py` computes this post-hoc diagnostic
from the saved scores. Within each 4,096-query trial, it subtracts the matched
clean student's score vector before taking signs. Its mean bit accuracy is
90.63 +/- 0.00% for balanced full training and 93.44 +/- 1.28% for LoRA. Both
models are queried at the same per-model count as the ordinary decoder, but the
diagnostic additionally needs the clean-trained shadow student. A separately
reported consensus pools all 20 query trials; it is not mixed with the
per-trial results. Zero score differences return no contrast message.

This diagnostic uses matched teacher-generation noise, student-training
randomness and verification queries. It measures message-related differences
between the paired image/student experiments, not reference-free extraction
from an ordinary iid image API with unknown seeds. It shows that the balanced
prototype's near-chance raw decoding is not an exact per-seed endpoint-isolation
result. The ideal conjugacy theorem remains valid; the learned fields retain
nonzero, structured approximation errors.

```bash
python tools/analyze_endpoint_loop_distillation.py
```

Final per-student outcomes are in the study's `report.md` and `results.json`;
the separately labeled diagnostic and its source/results hashes are in
`analysis.json`. Training and diagnostic protocol checks pass in
`test_endpoint_loop_distillation.py`.
