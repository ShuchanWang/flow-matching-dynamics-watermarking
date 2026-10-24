# Saved 32-Bit Baseline Screen

User amendment: stop five-bit development and do not launch five-key confirmation.
Use only saved 32-bit development models that decoded perfectly on the original
verification trials. Select the lowest FID/control candidate per method among
those models. Preserve all rejected trials; this is conditional one-key screening,
not independent confirmation or evidence for a population robustness claim.

Selected strengths: dynamics epsilon 1.5, weight regularization .1, trigger
regularization .1, naive epsilon 3. Initial FID/control differs across methods;
the dynamics and naive selections exceed the original 1.05 quality target.
Reverify selected baselines on fresh detection queries without filtering them
again if fresh verification is imperfect. No retraining of these baselines.

Compare whole-model pruning .25/.5, simulated quantization 8/4/2 bits, and full
clean fine-tuning 1,000 updates at lr 1e-5/1e-4. Distill using velocity or samples
into the clean-pretrained same architecture and into a randomly initialized
32-channel UNet with two residual blocks (same 32x32 state and time interface).
Use AdamW, 1,000 updates, batch 64, lr 1e-4 for distillation; sample-only training
uses 2,000 class-balanced teacher-generated images. Distillation receives no key.
Train matched clean-teacher controls with paired inputs and student initialization.
Report smaller-student parameter ratios and utility: failure to learn useful
generation must not be interpreted as selective watermark removal.

Each condition independently starts from the saved marked/clean-control pair.
Verification uses 20 trials and 4,096 velocity queries per trial, or 20 image
triggers, with detection seeds distinct from development. Weight extraction is
deterministic and N/A for the smaller architecture. Quality uses the original
5,000-image/100-step generation and KID protocol and applies the same modification
to the clean control. Also retain the original unmodified-control FID denominator
so modification-induced quality collapse is visible even if both models collapse.

The screen contains 12 conditions per method, including baseline: 48 cells.
No actual int8 backend or full five-key robustness claim is part of this screen.
Smoke tests use separate output folders and reduced counts. Source/config hashes,
selection, full per-trial outcomes, costs and sample features are retained.
