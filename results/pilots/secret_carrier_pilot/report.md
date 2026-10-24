# Secret Carrier Pilot

State: complete

One H32 message/key on the CIFAR UNet. Injected = exact analytic perturbation; trained = new short LoRA embedding. These are different controls, not pooled results. FID uses a small paired pilot sample, not a publication-quality estimate. Input features are centered globally on independent calibration queries; zero mean at each time and density preservation are not assumed. Attacks use only marked responses, not owner keys or clean responses. All attack choices are reported, with no owner-guided selection.

| Field | Carrier | Attack | Exact (%) | Bits (%) | FID/clean | Velocity relative MSE |
|---|---|---|---:|---:|---:|---:|
| clean | sine | none | 0.0 | 50.8 | 1.000 | 0.000000 |
| injected | sine | marked | 100.0 | 100.0 | 1.001 | 0.000348 |
| injected | sine | joint_spectrum | 0.0 | 47.7 | 0.998 | 0.002012 |
| injected | sine | time_svd_rank1 | 0.0 | 57.0 | 0.999 | 0.000178 |
| injected | sine | time_svd_rank4 | 0.0 | 53.1 | 0.997 | 0.000713 |
| injected | sine | input_ridge | 0.0 | 47.7 | 1.027 | 0.067709 |
| trained | sine | marked | 75.0 | 99.2 | 1.032 | 0.024569 |
| trained | sine | joint_spectrum | 0.0 | 46.1 | 1.032 | 0.012018 |
| trained | sine | time_svd_rank1 | 0.0 | 53.9 | 1.033 | 0.011039 |
| trained | sine | time_svd_rank4 | 0.0 | 50.0 | 1.031 | 0.010697 |
| trained | sine | input_ridge | 0.0 | 47.7 | 1.051 | 0.077584 |
| clean | multitone | none | 0.0 | 53.9 | 1.000 | 0.000000 |
| injected | multitone | marked | 100.0 | 100.0 | 1.000 | 0.000350 |
| injected | multitone | joint_spectrum | 0.0 | 45.3 | 0.998 | 0.002012 |
| injected | multitone | time_svd_rank1 | 25.0 | 96.9 | 1.000 | 0.000187 |
| injected | multitone | time_svd_rank4 | 0.0 | 96.1 | 0.997 | 0.000759 |
| injected | multitone | input_ridge | 0.0 | 43.8 | 1.027 | 0.067709 |
| trained | multitone | marked | 0.0 | 86.7 | 1.039 | 0.007992 |
| trained | multitone | joint_spectrum | 0.0 | 43.8 | 1.040 | 0.006520 |
| trained | multitone | time_svd_rank1 | 0.0 | 59.4 | 1.039 | 0.005707 |
| trained | multitone | time_svd_rank4 | 0.0 | 49.2 | 1.040 | 0.005193 |
| trained | multitone | input_ridge | 0.0 | 42.2 | 1.063 | 0.072199 |
| clean | input | none | 0.0 | 53.9 | 1.000 | 0.000000 |
| injected | input | marked | 100.0 | 100.0 | 1.000 | 0.000342 |
| injected | input | joint_spectrum | 100.0 | 100.0 | 0.998 | 0.002355 |
| injected | input | time_svd_rank1 | 100.0 | 100.0 | 0.998 | 0.000604 |
| injected | input | time_svd_rank4 | 100.0 | 100.0 | 0.997 | 0.001059 |
| injected | input | input_ridge | 100.0 | 100.0 | 1.027 | 0.068015 |
| trained | input | marked | 0.0 | 61.7 | 0.999 | 0.002107 |
| trained | input | joint_spectrum | 0.0 | 60.9 | 0.997 | 0.004055 |
| trained | input | time_svd_rank1 | 0.0 | 61.7 | 0.997 | 0.002226 |
| trained | input | time_svd_rank4 | 0.0 | 61.7 | 0.997 | 0.002751 |
| trained | input | input_ridge | 0.0 | 57.0 | 1.029 | 0.069513 |
