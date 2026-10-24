# Watermark Security: Targeted Removal and Transfer

State: complete

Saved CIFAR-10 O5 and H32 teachers, original owner keys, no watermark retraining. The five messages per encoding are not independent secret-key replications. Attacks do not receive keys or target messages. Carrier removal estimates the ambient carrier-correlated velocity using marked-model queries only, with no clean-model subtraction. Estimation and verification use independent random queries from the same original 20,000-image interpolation distribution. Strengths are prespecified, not selected using owner decoding.

Students start from the same public clean backbone. Sample-only transfer fits flow matching to balanced completed teacher samples; velocity transfer fits teacher responses on real-data interpolations. These are distinct supervision interfaces, not matched input distributions. Full-backbone clean continuation starts from the exactly materialized marked teacher, so dropping an adapter cannot explain removal. All updates are full-parameter and key-free. Fidelity of velocity imitation is audited on independent queries.

Quality uses balanced class conditions and a balanced held-out CIFAR-10 reference, rather than the older fixed-class quality protocol. Every condition uses paired generation noise and verification queries. Ratios refer to newly measured clean and marked baselines. Per-trial decodes, features and images are saved. Student conclusions must be interpreted with their achieved generation quality and teacher-velocity error; a finite training schedule alone cannot establish that transfer is impossible.

| Scheme | Condition | Messages | Recovery (%) | FID / Clean | FID / Teacher |
|---|---|---:|---:|---:|---:|
| H32 | clean | 5 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.963 +/- 0.010 |
| H32 | full_finetune | 5 | 0.000 +/- 0.000 | 0.971 +/- 0.001 | 0.935 +/- 0.010 |
| H32 | marked | 5 | 100.000 +/- 0.000 | 1.039 +/- 0.011 | 1.000 +/- 0.000 |
| H32 | sample_only | 5 | 4.000 +/- 2.000 | 1.080 +/- 0.016 | 1.040 +/- 0.007 |
| H32 | subtract_0.5 | 5 | 100.000 +/- 0.000 | 1.037 +/- 0.009 | 0.998 +/- 0.002 |
| H32 | subtract_1 | 5 | 0.000 +/- 0.000 | 1.038 +/- 0.008 | 0.999 +/- 0.006 |
| H32 | subtract_1.5 | 5 | 0.000 +/- 0.000 | 1.038 +/- 0.008 | 0.999 +/- 0.006 |
| H32 | velocity | 5 | 100.000 +/- 0.000 | 1.039 +/- 0.011 | 1.000 +/- 0.001 |
| O5 | clean | 5 | 2.000 +/- 2.449 | 1.000 +/- 0.000 | 0.966 +/- 0.008 |
| O5 | full_finetune | 5 | 6.000 +/- 4.899 | 0.972 +/- 0.001 | 0.939 +/- 0.008 |
| O5 | marked | 5 | 100.000 +/- 0.000 | 1.035 +/- 0.009 | 1.000 +/- 0.000 |
| O5 | sample_only | 5 | 100.000 +/- 0.000 | 1.075 +/- 0.007 | 1.039 +/- 0.006 |
| O5 | subtract_0.5 | 5 | 100.000 +/- 0.000 | 1.032 +/- 0.007 | 0.997 +/- 0.003 |
| O5 | subtract_1 | 5 | 2.000 +/- 2.449 | 1.031 +/- 0.008 | 0.997 +/- 0.008 |
| O5 | subtract_1.5 | 5 | 2.000 +/- 2.449 | 1.031 +/- 0.008 | 0.997 +/- 0.008 |
| O5 | velocity | 5 | 100.000 +/- 0.000 | 1.035 +/- 0.008 | 1.000 +/- 0.001 |
