# Endpoint-Preserving Translation Loop

State: complete

Analytic wrapper around the actual clean CIFAR checkpoint, not a trained watermark. Canonical sampling knows the private translation internally; this structural guarantee does not apply to a generic standalone learned approximation or an arbitrary sampler. Decoding uses marked velocities only. Clean subtraction is used only to diagnose the signal identity. Independent trajectories, not time steps, are replication units. All endpoint errors are measured before final image clipping as well as afterward. LPIPS is not required: exact tensor equality and endpoint errors are stronger checks of exact preservation. No claim of removal resistance is made.

| Steps | Construction | Bitwise equal | Max error | RMSE | PSNR | Exact % | Bits % |
|---:|---|---|---:|---:|---:|---:|---:|
| 25 | additive_sine | False | 0.0181 | 0.00225 | 59.00 | 100.0 | 100.0 |
| 25 | continuous_euler | False | 0.00137 | 0.000205 | 79.82 | 100.0 | 100.0 |
| 25 | grid_conjugate | False | 9.39e-07 | 8.05e-08 | 147.95 | 100.0 | 100.0 |
| 25 | canonical_exact | True | 0 | 0 | inf | 100.0 | 100.0 |
| 50 | additive_sine | False | 0.0193 | 0.00228 | 58.88 | 100.0 | 100.0 |
| 50 | continuous_euler | False | 0.000724 | 9.56e-05 | 86.46 | 100.0 | 100.0 |
| 50 | grid_conjugate | False | 7.53e-07 | 8.45e-08 | 147.56 | 100.0 | 100.0 |
| 50 | canonical_exact | True | 0 | 0 | inf | 100.0 | 100.0 |
| 100 | additive_sine | False | 0.0199 | 0.00231 | 58.77 | 100.0 | 100.0 |
| 100 | continuous_euler | False | 0.00038 | 4.7e-05 | 92.63 | 100.0 | 100.0 |
| 100 | grid_conjugate | False | 1.19e-06 | 1.03e-07 | 145.83 | 100.0 | 100.0 |
| 100 | canonical_exact | True | 0 | 0 | inf | 100.0 | 100.0 |
