# Standalone Frozen-Teacher Loop Training

State: complete

H32, one saved key. Decoder uses only suspect on-policy velocities. Teacher subtraction is used only for fitting diagnostics. All learned models use ordinary Euler with fixed clipping, without a keyed wrapper. Endpoint PSNR is paired against the original clean checkpoint, not FID.

| Arm | Updates | Effective epsilon | Target MSE | Exact % | Bits % | Endpoint RMSE | PSNR | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 100 | 0.0000 | 0 | 0.0 | 51.56 | 0 | inf | 19.4 |
| control | 300 | 0.0000 | 0 | 0.0 | 51.56 | 0 | inf | 55.1 |
| control | 1000 | 0.0000 | 0 | 0.0 | 51.56 | 0 | inf | 181.5 |
| additive | 100 | 0.7958 | 0.0138218 | 0.0 | 92.97 | 0.157737 | 22.084515128852207 | 18.3 |
| additive | 300 | 1.2642 | 0.0179583 | 100.0 | 100.00 | 0.142094 | 22.99030663475243 | 54.9 |
| additive | 1000 | 1.3706 | 0.012883 | 100.0 | 100.00 | 0.0940582 | 26.570165407619758 | 184.5 |
| loop | 100 | 0.9538 | 0.0160486 | 0.0 | 88.28 | 0.210198 | 19.593753007753484 | 20.0 |
| loop | 300 | 1.2560 | 0.0165611 | 50.0 | 98.44 | 0.175275 | 21.167166518902107 | 59.8 |
| loop | 1000 | 1.3771 | 0.0113777 | 100.0 | 100.00 | 0.118879 | 24.536777282718134 | 199.4 |
