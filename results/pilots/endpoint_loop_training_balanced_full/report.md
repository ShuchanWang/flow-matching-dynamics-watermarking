# Standalone Frozen-Teacher Loop Training

State: complete

H32, one saved key. Decoder uses only suspect on-policy velocities. Teacher subtraction is used only for fitting diagnostics. All learned models use ordinary Euler with fixed clipping, without a keyed wrapper. Endpoint PSNR is paired against the original clean checkpoint, not FID.

| Arm | Updates | Effective epsilon | Target MSE | Exact % | Bits % | Endpoint RMSE | PSNR | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| additive | 100 | 0.3117 | 0.000791049 | 25.0 | 85.94 | 0.0193391 | 40.306212720867485 | 22.3 |
| additive | 300 | 0.5622 | 0.00103417 | 50.0 | 97.66 | 0.0236057 | 38.5764220030442 | 63.0 |
| additive | 1000 | 1.0699 | 0.000870276 | 100.0 | 100.00 | 0.0215377 | 39.37056960207785 | 205.3 |
| loop | 100 | 0.5220 | 0.000826312 | 25.0 | 88.28 | 0.0255533 | 37.88416527103033 | 20.3 |
| loop | 300 | 0.7544 | 0.0010615 | 75.0 | 99.22 | 0.0287478 | 36.86273101578322 | 60.3 |
| loop | 1000 | 1.1475 | 0.000877799 | 100.0 | 100.00 | 0.0232389 | 38.710300962234186 | 192.3 |
