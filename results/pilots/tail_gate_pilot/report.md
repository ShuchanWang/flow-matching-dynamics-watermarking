# Tail-Gate Pilot

State: complete

One private H32 key; 300-step mixed ordinary/trigger LoRA training by default. Controls receive identical queries and losses with zero watermark target. Owner decoding uses only marked velocities and normalized trigger demodulation. Clean-model subtraction is used only for analysis of signal strength/leakage. Each attack fits independently to marked responses without the owner key. Gradient probing requires checkpoint/input-gradient access; random probing uses velocities only. OOD search scores can reflect ordinary model behavior: matched clean-control probes are included. The 100-image paired FID is a pilot screen. Repeated trials are not independent keys.

| Gate | Model | Condition | Exact % | Bits % | FID/clean | FID/control | Leakage ratio |
|---|---|---|---:|---:|---:|---:|---:|
| tau2.5_slope2 | control | baseline | 0.0 | 54.7 | 0.995 | n/a | 1.0819 |
| tau2.5_slope2 | clean | baseline | 0.0 | 54.7 | 1.000 | 1.005 | 0.0000 |
| tau2.5_slope2 | injected | baseline | 100.0 | 100.0 | 1.000 | 1.005 | 0.0386 |
| tau2.5_slope2 | injected | ordinary_regression | 100.0 | 100.0 | 0.998 | 1.003 | 0.0073 |
| tau2.5_slope2 | injected | ood_random_regression | 0.0 | 25.0 | 1.000 | 1.005 | 9.7911 |
| tau2.5_slope2 | injected | ood_gradient_regression | 0.0 | 14.1 | 0.999 | 1.004 | 7.6571 |
| tau2.5_slope2 | trained | baseline | 25.0 | 97.7 | 1.015 | 1.021 | 0.9569 |
| tau2.5_slope2 | trained | ordinary_regression | 0.0 | 52.3 | 1.013 | 1.018 | 0.8123 |
| tau2.5_slope2 | trained | ood_random_regression | 50.0 | 97.7 | 1.014 | 1.020 | 0.9570 |
| tau2.5_slope2 | trained | ood_gradient_regression | 50.0 | 98.4 | 1.017 | 1.022 | 0.9562 |
| tau2.5_slope4 | control | baseline | 0.0 | 54.7 | 0.995 | n/a | 1.0819 |
| tau2.5_slope4 | clean | baseline | 0.0 | 54.7 | 1.000 | 1.005 | 0.0000 |
| tau2.5_slope4 | injected | baseline | 100.0 | 100.0 | 1.000 | 1.006 | 0.0140 |
| tau2.5_slope4 | injected | ordinary_regression | 100.0 | 100.0 | 0.998 | 1.004 | 0.0065 |
| tau2.5_slope4 | injected | ood_random_regression | 0.0 | 72.7 | 0.999 | 1.004 | 15.7046 |
| tau2.5_slope4 | injected | ood_gradient_regression | 0.0 | 36.7 | 0.999 | 1.005 | 18.1274 |
| tau2.5_slope4 | trained | baseline | 25.0 | 97.7 | 1.019 | 1.025 | 0.9573 |
| tau2.5_slope4 | trained | ordinary_regression | 0.0 | 53.1 | 1.018 | 1.023 | 0.6606 |
| tau2.5_slope4 | trained | ood_random_regression | 25.0 | 97.7 | 1.019 | 1.024 | 0.9571 |
| tau2.5_slope4 | trained | ood_gradient_regression | 25.0 | 97.7 | 1.020 | 1.025 | 0.9589 |
| tau4_slope2 | control | baseline | 0.0 | 53.9 | 0.995 | n/a | 1.1418 |
| tau4_slope2 | clean | baseline | 0.0 | 53.9 | 1.000 | 1.005 | 0.0000 |
| tau4_slope2 | injected | baseline | 100.0 | 100.0 | 1.000 | 1.005 | 0.0030 |
| tau4_slope2 | injected | ordinary_regression | 100.0 | 100.0 | 0.998 | 1.003 | 0.0069 |
| tau4_slope2 | injected | ood_random_regression | 0.0 | 47.7 | 1.000 | 1.005 | 44.9064 |
| tau4_slope2 | injected | ood_gradient_regression | 0.0 | 25.8 | 1.000 | 1.005 | 7.9141 |
| tau4_slope2 | trained | baseline | 0.0 | 94.5 | 1.016 | 1.021 | 0.9154 |
| tau4_slope2 | trained | ordinary_regression | 0.0 | 53.1 | 1.011 | 1.016 | 0.3005 |
| tau4_slope2 | trained | ood_random_regression | 0.0 | 95.3 | 1.016 | 1.021 | 0.9181 |
| tau4_slope2 | trained | ood_gradient_regression | 50.0 | 96.9 | 1.016 | 1.022 | 0.9168 |
| tau4_slope4 | control | baseline | 0.0 | 53.9 | 0.995 | n/a | 1.1418 |
| tau4_slope4 | clean | baseline | 0.0 | 53.9 | 1.000 | 1.005 | 0.0000 |
| tau4_slope4 | injected | baseline | 100.0 | 100.0 | 1.000 | 1.005 | 0.0003 |
| tau4_slope4 | injected | ordinary_regression | 100.0 | 100.0 | 0.998 | 1.003 | 0.0062 |
| tau4_slope4 | injected | ood_random_regression | 0.0 | 68.0 | 0.999 | 1.005 | 14.2010 |
| tau4_slope4 | injected | ood_gradient_regression | 0.0 | 40.6 | 0.999 | 1.004 | 18.5294 |
| tau4_slope4 | trained | baseline | 0.0 | 95.3 | 1.020 | 1.026 | 0.9189 |
| tau4_slope4 | trained | ordinary_regression | 0.0 | 53.9 | 1.016 | 1.022 | 0.2600 |
| tau4_slope4 | trained | ood_random_regression | 0.0 | 95.3 | 1.020 | 1.025 | 0.9193 |
| tau4_slope4 | trained | ood_gradient_regression | 50.0 | 97.7 | 1.022 | 1.027 | 0.9199 |
