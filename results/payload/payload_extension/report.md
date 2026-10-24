# Fixed-Dimension Payload and Query Study

K=128, epsilon=1.5, 1000 training steps, same rank-16 LoRA recipe. Five messages per payload, original saved keys and adapters, no clean-model compensation. 96-bit messages are preselected prefixes of the five existing 128-bit messages. Old run files are unchanged.

Mean +/- population standard deviation across messages. Bit metrics and full signed coordinate arrays are retained per trial. All query counts use prefixes of the same trial.

| Bits | Queries | Exact recovery (%) | Bit accuracy (%) | Mean minimum signed margin | Nonpositive bit means | FID ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1024 | 59.0000 +/- 15.5403 | 99.2031 +/- 0.3619 | 0.0031 +/- 0.0053 | 0.0000 +/- 0.0000 | 1.0522 +/- 0.0140 |
| 64 | 4096 | 93.5000 +/- 8.3066 | 99.8984 +/- 0.1298 | 0.0157 +/- 0.0047 | 0.0000 +/- 0.0000 | 1.0522 +/- 0.0140 |
| 64 | 16384 | 99.5000 +/- 1.0000 | 99.9922 +/- 0.0156 | 0.0210 +/- 0.0048 | 0.0000 +/- 0.0000 | 1.0522 +/- 0.0140 |
| 96 | 1024 | 3.0000 +/- 4.8477 | 96.5312 +/- 0.8027 | -0.0268 +/- 0.0061 | 1.4000 +/- 0.8000 | 1.0448 +/- 0.0143 |
| 96 | 4096 | 16.0000 +/- 22.9456 | 98.1823 +/- 0.8110 | -0.0153 +/- 0.0093 | 1.4000 +/- 0.8000 | 1.0448 +/- 0.0143 |
| 96 | 16384 | 18.0000 +/- 32.3805 | 98.4271 +/- 0.7768 | -0.0122 +/- 0.0095 | 1.4000 +/- 0.8000 | 1.0448 +/- 0.0143 |
| 128 | 1024 | 1.0000 +/- 2.0000 | 95.4922 +/- 0.8748 | -0.0399 +/- 0.0101 | 3.6000 +/- 2.0591 | 1.0444 +/- 0.0087 |
| 128 | 4096 | 1.0000 +/- 2.0000 | 97.0742 +/- 1.2722 | -0.0290 +/- 0.0130 | 3.2000 +/- 2.0396 | 1.0444 +/- 0.0087 |
| 128 | 16384 | 0.0000 +/- 0.0000 | 97.3477 +/- 1.3188 | -0.0262 +/- 0.0134 | 3.0000 +/- 1.6733 | 1.0444 +/- 0.0087 |
