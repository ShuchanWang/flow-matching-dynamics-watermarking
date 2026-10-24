# Embedding Strategy and Whole-Model Robustness

State: complete

Both arms start from the same public clean CIFAR-10 UNet. Original O5/H32 keys and messages are reused. LoRA updates the original adapters plus time/output-selected layers; full embedding updates all parameters. Training draws, objectives, learning rates, batch size, and update counts are paired. This compares embedding into a pretrained backbone, not training a new generator from scratch. The matched training stream samples with replacement from the original 20,000 training images.

LoRA is merged and equivalence checked before attacks. Compression affects all floating-point parameters, including bias and normalization. Pruning thresholds include ties; actual zero counts are saved. Quantization is simulated per-tensor symmetric rounding/dequantization, not integer inference. Each condition independently reloads its unchanged marked baseline. Full fine-tuning updates all parameters using clean flow matching. Sine/cosine regression attacks use marked-model responses only.

Baseline exact recovery and FID/clean must be inspected before comparing attack survival. A common training budget is not evidence of matched initial performance. No model or message is discarded for poor recovery or quality. The five messages reuse the saved projection seeds, not five independent key replications. FID/marked uses each arm's own unmodified marked baseline. Verification and generation noise are paired across arms and attacks; quality uses balanced classes and a balanced test reference. Existing manuscript results are not overwritten.

| Code | Embedding | Condition | Messages | Exact (%) | Bit (%) | FID/clean | FID/marked |
|---|---|---|---:|---:|---:|---:|---:|
| H32 | full | full_ft_lr0.0001_1000 | 5 | 0.000 +/- 0.000 | 72.188 +/- 1.515 | 0.967 +/- 0.002 | 0.971 +/- 0.004 |
| H32 | full | full_ft_lr1e-05_1000 | 5 | 19.000 +/- 8.000 | 95.719 +/- 0.234 | 0.987 +/- 0.001 | 0.991 +/- 0.004 |
| H32 | full | marked | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 0.997 +/- 0.004 | 1.000 +/- 0.000 |
| H32 | full | prune_0.25 | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 0.992 +/- 0.003 | 0.996 +/- 0.003 |
| H32 | full | prune_0.5 | 5 | 0.000 +/- 0.000 | 64.656 +/- 8.440 | 1.763 +/- 0.014 | 1.769 +/- 0.015 |
| H32 | full | quant_2bit | 5 | 0.000 +/- 0.000 | 50.219 +/- 7.760 | 3.758 +/- 0.015 | 3.771 +/- 0.023 |
| H32 | full | quant_4bit | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.149 +/- 0.022 | 1.153 +/- 0.022 |
| H32 | full | quant_8bit | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 0.998 +/- 0.004 | 1.001 +/- 0.001 |
| H32 | full | subtract_cosine | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 0.997 +/- 0.004 | 1.000 +/- 0.000 |
| H32 | full | subtract_sine | 5 | 0.000 +/- 0.000 | 47.844 +/- 2.186 | 0.998 +/- 0.002 | 1.002 +/- 0.003 |
| H32 | lora | full_ft_lr0.0001_1000 | 5 | 0.000 +/- 0.000 | 51.625 +/- 1.775 | 0.971 +/- 0.001 | 0.936 +/- 0.008 |
| H32 | lora | full_ft_lr1e-05_1000 | 5 | 0.000 +/- 0.000 | 65.000 +/- 3.288 | 0.988 +/- 0.002 | 0.952 +/- 0.007 |
| H32 | lora | marked | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.038 +/- 0.009 | 1.000 +/- 0.000 |
| H32 | lora | prune_0.25 | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.043 +/- 0.008 | 1.005 +/- 0.005 |
| H32 | lora | prune_0.5 | 5 | 0.000 +/- 0.000 | 67.531 +/- 7.416 | 1.980 +/- 0.048 | 1.907 +/- 0.040 |
| H32 | lora | quant_2bit | 5 | 0.000 +/- 0.000 | 50.125 +/- 7.999 | 3.707 +/- 0.050 | 3.571 +/- 0.062 |
| H32 | lora | quant_4bit | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.172 +/- 0.035 | 1.129 +/- 0.032 |
| H32 | lora | quant_8bit | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.038 +/- 0.010 | 1.000 +/- 0.001 |
| H32 | lora | subtract_cosine | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.038 +/- 0.009 | 1.000 +/- 0.000 |
| H32 | lora | subtract_sine | 5 | 0.000 +/- 0.000 | 47.250 +/- 2.029 | 1.036 +/- 0.009 | 0.998 +/- 0.008 |
| O5 | full | full_ft_lr0.0001_1000 | 5 | 79.000 +/- 15.620 | 89.200 +/- 9.432 | 0.969 +/- 0.001 | 0.970 +/- 0.003 |
| O5 | full | full_ft_lr1e-05_1000 | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 0.985 +/- 0.002 | 0.987 +/- 0.004 |
| O5 | full | marked | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 0.998 +/- 0.004 | 1.000 +/- 0.000 |
| O5 | full | prune_0.25 | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 0.998 +/- 0.004 | 1.000 +/- 0.002 |
| O5 | full | prune_0.5 | 5 | 14.000 +/- 15.297 | 60.600 +/- 10.670 | 1.737 +/- 0.030 | 1.740 +/- 0.032 |
| O5 | full | quant_2bit | 5 | 1.000 +/- 2.000 | 46.800 +/- 4.490 | 3.730 +/- 0.033 | 3.735 +/- 0.029 |
| O5 | full | quant_4bit | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.147 +/- 0.018 | 1.149 +/- 0.022 |
| O5 | full | quant_8bit | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 0.999 +/- 0.004 | 1.001 +/- 0.001 |
| O5 | full | subtract_cosine | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 0.998 +/- 0.004 | 1.000 +/- 0.000 |
| O5 | full | subtract_sine | 5 | 2.000 +/- 2.449 | 49.200 +/- 4.445 | 0.999 +/- 0.004 | 1.000 +/- 0.003 |
| O5 | lora | full_ft_lr0.0001_1000 | 5 | 5.000 +/- 5.477 | 49.800 +/- 7.859 | 0.972 +/- 0.002 | 0.940 +/- 0.005 |
| O5 | lora | full_ft_lr1e-05_1000 | 5 | 50.000 +/- 16.125 | 74.800 +/- 7.414 | 0.988 +/- 0.002 | 0.955 +/- 0.004 |
| O5 | lora | marked | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.035 +/- 0.006 | 1.000 +/- 0.000 |
| O5 | lora | prune_0.25 | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.040 +/- 0.008 | 1.005 +/- 0.006 |
| O5 | lora | prune_0.5 | 5 | 42.000 +/- 45.453 | 70.200 +/- 27.029 | 1.935 +/- 0.129 | 1.870 +/- 0.126 |
| O5 | lora | quant_2bit | 5 | 0.000 +/- 0.000 | 48.800 +/- 4.308 | 3.670 +/- 0.022 | 3.547 +/- 0.026 |
| O5 | lora | quant_4bit | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.154 +/- 0.031 | 1.115 +/- 0.027 |
| O5 | lora | quant_8bit | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.035 +/- 0.007 | 1.000 +/- 0.002 |
| O5 | lora | subtract_cosine | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.035 +/- 0.006 | 1.000 +/- 0.000 |
| O5 | lora | subtract_sine | 5 | 2.000 +/- 2.449 | 50.000 +/- 5.657 | 1.034 +/- 0.007 | 0.999 +/- 0.005 |
