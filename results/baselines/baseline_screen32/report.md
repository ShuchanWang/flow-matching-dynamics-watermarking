# Saved 32-Bit Modification and Transfer Screen

State: complete

One development key per method, selected for perfect original recovery. Fresh-query baselines are retained regardless of their recovery. No five-bit or five-key confirmation is running. Quality and transfer must be interpreted jointly; weight extraction is N/A on the smaller architecture.

| Method | Condition | Exact % | Bits % | FID/modified control | FID/original control | Parameter ratio |
|---|---|---:|---:|---:|---:|---:|
| dynamics | baseline | 100.0 | 100.0 | 1.0547 | 1.0547 | 1.000 |
| weight | baseline | 100.0 | 100.0 | 1.0025 | 1.0025 | 1.000 |
| trigger | baseline | 100.0 | 100.0 | 1.0057 | 1.0057 | 1.000 |
| naive | baseline | 100.0 | 100.0 | 1.0515 | 1.0515 | 1.000 |
| dynamics | prune25 | 100.0 | 100.0 | 1.0421 | 1.0484 | 1.000 |
| weight | prune25 | 100.0 | 100.0 | 0.9928 | 0.9874 | 1.000 |
| trigger | prune25 | 100.0 | 100.0 | 1.0062 | 0.9994 | 1.000 |
| naive | prune25 | 100.0 | 100.0 | 1.0488 | 1.0461 | 1.000 |
| dynamics | prune50 | 0.0 | 70.6 | 1.0096 | 2.8446 | 1.000 |
| weight | prune50 | 100.0 | 100.0 | 0.9837 | 2.9246 | 1.000 |
| trigger | prune50 | 0.0 | 65.6 | 0.8533 | 2.5485 | 1.000 |
| naive | prune50 | 0.0 | 59.7 | 0.8343 | 2.4669 | 1.000 |
| dynamics | quant8 | 100.0 | 100.0 | 1.0529 | 1.0577 | 1.000 |
| weight | quant8 | 100.0 | 100.0 | 0.9923 | 1.0007 | 1.000 |
| trigger | quant8 | 100.0 | 100.0 | 0.9977 | 1.0020 | 1.000 |
| naive | quant8 | 100.0 | 100.0 | 1.0471 | 1.0505 | 1.000 |
| dynamics | quant4 | 100.0 | 100.0 | 1.0320 | 1.4312 | 1.000 |
| weight | quant4 | 100.0 | 100.0 | 0.9864 | 1.3753 | 1.000 |
| trigger | quant4 | 100.0 | 100.0 | 0.9572 | 1.2934 | 1.000 |
| naive | quant4 | 100.0 | 100.0 | 0.9963 | 1.3427 | 1.000 |
| dynamics | quant2 | 0.0 | 49.1 | 1.0110 | 6.9165 | 1.000 |
| weight | quant2 | 0.0 | 78.1 | 0.9936 | 6.9771 | 1.000 |
| trigger | quant2 | 0.0 | 40.6 | 0.9786 | 6.9704 | 1.000 |
| naive | quant2 | 0.0 | 49.2 | 1.0553 | 7.2598 | 1.000 |
| dynamics | ft_low | 35.0 | 96.6 | 1.0307 | 1.0021 | 1.000 |
| weight | ft_low | 100.0 | 100.0 | 1.0018 | 0.9619 | 1.000 |
| trigger | ft_low | 100.0 | 100.0 | 1.0078 | 0.9677 | 1.000 |
| naive | ft_low | 0.0 | 65.9 | 1.0360 | 1.0121 | 1.000 |
| dynamics | ft_high | 0.0 | 72.7 | 1.0141 | 0.9694 | 1.000 |
| weight | ft_high | 100.0 | 100.0 | 1.0002 | 0.9445 | 1.000 |
| trigger | ft_high | 0.0 | 56.2 | 1.0078 | 0.9511 | 1.000 |
| naive | ft_high | 0.0 | 46.6 | 1.0173 | 0.9761 | 1.000 |
| dynamics | same_velocity | 100.0 | 100.0 | 1.0532 | 1.0569 | 1.000 |
| weight | same_velocity | 100.0 | 100.0 | 1.0023 | 1.0111 | 1.000 |
| trigger | same_velocity | 0.0 | 58.9 | 1.0052 | 1.0136 | 1.000 |
| naive | same_velocity | 100.0 | 100.0 | 1.0490 | 1.0524 | 1.000 |
| dynamics | same_sample | 0.0 | 80.6 | 1.0404 | 1.2085 | 1.000 |
| weight | same_sample | 0.0 | 62.5 | 1.0028 | 1.1680 | 1.000 |
| trigger | same_sample | 0.0 | 56.2 | 1.0109 | 1.1769 | 1.000 |
| naive | same_sample | 100.0 | 100.0 | 1.0305 | 1.1979 | 1.000 |
| dynamics | small_velocity | 100.0 | 100.0 | 1.0056 | 3.9095 | 0.251 |
| weight | small_velocity | N/A | N/A | 1.0013 | 4.0437 | 0.251 |
| trigger | small_velocity | 0.0 | 43.8 | 1.0017 | 4.0446 | 0.251 |
| naive | small_velocity | 100.0 | 100.0 | 1.0117 | 3.9541 | 0.251 |
| dynamics | small_sample | 0.0 | 68.6 | 1.0062 | 4.2525 | 0.251 |
| weight | small_sample | N/A | N/A | 1.0006 | 4.6320 | 0.251 |
| trigger | small_sample | 0.0 | 46.9 | 1.0082 | 4.6646 | 0.251 |
| naive | small_sample | 0.0 | 92.5 | 1.0107 | 4.3356 | 0.251 |
