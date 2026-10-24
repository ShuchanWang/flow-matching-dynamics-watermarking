# Sample-Only Transfer of Trained Translation Loops

State: complete

One H32 key; independent student training seeds, not independent keys. Each student starts from the public clean checkpoint. Training receives only 2,000 completed teacher images and their class labels under the default protocol. No teacher velocities, intermediate states, marked weights or watermark key enter student training. Clean and marked conditions share generation noise, training randomness and evaluation queries. Query repetitions do not turn the same 32 message bits into independent payload trials. FID uses a matched small-sample reference; its count is recorded in config.json.

| Teacher | Seed | Bit accuracy % | Exact % | FID | Bit gain over clean (pp) | Consensus correct / 32 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 61731 | 45.000 | 0.0 | 101.801 | - | - |
| loop_full | 61731 | 49.062 | 0.0 | 103.036 | 4.062 | 16 |
| loop_lora | 61731 | 81.562 | 0.0 | 111.892 | 36.562 | 26 |
| clean | 62731 | 44.531 | 0.0 | 102.382 | - | - |
| loop_full | 62731 | 49.531 | 0.0 | 103.572 | 5.000 | 14 |
| loop_lora | 62731 | 81.562 | 0.0 | 111.688 | 37.031 | 29 |
| clean | 63731 | 44.375 | 0.0 | 100.857 | - | - |
| loop_full | 63731 | 49.688 | 0.0 | 102.097 | 5.312 | 15 |
| loop_lora | 63731 | 82.031 | 0.0 | 110.279 | 37.656 | 27 |

Group standard deviations below are across student training seeds, not query repetitions.

| Teacher | Student seeds | Bit accuracy mean +/- SD (%) | Exact mean (%) | FID mean |
|---|---:|---:|---:|---:|
| clean | 3 | 44.635 +/- 0.325 | 0.0 | 101.680 |
| loop_full | 3 | 49.427 +/- 0.325 | 0.0 | 102.902 |
| loop_lora | 3 | 81.719 +/- 0.271 | 0.0 | 111.286 |
