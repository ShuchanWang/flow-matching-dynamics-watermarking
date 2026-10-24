# Mismatched-Carrier Removal

State: complete

The marked models use sine. The attacker regresses onto cosine and subtracts that cosine component at strength one, without receiving the owner key, message or clean-model responses. Owner verification remains unchanged. This is a fixed wrong-carrier control, not an adaptive unknown-carrier search. All query seeds, counts, generated-image seeds and FID references match the original carrier-removal study.

| Code | Messages | Exact recovery (%) | Bit accuracy (%) | FID / marked teacher |
|---|---:|---:|---:|---:|
| H32 | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.000 +/- 0.000 |
| O5 | 5 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 1.000 +/- 0.000 |
