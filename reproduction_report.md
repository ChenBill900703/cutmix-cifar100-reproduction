# CutMix CIFAR-100 reproduction report

## Scope and protocol

This project reproduces only CIFAR-100 classification with bottleneck PyramidNet-200 (alpha=240), comparing the baseline and CutMix. ImageNet, detection, captioning, and full OOD experiments are intentionally excluded. Both methods use the same optimizer, schedule, augmentation, effective batch size, and evaluation code; only CutMix differs.

The paper's Table 5 reports the mean of each run's best checkpoint across three runs: baseline 16.45% Top-1 error and CutMix 14.47%. The repository's 14.23% number is a single published checkpoint/repository result, not the paper's three-run mean.

## Results

| run | method | seed | status | best Top-1 err | best Top-5 err | best epoch | last Top-1 err | last Top-5 err | hours | peak reserved GiB | micro-batch | accumulation | Top-1 gap (pp) | Top-5 gap (pp) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pyramidnet200_baseline_seed0 | baseline | 0 | complete | 16.73 | 3.5 | 295 | 16.91 | 3.6 | 13.063 | 5.334 | 64 | 1 | 0.28 | -0.19 |
| pyramidnet200_baseline_seed1 | baseline | 1 | complete | 16.83 | 3.7 | 282 | 17.05 | 3.59 | 12.797 | 5.314 | 64 | 1 | 0.38 | 0.01 |
| pyramidnet200_baseline_seed2 | baseline | 2 | complete | 16.66 | 3.66 | 298 | 16.96 | 3.65 | 12.942 | 5.33 | 64 | 1 | 0.21 | -0.03 |
| pyramidnet200_cutmix_seed0 | cutmix | 0 | complete | 14.72 | 3.02 | 244 | 15.49 | 3.39 | 13.074 | 5.334 | 64 | 1 | 0.25 | 0.05 |
| pyramidnet200_cutmix_seed1 | cutmix | 1 | complete | 15.13 | 3.47 | 268 | 15.57 | 3.48 | 12.934 | 5.305 | 64 | 1 | 0.66 | 0.5 |
| pyramidnet200_cutmix_seed2 | cutmix | 2 | complete | 14.8 | 3.11 | 239 | 15.82 | 3.38 | 13.098 | 5.33 | 64 | 1 | 0.33 | 0.14 |

- Baseline best Top-1 mean ± sample SD: 16.740 ± 0.085% (n=3)
- Baseline best Top-5 mean ± sample SD: 3.620 ± 0.106% (n=3)
- CutMix best Top-1 mean ± sample SD: 14.883 ± 0.217% (n=3)
- CutMix best Top-5 mean ± sample SD: 3.200 ± 0.238% (n=3)
- Absolute CutMix improvement: 1.857 percentage points
- Total training time across all runs: 77.908 hours

## Validation evidence

- Unit tests: shape, lambda bounds, corrected area ratio, label permutation, integer targets, model output/parameter count, and RNG round-trip.
- VRAM preflight: 64 (accumulation 1), validation batch 64; peak reserved 5.309 GiB versus safety limit 7.200 GiB.
- Two-epoch CIFAR-100 smoke/resume test: resume completed through epoch 2; train loss 4.6273 -> 4.5470.
- Timing gate: 337.3 seconds for the first full epoch; linear estimate for two 300-epoch runs: 56.2 hours. The phase-1 run proceeds only because this is under the six-day budget.

## Interpretation

No incomplete or single-seed result is presented as a three-seed mean. The selected micro-batch was 64 with accumulation 1, so this reproduction has no BatchNorm-statistics difference caused by reducing the micro-batch locally. Had preflight required 32, 16, or 8, BatchNorm would have observed that smaller micro-batch even though accumulation preserved effective batch 64, potentially changing running statistics and final accuracy. Results should be compared by both the absolute paper gap and the baseline-to-CutMix improvement under this common local protocol.

## Artifacts

- `outputs/experiments/comparison.csv`
- `outputs/experiments/error_curves.svg`
- `outputs/experiments/loss_curves.svg`
- Per-run `metrics.csv`, `summary.json`, `metadata.json`, `last.pt`, and `best.pt`
