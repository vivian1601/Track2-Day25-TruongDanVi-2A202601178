# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cleanest region: europe-north1

## Findings and prioritized actions

1. **Move stable demand to reserved capacity and checkpoint interruptible jobs on spot.** This is the largest measured lever at $10,040/month; validate utilization before making a long commitment.
2. **Apply inference routing, prompt caching, and batching.** This saves $1,212/month and is a fast, low-risk software change that does not require a capacity commitment.
3. **Profile and right-size the GPU-Util lies.** A high GPU-Util value only means the GPU clock was active; memory stalls, weak tensor-core occupancy, and kernel-launch overhead can keep useful FLOPs low. `gpu-h100-4` therefore bills as a full H100 while delivering about 20% MFU.
4. **Auto-stop idle instances.** This is a low-risk guardrail and removes $600/month from the current sample.

## Extension: reasoning budget

Reasoning is 8.4% of requests but consumes 16.5% of optimized inference cost and 94.0% of estimated energy. Enforce a 5% traffic budget: reserve reasoning for evaluation or high-complexity tasks; route excess requests to the small model after a cheap complexity check.
On this dataset that reroutes 81 requests/day and is projected to save $1.27/day plus 14,417.2 Wh/day.

| Traffic | Requests | Tokens | $/1M tokens | Wh/1M tokens |
|---|---:|---:|---:|---:|
| Reasoning | 201 | 1,241,156 | $1.125 | 24,000.0 |
| Standard | 2,199 | 6,291,871 | $1.127 | 300.0 |

At a 5% cap, the 30-day projection is $37.99 and 432.5 kWh saved.

## Extension: carbon-aware scheduling

The interruptible pool uses 1,789.0 kWh per workload cycle. Moving it from us-east-1 to europe-north1 saves 626.15 kgCO2e (92.1%) and reduces estimated electricity cost by $53.67.

| Region | $/kWh | gCO2/kWh | Electricity cost | Carbon (kg) |
|---|---:|---:|---:|---:|
| us-east-1 | 0.120 | 380 | $214.68 | 679.82 |
| us-west-2 | 0.070 | 120 | $125.23 | 214.68 |
| europe-north1 | 0.090 | 30 | $161.01 | 53.67 |
| europe-central2 | 0.180 | 660 | $322.02 | 1,180.74 |
| us-east-wa | 0.055 | 90 | $98.39 | 161.01 |

| Interruptible job | GPU | Energy (kWh) | Source carbon (kg) | Cleanest carbon (kg) | Saved (kg) |
|---|---|---:|---:|---:|---:|
| job-train-llm | H100 | 1,568.0 | 595.84 | 47.04 | 548.80 |
| job-train-embed | A100 | 80.0 | 30.40 | 2.40 | 28.00 |
| job-finetune | H100 | 25.2 | 9.58 | 0.76 | 8.82 |
| job-dev-sandbox | A10G | 52.8 | 20.06 | 1.58 | 18.48 |
| job-batch-eval | H100 | 63.0 | 23.94 | 1.89 | 22.05 |

Use **europe-north1** when carbon is the priority, **us-east-wa** for minimum electricity cost, or **us-east-wa** for the simple cost-carbon balance. Validate data residency, capacity, and user latency before moving online inference; the recommendation applies first to checkpointable batch/training jobs.

_Figures are June-2026 as-of snapshots; re-baseline before acting._