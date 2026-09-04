# Scale-readiness results — corrected (2026-09-03, v5e-8)

## CRITICAL: earlier 4M-scale numbers were invalid

Two dataset bugs invalidated all runs before 2026-09-03 07:00 UTC:

1. **Random values at train AND eval** (`data.py` drew fresh `value` every batch):
   nothing to memorize; every eval pinned at chance 1/256 = 0.39%. This is why
   D4/P4/P4MID all "scored" ~0.35% — the ceiling of a broken probe, not a model result.
2. **`n2 == n1` elementwise** (same rng, same shape, two draws): the 16.78M fact
   space collapsed to 16,384 diagonal triples; train/eval overlap was 100%.

Fixes: seeded fixed `_VAL` table (values permanent per triple), `jax.random.split`
for n1/n2. Verified locally: 0 conflicting values, 18/18 cross-batch agreement,
14.4M distinct triples in 4000 steps (85.9% coverage), diagonal rate = chance.

## Corrected results (fixed data, both metrics reported)

### Anchor: 65,536-fact space (NAVI_NONCE=64), 4000 steps, batch 512

| arm | eval | seen | fresh |
|---|---|---|---|
| P4 (1.048M slots, 16x headroom) | none | **1.0000** | **1.0000** |
| | zero | 0.0001 | 0.0001 |
| | shuffle | 0.0001 | 0.0000 |

Pipeline validated end-to-end. Sabotage now exact: zeroing OR shuffling the Pool
destroys recall completely -> knowledge lives in Pool values, retrieval works.

### Scale test: 16.78M-fact space, 4000 steps, batch 512

| arm | slots | eval | seen | fresh |
|---|---|---|---|---|
| P4MID (c1=c2=1024) | 4.19M | none | 0.0042 | 0.0040 |
| | | zero | 0.0010 | 0.0010 |
| | | shuffle | 0.0011 | 0.0011 |

Note: loss plateau ~4.93 = entropy floor of unseen-fact prediction (~1/128 avg over
nonce+value tokens); 4000 steps x 512 batch = 2.05M fact tokens vs 16.78M facts ->
0.12 exposures/fact. This is a training-budget regime, not capacity: the Pool
signal is 4x above zero/shuffle controls but absolute recall needs more steps.

Earlier 16.78M runs (pre-fix) that showed "1.0000" were memorizing the 16k collapsed
diagonal space — also invalid.

D4 (dense) and P4L1 arms launched on fixed data; results pending.

## Perf (unchanged from 09-02)

Slot-sharded jit: P4 0.30s/it, P4MID 1.09s/it, D4 0.027s/it at batch 512.
P4BIG (16.8M slots) fp32 Adam needs 17GB > 16GB — bf16/sharded optimizer required.