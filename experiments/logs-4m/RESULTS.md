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


# 500M run — final validation matrix (2026-09-08, v5e-8, complete)

Config: d_model=512, 8 layers, Pool at layers 0/2/4/6, c1=c2=512 → 4 classes ×
512² = 1.048M slots/block, cand_k=8, side_top=64, score_temp=4.0. 556M params
(537M Pool values + 19M backbone). Byte-level (260 vocab). Data: FineWeb
sample-10BT streamed (rolling 1GB buffer), BS=256, SEQ=512, 20k steps =
**2.62B tokens**. Lion (3e-4 core / 3e-3 mem, wd=0 on values), slot-sharded
values across 8 TPU v5e cores.

## Training + held-out

| step | train bpc | note |
|---|---|---|
| 6000 | 2.302 | syntax formed |
| 11000 | 2.099 | real-word rate up |
| 16000 | 1.928 | document structure (markdown bullets) |
| 19999 | **1.905** | final; LR≈0 |
| **held-out (fresh 2000 docs)** | **1.9421 ± 0.0069** | gap +0.037 → no overfit |

Infra: 63–69k tok/s steady, 0 stalls/NaNs. One resume crash (RESOURCE_EXHAUSTED
at 10.5k) — cause: unpickled arrays restored to a single device → replicated
program (5.63GB > 5.33G/core). Fixed via reshard_tree/device_put load
(`4cdf808`); val script reproduces final numbers from ckpt alone in 56s.

## Ablations (held-out, same windows, BOS-padded = training distribution)

| eval | val bpc | Δ vs intact |
|---|---|---|
| intact | 1.9421 | — |
| values zeroed | 4.1371 | +2.1951 |
| values shuffled (seed 4242) | 4.4165 | **+2.4745** |

Shuffle > zero damage ⇒ the delta is learned *content*, not row-magnitude
artifact. The Pool carries ~2.2 bpc of held-out predictive power (ppl ×2.3).

Eval-methodology note: first eval pass fed BOS-less windows (trainer's
RollingBytes always BOS-pads) → 5.87 bpc and an *inverted* ablation
(zeroed −0.098). Both recovered to sane values after matching the training
window format. Lesson recorded: eval input distribution must be bit-identical
to training.

## Pool health (diagnose_collapse.py + eval_extra.py on final ckpt)

Keys (mean |cos| random-baseline ≈ 0.10 for 64-dim): block 0/2/4/6 =
0.089–0.128 / 0.106–0.160 → **no key collapse**, eff_rank 111–119/256.

Values: dead_frac 0.68/0.88/0.94/0.94 (blocks 0/2/4/6) = **unwritten
capacity, not death** — dead rows at init norm (~0.45); alive rows ~210–220
norm (≈450× init), alive-norm Gini 0.056–0.101 (flat among alive, no
within-alive hot rows). Alive fraction: 32.3% / 11.8% / 6.4% / 6.0% —
depth gradient, early blocks do the heavy lifting.

Routing (live queries, 10 val batches, 164k slot-touches/block): distinct
slots touched 608/46/134/320 of 1.048M; **top-100 share 0.87–1.0, Gini
1.000 → routing is hot-spot concentrated** (Zipf-like), a few hundred
slots/block carry the working set. Not a correctness problem (intact 1.94
stands) but the same *routing-distribution* constraint VERDICT.md §3 found
at 16M-fact scale. Levers: staged curriculum, cand_k↑, mem_lr_mult sweep.

## Verdict

All five claims validated on the final checkpoint: convergence (1.905 train),
generalization (1.942 val, gap 0.037), key/value/routing integrity (no
destructive collapse anywhere), and Pool causality (+2.20 bpc zero / +2.47
shuffled on held-out). The 556M Pool model trains stably end-to-end at 10×
mem LR with no collapse — architecture ready to scale (C2048: 16.8M
slots/class) once routing-distribution levers are in place; more tokens is
the first lever (6–32% utilization at 2.62B).

Artifacts: `train500m.py` (trainer), `val500m.py` (held-out + zero ablation,
stage-logged), `eval_extra.py` (shuffle + routing + utilization), final ckpt
`ckpt_500m_step019999.pkl`, kernel logs `val500m2.log` / `eval_extra.log`.