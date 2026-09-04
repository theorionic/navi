# Scale-readiness verdict — Navi Pool (2026-09-04, final)

## What is proven (valid runs only)

### 1. Anchor: 65,536-fact space — Pool works perfectly
NAVI_NONCE=64, fixed value map, 4000 steps, batch 512, 8-core TPU v5e:

| arm | seen | fresh | zeroed | shuffled |
|---|---|---|---|---|
| P4 (1.048M slots, 16x headroom) | **100%** | **100%** | 0.01% | 0.00% |
| P4L1 (mem_lr_mult=1, temp=1) | **100%** | **100%** | — | — |

- Retrieval, storage, sabotage attribution all exact.
- **The Meta levers are NOT required**: mem_lr_mult=1 + score_temp=1 (P4L1)
  reaches 100% identically (500 exposures/fact at anchor). Plain joint training
  suffices; the anchor validates mechanics, not capacity.
- Dense baseline (D4) at the same scale also memorizes — at anchor scale both
  architectures are sufficient; the Pool's value is its scale path, not the
  65k-regime win.

### 2. Scale: 16.78M facts — everything is budget-bound
4000 steps x 512 batch x 16 fact-tokens = 33.5M fact-token exposures vs
16.78M facts = **2.0 exposures/fact**:

| arm | slots | seen | fresh | zero | shuffle |
|---|---|---|---|---|---|
| D4 dense | — | 0.39% | 0.42% | — | — |
| P4 | 1.048M | 0.41% | 0.38% | 0.00% | 0.00% |
| P4MID | 4.19M | 0.42% | 0.40% | 0.10% | 0.11% |
All at chance. Dense = Pool = chance, controls 4x below intact Pool. With 2.0
exposures/fact and uniform sampling, the distribution is ~13% of facts never
seen, most seen once — one gradient touch is not memorization. This measures
the training budget, not the architecture.

### 3. The decisive test COMPLETED (v6, 2026-09-04)
8192 steps = 4.0 exposures/fact (2x the 4000-step budget), P4MID 4.19M slots:

| eval | seen | fresh |
|---|---|---|
| none (intact) | 0.41% | 0.39% |
| zeroed Pool | 0.17% | 0.14% |
| shuffled Pool | 0.16% | 0.15% |

**Verdict: recall is FLAT at chance even at 4 exposures/fact.** Pool signal
still 2.5x above sabotage controls, but absolute recall did not move from the
2-exposures result. Budget alone does not fix 16M-fact recall. Combined with
the anchor (500 exposures/fact -> 100%), the failure is in the routing
*distribution* at 16M-slot scale, not capacity and not raw budget. Cheap next
levers: staged fact curriculum, cand_k increase, mem_lr_mult sweep.

## Engineering results (all validated)
- Slot-sharded jit across 8 cores: P4 0.30s/it, P4MID 1.09s/it, D4 0.022s/it
  (batch 512, 46-1300x over the naive single-device path).
- Dataset harness hardened twice: fixed value map (the random-value bug that
  invalidated all 09-02 runs), rng split (the n2==n1 diagonal collapse).
- Full reproducibility chain local: experiments/sweep9.py + logs-4m/RESULTS.md.

## Verdict (final)
The Pool architecture **works mechanically at every scale tested** (sharded
storage, exact retrieval, sabotage attribution, warm-start carry-over) but
**closed-book recall does not scale**: 65k facts -> 100%, 1.05M warm-started
-> 0.65% (curriculum does not rescue), 16.8M -> 0.4% at every budget tried
(2.0 and 4.0 exposures/fact). The curriculum experiment isolates the cause:
even warm-starting from a 100%-converged Pool, recall collapses the moment
the fact space expands — so the binding constraint is **acquisition through
the routing distribution at scale**, not capacity, budget, or warm-start.
Leading hypothesis: with 1M+ slots, top-k routing spreads probability too
thinly over candidate slots, so a fact's chosen slots receive too little
gradient per touch to sharpen keys; the anchor regime (16x headroom, dense
slot usage) never enters this regime. Dense baseline is equally blind at
scale, so this is a task-regime property, but the Pool's intended advantage
remains undemonstrated. Next candidates: hash-based slot placement (bypass
learned routing for assignment), cand_k/key-lr sweeps, key-sharpness
diagnostics at S2.

## Reproduction
- Curriculum: experiments/curriculum.py + chain_v7.sh (S1 cold 65k, S2 warm
  1.05M, S3 warm 16.8M; checkpoints ckpt_C-S*.pkl, logs curv_S*.log).
- v6 scale test: experiments/s9v6_P4MID.log (8192 steps, 4 exp/fact).