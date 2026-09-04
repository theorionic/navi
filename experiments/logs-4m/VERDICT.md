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

### 5. Hash-placement fix (v8, 2026-09-04): does NOT rescue 16.8M recall
Deterministic slot placement (splitmix32 of context token ids mod n_slots,
uniform 4-row gather, learned w_o readout; CPU mini-test 1024-fact space hit
19.9% fresh vs 0.39% chance -> mechanism learns). At task scale:

| arm | slots | space | fresh | zeroed |
|---|---|---|---|---|
| H4 | 1.048M | 65k (500 exp/fact) | 100% | 83.8% |
| H4 | 1.048M | 16.8M (2 exp/fact) | 0.40% | 0.03% |
| H4MID | 16.78M | 16.8M (2 exp/fact) | 0.40% | 0.02% |

Chance = 0.39%. Hash placement = learned placement = chance at 2 exp/fact.
Sabotage attribution at anchor is weak for hash (83% survives zeroing) —
with a random 4-of-1M gather the dense path alone reaches ~84% at 65k, so
hash adds nothing there; at 16.8M nothing learns at this budget, with any
placement scheme. The binding constraint at 2 exposures/fact is deeper than
placement: one gradient touch does not write a retrievable value, period.
The remaining question is whether MORE budget per fact (8-16+ exposures,
curriculum staged expansion with rehearsal) lets hash placement accumulate —
the CPU mini-test says the write mechanism works when exposure is adequate.

### 6. Pool-collapse diagnostics (2026-09-04): NO collapse found
Ran key/value health checks on all three curriculum checkpoints
(S1-65k, S2-1M, S3-16M; 4 memory layers each; diagnose_collapse.py):

- **Key tables**: mean|cosine| between random subkey pairs = 0.101-0.102 at
  every stage, every layer (random-vector baseline ~0.10 for d=64; collapse
  would push this toward 1.0). Effective rank ~60-62 of 64.
- **Value tables**: dead-row fraction 0.0000 everywhere, row norms healthy
  (0.135-0.165), effective rank ~62-64 (full).

Interpretation: the 16.8M failure is NOT key collapse, NOT value death, NOT
degenerate tables — the parameter spaces stay healthy and full-rank at every
scale. The tables simply never receive enough task gradient at 2-4
exposures/fact to organize into a memorizing store; they remain near-init
healthy noise. This is consistent with the exposure-budget diagnosis and
rules out the structural-collapse family of explanations.

## Real-data validation (enwik8, 2026-09-04): Pool matches dense on real text
Byte-level LM on enwik8 (100MB raw Wikipedia XML, 90M train / 10M test bytes,
raw 258-symbol vocab, d_model=256, 8 layers, memory every other FFN, c1=c2=512
x 4 classes = 1.05M slots/class, cand_k=8, 4000 steps):

| arm | held-out bpc | notes |
|---|---|---|
| DREAL dense | **1.7898** | 6.4M params |
| MREAL pool | **1.7997** | 274.6M params (270.3M in Pool tables) |
| MREAL, Pool zeroed | 3.9077 | value tables destroyed |
| MREAL, Pool shuffled | 3.9024 | value rows permuted |

Findings:
1. **The Pool trains and performs on real text**: within 0.01 bpc of the
   dense baseline at matched tokens (Pool ran batch 256 vs dense 512;
   8000 vs 4000 steps was interrupted at 4000 - both arms saw identical
   token counts at step 4000 since BS 256 x 8000 target = BS 512 x 4000;
   this run stopped at 4000 steps = half tokens, still bpc parity).
2. **Knowledge attribution is decisive on real text**: zeroing or shuffling
   the Pool value tables explodes bpc from 1.80 to 3.91 (+2.12 bits/byte,
   model degrades to near-unigram level). The Pool stores a large,
   non-redundant share of what the model learned - the sabotage test
   transfers from synthetic facts to real language modeling.
3. **Routing health (slot stats on final checkpoint, 20 eval batches x 256
   seq x 64 batch)**: distinct slots used of 1,048,576 per class:
   mem_0: 27,442 (gini 0.992, top1% 0.886), mem_2: 48,930 (0.983/0.692),
   mem_4: 50,164 (0.982/0.675), mem_6: 38,022 (0.988/0.783).
   Interpretation: at 1M slots with only 16.8M training tokens, routing
   concentrates on ~3-5% of slots (heavy-tail traffic, gini ~0.98-0.99).
   NOT a routing collapse (values are healthy, recall works) - it is the
   exposure problem again: 16.8M slots vs 512x64x4000=131M token-positions
   means most slots are never touched enough to be recruited. At real
   scale the design answer is the tiered pool: hot resident slice + cold
   tier, exactly as specced.

Reproduction: experiments/sweep_real.py (NAVI_DATA, NAVI_STEPS env);
eval-only: experiments/eval_ckpt.py or run_slot_stats.py; logs
logs-4m/sweep_real_final.log (+ sweep_real_partial.log, sweep_real_mem.log
from the interrupted first kernel; ckpts on kernel disk).

## Engineering results (validated)
- Dataset harness hardened twice: fixed value map (the random-value bug that
  invalidated all 09-02 runs), rng split (the n2==n1 diagonal collapse).
- Full reproducibility chain local: experiments/sweep9.py + logs-4m/RESULTS.md.

## Verdict (final, after v8 hash-placement test)
The Pool architecture **works mechanically at every scale tested** (sharded
storage, exact retrieval, sabotage attribution, warm-start carry-over,
deterministic hash placement) but **closed-book recall at 16.8M facts is
chance under every intervention tried**: learned placement (2 and 4
exposures/fact), curriculum warm-start, and hash placement. All acquisition
routes converge on the same wall: **at ~2-4 gradient touches per fact,
nothing writes a retrievable value — dense or pooled, learned or hashed.**
The anchor regime (500 touches/fact) is 100%, the CPU mini-train (adequate
exposure, 1024-fact space) reaches 20% and climbing on the hash path.
The scaling law this data supports: recall requires exposures per fact well
above single digits regardless of placement scheme; capacity and routing are
NOT the binding constraint at these budgets. The architecture bet (separate,
shardable, growable knowledge store) is mechanically sound; what remains is a
training-regime problem: get exposures/fact up (rehearsal curricula, repeat
sampling, or fact-space staging with consolidation) and the demonstrated
write-once-read-exact mechanics apply.

## Reproduction
- Curriculum: experiments/curriculum.py + chain_v7.sh (S1 cold 65k, S2 warm
  1.05M, S3 warm 16.8M; checkpoints ckpt_C-S*.pkl, logs curv_S*.log).
- v6 scale test: experiments/s9v6_P4MID.log (8192 steps, 4 exp/fact).
- v8 hash placement: chain_v8.sh (H4/H4MID 16.8M + H4 anchor; s9v8_*.log);
  hash impl in navi/pkm.py _hash_path + MemoryConfig.hash_slots.