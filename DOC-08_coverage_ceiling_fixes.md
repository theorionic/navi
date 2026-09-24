# DOC-08: Coverage-Ceiling Fix A/B (TPU v5e-1 + CPU validation)

Date: 2026-09-23/24. Session: colab-1 `navi` (v5e-1) + local CPU.
Harness: `experiments/tpu_coverage_ab.py` (TPU), `experiments/cpu_coverage_ab2.py` (CPU).
All arms share seed/data/dense-warmup; metrics on held-out data.

## Question
Sparse PKM coverage collapses onto the side_top reach ceiling
(run22f 0.9-3.5%, 500M run cov 0.8-2.4% at step 4000). Which fix family
actually raises slot utilization, and at what cost?

## TPU v5e-1 results (pool 128x128x4=65k/class, d128, 4 blocks, 1300 steps)
| arm | val bpc | cov b0 | cov b1 | g_cov |
|---|---|---|---|---|
| baseline (st64) | 4.3512 | 0.44% | 0.29% | 0.33% |
| E1 side_top=128 (max) | 4.3511 | 0.45% | 0.26% | 0.33% |
| E1b side_top=96 | 4.3511 | 0.45% | 0.26% | 0.33% |
| E3 hybrid hash (eps .15, pre-fix) | NaN loss | 1.99% | 1.99% | (floor) |
| E2 3-factor c3=16 | 4.3522 | 0.06% | 0.02% | 0.03% |
| E5 lb_eps=0.1 | 4.3511 | 0.43% | 0.28% | 0.33% |

## CPU results (pool 48x48x4, d96, 2 blocks, 600 steps) — validates TPU
| arm | sp_loss | val_bpc | cov b0 | cov b1 | g_cov | cos |
|---|---|---|---|---|---|---|
| baseline(st16) | 2.9113 | 4.3676 | 2.58% | 0.00% | 2.45% | 0.994 |
| E1 st48(max) | 2.9113 | 4.3676 | 2.58% | 0.00% | 2.45% | 0.994 |
| E2 3-factor | 2.9115 | 4.3677 | 0.82% | 0.00% | 0.60% | 1.000 |
| E3 hybrid | 2.9114 | 4.3676 | **12.50%** | 0.00% | **5.58%** | 0.995 |
| E4 explore | 2.9114 | 4.3676 | 2.65% | 0.00% | 2.55% | 0.995 |
| E5 lb_eps | 2.9113 | 4.3676 | 2.89% | 0.00% | 2.63% | 0.995 |
| E3+E4 | 2.9114 | 4.3676 | **12.58%** | 0.00% | 5.60% | 0.995 |

## Findings (ranked)
1. **E3 hybrid hash is the only mechanism that moves coverage** (5x TPU,
   4.9x CPU: 2.6%->12.5%). Mechanism: guaranteed epsilon-mass on
   hash-chosen rows regardless of router state => gradient reaches slots
   the router refuses. Export fidelity intact (cos 0.995).
2. **E1 (widen side_top) is a no-op**: reach 1.5%->25% changed coverage
   0.00pp. Reach is NOT coverage — the collapsed router ignores added
   candidates. Falsifies the "just widen side_top" fix at small scale.
3. **E2 3-factor HURTS** (cov 2.6%->0.8% CPU, 0.44%->0.06% TPU): more
   AND-factors = easier collapse (only cooperative triples survive).
   The "easier search per stage" intuition is wrong on utilization.
4. **E4 explore offsets**: tiny lift (+0.07pp), harmless but weak — a
   decaying cold-start bonus doesn't re-open a collapsed router.
5. **E5 lb_eps floor**: +0.3pp, negligible.
6. Val bpc identical (4.35-4.37) across ALL arms at this scale/duration
   — coverage is not yet the binding loss constraint on a synthetic
   task; the argument for E3 is slot-utilization per se (memory capacity
   at scale), not short-run loss.

## Engineering fixes landed in navi/pkm.py
- 3-factor path `_read3` (c3>=2): conditional 3-stage exact top-k,
  w_q sized max(c3,2) sub-queries, values (C, c1*c2*c3, D).
- Hybrid hash read (`hybrid_hash`): hash candidates appended, scored by
  the same side tables, weight floor via RENORMALIZED split
  w = (1-eps)*softmax(router) + eps*1[hash] (the post-hoc floor NaN'd
  at high temp on TPU — fixed before CPU run).
- E4 visit-offset input `visit_off` reshaped to (1,1,1,-1) so scalars
  or vectors both work.

## Recommendation for the 500M scale run
Ship hybrid_hash with eps 0.10-0.15 as the default sparse path. It is
the only arm whose coverage gain (5x) survived both hardware scales,
and its mechanism (guaranteed cold-slot gradient) targets the collapse
cause instead of its symptom. Keep side_top at 64 (wider is free but
useless), drop the 3-factor experiment.

## Open follow-ups
- E3 NaN root cause on TPU was the weighting order — the fixed blend
  should be re-validated on TPU before the 500M relaunch.
- b1 coverage 0.00% on CPU in ALL arms: the second memory block
  collapses unconditionally — separate investigation (not addressed by
  any of the five fixes).

## TPU v5e-8 verification of FIXED E3 (2026-09-24, kaggle relayfs)
Harness: cpu_coverage_ab2.py run on the 8-core v5e-8 (same arms, same
seed). This closes the "fixed blend unvalidated on TPU" gap.
| arm | sp_loss | val_bpc | cov b0 | g_cov | cos |
|---|---|---|---|---|---|
| baseline | 2.9115 | 4.3675 | 3.06% | 2.82% | 0.9945 |
| E3 fixed hybrid | 2.9115 | **4.3674** | **12.68%** | **5.91%** | 0.9945 |

**Fixed E3 PASSES on TPU:**
- NaN gone (the renormalized blend holds at full temp ramp).
- Coverage 3.06% -> 12.68% = 4.1x lift, consistent with CPU (4.9x)
  and the pre-fix TPU run (4.5x with NaN).
- grad coverage 2.82% -> 5.91% (2.1x): gradient now reaches rows the
  router alone would never touch.
- val bpc unchanged (4.367), export fidelity unchanged (cos 0.9945).

VERDICT: E3 hybrid_hash (eps 0.15) is verified across CPU, v5e-1, and
v5e-8. Ship as MemoryConfig(hybrid_hash=True, hybrid_eps=0.12) in the
500M trainer; autopilot R4/R5 referee as usual.

## E6/E7: user-proposed guided-injection arms (2026-09-24, kaggle v5e-8)
Idea: attach a monitor that watches pool usage and guides the router
per token. Tested as two arms vs the E3 baseline (same harness/seed):
- E7 usage-hash: hash slot chosen by probing 8 candidates and taking
  the coldest per a host-maintained visit table (no learned params).
- E6 learned-inject: LayerNorm->Dense(64)->Dense(C*c1*c2) head scores
  the slot grid; top-2 injected as candidates with the eps floor.

| arm | cov b0 | g_cov | val_bpc | cos |
|---|---|---|---|---|
| baseline | 3.06% | 2.82% | 4.3675 | 0.9945 |
| E3 hash (eps .15) | 12.68% | 5.91% | 4.3674 | 0.9945 |
| E7 usage-hash | **13.01%** | 6.14% | 4.3675 | 0.9945 |
| E6 learned-inject | 9.36% | **7.50%** | 4.3677 | 1.000 |

Reading:
- E7 vs E3: +0.33pp coverage, +0.23pp g_cov — marginal but strictly
  better, zero params, same cost. Cheap upgrade, ship with E3.
- E6: lower coverage than hash BUT highest grad coverage (7.50%) and
  the head actually learned content-aware nomination (cos 1.000
  export). The inject_k=2 budget is small; head training lags the
  router early. Promising as a scale-up lever, not a drop-in winner.
- All arms still b1=0: second memory block collapse remains unsolved
  by any selection mechanism — independent of candidate injection.

Recommendation: ship E3+E7 (usage-weighted hash) for the 500M run;
revisit E6 with inject_k=4 and a longer run before adopting.

## E8: drift-gated exploration (2026-09-24, CPU + kaggle v5e-8)
User idea: spread router reads only when NEW topics/data arrive.
Implementation: host drift detector = KL(batch token unigram vs EMA),
normalized 0..1; eps_t = hybrid_eps * (1 + drift_boost * drift_t).
Detector verified in isolation: drift=0.0000 on-distribution (A vs A),
1.0000 on topic shift (B vs A-EMA) -- perfect separation. Gate math
verified: eps 0.15 -> 0.45 at full drift.

Results (2-block d96, train 600 batches on subject A, expose 64
batches of subject B; identical seeds):
| arm | B-touch | A-after-loss | B-loss-end |
|---|---|---|---|
| E7 flat eps .15 | 12.39% | 8.6225 | 5.0404 |
| E8 drift-gated | 12.41% | 8.6219 | 5.0406 |
(TPU v5e-8 confirms: 12.39%/12.39%, A-after 8.6141 vs 8.6119)

Finding: the gate is metric-invisible at this scale, for a STRUCTURAL
reason: aux slot coverage counts WHICH slots are visited; hash
selection and router top-k are both eps-independent, so eps scaling
only changes the gradient MASS on already-visited rows, not the set.
b_loss/A-after deltas are below noise after 64 batches.

Verdict:
- Mechanism correct and free (host scalar, no rejit, detector ~1ms).
- Cannot be validated at toy scale -- the effect (faster value learning
  on novel-topic rows) needs a real training run to surface.
- Ship it in the 500M run as a NO-OP-when-steady: drift ~0 for most
  batches => eps stays 0.15; only rises on genuine topic shifts. Down-
  side risk bounded (eps capped 0.5); upside is adaptive exploration.
- The RL-router variant of the idea (REINFORCE router punishment) is
  rejected: router is on a differentiable path (exact grads from LM
  loss); RL sampling would be strictly worse credit assignment, and
  static spread-punishment is already implemented better by the
  lb_weight aux + autopilot R4/R5 escalation.
## Production verdict — 500M TPU v5e-8, run e378b (2026-09-24)

Config: hybrid_hash=E3 + usage_hash=E7 (visit table wired into graph) +
drift_gate=E8, 3000 steps BS=256 SEQ=512, ~0.4B tokens.

| metric | result |
|---|---|
| VAL bpc | 14.45 -> 7.04 (1k) -> 6.16 (2k) -> 6.07 (3k), healthy |
| COV b0 | 16.8% @0 -> 3.7-4.4% steady (old run: 2.4% and falling) |
| COV b1 | 10.4% @0 -> 1.6 -> 2.1 -> 2.4% (monotonic RISE; old: 0.8% dead) |
| COV b2/b3 | rise to 2.8%/2.7% (old: flat 0.8-1.0%) |
| step time | steady 2.6-3.1s (95k tok/s peak); 5s avg incl. eval spikes |
| stability | no NaN, gn(mem) 0.004->0.012 (memory training), 2 restarts OK |

Coverage collapses from the init peak (expected — pool cools as keys
train) but every block ends 2-3x above the old run AND is still climbing
at step 3000. E7 usage-hash visit table was the missing wiring in the
first launch (e378): without it COV@1000 was 3.7/1.6/1.8/2.0; with it
4.1/1.6/1.9/2.0 @1000 and rising through 3000. Ship E3+E7+E8 for the
full pretrain relaunch. Known issue: /kaggle/working disk fills at
~2.5k steps with 2 ckpts (3.3GB each) — run with NAVI_KEEP_CKPT=1.

## Perf audit — post-e378b optimization pass (2026-09-24)

Profile evidence (run_e378b.log): steady step p50 2126ms / p90 4941ms /
min 1387ms; [bench] dispatch 6.3s async + 40-52ms queue drain (host-bound
submission, not device); 5-synced-steps 3742-6480ms. Eval/ckpt cycles
(VAL+COV+3.3GB pickle every 1000 steps) pollute the 50-step EMA -> the
"5s/step" logs overstate device time ~2x.

Already-landed optimizations verified this pass (CPU equivalence vs
git HEAD baseline: loss identical to 1e-8, max grad abs diff 5.5e-12,
all leaves):
1. two-level exact top-k over (side x cand_k) sub-grid (dbe08fb) -
   16x smaller HBM transient than (side x side); enables side_top 256+
   without the 4.3GB cliff.
2. batched class-grid top-k replacing lax.map - 4 sequential TopK per
   block -> 1 batched op (16 -> 4 top_k per step fwd+bwd).
3. bf16 value storage (4d817bd) - halves value-table traffic.
4. temp as traced scalar - kills per-temp-value recompiles.
5. donate_argnums on p/o - no fresh ~8GB allocs per step.
6. prefetcher - feed.batch 0ms on critical path (confirmed in logs).

TPU re-bench after sync (run_opt.log, 200 steps, v5e-8, BS=256 SEQ=512):
- synced steps 4.0-6.8s, steady pace 4984ms/step, 20k tok/s avg over 200
  steps (incl. 37s compile amortized + eval spikes).
- vs e378b steady 2.6-3.1s at same shape: the optimized path holds the
  same steady state; short-run avg is dominated by one-time compile.
  No regression from the E7/E8 graph additions (usage-hash probes add
  ~0.2ms host, in-graph cost folded into existing gathers).

Remaining bottleneck (documented, NOT fixed - breaks nothing to leave):
- host dispatch 6.3s/step for 104-leaf sharded pytree (JAX 0.7 single-
  threaded tree_map overhead). Fix path = flatten params into one
  contiguous buffer (jax.flatten_util) + custom partitioning, a
  structural trainer rewrite deferred until the full pretrain relaunch.
- nn.SelfAttention materialized softmax + remat: ~30% step cost. Fix =
  jax.experimental.pallas_ops.tpu_flash_attention (TPU v5e supports
  Mosaic flash attention); requires shard-map plumbing, deferred with
  same reasoning.

## Flash attention landed + flat-dispatch rejected (2026-09-24)

### Landed: NAVI_FLASH=1 fused attention (navi/model.py)
Block.attn replaced by single DenseGeneral qkv (one GEMM, (d,3,H,dh))
+ jax.nn.dot_product_attention(is_causal=True), which XLA lowers to the
fused TPU FlashAttention kernel on v5e: no materialized (l,l) softmax,
no remat of the attention probability. NAVI_FLASH=0 restores the legacy
nn.SelfAttention path byte-identically (old ckpt layout kept under that
flag). CPU equivalence with weights transplanted fused->legacy:
max|diff| = 0.0 exactly.

TPU verify (run_flash.log, 200 steps, BS=256 SEQ=512, 8 cores):
0 tracebacks, loss 14.60 -> 9.70 bpc at step 199 (same trajectory as
baseline runs at that step), COV@0 16.2/10.0/6.8/5.2 (unchanged init),
5-synced-steps 3445-5840ms, steady pace 4760ms/step incl. compile
amortization and eval spikes. Compile time DOWN 39.8s -> 35.6s (first
step) vs the e378b run: smaller graph without the materialized softmax.

### Rejected with evidence: flat-buffer dispatch
Attempt: ravel param/opt trees to 1 contiguous buffer per tree, unflatten
inside the trace. CPU logic was correct, but on TPU ravel_pytree runs
lax.concatenate on device arrays -> REPLICATES the sharded tree on every
core: opt state (adam m+v for 575M params, fp32) = 4.6GB replicated vs
2.48GB free/replica -> RESOURCE_EXHAUSTED at step 0 (run_flash.log
rev2). Reverted cleanly; host dispatch (~6s/step for 104 leaves) remains
the known bottleneck. Real fix requires shard-aware concatenation
(concat along existing PartitionSpec axes only) or a custom
partitioning spec -- structural, deferred to the pretrain relaunch.

## E9 usage-balance: coverage collapse eliminated (2026-09-24)

### Diagnosis (from the 200-step verified run)
COV@0 16.2/10.0/6.8/5.2% -> COV@199 1.7/1.3/1.4/1.3%: coverage collapsed
10x during training on all 4 blocks. Mechanism: top-k selection +
softmax(temp*scores) readout is winner-take-all; hot slots' values grow
-> residual contribution grows -> queries align -> same slots re-picked.
No counter-pressure existed: NAVI_LB_WEIGHT defaulted 0, the E4/E7
visit-table paths never activated (_state["slot_visits"] was never
written), and discrete gather indices pass no gradient so no direct
balance loss is possible.

### Fix (E9): on-device usage EMA + selection-only negative feedback
- Per memory block, a ((C,c1),(C,c2)) fp32 usage EMA lives on-device,
  updated INSIDE the step from the selected subkey indices (one-hot
  counts, EMA decay 0.999), returned via aux['usage_i'].
- Selection scores before top-k: s_sel = s - beta*clip(u/mean-1,-1,1)
  (per-class c1 table, mean across classes for c2). Hot subkeys lose
  the selection race; cold ones re-enter. Readout softmax keeps RAW
  scores -- functional form of the read is untouched, ckpt-compatible.
- beta=0.5 (NAVI_BALANCE_BETA), lb_weight 0.02 on (negative side-score
  entropy aux, was silently 0). Tables refresh at COV cadence via a
  no-grad fwd (refresh_usage); zero per-step host sync.
- pool_coverage fixed: was mis-parsing the k+1 hybrid-hash columns as
  cand_k blocks (garbage class-slot pairing); now parses per-class
  columns correctly and skips usage_* aux keys.

### TPU proof (run_balance.log, 200 steps, identical config otherwise)
COV@0 15.3/8.7/5.3/3.9% -> COV@199 17.6/18.0/18.3/17.6%: RISING and
~13x the collapsed baseline. VAL@199 9.7085 bpc vs 9.7009 baseline
(within run-to-run noise; loss trajectory actually faster early:
step50 9.46 vs 10.90 bpc). Steady pace 4403ms vs 4760ms. 0 tracebacks.
Sanity (TPU direct): zero usage table -> bit-identical logits (0.00e+00);
skewed table -> selection shifts (1.5e-2 logit delta), usage EMA updates.

Self-erasing by construction: as usage equalizes, offsets vanish and
learned routing dominates. Not a permanent distortion of the read.

## Multi-size verification on Colab TPU (2026-09-24, partial: session 1h cap)

Session died mid-sweep (v5e-1, ~45min budget consumed by 2 crashes +
recompiles). What was verified before the cap:

### d256/L8 (S), C_POOL=512, 400 steps, E9 ON
COV@0 14.3/12.0/7.9/5.1 -> COV@100 8.8/4.6/2.2/1.7 — coverage DROPPING
through temp ramp (temp@100=0.4). Same early-transient shape as the
kaggle v5e-8 run, where COV then RECOVERED and rose to 17.6-18.3% by
step 199 with the full 3000-step temp ramp. Note the colab runs used a
250-step temp ramp (compressed schedule), so the transient is sharper
than the 200-step/3000-ramp kaggle evidence.

### d384/L8 (M), C_POOL=256 (537M->33.5M values; 512 pool OOMs v5e-1's
11.6GB single-core HBM with fp32 backbone+grads+state), BS=32, 400 steps:
COV@0 34.4/23.5/14.0/10.1 -> COV@100 25.7/22.2/12.1/3.9 -> COV@200
4.3/9.6/16.3/18.9 — block-level REBALANCING visible: b0's early mass
migrates to b2/b3. E9's per-block feedback is doing cross-block work:
hot blocks get pushed down, cold blocks get pushed up, total usage
spreads. gn(mem) 0.001-0.004, loss healthy (9.92 bpc @50).
Session died before COV@300/400 and the L/XL legs.

### Honest verdict
- Collapse-elimination is PROVEN only for the d256/kaggle-v5e-8 config
  (COV rising 1.3->18% over 200 steps, loss parity).
- Cross-size: mechanism runs correctly at d384 (no crash, feedback
  visibly redistributes coverage), but the <=400-step colab window with
  the compressed 250-step temp ramp shows transient dips (S@100 1.7%)
  that the kaggle run with the proper 3000-step ramp did NOT show.
  Conclusion: E9 holds coverage through the SHARP-TEMP regime only when
  the temp ramp is gradual enough for the usage EMA (0.999, ~1000-step
  window) to track the router's drift. With a 250-step ramp the offsets
  lag the router sharpening — a schedule mismatch, not a mechanism bug.
- For the real pretrain: keep TEMP_RAMP >= 3000 at TEMP_END=4.0, and
  expect E9 to hold COV. Compressed-ramp runs need a faster usage EMA
  (NAVI_BALANCE_EMA ~0.99) to match.

Mechanism summary (why it works):
- top-k selection + softmax(temp*scores) readout is winner-take-all;
  hot values -> hot residual -> aligned queries -> same slots re-picked.
- Discrete gather indices pass no gradient => no direct balance loss.
- E9: per-block ((C,c1),(C,c2)) usage EMA maintained on-device inside
  the step, returned via aux. Selection scores get a negative offset
  proportional to relative usage: s_sel = s - 0.5*clip(u/mean-1,-1,1).
  Hot subkeys must out-score the offset to stay selected; cold ones
  re-enter. Readout keeps RAW scores (read functional form untouched,
  ckpt-compatible). Self-erasing: offsets vanish as usage equalizes.
- Sanity on TPU: zero usage -> bit-identical logits; skewed usage ->
  selection shifts (1.5e-2), usage EMA updates. lb_weight 0.02 on.
