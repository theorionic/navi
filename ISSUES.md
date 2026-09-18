# Navi — Known Issues & Debugging Notes
_Created 2026-09-16. Status as of the small-model validation campaign._
_Updated 2026-09-18: ISSUE-01/02/04 root cause resolved by the router
temperature-start fix — see `FIX-01_pool_temp_start.md` for the validated
fix and ablation evidence. Net pool contribution on the 575M model is now
+120 mbpc (base beats zero/random/shuffle)._

This file tracks every open problem in the Navi training recipe, with
evidence, root cause (where known), and status. Order: severity, most
blocking first.

---

## ISSUE-01: Pool values learn 40× too slowly (net-negative at 4k steps)

**Severity: CRITICAL — blocks the core architecture thesis.**

The Pool (product-key memory value matrix) is *training* — gradients flow
(`gn(mem)` 0.001–0.006 every logged step), value norms grow ~30× over init
(0.016 → 0.47 by 4k), 60–70% of rows have norms far above init — but its
*contents* are worse than useless: on the step-4000 checkpoint, **zeroing
the entire pool beats the trained pool by 0.28 bpc** on held-out val
(base 5.655 → zero 5.376, 741 tokens held out). Trained values beat
random-init (+0.05) and beat shuffled (+0.18), so contents are
slot-specific and carry *some* signal — but the net contribution is
negative. The model would be better with the pool ripped out.

**Root cause (confirmed by code reading + frozen-gap experiments):**
1. **LR mis-wiring.** The trainer applies a `×30 grad-scale` to mem grads
   *before* the Lion optimizer. Lion is sign-based — `sign(30·g) ==
   sign(g)` — so the intended 30× value-learning speedup never happened.
   The values group effectively trains at the same 3e-4 as everything
   else, against a 1M-slot sparse table where each slot only gets touched
   a few hundred times per epoch.
2. **Sparse-touch amplification.** With top-k routing (k=8 of 1M slots),
   each slot receives gradient rarely; consolidation per touch needs
   *larger* steps, not the same step as the dense core.

**Fix in flight:** `NAVI_MEM_LR` env (committed, `experiments/
train500m_bpe.py` + `train500m_run23.py`) decouples the mem-group LR.
Validation plan: small-model A/B — baseline (mem-lr 3e-4) vs fixed
(3e-3), 2k steps each, compare zero-vs-base gap and coverage. The full
rebuild (run22f→run23 lineage) carries the fix only after the small A/B
proves the gap flips positive.

**Watch item:** if 3e-3 doesn't flip the gap positive within ~6k steps,
the problem is deeper than LR — suspect the value-update rule itself
(consider per-slot accumulated momentum, or bigger effective steps per
touch, e.g. accumulated per-slot momentum rather than per-step sign).

---

## ISSUE-02: Router concentrates faster than values consolidate

**Severity: HIGH — will cause collapse if left unchecked.**

Coverage metrics on the rebuild run: block b1 distinct-slot coverage
13.4% → 5.2% → 7.4% while Gini rose 0.73 → 0.86 (thousands of distinct
slots → fewer). The router is learning *where to look* (same-content
reads hit same slots, Jaccard 0.89 → 0.92) much faster than the values
learn *what to say*. If the router collapses to a handful of hot slots
before those slots' values are useful, training plateaus in a
self-reinforcing bad state.

**Mitigations available:** entropy regularization on routing weights,
load-balancing aux loss (MoE-style), or router temperature decay. Not yet
wired. Alarm thresholds set in notes: top-100 slots > 60% of reads, or
distinct slots < 1% on any block → intervene.

**Current status:** at 8k (rebuild) it had NOT crossed the alarm line
(top-100 ~ 23–25%, b2 watch-item tightening slowly). Healthy-but-
watched.

---

## ISSUE-03: Small-model validation runs are overhead-bound, not compute-bound

**Severity: MEDIUM — slows every experiment, wastes TPU budget.**

The 14M-param validator ran at 2–3k tok/s steady-state (~4.8 s/step at
BS=64×SEQ=256) vs the 575M rebuild's 50k tok/s — 4× slower per token with
40× smaller model. Causes identified:

1. **Per-block `nn.remat` recompute** — pure waste when activations fit.
   Fix: `NAVI_REMAT=0` env (committed, `navi/model.py`).
2. **Tiny per-core batch (8 seqs of 64 across 8 cores)** — latency-bound.
   Fix: relaunch at BS=256 (32 seqs/core).
3. **Cold-start XLA recompilation every relaunch** (~3–6 min each).
   Fix: persistent compilation cache — `JAX_COMPILATION_CACHE_DIR=/
   kaggle/working/jax_cache` with min-compile-time 0 / min-entry-size −1
   (committed in both trainers).

**Status:** fixes applied; the optimized baseline run (sml_base4) hit
7k tok/s by step 100 and still climbing. Remaining structural option if
still slow: `lax.scan` over blocks (compile body once) — refactor,
deferred.

---

## ISSUE-04: A/B validation protocol not yet completed

**Severity: MEDIUM — decision gate for the big-model fix.**

The mem-lr fix must show the **base-vs-zero ablation gap crossing
positive** on the small validator before committing the full rebuild to
it. Protocol:

- Arm A: small model, mem-lr 3e-4 (baseline) — 2k steps
- Arm B: small model, mem-lr 3e-3 (fixed) — 2k steps
- Compare: full-pool ablation battery (base/zero/random/shuffled) +
  coverage metrics at 2k.

**Status:** Arm A relaunched (pid 33515, smlA.log) after two false starts
(TPU device-busy after a stale process; `python3` died silently once —
kernel may have been recycled mid-poll). Arm B queued behind A.
If the gap doesn't cross positive within ~6k steps on the *big* model
later, escalate to ISSUE-01's "deeper than LR" branch.

---

## ISSUE-05: Full rebuild still mid-flight with old recipe (transplant source)

**Severity: INFO — contextual.**

The 8k→20k-step rebuild (run22f lineage, 575M params) was killed at ~8k2
to free the TPU for the small A/B (user decision: stop burning long
training; validate small first). Its ckpt remains the transplant source
for run23. Run23 (mem-lr 3e-3 + anti-collapse stack + depth growth)
launches only after the A/B verdict. All its code is pushed and
commit-ready.

---

## Open design questions (not bugs, tracked for completeness)

1. **Value-update rule.** If per-slot touch counts stay this sparse, does
   Lion-sign even make sense for values, vs accumulated per-slot momentum
   (counts-weighted)? Touch count is available at routing time — could
   scale updates by 1/sqrt(touch_count).
2. **Pool write-path.** All learning so far is gradient descent into a
   fixed-size table. No mechanism for *appending* new knowledge at
   inference/fine-tune time without retraining.
3. **Multi-depth injection.** Memory every 2 layers × 4 blocks — is that
   the right depth distribution for knowledge, or does the pool need
   different content at different depths?
4. **Eval floor.** Held-out val set is small (741 tokens); bpc deltas of
   ±0.05 are within noise. Before trusting small A/B verdicts, widen the
   eval window (multiple offsets, ≥4k tokens).