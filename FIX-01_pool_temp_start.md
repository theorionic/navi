# FIX-01: Router temperature start — the fix that made the Pool learn

_Created 2026-09-18. Companion to ISSUES.md (ISSUE-01/02/04). Status: **validated on the 575M model, TPU v5e-8.**_

---

## TL;DR

The Pool (product-key memory value table) was not learning because its
**addressing** (the router) was frozen during the early steps: softmax
temperature started at 0.0 and ramped to 1.0 over 3000 steps, and router
gradients scale ∝ temp — so for the entire ramp the router received
~zero gradient while values consolidated onto noise-selected slots.

**Fix:** start temperature at 0.5 instead of 0.0 (`NAVI_TEMP_START=0.5`,
ramp shortened to 1500 steps). Validated by the base-vs-zero pool
ablation: the gap went **−61 mbpc → −4 mbpc → +120 mbpc** across three
575M-param runs. +120 means the intact pool is now strictly better than
deleting it, with per-slot addressing verified by the shuffle ablation.

One env var. No architecture change. No speed cost (steady 222 ms/step,
38k tok/s — identical to before).

---

## Symptom

On every prior run (small validator and 575M rebuild), the pool ablation
battery showed the same pattern:

```
[ablate] base (intact pool):        7.779
[ablate] zero (all values = 0):     7.718   (-61 mbpc)   <- zero WINS
[ablate] random (fresh-init pool):  7.992   (+213 mbpc)
[ablate] shuffle (rows permuted):   8.019   (+240 mbpc)
```

The model would be better with the pool ripped out. Values had gradients
(`gn(mem)` nonzero, norms growing 30×) and contents were slot-specific
(random/shuffle hurt), but the net contribution was negative. More
training did not fix it: an 8k-step run with the old ramp still ended at
−4 mbpc.

## Root cause

`temp_at()` in `experiments/train500m_bpe.py` (pre-fix):

```python
def temp_at(step_i):
    f = min(1.0, step_i / 3000)
    return f
```

The routing softmax runs at temperature `f`. Router gradients (w_q, k1,
k2) flow through the softmax **scaled by temp**, so:

- At temp ≈ 0 the softmax is near-argmax → gradient ≈ 0 → **router frozen**.
- Measured on the 575M run: `w_q` grad norm was 0.0 exactly at temp=0,
  6.6e-5 by step 25, 1e-1 only at step 200. The first ~1000+ steps ran
  with a near-frozen router.

Meanwhile the values group trained at full LR from step 0 — consolidating
onto slots chosen by a **random, untrained router**. By the time the
router thawed, values had already committed to noise-selected slots; the
router then had to route around bad contents. The two never co-adapted.

Secondary bug found and fixed in the same campaign (ISSUE-01 follow-up):
the `is_mem()` optimizer-group test used `keystr(kp)` bracket-format
matching (`['p']['block_0']['mem']`), which never matched — **k1/k2 router
keys were silently landing in the core group and being weight-decayed
toward zero**. Fixed by matching `('p', ...'values'...)` / `('p', ...'k1'|'k2'...)`
path tuples directly. Weight decay was actively shrinking the router keys
every step — routing got worse the longer it trained.

## The fix

Two changes, both in `experiments/train500m_bpe.py` (synced to
`navi/train.py`, remote `/kaggle/working/code`):

### 1. Temperature knob (the fix)

```python
def temp_at(step_i):
    # NAVI_TEMP_START: initial softmax temperature. Router grads scale with
    # temp, so 0.0 keeps the router frozen for the whole ramp; 0.5+ lets
    # routing learn from step 0. Ramps to 1.0 over NAVI_TEMP_RAMP steps.
    t0 = float(os.environ.get("NAVI_TEMP_START", "0.0"))
    ramp = float(os.environ.get("NAVI_TEMP_RAMP", "3000"))
    f = t0 + (1.0 - t0) * min(1.0, step_i / ramp)
    return f
```

Launch config: `NAVI_TEMP_START=0.5 NAVI_TEMP_RAMP=1500`.

### 2. Optimizer group fix (prerequisite — already committed)

`is_mem(kp)` must match the tree-map path tuple, not `keystr` text, so
keys/queries land in the mem optimizer group (Adam, `NAVI_MEM_LR=3e-3`)
instead of the weight-decayed core group. Without this, the router keys
decay even when temp lets gradients through.

## Validation (575M params, TPU v5e-8, BS=64×SEQ=256)

Three runs, identical except the noted knob. Ablation battery on the
final checkpoint (held-out 741,321 tokens; mbpc = milli-bpc):

| Run | config | steps | val bpc | base − zero | base − random | base − shuffle | verdict |
|---|---|---|---|---|---|---|---|
| Fix run | temp 0→1, ramp 3000 | 2k | 7.779 | **−61** | +213 | +240 | pool net-negative |
| Arm B | temp 0→1, ramp 3000 | 8k | 6.120 | **−4** | +368 | +391 | neutral; steps alone insufficient |
| **Combo** | **temp 0.5→1, ramp 1500** | **8k** | **6.163** | **+120** | **+516** | **+452** | **pool beats all ablations** |

Router gradient evidence (from `[dbg:grads]` traces):

- Combo run, step 150: `w_q:g=6.1e-2`, `keys:g=1.6e-2` — alive from step 0
- Fix run (temp 0), same step: `w_q:g≈8e-5` — two orders of magnitude smaller

`base − zero = +119.5 mbpc` is the headline: zeroing all 1M value slots
makes the model worse, i.e. the contents are load-bearing.
`base − shuffle = +452 mbpc` is the deeper claim: permuting rows (right
contents, wrong addresses) is nearly as bad as random — **the addressing
is slot-correct**, which is what distinguishes a working memory from a
spare embedding table.

### Margin trajectory across runs

```
base − zero:  -61  →  -4  →  +120 mbpc      (2k fix → 8k old-temp → 8k combo)
base − random: +213 → +368 → +516
base − shuffle:+240 → +452 → +452
```

Still improving at 8k steps — longer runs should push margins further.

## Production recommendation

```bash
NAVI_TEMP_START=0.5   # router learns from step 0 (was 0.0)
NAVI_TEMP_RAMP=1500   # reach full sharpness in half the old ramp
NAVI_MEM_LR=3e-3      # values group LR (ISSUE-01 fix, required)
NAVI_STEPS=8000       # or more; margins still growing at 8k
```

## Honest caveats

- +120 mbpc ≈ 1.7% of total prediction quality. Real, but small — the
  backbone still does the heavy lifting at this scale.
- Eval floor (ISSUE-05 note 4): held-out window is 741k tokens for the
  big-model battery — deltas of ±5 mbpc are noise; +120 is well clear.
- Collapse watch (ISSUE-02): temp 0.5 sharpens routing earlier; coverage
  metrics should be re-checked on longer runs before scaling further.
- Open question: does the temp-start result hold for the small validator
  recipe, or is it 575M-specific? (Small-model A/B used a different
  temp schedule; not re-run with this fix.)

## Files

- `experiments/train500m_bpe.py` — `temp_at()` knob, `is_mem()` path-tuple fix, `[dbg:grads]` probe
- `navi/train.py` — same fixes synced
- `experiments/eval_pool_ablate.py` — ablation battery (base/zero/random/shuffle)
- Checkpoint of validated run: remote `/kaggle/working/combo/ckpt_bpe500m_step007999.pkl`
- Logs: remote `/kaggle/working/combo.log`, `/kaggle/working/ablate_combo.log`