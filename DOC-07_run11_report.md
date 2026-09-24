# bpe500m Run Report — dense-warmup training (kernel session ended)

**Run**: train500m_bpe.py, v5e-8, d512/L8, pool 512²×4, BS=256/SEQ=512,
NAVI_TEMP_START=0.5 RAMP=2000, NAVI_DENSE_WARMUP=3000 (never reached —
session ended at step ~1500).

**Outcome**: kernel died at ~step 1500/20000 (23:18 UTC). Clean log end,
no traceback, no OOM — consistent with the Kaggle session timeout.
Per instruction: no retry, results written from disk.

## What was accomplished

### 1. Startup wedges root-caused and fixed (the "compile is slow" issue)
- The trainer never hung in XLA compilation or data processing. Stack
  dump (PYTHONFAULTHANDLER + SIGABRT) pinned it:
  `xla_bridge.make_tpu_client → initialize_pjrt_plugin('tpu')`.
- Cause: the daemon launcher passed a minimal env that dropped Kaggle's
  TPU vars (`TPU_PROCESS_ADDRESSES=local`, `TPU_SKIP_MDS_QUERY=1`,
  `TPU_CHIPS_PER_HOST_BOUNDS`, `XRT_TPU_CONFIG`, ...). libtpu then
  queried the platform metadata server → `DEADLINE_EXCEEDED` retries
  for 10+ min per startup.
- Fix: inherit the full environment. Result: mesh init 10.7 s,
  first jit step 7.3 s. Boot/timing marks now permanently in the
  trainer (`[boot]` / `[timing]` lines).
- Fast-start also applied: training starts after 4 batches are buffered
  (was: 512MB prefill ≈ 8 min stall).

### 2. Training results (steps 0 → 1500, dense warmup phase)

| step | loss | bpc | val bpc | pace |
|---|---|---|---|---|
| 0 | 10.02 | 14.46 | 14.29 | 7.7 s (compile) |
| 250 | 6.96 | 10.04 | — | 5.1 s (host warmup) |
| 1000 | 4.82 | 6.95 | **7.06** | 1.3 s |
| 1500 | 4.36 | 6.29 | — | 1.4 s steady |

Steady state reached ~95-103k tok/s (~40 ms/step device time after
host overhead); the 1.3-1.4 s/step at this phase is host dispatch of
BS=256×SEQ=512 batches, not TPU compute (the optimized dense PKM step
itself measured 213 ms; the step at this batch shape is
host-bound on a 575M-param model).

### 3. Checkpoint at step 1000 exists
- `experiments/ckpt_bpe500m_step001000.pkl` (3.54 GB, params + Lion
  state + histories) + `data_state.json` (exact resume point).
- Generation @1000 is coherent-ish English ("The ultimate an impact of
  my family owned teammates...") — model is learning.

## Warning sign for the dense→sparse flip (NOT yet validated)
- COV@1000: b0=2.8% b1=0.4% b2=0.3% b3=0.3% — sparse-path coverage of
  the 1.048M-slot pool is *at the side_top=64 reach ceiling* already
  during DENSE warmup (expected: dense doesn't sharpen the router, so
  sparse top-k still lands on ~uniform scores).
- The decisive test remains COV@3000 (dense end) vs COV@3100+
  (sparse resume): if it slides from whatever dense achieved back to
  <1.5%, the router re-concentrated and the warmup strategy needs
  side_top↑ or lb_weight>0. **This verification has not run yet.**

## Artifacts preserved locally
- `runs/bpe500m_run11_dense_warmup.log` — full training log.
- On-kernel (if session returns): the two checkpoints + data_state.
- `experiments/ablation.py` — ready-to-run post-flip ablation harness
  (export eval, continue-train A/B, per-layer hit rates).

## To resume when a session is available
```
NAVI_RESUME=1 NAVI_TEMP_START=0.5 NAVI_TEMP_RAMP=2000 \
NAVI_DENSE_WARMUP=3000 python3 -u code/train500m_bpe.py
```
It resumes exactly from step 1000 (ckpt + data state). Note the temp
schedule and dense-flip are step-indexed, so resume at 1000 is correct
as-is.