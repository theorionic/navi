# TPU step-time profile — 500M Pool config (2026-09-14, v5e-8)

Script: `experiments/profile_pool.py` (kernel copy under
`/kaggle/working/code/profiling/`, synced via FUSE mount). Ablations are
full fwd+bwd jit steps, identical backbone, one change each. Full step at
BS 256 is the train500m production config (cand_k=16, side_top=64,
lb=0.01, temp=4.0).

## Measurements

| config | ms/step | note |
|---|---|---|
| full, bs=256 | **2093** | production shape; 63k tok/s matches train500m log |
| full, bs=128 | 1056 | 2.0× speedup |
| full, bs=64 | 555 | 3.8× speedup → near-linear in batch |
| B: no Pool (dense FFN) | **58.6** | the backbone alone |
| C: Pool, lb off | 2077 | lb aux ≈ free (−0.8%) |
| D: values frozen (no scatter-add grad) | 2107 | scatter-add ≈ free (+0.7%) |

HLO census (post-schedule dump, 21.5k lines): **77 all-gathers of the
full value table `f32[4,262144,128]` (512MB each)** — 21 per Pool layer
for blocks 2/4/6, 14 for block 0 — plus 223 scatter-adds (mostly the
transposed-JVP value grads) and 286 top_k / 76 sorts. The schedule tail
(last 26% of lines) is 1188 get-tuple-element + 1080 slice-done + 539
copy-done: async-collective/DMA drain, not compute. The backbone (dense
blocks + attention) is 219 convolutions (XLA maps matmul→conv on TPU) and
fits entirely in phase 1.

## Analysis

1. **The Pool is 97% of the step.** 2093 vs 59ms. Any kernel work should
   target the Pool path and nothing else.
2. **What inside the Pool:** the HLO dump shows XLA materializing the
   *entire* value table via all-gather before every gather/scatter —
   77 × 512MB ≈ 41GB of HBM traffic + 8-core collective sync per step.
   That is the elephant. A `cand_k=16` gather should touch 16×128×4B×2
   sides ≈ 16KB/token, not 512MB×77.
3. **What it is NOT:** top-k/sort (286 ops but in fused kernels), the
   scatter-add grad (ablation D: +0.7%), the lb entropy aux (C: −0.8%),
   or the dense backbone (B: 59ms). Batch-linear scaling confirms the
   bound is per-token data movement through the collective path, not
   MXU compute.
4. **The 25% gate is passed with room to spare:** the Pool path is 97%
   of step time. But the *target* changed: the win is not a Pallas
   gather→softmax→sum kernel (the local compute after the gather is
   trivial); it is **eliminating the full-table all-gather** — i.e.,
   making XLA gather directly from the slot-sharded table (emulated
   gather across the mesh, or layout/sharding annotations), so each core
   only pulls the rows it needs.

## Decision

- **Pallas readout kernel: NOT warranted.** The fused readout is ~2% of
  the step. Writing a kernel for it is the classic black-box-boundary
  trap for no measurable win.
- **Worth trying instead (cheap, no Pallas):** (a) re-shard values along
  the slot axis only and check XLA's `GatherScatterIndicesBitpacked` /
  async-collective config; (b) explicit `jax.lax.with_sharding_constraint`
  on the gathered rows; (c) try `side_top=32` — bigger side-candidate
  fan-in, same all-gather count. If the all-gather count per step drops
  from 77 to ~8 (one per layer per fwd/bwd phase), expected step time
  falls by up to ~40%: 2093 → ~1.2–1.4ms×1000.
- Fallback if XLA refuses: a Pallas kernel IS the right tool for a
  sharded-gather emulation (DMA per row-block), but only after the
  compiler-level fix is measured insufficient.

## Run integrity

- Earlier runs failed for environmental reasons: first un-sharded (HBM
  OOM 58.5G > 15.7G — params were replicated on one core), then a stale
  zipf watcher (pid 4880) held `/dev/vfio` until killed. Final numbers
  above are from the clean sharded run (log `profile_pool.log`,
  `PROF_EXIT_0` implied by full completion through ablation D).
- HLO census script initially returned empty counts (regex mismatch with
  the scheduled-dump format); corrected parse used awk field extraction +
  targeted greps. Raw dump kept at `/kaggle/working/step.hlo`.

## Reproduction

- Kernel: `cd /kaggle/working/code/profiling && python3 profile_pool.py`
- Sync method: relayfs FUSE mount at `/tmp/navi-kernel` (no git commits).