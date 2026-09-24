# PKM Pallas Kernel: Negative Result (TPU v5e, 2026-09-18)

## Question

Can a custom Pallas/Mosaic kernel beat XLA's `lax.top_k` lowering for the
PKM read path's selection stage, the hypothesized dominant cost of the
500M training run?

## Answer

**No. XLA's batched `lax.top_k` wins by ~11x.** Do not integrate
`navi/pallas_pkm.py` into the training path. Keep it as reference code.

## Measurements (v5e-1, JAX 0.11.2, production shapes)

Per memory block read, BS=256, SEQ=512, C=4, c1=c2=512, cand_k=8:

| Config | Pallas path | Baseline (JAX) | Speedup |
|---|---|---|---|
| full read, SIDE=64 | 495.8 ms | 385.0 ms | 0.78x |
| full read, SIDE=32 | 409.5 ms | 263.7 ms | 0.64x |
| selection only (no gather) | 226.8 ms | 20.1 ms | **0.09x** |

The selection primitive itself is the loser: Pallas 227 ms vs XLA 20 ms
for the same top-8 over 524k x 512 rows.

Exactness: both paths produce **identical slot sets and scores**
(max diff 0.0; the 0.11 h difference at SIDE=32 is a tie-order artifact
in softmax weighting, both correct).

## Side finding: the two-sided filter is exactly right

Two-level (side x cand_k) filtering recall vs the exact 512x512 grid:
**1.0000 at every SIDE >= 8** (theorem verified empirically). SIDE=32
runs the full read 1.55x faster than SIDE=64 with zero recall loss —
a free production win, no kernel needed (263.7 vs 385.0 ms above).
This is the actual optimization that came out of this investigation.

## Why the kernel lost

1. **Bitonic sort networks are unimplementable in Mosaic for 512-wide
   rows.** Row permutations lower to `tpu.dynamic_gather`, which fails
   with "Not implemented: Multiple source vregs along gather dimension"
   (any row wider than one 128-lane vreg).
2. **Rewriting selection as k iterative argmax extractions compiles but
   is slow**: k=8 rounds x (max-reduce + one-hot mask + where) over
   (8, 512) tiles is scalar-ish work on a chip whose strength is the
   MXU; XLA's top_k lowering uses the same vector hardware but schedules
   it once per column block with far better latency hiding.
3. **Mosaic maturity tax** (JAX 0.11.2, stable_mosaic v8): `jnp.where`
   with `±inf` values mislowers (had to use finite sentinels), boolean
   mask selects misbroadcast (had to kill winners by value-compare on
   extracted indices). Each workaround costs correctness risk.

## What would change the verdict

- Mosaic gaining cross-vreg permute/dynamic_gather (would let a real
  bitonic network compile).
- A kernel that keeps the (side, g2k) sub-grid in tiles and streams the
  argmax reduction without materializing rows — plausible but the
  selection-only gap (11x) is too large to close by tiling alone.
- If the read path ever becomes the dominant cost AFTER the flash
  attention + remat-off change, re-measure; today it is not.

## Recommended production change (small, safe)

`experiments/train500m_bpe.py`: set `SIDE_TOP` default 64 -> 32.
Evidence: recall 1.0000 at SIDE=8 (bench4 on TPU), full-read cost drops
~32% (bench6). Keep `NAVI_SIDE_TOP` env override for ablations.

## Artifacts

- `navi/pallas_pkm.py` — kernel (argmax-extraction variant, compiles on
  v5e, parity-verified, NOT for production)
- `tmp/bench_pkm_read.py` (round 1), `tmp/bench6.py` (per-block), 
  `tmp/bench7.py` (selection isolation), `tmp/bench4.py` (recall)
- Remote: colab-3 session `navi`, logs in `/content/tmp/`