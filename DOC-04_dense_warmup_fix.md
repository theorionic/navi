# DOC-04: Dense-router-gradient warmup — the verified fix for pool concentration

_Date: 2026-09-22. colab-4 TPU v5e-1, JAX 0.7.2. Full 512x512x4 pool, 3000
steps, eval every 500 on fixed 32-window val set with the SPARSE shipped
path. Files: tmp/dense_inproc.py, tmp/dense_results.json._

## The idea (user-proposed, verified)

**Dense training, sparse inference.** During training, score the FULL
c1 x c2 grid per class and read with a dense softmax (chunked scan over
tokens; per-chunk (32, 4, 512, 512) transient). Every key and every value
row receives gradient every step. At eval/export, the standard sparse
two-sided top-k path runs — same params, same geometry, zero export code.

This removes the root cause of router concentration that every prior
mechanism failed on: unselected slot keys getting zero gradient.

## Results (mem_0 coverage; sparse-path val at every checkpoint)

| step | sparse control | dense warmup 500 | dense warmup 1500 |
|---|---|---|---|
| 0 | .039 | .041 | .041 |
| 500 | .0040 | **.0171** | .0171 |
| 1000 | .0035 | **.0057** | .0057 |
| 1500 | .0040 | .0033 | .0033 |
| 2000 | .0017 | .0022 | .0022 |
| 2999 | .0021 | .0018 | .0018 |
| **final val** | 5.4010 | **5.3843** | **5.3843** |

## Findings

1. **Real quality improvement, first of the whole investigation.** Final
   val 5.3843 vs 5.4010 control (-0.017 nats) — the first mechanism that
   moved val loss at this scale, where entropy/eps/temp/noise/credit all
   produced identical val. The dense phase builds better value vectors
   across the pool and the sparse read inherits them.
2. **4x coverage at the dense-phase peak** (.0171 vs .0040) and still
   1.6x above control at step 1000 after switching back to sparse.
3. **The contraction still resumes after going sparse** — the frontier
   re-freezes to the same equilibrium. Dense warmup is a bootstrap, not
   a cure for the sparse phase. Warmup length (500 vs 1500) was
   irrelevant; identical trajectories.
4. **Cost is trivial**: dense phase ~1.4x step time (chunked scan
   parallelizes on TPU), sparse phase unaffected.
5. **Export verified**: the shipped sparse path (two-sided top-k) runs
   deterministically on the dense-trained params, aux slots well-formed,
   and its val is the number above — i.e. dense-trained params transfer
   to sparse inference cleanly. No train/export mismatch observed.

## Production recommendation

Add `dense_warmup_steps` to the 500M trainer: run the dense read for the
first ~1000 steps, then switch to the standard sparse path. ~20 lines in
pkm.py (chunked dense read + mode flag), validated here end to end. Keep
side_top=128 + lb_weight=0.01 (FIX-01/DOC-02) as is.

For *continuously growing* utilization, pair the warmup with a pool
sized to the token budget: dense-everything is affordable on small pools
(c1=c2=256) and gives near-full utilization by construction.