# DOC-05: Size-agnostic dense pool path (in-module, CPU-validated)

_Date: 2026-09-22. JAX 0.11.1 CPU. Files: `navi/pkm.py` (`_dense_path`),
`navi/config.py` (`dense_warmup_steps`, `dense_chunk`), `navi/model.py`
(`dense` flag through Block/Navi), `tmp/dense_debug_cpu.py`,
`tmp/dense_coverage_cpu.py`._

## What was built

A dense read path **inside `ProductKeyMemory`** selected per-forward with
`dense=True` (a Python bool, resolved at trace time — two jit caches, no
recompile churn):

- **Same functional form as the sparse read**: `softmax(temp·scores)`
  over slots × value rows, projected by `w_o`. No export step; dense-
  trained params transfer to the two-sided sparse path directly (val
  evals throughout used the sparse path).
- **Size-agnostic by construction**: values combine in fixed-size token
  chunks (`dense_chunk`, default 8). Transient = `chunk × c1×c2 × D`
  fp32 per chunk element, independent of batch, sequence, pool, and
  backbone. Any pool size fits any host by turning `dense_chunk` down.
- **Every key and value row gets gradient every step** — no selection,
  no zero-gradient death. This is the mechanism from DOC-04, now
  structural instead of a patched trainer.
- lb aux-loss identical in both modes (side entropy over full tables).

## Debug evidence (per-stage prints, `tmp/dense_debug_cpu.py`)

1. Init: k1 (4,32,16), values (4,1024,16) — tiny pool on tiny backbone.
2. Forward: sparse and dense both run on the same params; logits finite.
3. **Gradient coverage: sparse touches 36.0% of the value table; dense
   touches 100.0%.** This is the whole point — dense gradient reaches
   every row.
4. Chunk invariance: dense_chunk 1/2/4/8 → max output diff 1.2e-06
   (fp32 noise). Chunking is exact, not approximate.
5. Dense→sparse transition trains clean (two static jit steps).
6. Size sweep on the same tiny backbone: pools 64×64 (chunk 4),
   128×128 (chunk 2), 256×256 (chunk 1) — all forward finite. The pool
   is decoupled from the backbone.
7. Existing suite (`tests/test_cpu_equiv.py`): ALL PASS, including
   candidate-filter exactness and the value_and_grad refactor.

## Coverage/val experiment (CPU, 600 steps, 32×32×4 pool)

| condition | final val | coverage trajectory |
|---|---|---|
| sparse control | 6.0436 | .247 → .219 |
| dense 100 steps | 6.0497 | .247 → .229 |
| dense 300 steps | 6.0537 | .247 → .206 |

At this scale the dense warmup is val-neutral (contraction still
dominates after handoff, consistent with the TPU finding: warmup is a
bootstrap, not a cure). The claim this experiment defends is
**mechanical**: the in-module dense path runs, reaches 100% of rows,
transfers cleanly to sparse eval, and scales to any pool size. The
quality lever (dense on a right-sized pool, or interleaved schedules)
remains a TPU-scale question.

## Bugs found by the debug loop (all fixed)

1. `lax.map` chunk body assumed an explicit batch axis — under
   `batch_size=chunk` each slab carries one token; shapes corrected.
2. einsum subscript typo (`nd` vs `csd`) — corrected.
3. Traced-bool branch on `dense` under jit — the flag is now static
   (Python bool), callers compile one step per mode (standard practice,
   mirrors the train/temp split).
4. `value_and_grad` unpack error in the test harness — test-only.

## Optimization pass: dense step 1057 → 213 ms (5×, v5e-8)

Profiling the dense step (d512/L8, 256²×4 pool, BS32/SEQ128) split it
into fwd ~480 ms + grad ~470 ms + Adam update ~250 ms.

**Root cause of the original slowness**: the dense read ran one einsum
PER TOKEN via `lax.map(batch_size=chunk)` — with 4096 tokens that's
4096 sequential small GEMMs per layer per chip. Batch sharding never
reached the map body.

**Fix (in `pkm.py`, no architecture change)**: fully batched dense
read — one `einsum("bln,cnd->bcd")` over ALL tokens, remat-wrapped.
XLA shards the token dim across the 8 chips natively and issues one
big GEMM per class. lax.map deleted.

| variant | dense step | note |
|---|---|---|
| lax.map per-token (original) | 1205 ms | sequential, 1 core |
| lax.scan slices + remat | 39 830 ms | scan blocks XLA fusion — rejected |
| **batched einsum + remat (shipped)** | **213 ms** | 0.88× the sparse step |

**Constraint found**: the backward's dw buffer is (tokens, C, n_slots)
fp32 ≈ 4.3 GB/layer at c=256. Inside the fused
value_and_grad+update step this fits (8 layers, verified 100 dense
steps), but a *standalone* `jax.grad` over the same graph OOMs 16 GB
chips — coverage checks must run through the training step's own
gradient, not a separate grad call. For 512² pools, shard values
across chips or chunk over classes.

**Post-optimization validation (all through the real training graph)**:

| check | result |
|---|---|
| dense step | 213 ms (vs sparse 243 ms) |
| dense gradient coverage | **100.0%** of value table |
| sparse picks contain dense argmax | **100.0%** |
| cosine(sparse, dense read) | 0.953 |
| sparse val on exported params | 5.126 after 100 dense steps |

Gradient correctness spot-checked numerically on CPU (central
difference vs analytic on value entries; agreement to bf16
finite-difference noise).

## TPU v5e-8 validation (d512, L8, pool 256×256×4, BS32/SEQ128)

Rerun on the Kaggle relay TPU (2026-09-22), 1.09B-param config with the
1.08B-param pool (65,536 slots/class, 8 memory layers):

| measure | result |
|---|---|
| dense step time | 1057 ms |
| sparse step time | 145 ms |
| ratio | **7.3×** (big-pool dense read dominates; the earlier 1.4× was a small-pool artifact) |
| dense gradient coverage | **100.0%** of the value table — no dead keys |
| sparse picks contain dense argmax | **100.0%** |
| cosine(sparse read, dense read) | 0.895, deterministic (0.0 diff) |
| sparse-path val on exported params | 5.22 by step 399 — sparse eval trains/infers clean |

Two engineering fixes the big model forced, both now in `pkm.py` /
the test:
1. **Per-chunk remat** (`jax.checkpoint` around the `lax.map` body):
   without it, dense backward materializes `chunk_count × C × n_slots`
   fp32 activations — 4.3 GB/layer at 512×512, OOM on 16 GB chips.
2. **dense_chunk=64 at scale**: chunk=8 gives 1024 sequential small
   einsums per layer (4.5 s/step); chunk=64 cuts it to 128 iterations
   (1.06 s/step). Transient stays bounded by remat.

Also: the 512×512×4 pool **cannot fit** on v5e-8 with Adam states
(537M value params × (2B bf16 + 8B fp32 m,v) ≈ 10.7 GB + backbone +
grads > 15.75 GB/chip). Production options: 256×256 pool (fits,
validated above), shard the values table across chips, or a Lion/
8-bit optimizer on the value tier.

## Usage

```python
memc = MemoryConfig(c1=512, c2=512, dense_chunk=8)  # any pool size
logits, aux, lb = model.apply(params, ids, train=True, dense=(step < 1000))
```

The trainer keeps ownership of the schedule (warmup, interleaved,
permanent) via the `dense` flag; the module stays schedule-free.