# bpe500m run22f — RESULTS & VERDICT

**Run**: 2026-09-15, TPU (v5e?), 575,695,872 params (542,113,792 in the Pool = 94%).
**Config**: d_model 512, 8 layers, memory_every 2 (4 memory blocks), c1=512, c2=512,
cand_k=8, side_top=64, n_classes=4, vocab 16384 BPE (kernel-trained), FineWeb.
**Schedule**: LR warmup(1000) → cosine → 5% peak (core+pool keys/router); pool values
constant. Backbone lion lr 3e-4 peak. mem-grad ×10.

## Outcome: COMPLETED 10550→20000 (resumed from run21 @10499), DONE

- **Final train bpc 5.15** (loss 3.567 @19999; ~64k tok/s warm, ETA tracked 8.8h→0)
- **VAL bpc (held-out, 2.9MB)**: peaked 5.460@12000 → **5.332@19999**, monotone
  improvement through decay tail. Full hist in ckpt `val_hist`.
- No instability, no collapse, gn(core) 0.11–0.22 throughout, gn(mem) ~0.002-0.003.
- Checkpoints: `ckpt_bpe500m_step019999.pkl` (+019000 backup), 4.6GB each.
- Coverage: COV@19999 b0=2.8% b1=0.9% b2=0.7% b3=2.7% (2 batches) — flat all run,
  b3 crept 2.3→2.7% late (key sharpening as LR decayed).

## Final eval (eval_final_bpe.py on ckpt_19999)

**GEN@19999 (temp 0.8, 128 tok):** fluent English, correct grammar, zero factual
anchoring ("president of the United States" → hallucinated 1927 anecdote). Reads
like a small model mid-pretrain: syntax ✓, world-model ✗. Expected at 1.8B tokens
(~2.9GB of unique text through a 576M model).

**COV-final (8×16 fresh windows):**

| block | b0 | b1 | b2 | b3 |
|---|---|---|---|---|
| coverage | 4.75% | 1.39% | 1.11% | 4.87% |

Higher than the 2-batch trainer estimate (sampling artifact confirmed), but the
shape holds: b2/b1 ~3-4× narrower than b0/b3.

**TRAFFIC (4×16 windows, per block):**

| block | top-100 slot share | alive slots | % of 2.097M-slot space |
|---|---|---|---|
| mem_0 | 0.232 | 36,292 | 3.5% |
| mem_2 | **0.508** | 11,527 | **1.1%** |
| mem_4 | **0.558** | 9,147 | **0.9%** |
| mem_6 | 0.236 | 35,591 | 3.4% |

## VERDICT

1. **Training worked; the Pool is load-bearing but tiny in effect.** Loss fell
   5.40→5.15 bpc over the resumed segment with the Pool active, and val improved
   0.13 bpc under the cosine. But: 94% of parameters live in the Pool and read
   spread froze at 1-5% per block. The effective pool is ~30k-92k slots, not 542M
   params' worth of capacity. We paid 542M params to get what ~30-90M params of
   dense FFN would give at this data scale.

2. **Routing collapsed to a persistent core, not a bug, a quantization wall.**
   Alive slots 0.9-3.5% and top-100 share 0.23-0.56 means: every block has a
   small hot core doing most reads; mem_2/mem_4 are nearly degenerate (50%+ of
   traffic through 100 slots). This matches the product-key math: 2×(c2+1)=95
   coarse classes/block — the router's *coarse* space saturates long before the
   2.1M-slot fine space does. The pool's nominal capacity is a fiction; the
   binding constraint is the class space, not the slot table.

3. **The mid-run VAL bump was benign** (peak 5.460@12000 → 5.332@19999): the
   scheduled LR decay fixed it. gn(mem) 0.002-0.003 the whole run — memory
   gradients are 60-100× smaller than core gradients even with the ×10 boost.
   The pool learns slowly; that's structural (values get gradient only through
   ~top-8 gathered slots per token).

4. **What to change next run (in priority order):**
   a. **Shrink c2, widen classes**: c2=512→64 (pool 542M→68M params) and spend
      the budget on *more memory blocks* (4→8-12) instead of wider blocks.
      Coverage % will jump 8× from the same router behavior; knowledge per FLOP
      improves.
   b. **Raise router LR / add router-temperature annealing** (score_temp 4.0→2.0
      over training): b3's late coverage creep shows keys respond to decay; force
      exploration early instead.
   c. **mem-grad ×10 → ×30-50**, or normalize per-slot gradient: gn(mem) ~0.003
      vs gn(core) ~0.18 is a 60× imbalance; the pool is training at 1/60th speed.
   d. **Keep** the schedule (worked), keep-2 ckpt rotation (worked), resume path
      with opt-state migration (worked — this run survived 3 restarts).

5. **Infrastructure record** (all verified this run): opt-state migration across
   tx-structure changes (`load_ckpt_sharded(new_opt_state=)`), HBM-safe resume
   (`del fresh_o`), coverage logging, LR schedule with per-group LRs, 4.6GB ckpt
   with 2-slot rotation on a 20GB disk, ~64k tok/s steady-state on TPU.

## Files
- Trainer: `experiments/train500m_bpe.py` (final: commit 13d5257)
- Final eval: `experiments/eval_final_bpe.py` (commit 844a493, O(N log N) Gini)
- Kernel artifacts: `/kaggle/working/experiments/ckpt_bpe500m_step019999.pkl`,
  `/kaggle/working/train_bpe500m.log`, `/kaggle/working/eval_final.log`
- Raw numbers: VAL_HIST line in log; TRAFFIC/COV/GEN lines in eval_final.log