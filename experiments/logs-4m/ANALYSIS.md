# Navi Pool attribution — complete result matrix (2026-09-01)

Hardware: Kaggle TPU v5e-8 (8 cores, pmap data-parallel, batch 64 split 8×8).
Task: synthetic fact recall `(key, n1, n2) -> value`; accuracy scored on value
tokens only. Chance = 1/N_VALS. d_model=256, 8 layers, Pool = 4 memory layers
× 4 classes × 512×512 slots (268M value params).

## 262k-fact matrix (v5e-1, earlier session) + routing probes (v5e-8)

| Arm | acc | zeroed Pool | Verdict |
|---|---|---|---|
| dense-only | 25.6% | — | freeloader baseline |
| joint (FFN+Pool) | 25.65% | 25.60% | Pool ignored *as fact store* |
| frozen-pool | 25.27% | — | dense path alone suffices |
| pool-only (no FFN) | 25.27% | 2.6% | **Pool stores facts when forced** |
| two-stage (upcycling) | 25.38% | 25.35% | insertion order doesn't fix routing |

**Probe surprise (joint arm):** the memory read is NOT dormant. `||mem||/||x||`
≈ 0.5–1.1 (same magnitude as residual), softmax entropy 0.22–0.52 (peaked),
w_o-rescale ×16 destroys accuracy (25.75% → 8.65%). The joint model *uses*
the Pool heavily — but as a generic feature transformer, and parks facts in
the dense FFNs.

## 4M-fact crossover (this directory; 6000 steps, ~1.5 exposures/fact)

| Arm | acc | zeroed Pool | shuffled Pool |
|---|---|---|---|
| dense-only | 1.61% | — | — |
| pool-only | 1.50% | **0.0000%** | — |
| two-stage upcycle | 1.64% | **0.70%** | — |
| joint-from-scratch | 1.67% | 1.60% | 1.60% |

Chance = 0.024%. Read:

- **Dense ceiling collapsed** as designed (25.6% → 1.6%): at 4M facts the FFNs
  cannot hold the knowledge. The freeloader is bankrupt.
- **Pool-only still stores knowledge** (1.50%, 60× chance; zeroing → 0.0000%).
- **Two-stage: zeroing the Pool collapses accuracy 1.64% → 0.70%** — the
  sabotage test finally moves. The Pool is now *carrying* a measurable share
  of the facts (~40% of what the model recalls is Pool-held).
- **Joint-from-scratch still ignores it** (1.67% → 1.60% when zeroed):
  from-scratch, the dense path captures knowledge before the Pool organizes.

## Joint-4M probes (finisher run)

Same capture pattern at 4M: memory read heavily used (`||mem||/||x||` up to
4.1), but softmax entropy **0.01–0.05** — retrieval became near-one-hot, and
slot diversity collapsed (352–3900 distinct slots). Interpretation: with
scarcity the router sharpens drastically but distributes over few slots —
consistent with the Pool holding only a fraction of facts.

## Conclusions

1. **The Pool mechanism works at every scale tested** (pool-only > 50× chance
   at both 262k and 4M facts; zero-collapse is exact).
2. **The routing/storage failure is a *from-scratch dynamics* problem**: only
   the upcycling schedule makes the Pool carry knowledge when dense FFNs exist.
3. **Two-stage + scarcity = Pool becomes load-bearing** (zero-collapse 0.70%
   vs 1.64%). Not yet dominant (dense still holds most facts) — next levers:
   memory-specific LR (keys/values lr ≫ backbone), sharper retrieval
   temperature, FFN-path dropout during stage 2, bigger cand_k.

## Reproduction

- `experiments/batch4m.py --phase-a` (262k joint + pool-only + probes)
- `experiments/batch4m.py --phase-b` (4M dense / pool-only / two-stage)
- `finish4m.py` on the VM (joint-4M + probes; also in logs-4m/f.log)
- pmap data-parallel validated against single-core (identical acc at 300 steps)