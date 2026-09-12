# lb scale check — does entropy-lb rescue 16.8M-fact recall? (2026-09-12)

Question: the lb battery (`lb_battery.md`) showed the entropy-lb aux
breaks router monopoly on real text (mem_6 10× slot spread, gini 1.000 →
0.962, val bpc −0.15). Does the same lever lift the synthetic-fact recall
ceiling at 16.8M slots — the original VERDICT failure?

## Setup

`experiments/lb_scale.py`, two arms, TPU v3-8, 4000 steps each,
batch 512 × seq 64 = 131M token-positions, c1=c2=512 (4.19M slots total),
cand_k=8, side_top=16, score_temp=4.0, **lb_weight=0.01**, mem_lr_mult=10.
Same data module as VERDICT (16 keys × nonce² facts, fixed value table).

| arm | fact space | exposures/fact | result |
|---|---|---|---|
| lb-64 | 65,536 facts | ~500 | **fresh 100.0%**, zero 5.2%, shuffle 5.2% |
| lb-1024 | 16.78M facts | ~2 | **fresh 0.38%**, zero 0.20%, shuffle 0.19% |

Chance = 1/256 = 0.39%.

Routing battery (eval traffic, per class, 1.05M slots):

| arm | touched/class | top100 |
|---|---|---|
| lb-64 | 17k–56k (1.7–5.4%) | 0.05–0.35 |
| lb-1024 | 5.4k–7.4k (0.5–0.7%) | 0.19–0.70 |
| Sweep A ref (no lb, 4M slots) | 21–55k (0.5–1.3%) | top1% ~1.00 |

## Verdict: lb does NOT rescue 16.8M recall

1. **Anchor intact**: lb-64 = 100% fresh with clean sabotage attribution.
   The lb aux does not damage the regime where recall is achievable.
2. **Scale failure unchanged**: lb-1024 fresh = 0.38% = exactly chance.
   Sabotage gap ≈ nil — the Pool carries no retrievable signal at
   2 exposures/fact, with balanced routing.
3. **Router-level fix DID transfer**: no single-slot monopoly (top100
   max 0.70 vs ~1.00 without lb). The distribution is healthier than
   Sweep A's — and it still doesn't matter for recall.
4. **The binding constraint is exposures-per-fact, not router balance.**
   lb fixes the traffic distribution over slots; it cannot manufacture
   gradient touches that never happen. VERDICT's core diagnosis stands:
   at ~2 gradient touches per fact, nothing writes a retrievable value
   — dense or pooled, learned or hashed, balanced or collapsed.
5. Nuance: lb-1024 touched FEWER distinct slots than lb-64 (5–7k vs
   17–56k) despite identical slot count — at 16.8M facts, 131M token
   positions visit fewer distinct slot neighborhoods. Spread helps
   within what traffic exists; it is downstream of exposure.

## Where this leaves the program

- The lb aux is validated for what it targets: **router health on
  real-text training** (spread + faster value learning). Keep it.
- The 16.8M recall wall is confirmed (now under balanced routing) to be
  the exposure wall. The remaining routes are training-regime: rehearsal
  curricula (get exposures/fact up per fact-space stage), repeat sampling,
  or staged expansion with consolidation — as VERDICT already concluded.
- Scale ladder for a future decisive test: same lb-64 recipe but expand
  the fact space in stages with rehearsal (S1 65k → S2 1M → S3 16.8M,
  warm-started, curriculum.py protocol) with lb on. If S3 fresh recall
  > chance under warm-start + lb, expansion-with-consolidation works;
  that is the one hypothesis this run does not test (it trains 16.8M
  from scratch, uniform sampling).

## Reproduction

- Harness: `experiments/lb_scale.py` (md5 d20fcb14), runner
  `experiments/run_lb_scale.sh` (md5 83eef870).
- Logs: kernel `/kaggle/working/lb_scale_64.log`, `lb_scale_1024.log`
  (tee'd through `lb_scale_all.log`); ckpt `/kaggle/working/experiments/ckpt_lb_scale.pkl`
  (final = lb-1024).