# Routing levers battery — 500M Pool, 3k steps each (2026-09-11)

Matched protocol: 393M tokens per run, routing battery = 164k touches/block
(10 batches × 32 × 512), eval at score_temp=4.0, same FineWeb val stream.

| run | cand_k | temp | val bpc@3k | touched m0/m2/m4/m6 | top100 | Gini |
|---|---|---|---|---|---|---|
| baseline | 8 | 4.0 const | ~2.4 | 608/46/134/320 | 0.87–1.00 | 1.000 |
| levers-1 | 16 | 1.0 const | 2.386 | 2161/2198/4719/18602 | 0.32–0.50 | 1.000 |
| anneal | 16 | 1.0→4.0 @1500 | 3.144 | 3017/1800/999/1045 | 0.52–0.72 | 1.000 |

Alive% (values, norm > 2× init): m0 42 / m2 17 / m4 10 / m6 8.5 (anneal run).

## Findings
1. cand_k 8 → 16: 3–10× more distinct slots touched, top100 share down
   from ~0.9–1.0 to 0.5–0.7, no loss cost. **KEEP.**
2. Two-phase temp anneal (flat 1.0 → sharp 4.0 at step 1500): zero
   routing-spread benefit (pre- vs post-anneal ckpts nearly identical:
   2992 vs 3017 touched on m0), costs +0.76 bpc (3.144 vs 2.386). Flat
   start permanently damages value learning; sharp phase never recovers
   (VAL plateau 3.145 → 3.144 from step 2000 on). **DROP.**
3. Gini = 1.000 in every config: routing is winner-take-most regardless
   of temperature or cand_k. Structural fix is a load-balancing aux loss
   (switch-transformer style) or entropy regularizer on router logits.

## Caveat
Battery evals all ckpts at score_temp=4.0 regardless of training temp;
levers-1 (T=1.0) is evaluated off its training temperature.

## Next lever
Load-balance aux loss on router logits: aux = n_slots · Σ(f_i · P_i).
Target: Gini < 0.95, top100 < 0.3, no val bpc regression.