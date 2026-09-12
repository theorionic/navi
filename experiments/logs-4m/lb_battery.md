# Router load-balancing aux — 500M Pool, 3k steps (2026-09-12)

Run: cand_k=16, score_temp=4.0 const, **lb_weight=0.01** (entropy form:
mean negative entropy of the side-score softmaxes s1/s2), 393M tokens,
same protocol as the levers battery (164k touches/block, eval temp=4.0).

| metric | v1 (no lb) | **lb run** | delta |
|---|---|---|---|
| val bpc @3k | 2.386 | **2.239** | **−0.15** |
| distinct touched m0/m2/m4/m6 | 2161/2198/4719/18602 | 508/9251/13567/**189355** | m6 **10×** |
| top-100 share m0/m2/m4/m6 | ~0.65/0.46/0.70/0.32 | 0.992/0.269/0.202/**0.059** | m6 **5×** |
| Gini m6 | 1.000 | **0.962** | first sub-1.0 |
| alive% m0/m2/m4/m6 | 42/17/10/8.5 | 63/27/24/**66** | m6 **8×** |

## Findings
1. **The entropy-lb aux worked on the load-bearing layer.** mem_6 (deepest
   Pool layer): 10× more distinct slots, top-100 share 0.32→0.059 (traffic
   spread over ~1700 slots instead of ~100), Gini 0.962 — the first
   measurement to break the 1.000 wall. Value-table alive% 8.5%→65.6%.
2. **Loss improved too**: val bpc 2.239 vs 2.386 (−0.15). Balance and
   learning were NOT in tension here — better spread = better gradient
   coverage = faster value learning.
3. **Regression**: mem_0 hyper-collapsed (508 slots, top100 0.992). The
   network shifted Pool usage to deeper layers; layer-0 Pool may be
   reverting toward dense/FFN-like behavior absorbed by the backbone.
   Watch: if mem_0 stays dead at longer training, consider per-layer
   lb_weight ramp or pruning the layer-0 Pool outright.

## History of the design (what failed first)
- Switch-transformer `n_slots·Σ(fᵢ·Pᵢ)` is **vacuous for sparse top-k
  PKM**: fᵢ over the gathered cand_k subset is always 1/k — the term
  cannot distinguish balanced from collapsed routing. First tiny run
  showed it only as a huge constant offset (loss 41k at lb=0.01).
- Entropy over the candidate softmax `w` has the same flaw (mass only
  on the k gathered slots).
- **Working form: entropy of the side-score tables s1/s2** (softmax over
  ALL c1/c2 subkeys per class). Gradients reach every side key, scale
  O(1) (ln(c1)+ln(c2) per block), and lb-only SGD training raises
  distinct routed slots (147→493 in the local smoke).

## Caveats
- Eval battery always scores at temp=4.0; training temp matched here, so
  the comparison vs v1 (trained at temp=1.0, evaluated at 4.0) slightly
  favors the lb run.
- Single run, no seed replicate.

## Next
1. Scale check: 16.8M-key recall retest (the original VERDICT failure)
   with lb active — does spread survive 32× more slots?
2. mem_0 dead-zone: either lb ramp on shallow layers or drop layer-0 Pool.
3. lb_weight sweep {0.003, 0.03} to bracket 0.01.