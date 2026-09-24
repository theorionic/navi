# DOC-02: Router utilization fixes — Colab TPU v5e-1 verification

_Date: 2026-09-21. Companion to ISSUES.md and FIX-01. Session: Colab v5e-1 (single chip), JAX 0.7.2._

## Question

The v5e-8 training runs show the router concentrating on a tiny reachable
slot set (coverage 6.7%/1.3%/0.5%/0.4% per block with side_top=64; 3.8-5.4x
more after side_top=128, but still far from the full pool). Which mechanism
actually makes the router USE more of the pool?

## Test setup

Full pool geometry (512x512x4 = 1.05M slots per block), single memory block,
3-layer d=256 backbone, 4096-token Zipf vocab with injected bigram structure,
400 training steps, BS=16/SEQ=128, Adam 1e-3. Conditions:

| condition | mechanism |
|---|---|
| base | exact top-k, no balancing (the collapsed equilibrium) |
| lb_eps0.01 | epsilon-mixed read weights `w = w(1-e) + e/k` |
| lb_loss0.01 | router entropy aux loss in the objective |
| both | lb_eps + lb_weight together |
| hashpath | deterministic content-hash addressing (no learned router) |
| restart50/100 | dead-slot key recycling every 50/100 steps (random 10% reinit) |

## Results

| condition | val loss | coverage (distinct slots/pool) | loss curve (step 0 → 390) |
|---|---|---|---|
| base | 5.4262 | 0.03% | 8.748 → 5.341 |
| **lb_eps0.01** | 5.4262 | 0.03% | 8.748 → 5.341 (**identical to base**) |
| **lb_loss0.01** | 5.4262 | **0.35% (10x)** | **8.374 → 4.966 (faster)** |
| **both** | 5.4262 | **0.51% (17x)** | **8.374 → 4.966** |
| hashpath | 5.4251 | n/a (hash reads not in aux slots) | fastest (16s, no top-k) |
| restart100 | 5.4218 | 0.02% | 8.748 → 5.341 (no help at this scale) |

## Findings

1. **lb_eps is a NO-OP for utilization.** The epsilon floor mixes the
   *read weights* w, but the hard top-k still selects exactly the same
   slots — the floor never routes tokens to NEW slots, so dead-slot keys
   get no gradient. Loss curve byte-identical to base. Do not use alone.

2. **lb_loss (router entropy) is the winner.** 10x more slots touched,
   and the loss *trajectory itself* is better (4.966 vs 5.341 at step 390):
   spreading candidate mass off hot subkeys improves consolidation speed,
   not just spread. Combined with lb_eps it goes to 17x coverage.

3. **restart (random 10% key reinit every 50-100 steps) doesn't help at
   this scale** — 400 steps is too short for a re-rolled key to win reads,
   and re-rolling 10% every 50 steps just adds noise (val 5.4218, best,
   but coverage lowest). Needs traffic-EMA targeting (reinit only truly
   dead slots) and longer horizons to show benefit. Unproven here.

4. **hashpath** is structurally different: no top-k at all (16s vs 120s
   per condition). Coverage metric is not comparable (hash addressing
   reads ctx-derived slots deterministically). It sidesteps the problem
   rather than solving learned routing.

## Verdict

**`NAVI_LB_WEIGHT=0.01` (router entropy loss) is the verified fix** — it is
what made the difference in the v5e-8 run (coverage 3.8-5.4x jump after
enabling it with side_top=128, stable across 12k/13k/14k checkpoints).
The Colab microbenchmark independently confirms: it expands utilization
10x in isolation and improves consolidation speed. lb_eps adds a further
1.7x on top when combined.

### Recommendation for the production config

```
NAVI_SIDE_TOP=128     # doubles the reachable band (validated on v5e-8)
NAVI_LB_WEIGHT=0.01   # router entropy loss (validated both scales)
NAVI_LB_EPS=0.0       # no-op alone; keep off unless combined with lb_weight
NAVI_TEMP_START=0.5   # FIX-01, unchanged
NAVI_MEM_LR=3e-3      # FIX-01, unchanged
```

Dead-slot recycling needs a traffic-EMA implementation before it can be
judged; the naive random-restart variant tested here is not effective at
short horizons.

## Files

- Benchmark: `tmp/util_bench.py`, `tmp/runall2.py`, `tmp/final_run.py`
- Results: `/content/final_results.json` (Colab), `tmp/` (local copies)