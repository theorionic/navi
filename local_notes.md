# Launch 500M routing run

The 500M routing run was already launched in the previous exchange
(`routing_500m.log`, cfg DM=512 NL=8, CAND_K=16, TEMP 1.0). This turn:
monitor it to completion, then run routing metrics on the new ckpt and
compare against the baseline checkpoint (019999) with the same protocol.
## run23 smoke verdict (2026-09-15/16, kernel died mid-run)
- graft bug chain: (1) split('/') on keystr - keystr has NO '/' separators,
  so graft silently returned fresh init everywhere; earlier local 'carried
  True' was vacuous (same PRNG seed => identical fresh init). Fixed with
  bracket-regex parse (commit 9e574cb). Verified non-vacuously with
  different-seed ckpt: k1/ff carried True, block_8 stays fresh.
- remat per Block fixed the 11.9GB OOM (was attention softmax activations).
- bf16 values + dtype-cast graft: 12.9GB -> fits 9.5GB/core.
- SMOKE RESULT (100 steps): carried: True; step0 loss 2.6464; step50
  2.6324; step99 2.6816 (noise, single-digit steps); VAL@99 bpc 5.3320
  (down from 5.3756); COV@99 b1 14.0% b2 14.4% b3 14.1% b7 15.2% (from
  ~3-4% @0) - pool slots ARE being used now, routing not collapsed.
- ROUTING ISSUE: yes - coverage grew ~4x across b1-b3/b7 within 100 steps;
  two-level exact top-k + side_top=128 reachable = 8192/class... 

  verified with tiny local probe: fresh random router, 2x32 tokens ->
  ~900-1300 distinct slots of 16384 per mem layer (5-8%); matches the
  kernel COV@0 3-4% baseline. Routing machinery itself spreads.

## Step-1000 knowledge battery (rebuild ckpt, 2026-09-16)
- VALUES: b0 alive 98.2% norm 0.911 (init 0.016 -> 56x), Gini 0.089;
  b2/4/6 alive 45-53%, norm 1.0-1.2, Gini 0.11-0.18. Rows are written.
- ROUTING: mem_0 distinct 18.7% top100 3.2% Gini 0.665; mem_2/4/6
  distinct 5.8-7.9% top100 10-16% Gini 0.80-0.83. No collapse.
- REPRO: self-Jaccard 1.000; same-rare 0.88-0.90 vs rand 0.15-0.29
  => CONTENT-ADDRESSED reads (the router learned addressing).
- KILL @1000: base 7.0854, hot-kill 7.0043 (-81mbpc), rand-kill 6.9229
  (-162mbpc) - both kills IMPROVE: values still noise at step 1000,
  not yet load-bearing. Re-test at 5k/8k.
- Probe had 4 bugs (aux keys, window indexing, jax.nn->optax, stale
  outer ids closure) - all fixed, committed e9e493d..c5b10f5.
- Training rebuild: step 2400 loss 3.97, 41k tok/s, ETA ~15.7h.

## Utilization trend (1k -> 3k -> 4k batteries, rebuild run)
- alive%: b0 98.2->99.1, b2 52.7->78.3->78.4, b4 45.4->81.8->82.6,
  b6 51.1->85.0->87.2. Sharp growth to 3k, plateau after.
- routing distinct%: mem_0 18.7->13.5->13.4 (stable); mem_2 5.8->4.6->3.4
  (concentrating, top100 16->23.5->29.2%); mem_4 7.6->6.5->5.2;
  mem_6 7.9->8.0->7.4. Gini 0.67-0.83 -> 0.73-0.88. Addressing intact
  (Jaccard 0.84-0.90 stable).
- kill: hot vs rand = -119 vs -341 mbpc @4k (hot carries 2.9x more
  signal than random, both still negative => values not yet
  load-bearing at 4k).
- Watch item: mem_2 distinct 3.4% falling, top100 29%. Alarm thresholds:
  top100 >60% or distinct <1%.

## Full-pool ablation @ step 4000 (user-requested)
- base 5.6565 | zero 5.3767 (-280mbpc) | random 5.7070 (+50mbpc)
  | shuffle 5.8335 (+177mbpc)
- zero HELPS -280mbpc: current pool contents net-harmful on val at 4k
  (consistent with kill test). BUT:
- trained values beat RANDOM values (+50) and beat SHUFFLED values
  (+177, worst). Ordering: shuffled > random > zero > base.
  => the model DEPENDS on slot-specific value content: right values in
  wrong slots is maximally misleading. The pool stores slot-bound
  information the model uses; it is just not yet net-positive because
  read magnitudes mislead more than they inform at 4k.
- Expected trajectory: base should overtake zero when values become
  load-bearing (~8-12k). Zero-ablation delta is THE metric to track.

## 8k validation (user-directed checkpoint check)
- ablation: base 5.4403 | zero 5.1513 (-289mbpc) | random 5.4988 (+58)
  | shuffle 5.6180 (+178). ALL GAPS FROZEN vs 4k (-280/+50/+177).
  Pool values NOT consolidating into usable knowledge.
- probe: alive% b6 89.9 (slow growth); concentration rising everywhere:
  mem_2 distinct 2.0% top100 37.7% Gini 0.905; mem_0 8.7%/9.1%/0.790.
  Addressing intact (0.85-0.92). mem_2 trending toward alarm by ~20k.
- VAL 5.4487 @8k monotone.
- Root cause of frozen gaps: mem lr effectively 3e-4 (not 3e-3 as the
  docstring claims) + Lion sign-based update neutralizes the x10 grad
  scale. Action for run23: real mem lr 3e-3.
