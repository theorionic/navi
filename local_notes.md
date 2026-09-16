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
