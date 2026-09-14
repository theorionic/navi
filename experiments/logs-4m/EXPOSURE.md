# Exposure tests — Zipf frequency + staged rehearsal (2026-09-14, TPU v5e-8)

Two runs closing the question lb_scale.md left open: is the 16.8M-fact
recall wall an exposure-budget wall, and if so, do heavy-tailed sampling
or staged expansion with rehearsal get per-fact exposures above the
recruitment threshold? Chance = 1/256 = 0.0039 throughout. Same model/
protocol family as lb_scale.py (d_model=256, 8 layers, 4 Pool layers,
c1=c2=512, cand_k=8, lb=0.01, mem_lr_mult=10, 275M params).

## 1. Zipf arm: frequency alone writes knowledge (decisive)

Hypothesis (from the user's natural-text intuition): in real corpora some
patterns are always present and others never accumulate exposure. Uniform
sampling at 2 exposures/fact gave every fact the same starvation; Zipf
sampling at the SAME budget must split by exposure instead.

Protocol: same 16.78M-fact space, same 4000 steps x 512 batch x 64 seq =
131M token positions, but (n1, n2) drawn from a Zipf(a=1.5) heavy tail
(product of two Zipf marginals over the 1024 nonce values per side).

**Accuracy vs TRUE training exposure** (sampler replayed exactly, counting
(key,n1,n2) occurrences; eval = 16 fresh draws, 512x16 triples each;
script `experiments/eval_zipf_true.py`):

| true exposures/fact | accuracy |
|---|---|
| 0 (never touched) | 0.0109 |
| 1 | 0.0591 |
| 2–3 | 0.1739 |
| 4–15 | 0.7569 |
| 16–63 | 0.9980 |
| 64+ | 1.0000 |

Chance is 0.0039. Sabotage: zeroing or shuffling the Pool value tables
drops hot-bucket recall 0.9926 → 0.0000 (cold bucket 0.0004). The recall
lives in the Pool and is exposure-driven.

Findings:
1. **The exposure hypothesis is confirmed and quantified.** Recall is a
   monotone function of exposures/fact: 0 exposures → chance-level,
   ≥16 → ~100%. The single gradient touch was never the problem; the
   write threshold sits between ~2 and ~16 touches in this regime
   (steepness at 2–15 shows recruitment, not an all-or-nothing wall).
2. **Frequency is a sufficient substitute for budget.** 611k distinct
   facts were touched at least once; those with ≥16 exposures recall
   perfectly. The earlier "16.8M recall = chance" results were the
   *uniform-distribution* consequence (nobody reaches 16), not an
   architecture ceiling at scale.
3. **Caveat on the run's own buckets**: the hot/mid/tail/cold bands were
   computed from a 2M-draw reference table; the training run's 32.7M
   draws outran it, so the reported "cold 0.8190" is NOT a contradiction
   — those triples averaged 217 true exposures (85% seen). The
   true-exposure table above supersedes the reference buckets. The
   `tail` bucket printed 0.0000 because pct50 of the reference counts was
   0 → empty mask (artifact, not a measurement).
4. Training-curve check: zipf loss 1.3037 @4k steps vs uniform-16.8M
   4.93 plateau (lb_scale log) — heavy-tail concentration lets the LM
   loss drop much lower on the same token budget, consistent with most
   gradient mass landing on a recallable subset.

## 2. Staged arm: warm-start + rehearsal does NOT preserve knowledge

Protocol (stage_lb.py, all stages lb=0.01, cand_k=8, T=4, mem_lr_mult=10,
Pool warm-started + optimizer state carried, vocab layers re-inited):

| stage | nonce space | steps | replay | EVAL fresh | old-space | zero-fresh |
|---|---|---|---|---|---|---|
| S1 | 64 (65k facts) | 1500 | — | **1.0000** | — | — |
| S2 | 256 (1.05M) | 1500 | 30% S1 | **0.0003** | S1 0.0039 (chance) | 0.0012 |
| S3 | 1024 (16.8M) | 4000 | 15/15% S1+S2 | **0.0042** | S1 0.0000, S2 0.0000 | 0.0001 |

Findings:
1. **The rehearsal recipe failed outright.** S2 fresh is BELOW chance
   (0.0003 < 0.0039) and S1 knowledge was erased despite 30% replay of
   S1-space triples. S3 same: fresh ~chance, old spaces at exactly 0.
2. **Per-fact exposures in S2/S3 are the suspected cause, not warm-start
   mechanics.** S2: 1500 steps x 512 x 16 fact-tokens x 70% new = 8.6M
   touches over 1.04M NEW facts = ~8/fact — below the measured
   recruitment band (16+). Replay budget: S1 facts got 1500x512x16x0.30 /
   65,536 ≈ 56 exposures/fact, yet S1 recall still collapsed to chance.
   Interpretation: **the rehearsal touches were outcompeted** — new-space
   facts share keys/values (Pool is global across fact spaces), so
   continued training on a 16x larger space reorganizes routing faster
   than 56 diluted touches can re-consolidate; the network reallocates
   slot traffic to the dominant (new) distribution and old-space
   lookups land on rewritten slots.
3. Contrast with lb_scale anchor: same 65k space trained *from scratch*
   for 4000 steps (~500 exp/fact) = 100%. So S2/S3 collapse is not about
   stage sizing per se — it is again exposure, but now it shows that
   **exposure must be high *in the final configuration*; consolidating
   then perturbing (warm-start into a bigger space) destroys the
   consolidated store even with replay fractions up to 30%.**
4. Run integrity: S2/S3 sabotage rows present, ckpts saved after each
   eval (S3 ckpt save hit ENOSPC after evals printed — numbers above are
   from the log lines, unaffected; S1/S2 ckpts deleted to free 20G disk).

## 3. Combined verdict

- The 16.8M "recall wall" is fully explained as an **exposure
  distribution** effect: give a fact ≥16 touches and the Pool recalls it
  at ~100% even in a 16.78M-fact space. No new mechanism is required;
  the earlier uniform runs simply never gave any fact 16 touches.
- **Staged expansion with rehearsal is falsified as-is**: carrying a
  trained Pool into a larger fact space with 15–30% replay neither
  recalls the new space nor retains the old one. The Pool's value table
  is not a stable append-only store under this recipe; growth must either
  (a) keep per-fact exposure above recruitment in the FINAL space (i.e.,
  train at final scale with a Zipf-like frequency mix), or
  (b) use consolidation steps that do not re-run SGD over the union
  space (future work: per-slot freezing/EM-style value commit, or
  hash-placement append-only pools for the cold tier).
- Program consequence: the tiered-pool design (hot Zipf-ish resident
  slice + cold append-only tier) now has an empirical justification in
  BOTH halves — natural corpora are heavy-tailed (zipf arm shows the
  hot tier will hold), and re-SGD over an expanded space is destructive
  (staged arm shows the cold tier must not be retrained through the same
  SGD path).

## Reproduction

- Zipf: `experiments/zipf_test.py` (md5 8cbdb319, deployed = local),
  ckpt `ckpt_zipf.pkl` on kernel; true-exposure eval
  `experiments/eval_zipf_true.py` (md5 7e7045b4), log `eval_zipf_true.log`.
- Staged: `experiments/stage_lb.py` (md5 d018d30c, deployed = local),
  log `stage_lb.log`; ckpts S1/S2 deleted for disk, S3 partial.
- Kernel logs copied: `experiments/logs-4m/{zipf_final,eval_zipf_true,stage_lb}.log`.
- Notes: transfer-channel lesson re-learned — the printf/base64 pipe
  corrupts multi-chunk payloads; write_file (JSON) is byte-exact.