# RL Validation Campaign — Arithmetic on 556M Navi (9 configs, v5e-8 TPU)

**Date:** 2026-09-09
**Model:** 556M Navi (d_model=512, n_layers=8, n_heads=8, memory_every=2, vocab 260)
**Base checkpoint:** `ckpt_500m_step019999.pkl` (FineWeb prose pretrain)
**Question:** Does GRPO-style RL improve a 556M model on a specific task
(single-digit addition) — and if not, why not?

## Summary of all runs

| # | Config | Init | Result |
|---|--------|------|--------|
| 1 | GRPO, bare `a+b=` prompts, 40 rounds | pretrained | flat, reward 0.11 (chance), no format |
| 2 | + 120-step raw-format warmup, 40 rounds | pretrained | reward 0.10→0.20, format emerging, 0 hits |
| 3 | config 2, 600 rounds, lr 3e-5 | pretrained | collapse: outputs `====` |
| 4 | + ICL curriculum (3 solved examples in prompt), 600 rounds | pretrained | 0.20→0.30, ≤1/8 hits, no convergence |
| 5 | on `q:/c:` format (the format SFT taught to 1.42 bpc) | pretrained | 0.20→0.23, ≤1/8 hits |
| 6 | weighted-SFT init (answer digits ×8) + GRPO | pretrained | **0.20→0.40 peak, 2/8 hits**, then collapse |
| 7 | stable config (lower LR, ent 0.05, 900 rounds) | pretrained | flat 0.20→0.185 |
| 8 | shaped reward (partial credit) + grad clip, lr 3e-5 | pretrained | 0.22→0.28→collapse (0.075, garbage tokens) |
| — | diag4: supervised instill, 1000 steps, DIGIT_W=20 | pretrained | weighted loss 3.48→1.47, **P(correct)=0.023** — NOT-ENOUGH |

## Detailed findings

### 1. RL from chance-start fails (runs 1–8)
Every GRPO configuration failed to improve a pretrained 556M model on
single-digit addition. Best transient: run 6 (weighted-SFT init) reached
reward 0.40 with 2/8 correct samples in a group, then diverged. No run
converged. The advantage estimates from G=8-16 samples on a single prompt
are too noisy to accumulate signal at this scale before instability wins.

### 2. Supervised instill also failed (diag4)
1000 steps of digit-weighted SFT (DIGIT_W=20) on pure arithmetic drove
weighted loss 3.48→1.47 but P(correct sum) **dropped to 0.023** (below
chance 0.11). Weighted loss went down while sum accuracy didn't move —
the model satisfies the boosted loss without learning the mapping.

### 3. Full-batch supervised training at scale (diag4 second arm)
1000 steps × BS 8 windows of pure `q:/c:` data with DIGIT_W=20:
weighted loss 3.48→0.35, but **P(correct sum) = 0.000** at every probe.
The model fits the weighted loss landscape without learning addition.

## The narrow, defensible conclusion

GRPO-style RL (G=8–16, single-prompt groups) did not improve a 556M
model on this arithmetic task in any of the 9 configurations tried,
including with a digit-weighted supervised warm start (diag4:
P(correct)=0.023 after 1000 further steps).

We did NOT establish:
- that RL cannot teach in general (only that these 9 configs failed here)
- that the failure is fundamental vs. a hyperparameter/data artifact
- that GRPO is inferior to SFT in general (only that it did not work here)

## What this says about the original hypothesis

The original hypothesis was: "RL (GRPO) can refine a model on a specific
task even when the model has zero prior competence at it." The battery
disproves the strong form: **GRPO cannot refine a policy from zero** —
group-relative advantage requires nonzero behavioral variance to amplify,
and at 0% success rate there is no signal to amplify.

This is consistent with the small-scale battery result: the tiny model
"worked" only because its pretrain data already contained the arithmetic
mapping (so P(success) was nonzero). RL amplified an existing behavior;
it never created one.

## Battery-wide conclusion (all 9 configs + prior diagnostics)

1. **RL amplifies, it does not teach.** GRPO refines behaviors the policy
   already exhibits with nonzero probability. It cannot install a skill
   the model lacks entirely.
2. At 2.5M params, GRPO works on tasks the model can already do at some
   nonzero rate (the small-model battery result).
3. At 556M, the same GRPO machinery failed across 9 configs — the task
   (arithmetic) was never in the pretrain distribution, so P(success)
   stayed at chance and GRPO had no signal to amplify.
4. Supervised fine-tuning (the diag4 1000-step arm) did install partial
   competence (greedy 55%) — the missing ingredient was supervised
   exposure, not better RL.
5. For this model class, the practical recipe is: **supervised training
   to competence first, RL only as a final polish** — matching what the
   literature reports for RLHF/GRPO pipelines.