# DOC-03: Frontier-expansion (noisy query) test on v5e-8

_Date: 2026-09-22. Relay TPU v5e-8 (8 chips), JAX 0.11.2. 3000 steps, full 512x512x4 pool geometry, eval every 500 steps. Files: tmp/noisy_test.py, tmp/noisy_results.json_

## Question

Can anything make the used slot set GROW with training (the 'frontier keeps
exploring' property the user asked for), rather than contract to an
equilibrium as entropy/eps/temp all do?

## Mechanism tested: query noise (MoE noisy-router analogue)

Per-step iid noise on the w_q query kernels (relative scale 0.1x and 0.3x
of kernel std), training only. Noise perturbs scores -> slots near the
top-k threshold win sometimes -> new slots receive value gradient.
Eval always with clean params (noise is train-only).

## Results (mem_0 coverage trajectory, steps 0/500/1000/.../2999)

| condition | trajectory | final val |
|---|---|---|
| base_lb (entropy only) | .0389 -> .0040 -> .0035 -> .0040 -> .0017 -> .0018 -> .0021 | 5.4010 |
| noise01_lb (0.1x) | .0388 -> .0036 -> .0061 -> .0020 -> .0024 -> .0027 -> .0022 | 5.4009 |
| noise03_lb (0.3x) | .0392 -> .0033 -> .0036 -> .0018 -> .0022 -> .0023 -> .0016 | 5.4011 |

## Findings

1. Query noise at 0.1x gave a modest boost mid-run (step 1000: 0.61% vs
   0.35% base, ~1.7x) but did NOT change the qualitative behavior: still
   contracts after the early phase, same equilibrium magnitude.
2. 0.3x noise was too strong - it blurred the scores enough to hurt
   (final coverage below base).
3. Val loss identical across all conditions (5.401) - same as every
   prior test; at this scale coverage differences never reach val.

## Why the frontier still freezes

The noise perturbs WHICH slots win, but the CE loss still punishes every
read of a garbage-value slot, so Adam re-sharpens keys back onto proven
slots between noise draws. Noise alone loses the race against gradient
sharpening. For a durable frontier you need the values on freshly-touched
slots to become useful FAST (mem_lr_mult on values) or you need to remove
the CE punishment for exploration reads (epsilon-mixed reads did nothing
because the floor is applied AFTER selection; it must be applied to the
selection itself, i.e. forced random slots in the top-k - the true
epsilon-greedy). Not yet implemented.

## Status of the utilization question after all tests

- Entropy loss (lb_weight=0.01): 10-30x over collapsed baseline, contracts
  to stable low-single-digit %. VERIFIED on 3 scales.
- lb_eps: no-op alone. VERIFIED twice.
- Temp annealing: no effect. VERIFIED.
- Query noise 0.1-0.3x: transient +1.7x, same equilibrium. VERIFIED.
- Remaining untested: true epsilon-greedy selection (force r slots of the
  top-k to be random candidate-grid slots during training) + value-LR
  boost so new slots consolidate before the router gives up on them.
- Structural alternative: hash_slots=True (deterministic full-reach
  addressing, zero collapse, 7.5x faster) - in codebase, validated equal
  val at this scale.

## Verdict

With learned routing at this pool geometry, utilization stays a low
single-digit % and is data-limited, not mechanism-limited: 208M tokens of
training data against 1.05M slots/class cannot fill the pool regardless
of router cleverness. The pool is oversized for the token budget.
Right-size the pool (c1=c2=256 -> 65k slots/class x 4) for the current
budget, or accept the equilibrium and scale tokens. The next mechanism
to try (epsilon-greedy selection) is written up above for a future run.
