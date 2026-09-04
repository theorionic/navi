"""Synthetic fact-recall dataset: the minimal test the Pool must pass.

Each sequence encodes `key n1 n2 value` triples with 16 shared keys.
Predicting a value token requires storing (key, n1, n2) -> value
associations in weights — a pure knowledge-capacity task, isomorphic to
factual QA at real scale.

The value of a triple is FIXED: drawn once from a seeded table. Without
a fixed map there is nothing to memorize and recall ceilings at chance
(1/256) — the bug that silently invalidated the first 4M-scale runs.

Fact space: 16 x N_NONCE x N_NONCE triples (16.78M at N_NONCE=1024).
NAVI_NONCE=64 shrinks to the original 65,536-fact regime used for the
262k-era anchor results. Eval = closed-book recall: fresh random draws
from the same fact space the model trained on, scored on val positions.
"""

import os

import jax
import numpy as np

N_KEYS = 16
N_NONCE = int(os.environ.get("NAVI_NONCE", "1024"))
N_VALS = 256
KEY0 = 3
NONCE0 = KEY0 + N_KEYS
VAL0 = NONCE0 + N_NONCE
VOCAB = VAL0 + N_VALS  # 3 + 16 + N_NONCE + 256
_VAL_SEED = 1234


def _val_table() -> np.ndarray:
    rng = np.random.default_rng(_VAL_SEED)
    return rng.integers(0, N_VALS, size=(N_KEYS, N_NONCE, N_NONCE), dtype=np.int16)


_VAL = _val_table()


def _draw(rng: jax.Array, batch: int, seq_len: int) -> tuple[np.ndarray, ...]:
    n_triples = max(1, seq_len // 4)
    keys = np.asarray(jax.random.randint(rng, (batch,), 0, N_KEYS))[:, None]
    # split: same rng+shape twice would make n2 == n1 elementwise
    r1, r2 = jax.random.split(rng)
    n1 = np.asarray(jax.random.randint(r1, (batch, n_triples), 0, N_NONCE))
    n2 = np.asarray(jax.random.randint(r2, (batch, n_triples), 0, N_NONCE))
    val = _VAL[keys, n1, n2]
    return keys, n1, n2, val, n_triples

def sample_batch(rng: jax.Array, batch: int, seq_len: int, train: bool = True) -> np.ndarray:
    keys, n1, n2, val, n_triples = _draw(rng, batch, seq_len)
    seqs = np.full((batch, n_triples * 4), 2, dtype=np.int32)
    for i in range(n_triples):
        seqs[:, 4 * i + 0] = np.asarray(keys[:, 0]) + KEY0
        seqs[:, 4 * i + 1] = n1[:, i] + NONCE0
        seqs[:, 4 * i + 2] = n2[:, i] + NONCE0
        seqs[:, 4 * i + 3] = val[:, i] + VAL0
    return seqs


# closed-book eval draws from the same fact space; separate name keeps
# the sweep runners' imports stable
sample_eval_batch = sample_batch


def fact_recall_acc(logits, targets):
    """Accuracy on value-token predictions only.

    Value tokens sit at positions 4i+3; causal logits at position 4i+2
    predict them. logits/targets here are already the shifted pair
    (logits over s[:-1], targets = s[1:]), so both are indexed at 4i+2.
    """
    pred_vals = logits[:, 2::4].argmax(-1)
    tgt_vals = targets[:, 2::4]
    return (pred_vals == tgt_vals).mean()