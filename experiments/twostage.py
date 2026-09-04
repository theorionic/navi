"""Two-stage upcycling on the 262k-fact task.

Stage 1: train dense-only backbone (fresh — the pruned session took the last one).
Stage 2: insert the Pool (memory layers replace every other FFN), copy all dense
         params, keep Pool fresh, continue training the SAME budget.

Verdict rule: if zeroing the Pool after stage 2 collapses accuracy (vs the
untouched eval), knowledge now flows through the Pool -> the freeloading
equilibrium is broken and the upcycling recipe works.
"""

import os
import sys
import time

sys.path.insert(0, os.environ.get("NAVI_ROOT", "/content"))

import jax
import jax.numpy as jnp

import attribution as A
from navi.config import ModelConfig, TrainConfig
from navi.model import Navi
from navi.train import init_params

DENSE = ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=0)
POOLED = ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2)
CFG_T = TrainConfig(total_steps=A.STEPS, batch_size=A.BS, seq_len=A.SEQ, log_every=300)


def is_dense(kp) -> bool:
    return "values" not in jax.tree_util.keystr(kp)


def transfer_to_pooled(dense_params, pooled_model):
    """Stage-2 init: every dense param copied 1:1 by path; Pool arrays fresh.
    Dense and pooled trees differ structurally (mem replaces ff on even
    layers), so match leaves by path string, not by tree_map."""
    fresh = init_params(pooled_model, A.SEQ, jax.random.PRNGKey(123))
    dense_leaves = {
        jax.tree_util.keystr(kp): v
        for kp, v in jax.tree_util.tree_flatten_with_path(dense_params)[0]
    }
    merged_leaves = []
    for kp, f in jax.tree_util.tree_flatten_with_path(fresh)[0]:
        ks = jax.tree_util.keystr(kp)
        if ks in dense_leaves and dense_leaves[ks].shape == f.shape:
            merged_leaves.append(dense_leaves[ks])
        else:
            merged_leaves.append(f)
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(fresh), merged_leaves
    )


def main():
    print("== two-stage upcycling ==", flush=True)

    # ---- stage 1: dense-only
    print("[s1] training dense-only backbone", flush=True)
    dense_model = Navi(DENSE, A.CFG_MEM)
    dense_params = A.train(dense_model, CFG_T, tag="s1-dense")
    acc1 = A.evaluate(dense_model, dense_params, "s1-dense", "none")
    print(f"[s1] RESULT dense-only acc {acc1:.4f}", flush=True)

    # ---- stage 2: insert pool, continue training
    print("[s2] inserting Pool and continuing training", flush=True)
    pooled_model = Navi(POOLED, A.CFG_MEM)
    params2 = transfer_to_pooled(dense_params, pooled_model)
    params2 = A.train(pooled_model, CFG_T, tag="s2-pool", params=params2)

    r = {}
    for label, p in (("none", params2), ("zero", A.zero_pool(params2)),
                     ("shuffle", A.shuffle_pool(params2, jax.random.PRNGKey(4242)))):
        r[label] = A.evaluate(pooled_model, p, "s2-pool", label)

    print("== TWO-STAGE SUMMARY ==", flush=True)
    print(f"stage1 dense-only : {acc1:.4f}", flush=True)
    print("stage2 pooled     : " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)


if __name__ == "__main__":
    main()