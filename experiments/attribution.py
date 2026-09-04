"""Pool-attribution matrix: does the Pool store the knowledge, or do the
dense layers memorize everything?

Arms (all on the 262,144-fact dataset):
  D  dense-only    memory_every=0                      -> capacity ceiling of dense path
  A  pool          memory_every=2, trained normally    -> eval normal / zeroed / shuffled
  B  frozen-pool   same arch, Pool values get NO grad  -> can dense alone reach 100% in
                                                           the Pool-arm architecture?
  C  pool-only     memory_every=1 (zero FFN blocks)    -> knowledge has nowhere else to live

Sabotage at eval time only:
  zero    values := 0          -> memory read contributes nothing (residual untouched)
  shuffle values := permuted    -> right slots, wrong contents
Chance accuracy = 1/256 = 0.39%.
"""

import os
import sys
import time

sys.path.insert(0, os.environ.get("NAVI_ROOT", "/content"))


import jax
import jax.numpy as jnp
import optax

from navi.config import MemoryConfig, ModelConfig, TrainConfig
from navi.data import fact_recall_acc, sample_batch
from navi.model import Navi
from navi.train import init_params, make_tx
STEPS = 1500
BS = 64
SEQ = 64
EVAL_BATCH = 256
N_EVAL = 8
EVAL_SEED = 555
CFG_MEM = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4)

if "--smoke" in sys.argv:
    # CPU smoke: shrink slot space so the Pool arm is not 300s/it
    STEPS = 30
    EVAL_BATCH = 32
    N_EVAL = 2
    CFG_MEM = MemoryConfig(c1=64, c2=64, cand_k=8, side_top=64, n_classes=4)


def is_values(kp) -> bool:
    return "values" in jax.tree_util.keystr(kp)


def pool_touched(p):
    """(n_pool_value_arrays, total pool value params) for logging."""
    n = sz = 0
    for kp, leaf in jax.tree_util.tree_flatten_with_path(p)[0]:
        if is_values(kp):
            n += 1
            sz += leaf.size
    return n, sz


def zero_pool(p):
    return jax.tree_util.tree_map_with_path(
        lambda kp, x: jnp.zeros_like(x) if is_values(kp) else x, p
    )


def shuffle_pool(p, rng):
    """Permute value rows within each class; slot ids stay valid, contents wrong."""
    out = []
    parts = jax.tree_util.tree_flatten_with_path(p)[0]
    for i, (kp, x) in enumerate(parts):
        if is_values(kp):
            r = jax.random.fold_in(rng, i)
            idx = jax.random.permutation(r, x.shape[1])
            out.append(x[:, idx, :])
        else:
            out.append(x)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(p), out)


def loss_fn(model, p, b):
    logits = model.apply(p, b[:, :-1], train=True)
    return optax.softmax_cross_entropy_with_integer_labels(logits, b[:, 1:]).mean()


def train(model, cfg_t, train_values=True, tag="", params=None):
    rng = jax.random.PRNGKey(cfg_t.seed)
    if params is None:
        params = init_params(model, cfg_t.seq_len, jax.random.fold_in(rng, 1))
    tx = make_tx(cfg_t)
    opt = tx.init(params)

    @jax.jit
    def step(p, o, b):
        g = jax.grad(lambda pp, bb: loss_fn(model, pp, bb))(p, b)
        if not train_values:
            # freeze Pool contents: no gradient may touch `values`
            g = jax.tree_util.tree_map_with_path(
                lambda kp, x: jnp.zeros_like(x) if is_values(kp) else x, g
            )
        u, o2 = tx.update(g, o, p)
        return optax.apply_updates(p, u), o2

    t0 = time.time()
    for i in range(cfg_t.total_steps):
        rng, d = jax.random.split(rng)
        b = jax.device_put(sample_batch(d, cfg_t.batch_size, cfg_t.seq_len))
        params, opt = step(params, opt, b)
        if i % 300 == 0 or i == cfg_t.total_steps - 1:
            l = float(loss_fn(model, params, b))
            print(f"[{tag}] step {i:5d} loss {l:.4f} ({(time.time()-t0)/(i+1):.3f}s/it)", flush=True)
    return params


def evaluate(model, params, tag, label, n_batches=N_EVAL):
    @jax.jit
    def eval_step(p, b):
        return fact_recall_acc(model.apply(p, b[:, :-1], train=False), b[:, 1:])

    accs = []
    for i in range(n_batches):
        b = jax.device_put(
            sample_batch(
                jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), i),
                EVAL_BATCH, SEQ,
            )
        )
        accs.append(float(eval_step(params, b)))
    acc = sum(accs) / len(accs)
    print(f"[{tag}] EVAL {label:10s} acc {acc:.4f}", flush=True)
    return acc


def run_arm(name, cfg_m, tag, train_values=True, sabotages=("none",)):
    model = Navi(cfg_m, CFG_MEM)
    cfg_t = TrainConfig(total_steps=STEPS, batch_size=BS, seq_len=SEQ, log_every=300)
    n, sz = pool_touched(init_params(model, SEQ, jax.random.PRNGKey(0)))
    print(f"[{tag}] == arm {name}: pool arrays {n}, pool params {sz:,}", flush=True)
    params = train(model, cfg_t, train_values=train_values, tag=tag)
    accs = {}
    for s in sabotages:
        p = params
        if s == "zero":
            p = zero_pool(params)
        elif s == "shuffle":
            p = shuffle_pool(params, jax.random.PRNGKey(4242))
        accs[s] = evaluate(model, p, tag, s)
    print(f"[{tag}] RESULT {name}: " + " ".join(f"{k}={v:.4f}" for k, v in accs.items()), flush=True)
    return accs


def main():
    smoke = "--smoke" in sys.argv
    label = "SMOKE (64x64 slots)" if smoke else "FULL (512x512 slots)"
    print("== Pool attribution matrix ==", flush=True)
    print(f"config: d256/8L, memory {label}, {STEPS} steps, 262k facts", flush=True)

    r = {}
    # D: dense-only reference — the freeload ceiling
    r["dense"] = run_arm(
        "dense-only", ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=0),
        "dense", sabotages=("none",),
    )
    # A: normal Pool model, sabotage at eval
    r["pool"] = run_arm(
        "pool", ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2),
        "pool", sabotages=("none", "zero", "shuffle"),
    )
    # B: frozen-random Pool, same architecture
    r["frozen"] = run_arm(
        "frozen-pool", ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2),
        "frozen", train_values=False, sabotages=("none",),
    )
    # C: Pool-only, no FFN anywhere
    r["ponly"] = run_arm(
        "pool-only", ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=1),
        "ponly", sabotages=("none", "zero"),
    )

    print("== SUMMARY ==", flush=True)
    for arm, accs in r.items():
        print(f"{arm:12s}: " + " ".join(f"{k}={v:.4f}" for k, v in accs.items()), flush=True)


if __name__ == "__main__":
    main()