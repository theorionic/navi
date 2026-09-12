"""Scale check: does router entropy-lb rescue recall at 16.8M slots?

The original VERDICT failure (16.8M facts, 2-4 exposures/fact): recall flat
at chance under every intervention -- learned placement, curriculum
warm-start, hash placement. Sweep A on real text added the exposure wall:
4.19M slots at 66M tokens = 0.5-1.3% slots touched, gini 0.998-0.999,
one slot per layer taking ~100% of traffic. Recruitment collapses as
slots/token grows.

The lb battery (lb_battery.md) showed the entropy-lb aux breaks that
pattern at 4.19M-slot real-text scale: mem_6 distinct slots 18.6k ->
189k (10x), top100 share 0.32 -> 0.059, gini 1.000 -> 0.962, alive 8.5%
-> 66% -- AND val bpc improved 0.15. Hypothesis: spread = gradient
coverage = recruitment. If the same lever lifts the synthetic-fact
recall ceiling at 16.8M slots, the original VERDICT failure was (partly)
the router, not just exposure budget.

Protocol (both arms identical except lb_weight; c1=c2=512 -> 4 slots
planes x 262k = 1.05M/class, 4.19M total; same as VERDICT P4/P4MID but
4x the slots of the 65k anchor):
  lb-64:    NAVI_NONCE=64   65,536-fact space  ~500 exp/fact  4000 steps
  lb-1024:  NAVI_NONCE=1024 16.78M-fact space  ~2 exp/fact    4000 steps
Reference numbers from VERDICT.md (no lb):
  lb-64 analog  (P4,  65k):  fresh 100%  (anchor)
  lb-1024 analog (P4/P4MID): fresh 0.38-0.42% = chance
Success: lb-1024 fresh recall meaningfully above 0.4% chance.

Env: NAVI_LB (default 0.01), NAVI_STEPS (4000), NAVI_NONCE set per arm
by the runner script; eval seeds fixed, sabotage controls included.
"""
import sys
import time

sys.path.insert(0, "/kaggle/working/code")
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

import navi.data as D
from navi.config import MemoryConfig, ModelConfig, TrainConfig
from navi.model import Navi
from navi.train import init_params

TAG = "lb-scale"
T0 = time.time()
STEPS = int(os.environ.get("NAVI_STEPS", "4000"))
PBS = 512
SEQ = 64
LR = 3e-3
LB = float(os.environ.get("NAVI_LB", "0.01"))
EVAL_SEED = 777
N_EVAL = 16

# 512x512x4 = 1.05M slots/class; 4 planes -> 4.19M total (VERDICT P4MID)
CFG_MEM = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=16, n_classes=4,
                       score_temp=4.0, lb_weight=LB)
MODEL = ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2,
                    vocab_size=D.VOCAB)

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))


def shard_tree(tree):
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


def log(m):
    print(f"[{TAG}] {m}", flush=True)


def is_mem(kp):
    ks = jax.tree_util.keystr(kp)
    return "values" in ks or "/k1" in ks or "/k2" in ks


def make_tx(params):
    core = optax.adamw(
        optax.schedules.warmup_cosine_decay_schedule(
            init_value=LR * 0.02, peak_value=LR,
            decay_steps=STEPS, warmup_steps=200),
        b1=0.9, b2=0.95, weight_decay=0.01)
    mem = optax.adamw(
        optax.schedules.warmup_cosine_decay_schedule(
            init_value=LR * 0.02, peak_value=LR * 10.0,
            decay_steps=STEPS, warmup_steps=200),
        b1=0.9, b2=0.95, weight_decay=0.01)
    labels = jax.tree_util.tree_map_with_path(
        lambda kp, x: "mem" if is_mem(kp) else "core", params)
    return optax.multi_transform({"core": core, "mem": mem}, labels)


def zero_pool(p):
    return jax.tree_util.tree_map_with_path(
        lambda kp, x: jnp.zeros_like(x) if "values" in jax.tree_util.keystr(kp) else x, p)


def shuffle_pool(p, rng):
    def perm(kp, x):
        if "values" in jax.tree_util.keystr(kp):
            n = x.shape[-2]
            return x[..., jax.random.permutation(jax.random.fold_in(rng, n), n), :]
        return x
    return jax.tree_util.tree_map_with_path(perm, p)


def main():
    log(f"== lb scale check: nonce={D.N_NONCE} facts={D.N_KEYS * D.N_NONCE**2:,} "
        f"LB={LB} steps={STEPS} bs={PBS} cores={jax.device_count()} ==")
    log(f"slots: {CFG_MEM.n_classes} x {CFG_MEM.c1} x {CFG_MEM.c2} = "
        f"{CFG_MEM.n_classes * CFG_MEM.c1 * CFG_MEM.c2:,}")

    model = Navi(MODEL, CFG_MEM)
    params = init_params(model, SEQ, jax.random.PRNGKey(0))
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    msz = sum(x.size for k, x in flat if "mem" in jax.tree_util.keystr(k))
    log(f"params: total {sum(x.size for _, x in flat):,} mem {msz:,}")

    p = shard_tree(params)
    tx = make_tx(params)
    o = shard_tree(tx.init(params))
    del params
    import gc
    gc.collect()

    model_ra = Navi(MODEL, CFG_MEM, return_aux=True)

    # temp must be a closed Python float under grad (flax 0.12 dynamic
    # kwarg issue); score_temp=4.0 is baked in CFG_MEM -- pass nothing.
    @jax.jit
    def step(p, o, b):
        def loss(pp):
            logits, _aux, lb = model_ra.apply(pp, b[:, :-1], train=True)
            ce = optax.softmax_cross_entropy_with_integer_labels(
                logits, b[:, 1:]).mean()
            return ce + LB * lb
        g = jax.grad(loss)(p)
        u, o2 = tx.update(g, o, p)
        return optax.apply_updates(p, u), o2

    @jax.jit
    def loss_scalar(p, b):
        logits = model.apply(p, b[:, :-1], train=False)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, b[:, 1:]).mean()

    @jax.jit
    def acc(pp, b):
        lg = model.apply(pp, b[:, :-1], train=False)
        return D.fact_recall_acc(lg, b[:, 1:])

    def ev(pp, seed_base):
        s = 0.0
        for i in range(N_EVAL):
            r = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), seed_base + i)
            b = D.sample_batch(r, PBS, SEQ)
            s += float(acc(pp, jax.device_put(b, BATCH)))
        return s / N_EVAL

    # routing battery inline: distinct-slot stats over 8 fresh batches
    @jax.jit
    def ev_aux(pp, ids):
        _, aux, _lb = model_ra.apply(pp, ids, train=False)
        return aux

    def routing_stats(pp):
        n_cls = CFG_MEM.n_classes
        n_sc = CFG_MEM.c1 * CFG_MEM.c2
        traffic = {k: np.zeros(n_cls * n_sc, dtype=np.int64)
                   for k in ("mem_0", "mem_2", "mem_4", "mem_6")}
        for i in range(8):
            r = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), 50_000 + i)
            b = D.sample_batch(r, 64, SEQ)
            aux = model_ra.apply(pp, jax.device_put(b, BATCH),
                                 train=False)[1]
            for k in traffic:
                s = np.asarray(aux[k]).reshape(64, SEQ, n_cls, -1)
                cls = np.arange(n_cls)[None, None, :, None]
                glob = (cls * n_sc + s).ravel()
                traffic[k] += np.bincount(glob, minlength=n_cls * n_sc)
        for k, tc in traffic.items():
            top100 = float(np.sort(tc)[::-1][:100].sum() / max(1, tc.sum()))
            log(f"ROUTING {k}: touched {int((tc > 0).sum()):,} "
                f"({(tc > 0).mean() * 100:.3f}%) top100 {top100:.3f}")

    t0 = time.time()
    for i in range(STEPS):
        rng = jax.random.fold_in(jax.random.PRNGKey(9), i)
        b = jax.device_put(D.sample_batch(rng, PBS, SEQ), BATCH)
        p, o = step(p, o, b)
        if i % 250 == 0 or i == STEPS - 1:
            l = float(loss_scalar(p, b))
            log(f"step {i:5d} loss {l:.4f} ({(time.time()-t0)/(i+1):.3f}s/it)")

    # fresh draws: seed_base 4_000_000 is outside every training range
    fr = ev(p, 4_000_000)
    log(f"EVAL fresh {fr:.4f}")
    zr = ev(zero_pool(p), 4_000_000)
    log(f"EVAL zero-fresh {zr:.4f}")
    sr = ev(shuffle_pool(p, jax.random.PRNGKey(4242)), 4_000_000)
    log(f"EVAL shuffle-fresh {sr:.4f}")
    routing_stats(p)

    with open("/kaggle/working/experiments/ckpt_lb_scale.pkl", "wb") as f:
        pickle.dump(jax.device_get(p), f)
    log(f"RESULT nonce={D.N_NONCE} fresh={fr:.4f} zero={zr:.4f} "
        f"shuffle={sr:.4f}")
    log(f"== DONE in {time.time()-T0:.0f}s ==")


if __name__ == "__main__":
    main()