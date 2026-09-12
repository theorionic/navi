"""Zipf exposure test: does frequency alone write retrievable Pool knowledge?

User hypothesis (confirmed reasoning): in natural text some patterns are
always present (grammar, common facts) and those vectors get recalled
every time, while rare ones never accumulate exposure. The uniform-
sampling 16.8M run gave every fact ~2 exposures -> recall = chance
(0.38%). If the exposure story is right, a Zipf-sampled run at the SAME
budget must split: hot facts (high exposure) recall well, cold facts
(few exposures) stay at chance.

Protocol: fact space = 16 keys x 1024^2 = 16.78M facts (identical to
lb_scale.py). Sampling: draw (n1, n2) from a Zipf(a=1.5) distribution
over nonce space instead of uniform. Zipf over 1024^2 pairs: top pairs
get thousands of exposures, tail gets ~0-2. Same 4000 steps, BS 512,
SEQ 64 -> 131M token positions.

Eval buckets by exposure band (measured empirically by replaying the
sampler and counting (n1,n2) occurrences):
  HOT  : top 1% most-sampled pairs
  MID  : 1-10%
  COLD : sampled 0 times in training (fresh, never seen)
  TAIL : sampled 1-2 times
Success = HOT recall >> chance AND COLD ~ chance AND zero/shuffle
sabotage destroys HOT but not COLD (attribution).

Env: NAVI_STEPS (4000), NAVI_LB (0.01), NAVI_ZIPF_A (1.5).
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

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi
from navi.train import init_params

TAG = "zipf"
T0 = time.time()
STEPS = int(os.environ.get("NAVI_STEPS", "4000"))
PBS = 512
SEQ = 64
LR = 3e-3
LB = float(os.environ.get("NAVI_LB", "0.01"))
Z_A = float(os.environ.get("NAVI_ZIPF_A", "1.5"))
EVAL_SEED = 777
N_EVAL = 16

# same space/slots as lb_scale.py
NN = 1024
N_KEYS = 16
VOCAB = 3 + N_KEYS + NN + 256
KEY0, NONCE0, VAL0 = 3, 3 + N_KEYS, 3 + N_KEYS + NN

CFG_MEM = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=16, n_classes=4,
                       score_temp=4.0, lb_weight=LB)
MODEL = ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2,
                    vocab_size=VOCAB)

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))

# fixed value map: identical scheme to navi/data.py
_VAL = np.random.default_rng(1234).integers(
    0, 256, size=(N_KEYS, NN, NN), dtype=np.int16)


def log(m):
    print(f"[{TAG}] {m}", flush=True)


def shard_tree(tree):
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


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


# ---- Zipf pair sampler ----
# Zipf rank->freq over 1024 nonce values per side; pair rank = r1*1024 + r2
# (product of two independent Zipf marginals is a valid 2-D heavy tail).
_zrng = np.random.default_rng(555)
_ranks = _zrng.zipf(Z_A, size=(2_000_000, 2))
_ranks = np.clip(_ranks, 1, NN) - 1  # zipf support starts at 1; map to [0, 1023]
_hot_n1 = _ranks[:, 0].astype(np.int32)
_hot_n2 = _ranks[:, 1].astype(np.int32)
# empirical pair-count table for exposure buckets (same generator, big draw)
_counts = np.zeros((NN, NN), dtype=np.int32)
np.add.at(_counts, (_hot_n1, _hot_n2), 1)
# threshold ranks for buckets (pair counts of the pooled marginals)
_pct = np.percentile(_counts.ravel(), [50, 90, 99])
log(f"zipf a={Z_A}: pair count pct50={_pct[0]:.0f} pct90={_pct[1]:.0f} "
    f"pct99={_pct[2]:.0f} max={_counts.max()}")


def draw_zipf(rng_seed, batch, n_triples):
    """Batch of (key, n1, n2, val) triples with Zipf-sampled nonces."""
    r = np.random.default_rng(int(rng_seed))
    keys = r.integers(0, N_KEYS, size=batch)
    idx1 = r.integers(0, len(_hot_n1), size=(batch, n_triples))
    idx2 = r.integers(0, len(_hot_n1), size=(batch, n_triples))
    n1 = np.where(r.random((batch, n_triples)) < 0.5,
                  _hot_n1[idx1], _hot_n2[idx2])
    n2 = np.where(r.random((batch, n_triples)) < 0.5,
                  _hot_n2[idx2], _hot_n1[idx1])
    val = _VAL[keys[:, None], n1, n2]
    return keys, n1, n2, val


def sample_batch(rng_seed, batch, seq_len):
    n_triples = max(1, seq_len // 4)
    keys, n1, n2, val = draw_zipf(rng_seed, batch, n_triples)
    seqs = np.full((batch, n_triples * 4), 2, dtype=np.int32)
    for i in range(n_triples):
        seqs[:, 4 * i + 0] = keys + KEY0
        seqs[:, 4 * i + 1] = n1 + NONCE0
        seqs[:, 4 * i + 2] = n2 + NONCE0
        seqs[:, 4 * i + 3] = val[:, i] + VAL0
    return seqs


def fact_acc(logits, targets):
    pred_vals = logits[:, 2::4].argmax(-1)
    tgt_vals = targets[:, 2::4]
    return (pred_vals == tgt_vals).mean()


def main():
    log(f"== zipf exposure test: a={Z_A} steps={STEPS} bs={PBS} "
        f"cores={jax.device_count()} ==")
    model = Navi(MODEL, CFG_MEM)
    params = init_params(model, SEQ, jax.random.PRNGKey(0))
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    log(f"params total {sum(x.size for _, x in flat):,}")

    p = shard_tree(params)
    tx = make_tx(params)
    o = shard_tree(tx.init(params))
    del params
    import gc
    gc.collect()

    model_ra = Navi(MODEL, CFG_MEM, return_aux=True)

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
        return fact_acc(lg, b[:, 1:])

    def ev_bucket(pp, bucket, seed_base):
        """Accuracy restricted to triples whose (n1,n2) count falls in bucket."""
        accs = []
        for i in range(N_EVAL):
            keys, n1, n2, val = draw_zipf(EVAL_SEED * 1000 + seed_base + i,
                                          PBS, SEQ // 4)
            c = _counts[n1, n2]
            if bucket == "hot":
                m = c >= _pct[2]
            elif bucket == "mid":
                m = (c >= _pct[0]) & (c < _pct[2])
            elif bucket == "tail":
                m = (c >= 1) & (c < _pct[0])
            else:  # cold: never sampled in training draws
                m = c == 0
            if m.sum() < 8:
                continue
            seqs = np.full((int(m.sum()), SEQ), 2, dtype=np.int32)
            seqs[:, 0::4] = keys[m][:, None] + KEY0
            seqs[:, 1::4] = n1[m][:, None] + NONCE0
            seqs[:, 2::4] = n2[m][:, None] + NONCE0
            seqs[:, 3::4] = val[m][:, None] + VAL0
            lg = model.apply(pp, jax.device_put(seqs[:, :-1], BATCH), train=False)
            a = fact_acc(lg, jax.device_put(seqs[:, 1:], BATCH))
            accs.append(float(a))
        return sum(accs) / max(1, len(accs))

    t0 = time.time()
    for i in range(STEPS):
        b = jax.device_put(sample_batch(i, PBS, SEQ), BATCH)
        p, o = step(p, o, b)
        if i % 250 == 0 or i == STEPS - 1:
            l = float(loss_scalar(p, b))
            log(f"step {i:5d} loss {l:.4f} ({(time.time()-t0)/(i+1):.3f}s/it)")

    log("EVAL by exposure bucket (chance 0.0039):")
    for name in ("hot", "mid", "tail", "cold"):
        log(f"  {name:5s}: {ev_bucket(p, name, 4_000_000):.4f}")

    zh = ev_bucket(zero_pool(p), "hot", 4_000_000)
    zc = ev_bucket(zero_pool(p), "cold", 4_000_000)
    log(f"  zero  hot {zh:.4f} cold {zc:.4f}")
    sh = ev_bucket(shuffle_pool(p, jax.random.PRNGKey(4242)), "hot", 4_000_000)
    sc = ev_bucket(shuffle_pool(p, jax.random.PRNGKey(4242)), "cold", 4_000_000)
    log(f"  shuff hot {sh:.4f} cold {sc:.4f}")

    with open("/kaggle/working/experiments/ckpt_zipf.pkl", "wb") as f:
        pickle.dump(jax.device_get(p), f)
    log(f"== DONE in {time.time()-T0:.0f}s ==")


if __name__ == "__main__":
    main()