"""Staged fact-space expansion with rehearsal: does knowledge survive growth?

Second exposure test. lb_scale.py proved uniform sampling at 16.8M facts
(2 exposures/fact) recalls at chance. curriculum.py (no lb) staged
65k -> 1M -> 16.8M WITHOUT rehearsal and S3 was not > chance either.
The open hypothesis (lb_scale.md): warm-start + in-stage rehearsal of
old fact spaces gets per-fact exposures above the recruitment threshold
at every stage, so knowledge accumulates instead of diluting.

Protocol (all stages lb=0.01, cand_k=8, T=4, mem_lr_mult=10):
  stage 1: nonce=64   (65k facts)   1500 steps, uniform within stage
  stage 2: nonce=256  (1.05M facts) 1500 steps, 70% new + 30% stage-1 replay
  stage 3: nonce=1024 (16.8M facts) 4000 steps, 70% new + 30% replay
                                                    (50/50 stage1/stage2)
Replay re-samples (n1,n2) pairs from the OLD nonce space; values come
from the same fixed value table (prefix property: value(key,n1,n2) is
nonce-table-identical, old facts keep their answers).

Evals after each stage:
  - fresh draws from current stage's space (never-seen pairs)
  - OLD-space replay accuracy (S2: S1 facts; S3: S1 and S2 facts)
    -> forgetting measured directly
  - zero/shuffle sabotage on current-space fresh

Success: S2/S3 old-space accuracy stays near its stage-end level (no
collapse), and S3 fresh > chance if exposures compound. Even if S3
fresh stays low, S1/S3 retention is the primary readout.

Env: NAVI_LB (0.01); stage sizes/steps fixed to match curriculum.py.
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

TAG = "stage-lb"
T0 = time.time()
LR = 3e-3
LB = float(os.environ.get("NAVI_LB", "0.01"))
PBS = 512
SEQ = 64
EVAL_SEED = 777
N_EVAL = 16
N_KEYS = 16
KEY0 = 3

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))

STAGES = [
    # (nonce, steps, replay) - replay: list of (fraction, old_nonce)
    (64, 1500, []),
    (256, 1500, [(0.30, 64)]),
    (1024, 4000, [(0.15, 64), (0.15, 256)]),
]

_val_cache = {}


def val_table(nn):
    """Fixed value map per nonce space; shared prefix across spaces."""
    if nn not in _val_cache:
        # seed 1234 base table; large enough for nn=1024, sliced down
        full = np.random.default_rng(1234).integers(
            0, 256, size=(N_KEYS, 1024, 1024), dtype=np.int16)
        _val_cache[nn] = full[:, :nn, :nn]
    return _val_cache[nn]


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


def make_tx(params, steps):
    core = optax.adamw(
        optax.schedules.warmup_cosine_decay_schedule(
            init_value=LR * 0.02, peak_value=LR,
            decay_steps=steps, warmup_steps=150),
        b1=0.9, b2=0.95, weight_decay=0.01)
    mem = optax.adamw(
        optax.schedules.warmup_cosine_decay_schedule(
            init_value=LR * 0.02, peak_value=LR * 10.0,
            decay_steps=steps, warmup_steps=150),
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


def sample_stage_batch(nn, rng_seed, batch, seq_len, replay=None, step_i=0):
    """Uniform batch from nonce space `nn`; if replay given, a fraction of
    triples comes from replay spaces (list of (fraction, old_nonce))."""
    n_triples = max(1, seq_len // 4)
    r = np.random.default_rng(rng_seed * 7919 + step_i)
    vt = val_table(nn)
    keys = r.integers(0, N_KEYS, size=(batch, n_triples))
    n1 = r.integers(0, nn, size=(batch, n_triples))
    n2 = r.integers(0, nn, size=(batch, n_triples))
    val = vt[keys, n1, n2]
    if replay:
        # per-triple source choice: new vs each replay space
        fracs = [f for f, _ in replay]
        ps = [1.0 - sum(fracs)] + fracs
        nns = [nn] + [old for _, old in replay]
        src = r.choice(len(nns), size=(batch, n_triples), p=ps)
        for si, old_nn in enumerate(nns[1:], start=1):
            m = src == si
            if not m.any():
                continue
            ovt = val_table(old_nn)
            o1 = r.integers(0, old_nn, size=(batch, n_triples))
            o2 = r.integers(0, old_nn, size=(batch, n_triples))
            # vectorized mask fill
            n1 = np.where(m, o1, n1)
            n2 = np.where(m, o2, n2)
            val = np.where(m, ovt[keys, o1.clip(0, old_nn - 1), o2.clip(0, old_nn - 1)], val)
    return keys, n1, n2, val


def to_seqs(keys, n1, n2, val, nn):
    batch = keys.shape[0]
    n_triples = keys.shape[1]
    nonce0 = KEY0 + N_KEYS
    val0 = nonce0 + nn
    seqs = np.full((batch, n_triples * 4), 2, dtype=np.int32)
    seqs[:, 0::4] = keys + KEY0
    seqs[:, 1::4] = n1 + nonce0
    seqs[:, 2::4] = n2 + nonce0
    seqs[:, 3::4] = val + val0
    return seqs


def model_for(nn):
    vocab = 3 + N_KEYS + nn + 256
    cfg_mem = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=16, n_classes=4,
                           score_temp=4.0, lb_weight=LB)
    return (ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2,
                        vocab_size=vocab), cfg_mem)


def main():
    log(f"== staged expansion with rehearsal, LB={LB} cores={jax.device_count()} ==")
    # params persist across stages; vocab grows so embed/head re-init each stage
    p = None
    old_state = None
    for si, (nn, steps, replay) in enumerate(STAGES, start=1):
        cfg_m, cfg_mem = model_for(nn)
        model = Navi(cfg_m, cfg_mem)
        model_ra = Navi(cfg_m, cfg_mem, return_aux=True)
        params = init_params(model, SEQ, jax.random.PRNGKey(100 + si))
        if p is not None:
            # carry every matching-shape tensor (Pool keys/values always match)
            old_flat = {}
            jax.tree_util.tree_map_with_path(
                lambda kp, x: old_flat.__setitem__(jax.tree_util.keystr(kp), x), old_state["params"])
            carried, reinited = [], []
            def carry(kp, new_x):
                ks = jax.tree_util.keystr(kp)
                ox = old_flat.get(ks)
                if ox is not None and ox.shape == new_x.shape:
                    carried.append(ks)
                    return ox
                reinited.append(ks)
                return new_x
            params = jax.tree_util.tree_map_with_path(carry, params)
            log(f"stage {si}: warm-start {len(carried)} carried, {len(reinited)} re-inited")
            assert not any("values" in k or "/k1" in k or "/k2" in k for k in reinited), \
                "Pool params lost in warm-start!"
        p = shard_tree(params)
        tx = make_tx(params, steps)
        o = shard_tree(tx.init(params))
        if old_state is not None and old_state.get("opt") is not None:
            # carry matching optimizer state too; shapes match for carried params
            of = {}
            jax.tree_util.tree_map_with_path(
                lambda kp, x: of.__setitem__(jax.tree_util.keystr(kp), x), old_state["opt"])
            def carry_o(kp, new_x):
                ks = jax.tree_util.keystr(kp)
                ox = of.get(ks)
                if ox is not None and ox.shape == new_x.shape:
                    return ox
                return new_x
            o = jax.tree_util.tree_map_with_path(carry_o, o)

        @jax.jit
        def step(pp, oo, b):
            def loss(q):
                logits, _aux, lb = model_ra.apply(q, b[:, :-1], train=True)
                ce = optax.softmax_cross_entropy_with_integer_labels(
                    logits, b[:, 1:]).mean()
                return ce + LB * lb
            g = jax.grad(loss)(pp)
            u, oo2 = tx.update(g, oo, pp)
            return optax.apply_updates(pp, u), oo2

        @jax.jit
        def acc(pp, ids, tg):
            lg = model.apply(pp, ids, train=False)
            pv = lg[:, 2::4].argmax(-1)
            tv = tg[:, 2::4]
            return (pv == tv).mean()

        def ev(pp, nn_ev, seed_base):
            s = 0.0
            n = 0
            for i in range(N_EVAL):
                k, a1, a2, v = sample_stage_batch(nn_ev, seed_base + i, PBS, SEQ)
                seqs = to_seqs(k, a1, a2, v, nn_ev)
                s += float(acc(pp, jax.device_put(seqs[:, :-1], BATCH),
                               jax.device_put(seqs[:, 1:], BATCH)))
                n += 1
            return s / n

        t0 = time.time()
        for i in range(steps):
            k, a1, a2, v = sample_stage_batch(nn, 1000 + si, PBS, SEQ,
                                              replay=replay, step_i=i)
            b = jax.device_put(to_seqs(k, a1, a2, v, nn), BATCH)
            p, o = step(p, o, b)
            if i % 250 == 0 or i == steps - 1:
                log(f"s{si} step {i:5d}/{steps} ({(time.time()-t0)/(i+1):.3f}s/it)")

        # evals: fresh current-space + replay old spaces + sabotage
        fr = ev(p, nn, 4_000_000 + si)
        log(f"stage {si} (nonce={nn}) EVAL fresh {fr:.4f}")
        old_spaces = [64] if si == 2 else ([64, 256] if si == 3 else [])
        for old_nn in old_spaces:
            old_acc = ev(p, old_nn, 4_000_000 + si)
            log(f"stage {si} EVAL old-space nonce={old_nn} acc {old_acc:.4f}")
        if si >= 2:
            zr = ev(zero_pool(p), nn, 4_000_000 + si)
            log(f"stage {si} EVAL zero-fresh {zr:.4f}")
        old_state = {"params": jax.device_get(p), "opt": jax.device_get(o)}
        with open(f"/kaggle/working/experiments/ckpt_stage_lb_s{si}.pkl", "wb") as f:
            pickle.dump({"params": old_state["params"], "opt": old_state["opt"],
                         "nonce": nn, "stage": si}, f)
        log(f"stage {si} ckpt saved")

    log(f"== ALL STAGES DONE in {time.time()-T0:.0f}s ==")


if __name__ == "__main__":
    main()