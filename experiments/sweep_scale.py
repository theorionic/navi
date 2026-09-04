"""Scaling sweeps on enwik8: does a bigger Pool buy knowledge?

Sweep A (capacity): pool slots in {0.5M, 1M, 2M, 4M} per class at fixed
tokens - bpc should drop with slots if capacity scales.
Sweep B (exposure): fixed pool, tokens in {262M, 524M} - slot recruitment
should rise with tokens if exposure is the binding constraint.

Env knobs: NAVI_C1 (subclass subkeys, default 512), NAVI_STEPS,
NAVI_TAG (arm label). One process = one arm; the launcher script chains
them. Checkpoints every 500 steps; slot stats at the end.
"""
import sys
sys.path.insert(0, "/kaggle/working")
import os, time, pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from navi.model import Navi
from navi.config import TrainConfig
from navi.train import init_params
from dataclasses import replace as _dc_replace
import sweep_real as S

C1 = int(os.environ.get("NAVI_C1", "512"))
TAG = os.environ.get("NAVI_TAG", f"C{C1}")
STEPS = int(os.environ.get("NAVI_STEPS", "4000"))
BS = 256  # global batch; sharded across the 8 TPU cores like sweep9

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))

def shard_tree(tree):
    # Slot tables (values) are sharded on their leading (slot) axis; the
    # rest of the model is replicated. Without this, 4M+ slot pools cannot
    # fit grads+moments on one core (the 9.44G > 8.59G OOM).
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)

def main():
    train_split, val_split = S.load_enwik8()
    mem_cfg = S.MemoryConfig(c1=C1, c2=C1, cand_k=8, side_top=64, n_classes=4,
                             score_temp=4.0)
    cfg_m = S.ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2,
                          vocab_size=S.VOCAB)
    print(f"== SCALE arm {TAG}: c1=c2={C1} slots/class={4*C1*C1:,} "
          f"steps={STEPS} BS={BS} tokens={STEPS*BS*S.SEQ:,} ==", flush=True)
    model = Navi(cfg_m, mem_cfg)
    cfg_t = TrainConfig(total_steps=STEPS, batch_size=BS, seq_len=S.SEQ,
                        warmup_steps=min(200, STEPS // 10), log_every=250,
                        mem_lr_mult=10.0)
    p0 = init_params(model, S.SEQ, jax.random.PRNGKey(0))
    flat = jax.tree_util.tree_flatten_with_path(p0)[0]
    sz = sum(x.size for _, x in flat)
    msz = sum(x.size for k, x in flat if "mem" in jax.tree_util.keystr(k))
    print(f"[{TAG}] == arm pool: params {sz:,} (mem {msz:,})", flush=True)
    p = shard_tree(p0)
    # Lion: momentum only (opt state = 1x params, AdamW is 2x) - the 1B+
    # pool OOMs HBM with replicated m/v. LR 3-10x smaller than AdamW's 3e-3.
    tx = optax.lion(learning_rate=3e-4, b1=0.9, b2=0.99, weight_decay=0.03)
    o = shard_tree(tx.init(p0))

    @jax.jit
    def step(pp, oo, ids, tg):
        g = jax.grad(lambda q, a, t: S.loss_fn(model, q, a, t))(pp, ids, tg)
        # mem params train 10x faster (Meta recipe), core gets weight decay 0
        g = jax.tree_util.tree_map_with_path(
            lambda kp, x: x * 10.0 if "mem" in jax.tree_util.keystr(kp) else x, g)
        u, oo2 = tx.update(g, oo, pp)
        return optax.apply_updates(pp, u), oo2

    # resume support: same pickle format as sweep_real
    ckpt = os.path.expanduser(f"~/experiments/ckpt_{TAG}.pkl")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    rng = jax.random.PRNGKey(0)
    start = 0
    if ckpt and os.path.exists(ckpt) and os.environ.get("NAVI_RESUME") == "1":
        with open(ckpt, "rb") as f:
            state = pickle.load(f)
        p, o, rng, start = state["params"], state["opt"], state["rng"], state["step"] + 1
        print(f"[{TAG}] RESUMED at step {start}", flush=True)

    losses = []
    t0 = time.time()
    for i in range(start, STEPS):
        rng, d = jax.random.split(rng)
        win = S.batch_iter(train_split, d, BS, S.SEQ)
        ids = jax.device_put(win[:, :-1], BATCH)
        tg = jax.device_put(win[:, 1:], BATCH)
        p, o = step(p, o, ids, tg)
        if i % 250 == 0 or i == STEPS - 1:
            l = float(S.loss_fn(model, p, ids, tg))
            losses.append(l)
            print(f"[{TAG}] step {i:5d} loss {l:.4f} bpc {l/np.log(2):.4f} "
                  f"({(time.time()-t0)/(i+1-start+1e-9):.3f}s/it)", flush=True)
        if ckpt and (i % 500 == 499 or i == STEPS - 1):
            # params-only: p+m+v for 1B+ pools blows the 20GB kernel disk
            # (13GB ckpt killed both C1024 attempts at step ~499/999)
            with open(ckpt + ".tmp", "wb") as f:
                pickle.dump({"params": p, "opt": None, "rng": rng,
                             "step": i, "losses": losses}, f)
            os.replace(ckpt + ".tmp", ckpt)
            print(f"[{TAG}] CKPT saved at step {i}", flush=True)

    res = {"bpc": S.evaluate(model, p, TAG, "none", val_split)}
    pz = S.zero_pool(p)
    res["zero"] = S.evaluate(model, pz, TAG, "zero", val_split)
    print(f"[{TAG}] RESULT pool: " +
          " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
    stats = S.slot_stats(cfg_m, mem_cfg, p, val_split)
    for k, (d, g, t) in stats.items():
        print(f"[{TAG}] slots {k}: {d} gini={g:.3f} top1%={t:.3f}", flush=True)
    with open(os.path.expanduser(f"~/experiments/scale_{TAG}.txt"), "w") as f:
        f.write(f"{TAG} c1={C1} steps={STEPS} " +
                " ".join(f"{k}={v:.4f}" for k, v in res.items()) + "\n")
        for k, (d, g, t) in stats.items():
            f.write(f"slots {k}: {d} gini={g:.3f} top1%={t:.3f}\n")

if __name__ == "__main__":
    main()