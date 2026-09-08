"""Extra validation battery for the completed 500m run -- the tests the
4M-scale VERDICT battery standardized that val500m.py did not cover:

1. SHUFFLE ablation (paired control for the zero ablation): permute value
   rows within each class plane (fixed seed) and re-evaluate. Zeroing
   proves the values carry magnitude; shuffling proves they carry
   *content* -- if shuffled ~= intact, the +2.19 bpc "Pool contribution"
   would be a norm artifact, not learned knowledge.
2. SLOT-TRAFFIC distribution (the one collapse mode never directly
   measured on this run): feed real val batches with return_aux=True,
   collect the routed slots, report distinct-slot count, top-100 share,
   and Gini per block. Routing collapse with loss compensation would show
   as top-100 share ~1.0 / Gini ~1.0.
3. VALUE UTILIZATION from the checkpoint alone: per-block row-norm stats
   (alive fraction, alive-norm mean, Gini of alive norms) to quantify the
   dead_frac=0.68-0.94 finding as a precise utilization number.

Env: NAVI_EVAL_BS (32), NAVI_EVAL_BATCHES (50), NAVI_TRAFFIC_BATCHES (10),
NAVI_VAL_DOCS (2000), NAVI_SEQ (512).

Usage: python3 /kaggle/working/navi/experiments/eval_extra.py
"""
import sys
import os
# make "navi" importable regardless of cwd (repo root = one level up)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import gc
import os as _os
import pickle
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi
from fineweb_data import BOS, FineWebFeed

TAG = "eval-extra"
T0 = time.time()
SEQ = int(_os.environ.get("NAVI_SEQ", "512"))
EVAL_BS = int(_os.environ.get("NAVI_EVAL_BS", "32"))
EVAL_BATCHES = int(_os.environ.get("NAVI_EVAL_BATCHES", "50"))
TRAFFIC_BATCHES = int(_os.environ.get("NAVI_TRAFFIC_BATCHES", "10"))
VAL_DOCS = int(_os.environ.get("NAVI_VAL_DOCS", "2000"))


def log(msg):
    print(f"[{TAG} +{time.time()-T0:6.1f}s] {msg}", flush=True)


def gini(counts):
    c = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(c)
    if c.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * c).sum()) / (n * c.sum()) - (n + 1) / n)


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


def main():
    # 1. checkpoint
    ckpt_dir = "/kaggle/working/experiments"
    ckpts = sorted(f for f in _os.listdir(ckpt_dir)
                   if re.match(r"ckpt_500m_step\d+\.pkl$", f))
    if not ckpts:
        raise SystemExit(f"no ckpt_500m_step*.pkl in {ckpt_dir}")
    ckpt_path = _os.path.join(ckpt_dir, ckpts[-1])
    log(f"stage 1/5: loading {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        st = pickle.load(f)
    p = shard_tree(st["params"])
    del st
    gc.collect()
    log("stage 1/5 done: params on mesh")

    # 2. model + fresh val stream (same readiness signal as val500m)
    log(f"stage 2/5: streaming {VAL_DOCS} fresh val docs")
    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                           score_temp=4.0)
    cfg_m = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=2,
                        vocab_size=260)
    model = Navi(cfg_m, mem_cfg)
    model_ra = Navi(cfg_m, mem_cfg, return_aux=True)

    feed = FineWebFeed(n_val_docs=VAL_DOCS)
    for _ in range(600):
        if len(feed.train_buf) > 0:  # val docs have drained through
            break
        time.sleep(2)
    val_bytes = feed.val.array()
    log(f"stage 2/5 done: val buffer {len(val_bytes)/1024/1024:.1f}MB")

    rng = np.random.default_rng(7)

    def windows(n_batches):
        outs = []
        for _ in range(n_batches):
            offs = rng.integers(0, len(val_bytes) - SEQ - 2, size=EVAL_BS)
            idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
            win = val_bytes[idx]
            win[:, 0] = BOS  # EXACT training distribution (see val500m fix)
            outs.append((win[:, :-1], win[:, 1:]))
        return outs

    @jax.jit
    def ev(pp, ids, tg):
        logits = model.apply(pp, ids, train=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()

    def bpc_of(params, batches):
        ces = [float(ev(params, jax.device_put(i, BATCH),
                        jax.device_put(t, BATCH))) for i, t in batches]
        return float(np.mean(ces)) / np.log(2)

    # 3. shuffle ablation: permute value rows WITHIN each class plane
    log("stage 3/5: shuffle ablation (value-row permutation, seed 4242)")
    batches = windows(EVAL_BATCHES)
    bpc_intact = bpc_of(p, batches)

    def shuffled_params(tree):
        def sh(kp, x):
            if "values" not in jax.tree_util.keystr(kp):
                return x
            host = np.asarray(x)  # (classes, c1*c2, dim)
            rng_s = np.random.default_rng(4242)
            out = host.copy()
            for c in range(host.shape[0]):
                out[c] = host[c][rng_s.permutation(host.shape[1])]
            return out
        return jax.tree_util.tree_map_with_path(sh, tree)

    p_shuf = shard_tree(shuffled_params(p))
    bpc_shuf = bpc_of(p_shuf, batches)
    log(f"stage 3/5 done: intact {bpc_intact:.4f} | shuffled {bpc_shuf:.4f} "
        f"-> shuffle-delta {bpc_shuf - bpc_intact:+.4f} bpc "
        f"(zero-delta was +2.1951)")

    # 4. slot-traffic distribution via live queries (routing-collapse check)
    log(f"stage 4/5: routing traffic over {TRAFFIC_BATCHES} batches "
        f"(return_aux; distinct slots / top-100 share / Gini per block)")
    tbatches = windows(TRAFFIC_BATCHES)

    @jax.jit
    def ev_aux(pp, ids):
        _, aux = model_ra.apply(pp, ids, train=False)
        return aux

    n_slots_cls = 512 * 512
    aux0 = ev_aux(p, jax.device_put(tbatches[0][0], BATCH))
    mem_blocks = sorted(aux0.keys())
    log(f"  aux blocks: {mem_blocks}")
    traffic = {k: np.zeros(4 * n_slots_cls, dtype=np.int64) for k in mem_blocks}
    for i, t in tbatches:
        aux = ev_aux(p, jax.device_put(i, BATCH))
        for k in mem_blocks:
            s = np.asarray(aux[k])  # (b, l, classes*cand_k)
            s = s.reshape(s.shape[0], s.shape[1], 4, -1)
            cls = np.arange(4)[None, None, :, None]
            glob = (cls * n_slots_cls + s).ravel()
            traffic[k] += np.bincount(glob, minlength=4 * n_slots_cls)
    log("  per-block routing (slots touched, top-100 share, Gini):")
    for k in mem_blocks:
        tc = traffic[k]
        touched = int((tc > 0).sum())
        top100 = float(np.sort(tc)[-100:].sum() / max(1, tc.sum()))
        g = gini(tc)
        log(f"  {k}: touched {touched}/{4*n_slots_cls} "
            f"({100*touched/(4*n_slots_cls):.2f}%) top100 {top100:.3f} Gini {g:.3f}")
    log("stage 4/5 done (healthy: top100 << 1.0, Gini ~ power-law not ~1.0)")

    # 5. value-utilization from checkpoint (row norms per block)
    log("stage 5/5: value row-norm utilization per memory block")
    flat = {}
    import jax.tree_util as tu
    with open(ckpt_path, "rb") as f:
        st2 = pickle.load(f)
    for kp, v in tu.tree_flatten_with_path(st2["params"])[0]:
        ks = tu.keystr(kp).replace("['", "/").replace("']", "")
        if ks.endswith("/values"):
            flat[ks] = np.asarray(v)
    init_norm = 0.02 * np.sqrt(512)
    for ks in sorted(flat):
        v = flat[ks].astype(np.float32)
        norms = np.linalg.norm(v, axis=-1).ravel()
        alive = norms > 2 * init_norm
        g = gini(norms[alive] if alive.any() else norms)
        log(f"  {ks}: alive {alive.mean()*100:.1f}% "
            f"alive-norm mean {norms[alive].mean() if alive.any() else 0:.1f} "
            f"(init ~{init_norm:.2f}) alive-norm Gini {g:.3f}")
    print(f"[{TAG}] DONE in {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()