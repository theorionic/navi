"""Minimal, self-contained routing battery for one checkpoint.

Runs live-query slot traffic (distinct / top100 / Gini per block) and
value-row utilization (alive% / norm Gini). No sabotage stages.
Usage: NAVI_EVAL_CKPT=/path/x.pkl python3 eval_routing2.py
"""
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi
from fineweb_data import FineWebFeed

TAG = "rout"
T0 = time.time()
ckpt_path = os.environ["NAVI_EVAL_CKPT"]
TRAFFIC_BATCHES = 10
EVAL_BS = 32
SEQ = 512


def log(m):
    print(f"[{TAG} +{time.time()-T0:6.1f}s] {m}", flush=True)


def gini(counts):
    c = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(c)
    if n == 0 or c.sum() == 0:
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


mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                       score_temp=4.0)
cfg_m = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=2,
                    vocab_size=260)
model_ra = Navi(cfg_m, mem_cfg, return_aux=True)

feed = FineWebFeed(n_val_docs=4000)
feed.wait_ready(min_bytes=8 * 1024 * 1024, timeout_s=300)
vb = np.asarray(feed.val.array(), dtype=np.uint8)
rng = np.random.default_rng(9)


def windows(n):
    out = []
    for _ in range(n):
        offs = rng.integers(0, len(vb) - SEQ - 1, size=EVAL_BS)
        idx = offs[:, None] + np.arange(SEQ)[None, :]
        out.append(vb[idx].astype(np.int32))
    return out


with open(ckpt_path, "rb") as f:
    p = shard_tree(pickle.load(f)["params"])
log(f"loaded {os.path.basename(ckpt_path)}")

tbatches = windows(TRAFFIC_BATCHES)


@jax.jit
def ev_aux(pp, ids):
    _, aux, _lb = model_ra.apply(pp, ids, train=False)
    return aux


n_slots_cls = 512 * 512
aux0 = ev_aux(p, jax.device_put(tbatches[0], BATCH))
blocks = sorted(aux0.keys())
traffic = {k: np.zeros(4 * n_slots_cls, dtype=np.int64) for k in blocks}
for b in tbatches:
    aux = ev_aux(p, jax.device_put(b, BATCH))
    for k in blocks:
        s = np.asarray(aux[k]).reshape(EVAL_BS, SEQ, 4, -1)
        cls = np.arange(4)[None, None, :, None]
        glob = (cls * n_slots_cls + s).ravel()
        traffic[k] += np.bincount(glob, minlength=4 * n_slots_cls)
log("ROUTING per block (of 4,194,304 slots):")
for k in blocks:
    tc = traffic[k]
    touched = int((tc > 0).sum())
    top100 = float(np.sort(tc)[-100:].sum() / max(1, tc.sum()))
    log(f"  {k}: touched {touched} ({100*touched/(4*n_slots_cls):.3f}%) "
        f"top100 {top100:.3f} Gini {gini(tc):.3f}")

import jax.tree_util as tu
with open(ckpt_path, "rb") as f:
    st = pickle.load(f)
for kp, v in tu.tree_flatten_with_path(st["params"])[0]:
    ks = tu.keystr(kp).replace("['", "/").replace("']", "")
    if ks.endswith("/values"):
        v = np.asarray(v).astype(np.float32)
        norms = np.linalg.norm(v, axis=-1).ravel()
        alive = norms > 2 * 0.02 * (512 ** 0.5)
        g = gini(norms[alive] if alive.any() else norms)
        log(f"  {ks}: alive {alive.mean()*100:.1f}% "
            f"alive-norm mean {norms[alive].mean() if alive.any() else 0:.1f} "
            f"alive-norm Gini {g:.3f}")
print(f"[{TAG}] DONE in {time.time()-T0:.0f}s", flush=True)