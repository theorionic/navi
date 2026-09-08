"""Standalone held-out validation for the completed 500m run.

Why this exists: the 20k training run crashed with RESOURCE_EXHAUSTED right
after the final checkpoint save, before the VAL bpc line -- so the final
held-out number was never printed. This script recovers it from the saved
checkpoint alone.

What it does:
1. finds the newest ckpt_500m_stepNNNNNN.pkl in /kaggle/working/experiments
2. loads params (opt state NOT loaded -- 2.24GB of HBM we don't need here;
   skipping it is exactly what the train-run validator failed on)
3. streams a FRESH val set from FineWeb (different from the 24MB in-kernel
   buffer the trainer used, and it died with the kernel anyway)
4. evaluates bpc on many random 512-token windows with a batch size that
   fits (NAVI_EVAL_BS, default 32 vs the trainer's 256 -- no optimizer
   state, no grads, activations only)
5. also reports loss WITHOUT the Pool contribution (zeroed value tables)
   so the Pool's contribution to held-out quality is measured, not assumed

Env: NAVI_EVAL_BS (32), NAVI_EVAL_BATCHES (50 -> 50*32*512 = 819k tokens,
~2x the trainer's val sample), NAVI_SEQ (512, must match training),
NAVl_VAL_DOCS (2000 fresh docs for the val buffer).

Usage: python3 /kaggle/working/navi/experiments/val500m.py
"""
import sys
sys.path.insert(0, "/kaggle/working")
import os
import glob
import gc
import re
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi
from fineweb_data import FineWebFeed

TAG = "val500m"
SEQ = int(os.environ.get("NAVI_SEQ", "512"))
EVAL_BS = int(os.environ.get("NAVI_EVAL_BS", "32"))
EVAL_BATCHES = int(os.environ.get("NAVI_EVAL_BATCHES", "50"))
VAL_DOCS = int(os.environ.get("NAVI_VAL_DOCS", "2000"))

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))


def shard_tree(tree):
    # identical placement to train500m.shard_tree: values sharded on the
    # slot axis, everything else replicated -- same program shape as training
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


def reshard_tree(tree):
    # same as above but for already-materialized arrays (from pickle load)
    return shard_tree(tree)


def main():
    # 1. newest checkpoint
    ckpt_dir = "/kaggle/working/experiments"
    ckpts = sorted(f for f in os.listdir(ckpt_dir)
                   if re.match(r"ckpt_500m_step\d+\.pkl$", f))
    if not ckpts:
        raise SystemExit(f"no ckpt_500m_step*.pkl found in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"[{TAG}] loading {ckpt_path}", flush=True)

    # 2. params only -- deliberately NOT loading "opt" (2.24GB we don't use;
    #    loading it is what OOM'd the end-of-training validation)
    with open(ckpt_path, "rb") as f:
        st = pickle.load(f)
    p = shard_tree(st["params"])
    step = st.get("step", "?")
    del st
    gc.collect()

    # 3. fresh held-out stream: the trainer's val buffer died with the kernel,
    # and a fresh stream is also a cleaner estimate (no overlap with any
    # doc the trainer's 24MB buffer happened to hold)
    print(f"[{TAG}] streaming fresh val docs ({VAL_DOCS})...", flush=True)
    feed = FineWebFeed(n_val_docs=VAL_DOCS)
    # n_val_docs is consumed by the producer thread; wait for the TRAIN
    # buffer to be ignored -- we only need val. FineWebFeed fills val first
    # (token_stream routes the first n_val_docs to val), so wait until the
    # val buffer reports full or enough bytes, then stop the feed.
    feed.wait_ready(min_bytes=1)  # producer started; val fills first
    for _ in range(600):  # up to ~20min: wait for val buffer to fill
        if feed.val.full or feed.val.total >= 8 * 1024 * 1024:
            break
        import time
        time.sleep(2)
    val_bytes = feed.val.array()
    print(f"[{TAG}] val buffer: {len(val_bytes)/1024/1024:.1f}MB", flush=True)

    # 4. model config must match training exactly
    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                           score_temp=4.0)
    cfg_m = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=2,
                        vocab_size=260)
    model = Navi(cfg_m, mem_cfg)

    @jax.jit
    def ev(pp, ids, tg):
        logits = model.apply(pp, ids, train=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()

    rng = np.random.default_rng(7)
    ces = []
    t0 = None
    for bi in range(EVAL_BATCHES):
        offs = rng.integers(0, len(val_bytes) - SEQ - 2, size=EVAL_BS)
        idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
        win = val_bytes[idx]
        ids = jax.device_put(win[:, :-1], BATCH)
        tg = jax.device_put(win[:, 1:], BATCH)
        ces.append(float(ev(p, ids, tg)))
        if t0 is None:
            t0 = __import__("time").time()
    vbpc = float(np.mean(ces)) / np.log(2)
    # per-batch spread: if batches disagree wildly, val is too small/hot
    spread = float(np.std(ces) / np.sqrt(len(ces))) / np.log(2)
    print(f"[{TAG}] VAL bpc {vbpc:.4f} +- {spread:.4f} "
          f"({EVAL_BATCHES} batches x {EVAL_BS} x {SEQ} = "
          f"{EVAL_BATCHES*EVAL_BS*SEQ/1e6:.2f}M tokens)", flush=True)

    # 5. Pool contribution on held-out data: zero the value tables, re-eval.
    # delta = zeroed - intact > 0 means the Pool helps generalization.
    def zero_pool(tree):
        def z(kp, x):
            return jnp.zeros_like(x) if "values" in jax.tree_util.keystr(kp) else x
        return jax.tree_util.tree_map_with_path(zero, tree)
    pz = shard_tree(zero_pool(p))
    ces_z = []
    for bi in range(EVAL_BATCHES):
        offs = rng.integers(0, len(val_bytes) - SEQ - 2, size=EVAL_BS)
        idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
        win = val_bytes[idx]
        ces_z.append(float(ev(pz, jax.device_put(win[:, :-1], BATCH),
                              jax.device_put(win[:, 1:], BATCH))))
    vbpc_z = float(np.mean(ces_z)) / np.log(2)
    print(f"[{TAG}] VAL bpc (Pool zeroed) {vbpc_z:.4f} "
          f"-> Pool contributes {vbpc_z - vbpc:+.4f} bpc", flush=True)
    print(f"[{TAG}] DONE", flush=True)


if __name__ == "__main__":
    main()