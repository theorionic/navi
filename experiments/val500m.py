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
import os
# make "navi" package importable regardless of cwd: this file lives in
# <repo>/experiments/, so the repo root is exactly one level up
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import gc
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


T0 = time.time()


def log(msg):
    """Timestamped stage log: every phase boundary and every wait tick is
    visible, so a slow phase is observable instead of a silent hang."""
    print(f"[{TAG} +{time.time()-T0:6.1f}s] {msg}", flush=True)


def main():
    # 1. newest checkpoint
    ckpt_dir = "/kaggle/working/experiments"
    ckpts = sorted(f for f in os.listdir(ckpt_dir)
                   if re.match(r"ckpt_500m_step\d+\.pkl$", f))
    if not ckpts:
        raise SystemExit(f"no ckpt_500m_step*.pkl found in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    log(f"stage 1/6: loading {os.path.basename(ckpt_path)} "
        f"({os.path.getsize(ckpt_path)/1e9:.2f}GB)")

    # 2. params only -- deliberately NOT loading "opt" (2.24GB we don't use;
    #    loading it is what OOM'd the end-of-training validation)
    with open(ckpt_path, "rb") as f:
        st = pickle.load(f)
    log("stage 2/6: checkpoint unpickled; placing params on mesh (device_put)")
    p = shard_tree(st["params"])
    step = st.get("step", "?")
    del st
    gc.collect()
    log(f"stage 2/6 done: params on mesh, step={step}")

    # 3. fresh held-out stream (the trainer's val buffer died with the
    # kernel). FineWebFeed's producer routes the first n_val_docs to val,
    # then streams train docs forever; val.full is the only stop condition.
    log(f"stage 3/6: starting FineWeb stream for {VAL_DOCS} val docs "
        f"(HF datasets init took ~4min on the training run - this is the "
        f"slow part, progress prints follow every 30s)")
    feed = FineWebFeed(n_val_docs=VAL_DOCS)
    log("  waiting for the val docs to drain through the producer "
        "(watch the TRAIN buffer: it starts growing only after all "
        "val docs have streamed through)")
    prev_train = -1
    for tick in range(600):  # hard cap 20min; real: ~60s total
        tb = len(feed.train_buf)
        # val docs stream FIRST; the moment train bytes appear, every val
        # doc has passed. (val.full can never fire: 2000 docs ~= 6MB <
        # the 24MB cap -- this exact dead-wait burned 20-30min in v2/v3.)
        if tb > 0:
            break
        if tb != prev_train and tick % 15 == 0:
            log(f"  val fill: {feed.val.total/1024/1024:.1f}MB so far")
            prev_train = tb
        time.sleep(2)
    val_bytes = feed.val.array()
    log(f"stage 3/6 done: val buffer {len(val_bytes)/1024/1024:.1f}MB "
        f"({VAL_DOCS} docs)")

    # 4. model config must match training exactly
    log("stage 4/6: building model + jit compile (one-time, ~30-60s)")
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

    def eval_pass(params, label):
        """EVAL_BATCHES forward passes. Single jit compile on the first
        batch (same shapes every batch -> no recompiles), then ~0.1s/batch.
        Per-batch logs for the first 3 batches expose a recompile if one
        ever sneaks in (recompile shows as batch 2/3 >> batch 1)."""
        ces = []
        t0 = time.time()
        for bi in range(EVAL_BATCHES):
            offs = rng.integers(0, len(val_bytes) - SEQ - 2, size=EVAL_BS)
            idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
            win = val_bytes[idx]
            # EXACT training-distribution windows: RollingBytes.sample()
            # overwrites the first byte of every window with BOS (256) --
            # the model never saw a BOS-less window in 20k steps, so eval
            # must match (v1 eval fed raw windows: OOD, 5.87 vs 1.91 bpc).
            win[:, 0] = BOS
            ids = jax.device_put(win[:, :-1], BATCH)
            tg = jax.device_put(win[:, 1:], BATCH)
            ces.append(float(ev(params, ids, tg)))
            if bi < 3:
                log(f"  {label}: batch {bi+1}/{EVAL_BATCHES} "
                    f"ce={ces[-1]:.4f} ({time.time()-t0:.1f}s elapsed)")
        dt = time.time() - t0
        return ces, dt

    log("stage 5/6: eval pass 1 (intact Pool) - first batch includes jit compile")
    ces, dt = eval_pass(p, "pass1")
    log(f"stage 5/6 done: {dt:.1f}s total ({dt/EVAL_BATCHES:.2f}s/batch)")
    vbpc = float(np.mean(ces)) / np.log(2)
    # per-batch spread: if batches disagree wildly, val is too small/hot
    spread = float(np.std(ces) / np.sqrt(len(ces))) / np.log(2)
    print(f"[{TAG}] VAL bpc {vbpc:.4f} +- {spread:.4f} "
          f"({EVAL_BATCHES} batches x {EVAL_BS} x {SEQ} = "
          f"{EVAL_BATCHES*EVAL_BS*SEQ/1e6:.2f}M tokens)", flush=True)

    # 6. Pool contribution on held-out data: zero the value tables, re-eval.
    # delta = zeroed - intact > 0 means the Pool helps generalization.
    log("stage 6/6: zeroing Pool value tables and re-evaluating")
    def zero_pool(tree):
        def z(kp, x):
            return jnp.zeros_like(x) if "values" in jax.tree_util.keystr(kp) else x
        return jax.tree_util.tree_map_with_path(z, tree)
    pz = shard_tree(zero_pool(p))
    ces_z, dt_z = eval_pass(pz, "pass2")
    log(f"stage 6/6 done: {dt_z:.1f}s total ({dt_z/EVAL_BATCHES:.2f}s/batch, "
        f"no recompile - same shapes)")
    vbpc_z = float(np.mean(ces_z)) / np.log(2)
    print(f"[{TAG}] VAL bpc (Pool zeroed) {vbpc_z:.4f} "
          f"-> Pool contributes {vbpc_z - vbpc:+.4f} bpc", flush=True)
    log(f"ALL DONE in {time.time()-T0:.0f}s "
        f"(ckpt load -> mesh | val fill | compile | 2 eval passes)")


if __name__ == "__main__":
    main()