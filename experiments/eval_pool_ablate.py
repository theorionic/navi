"""Full-pool ablation: what does the ENTIRE pool contribute at this step?

Conditions on one checkpoint (val bpc on held-out):
  base    - intact trained pool
  zero    - every value row zeroed (memory read contributes nothing)
  random  - values re-drawn from fresh init normal(0.02) (untrained pool)
  shuffle - existing value rows permuted among slots (keeps the learned
            value distribution, destroys which-content-maps-to-which)
  nomem   - use_memory=False forward (skip the PKM block entirely)

Reading: knowledge => base beats zero/random/shuffle. If base ~ random,
the pool is (still) interchangeable with noise at this step.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi

jax.config.update("jax_platform_name", "cpu")

TAG = "ablate"
SEQ = 512
VOCAB = 16384


def log(msg):
    print(f"[{TAG}] {msg}", flush=True)


def bpc_of(model, p, val, rng, hi, n_batches=6, bs=4):
    tot, n = 0.0, 0
    for _ in range(n_batches):
        offs = rng.integers(0, hi, size=bs)
        idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
        tg = jnp.asarray(val[idx][:, 1:])
        ids = jnp.asarray(val[idx][:, :-1])
        logits = model.apply(p, ids, train=False)
        l = optax.softmax_cross_entropy_with_integer_labels(logits, tg)
        tot += float(l.mean()); n += 1
    return tot / n / np.log(2)


def main():
    ckpt_path = os.environ["NAVI_EVAL_CKPT"]
    log(f"loading {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        st = pickle.load(f)
    p = st["params"]
    step = st.get("step", "?")

    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64,
                           n_classes=4, score_temp=4.0)
    # no-memory variant: same model class, but every Block has
    # use_memory=False - build a config with memory_every=0
    cfg_nomem = ModelConfig(d_model=512, n_layers=8, n_heads=8,
                            memory_every=0, vocab_size=VOCAB)
    model_nomem = Navi(cfg_nomem, mem_cfg)

    from grain_parquet_data import PhaseFeed
    feed = PhaseFeed(buffer_mb=64, val_docs=1000)
    feed.launch()
    feed.wait_ready(min_tokens=SEQ * 64, timeout=600)
    val = np.asarray(feed.val[:feed.val_end])
    hi = len(val) - SEQ - 2
    rng = np.random.default_rng(23)
    log(f"val tokens: {len(val):,}")

    base = bpc_of(model, p, val, rng, hi)
    log(f"base (intact pool):        {base:.4f}")

    # zero all values
    p_zero = jax.tree_util.tree_map(lambda x: x, p)
    pz = p_zero.get("params", p_zero)
    for blk in (0, 2, 4, 6):
        pz[f"block_{blk}"]["mem"]["values"] = jnp.zeros_like(
            pz[f"block_{blk}"]["mem"]["values"])
    zero = bpc_of(model, p_zero, val, rng, hi)
    log(f"zero (all values = 0):     {zero:.4f}  "
        f"({1000*(zero-base):+.1f} mbpc)")

    # fresh random values (same init statistics)
    p_rand = jax.tree_util.tree_map(lambda x: x, p)
    pr = p_rand.get("params", p_rand)
    rng2 = np.random.default_rng(101)
    for blk in (0, 2, 4, 6):
        shape = pr[f"block_{blk}"]["mem"]["values"].shape
        pr[f"block_{blk}"]["mem"]["values"] = jnp.asarray(
            rng2.normal(0, 0.02, size=shape), dtype=jnp.bfloat16)
    rand = bpc_of(model, p_rand, val, rng, hi)
    log(f"random (fresh-init pool):  {rand:.4f}  "
        f"({1000*(rand-base):+.1f} mbpc)")

    # shuffle rows among slots within each class (destroy addressing,
    # keep learned value distribution)
    p_shuf = jax.tree_util.tree_map(lambda x: x, p)
    ps = p_shuf.get("params", p_shuf)
    for blk in (0, 2, 4, 6):
        v = np.asarray(ps[f"block_{blk}"]["mem"]["values"], dtype=np.float32)
        flat = v.reshape(-1, v.shape[-1])
        perm = rng2.permutation(flat.shape[0])
        ps[f"block_{blk}"]["mem"]["values"] = jnp.asarray(
            flat[perm].reshape(v.shape), dtype=jnp.bfloat16)
    shuf = bpc_of(model, p_shuf, val, rng, hi)
    log(f"shuffle (rows permuted):   {shuf:.4f}  "
        f"({1000*(shuf-base):+.1f} mbpc)")

    # no memory at all
    nomem = bpc_of(model_nomem, p, val, rng, hi)
    log(f"nomem (PKM skipped):       {nomem:.4f}  "
        f"({1000*(nomem-base):+.1f} mbpc)")

    log(f"step {step} verdict: "
        f"pool better than zero: {base < zero}, "
        f"better than random: {base < rand}, "
        f"better than shuffled: {base < shuf}, "
        f"memory helps vs no-memory: {base < nomem}")


if __name__ == "__main__":
    main()