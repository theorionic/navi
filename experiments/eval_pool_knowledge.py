"""Pool knowledge-storage probe for the 500M bpe checkpoints.

Answers: does the Pool actually STORE knowledge (beyond routing spread)?

Four tests, CPU-only, one checkpoint:
  1. VALUES  - are value rows being written? alive% (norm > init noise
     floor), norm mean, norm Gini per memory block.
  2. ROUTING - live-query slot traffic: distinct slots, top-100 share,
     Gini of slot read counts per block (collapse detector).
  3. REPRO   - slot reproducibility: same content -> same slots? For
     repeated 16-token contexts, Jaccard overlap of selected slot sets
     vs a random-context baseline.
  4. KILL    - ablation: zero the values of the top-1% most-read slots
     and measure val bpc delta vs zeroing a random 1%. Knowledge means
     hot-slot removal hurts more than random removal.

Usage: NAVI_EVAL_CKPT=/path/ckpt.pkl python3 eval_pool_knowledge.py
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
from grain_parquet_data import PhaseFeed

jax.config.update("jax_platform_name", "cpu")

TAG = "know"
SEQ = 512
VOCAB = 16384


def log(msg):
    print(f"[{TAG}] {msg}", flush=True)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (x * idx).sum()) / (n * x.sum()) - (n + 1) / n)


def main():
    ckpt_path = os.environ["NAVI_EVAL_CKPT"]
    log(f"loading {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        st = pickle.load(f)
    p = st["params"]
    pp = p.get("params", p)
    step = st.get("step", "?")

    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64,
                           n_classes=4, score_temp=4.0)
    cfg = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=2,
                      vocab_size=VOCAB)
    model = Navi(cfg, mem_cfg)
    model_ra = Navi(cfg, mem_cfg, return_aux=True)

    # fresh val data: PhaseFeed holds out its own val buffer
    feed = PhaseFeed(buffer_mb=64, val_docs=1000)
    feed.launch()
    feed.wait_ready(min_tokens=SEQ * 64, timeout=600)
    val = np.asarray(feed.val[:feed.val_end])
    log(f"val tokens: {len(val):,}")

    rng = np.random.default_rng(23)
    hi = len(val) - SEQ - 2

    # ---------------- 1. VALUES ----------------
    log("== 1. value-row health (init-normal ~0.016, Gini healthy < 0.5)")
    for bi in range(0, 8, 2):
        v = np.asarray(pp[f"block_{bi}"]["mem"]["values"], dtype=np.float32)
        norms = np.linalg.norm(v.reshape(v.shape[0], -1, v.shape[-1]),
                               axis=-1).ravel()
        alive = norms > 2 * 0.02 * np.sqrt(v.shape[-1])
        log(f"  b{bi}: alive {alive.mean()*100:5.1f}% "
            f"norm-mean {norms[alive].mean() if alive.any() else 0:.3f} "
            f"alive-Gini {gini(norms[alive] if alive.any() else norms):.3f}")

    # ---------------- 2. ROUTING ----------------
    log("== 2. routing traffic (collapse = distinct ~ cand_k, Gini -> 1)")
    n_batches = 8
    slot_sets = {}
    read_counts = {}
    for bi in range(0, 8, 2):
        slot_sets[f"mem_{bi}"] = set()
        read_counts[f"mem_{bi}"] = {}
    for _ in range(n_batches):
        offs = rng.integers(0, hi, size=8)
        idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
        ids = jnp.asarray(val[idx][:, :-1])
        _, aux, _ = model_ra.apply(p, ids, train=False)
        for name in sorted(aux):
            arr = np.asarray(aux[name])          # (b, l, classes*cand_k)
            CAND = 8
            for cls in range(arr.shape[-1] // CAND):
                sl = arr[..., cls * CAND:(cls + 1) * CAND].reshape(-1)
                s = slot_sets[name]
                rc = read_counts[name]
                for v_ in sl:
                    key = (cls, int(v_))
                    s.add(key)
                    rc[key] = rc.get(key, 0) + 1
    for name in sorted(slot_sets):
        n_slots = 512 * 512 * 4
        counts = np.asarray(list(read_counts[name].values()))
        top100 = float(np.sort(counts)[-100:].sum() / counts.sum())
        log(f"  {name}: distinct {len(slot_sets[name]):6d} "
            f"({len(slot_sets[name])/n_slots*100:5.2f}%) "
            f"top100-share {top100*100:5.1f}% Gini {gini(counts):.3f}")

    # ---------------- 3. REPRO ----------------
    log("== 3. slot reproducibility (knowledge => repeat ctx reads same slots)")
    def slots_for(text_ids):
        ids = jnp.asarray(text_ids[None, :-1])
        _, aux, _ = model_ra.apply(p, ids, train=False)
        out = {}
        for name in aux:
            a = np.asarray(aux[name])[0]      # (l, classes*cand_k)
            out[name] = set(map(int, a.reshape(-1)))
        return out

    base = val[rng.integers(0, hi, size=SEQ + 1)]
    s1 = slots_for(base)
    s2 = slots_for(base)  # determinism sanity: must be identical
    for name in sorted(s1):
        j = len(s1[name] & s2[name]) / max(1, len(s1[name] | s2[name]))
        log(f"  {name}: self-Jaccard {j:.3f} (determinism check)")

    # repeated-content probe: replace 16 tokens of the window with a fixed
    # rare pattern; check whether the slot sets at those positions differ
    # from a different-content window less than random windows do
    rare = (VOCAB - 8 + np.arange(8)) % VOCAB
    w_a = base.copy(); w_a[100:108] = rare
    w_b = base.copy(); w_b[300:308] = rare
    w_rand = val[rng.integers(0, hi, size=SEQ + 1)]
    sa, sb, sr = slots_for(w_a), slots_for(w_b), slots_for(w_rand)
    for name in sorted(sa):
        jab = len(sa[name] & sb[name]) / max(1, len(sa[name] | sb[name]))
        jar = len(sa[name] & sr[name]) / max(1, len(sa[name] | sr[name]))
        log(f"  {name}: Jaccard(same-rare) {jab:.3f} vs Jaccard(rand) {jar:.3f} "
            f"{'-> content-addressed' if jab > jar + 0.02 else '~ content-neutral'}")

    # ---------------- 4. KILL ----------------
    log("== 4. hot-slot kill test (knowledge => hot 1% hurts > random 1%)")
    def val_bpc(params):
        tot, n = 0.0, 0
        for _ in range(4):
            offs = rng.integers(0, hi, size=4)
            idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
            ids = jnp.asarray(val[idx][:, :-1])
            l = optax.softmax_cross_entropy_with_integer_labels(logits, tg)
            tot += float(l.mean()); n += 1
        return tot / n / np.log(2)

    base_bpc = val_bpc(p)
    # hottest slots: union of top read-count keys across blocks, as slot ids
    hot_by_block = {}
    mem_blocks = [0, 2, 4, 6]
    for blk in mem_blocks:
        name = f"mem_{blk}"
        rc = read_counts[name]
        top = sorted(rc.items(), key=lambda kv: -kv[1])[:len(rc) // 100 or 100]
        hot_by_block[blk] = top
    p_hot = jax.tree_util.tree_map(lambda x: x, p)
    ph = p_hot.get("params", p_hot)
    killed = 0
    for blk in mem_blocks:
        v = ph[f"block_{blk}"]["mem"]["values"]
        cls_ids = np.asarray([k[0] for k, _ in hot_by_block[blk]])
        slot_ids = np.asarray([k[1] for k, _ in hot_by_block[blk]])
        v = v.at[cls_ids, slot_ids, :].set(0.0)
        ph[f"block_{blk}"]["mem"]["values"] = v
        killed += len(cls_ids)
    hot_bpc = val_bpc(p_hot)
    # random kill, same count
    p_rand = jax.tree_util.tree_map(lambda x: x, p)
    pr = p_rand.get("params", p_rand)
    rng2 = np.random.default_rng(7)
    for blk in mem_blocks:
        v = pr[f"block_{blk}"]["mem"]["values"]
        n_cls, n_slot = v.shape[0], v.shape[1]
        pick_c = rng2.integers(0, n_cls, size=len(hot_by_block[blk]))
        pick_s = rng2.integers(0, n_slot, size=len(hot_by_block[blk]))
        v = v.at[pick_c, pick_s, :].set(0.0)
        pr[f"block_{blk}"]["mem"]["values"] = v
    rand_bpc = val_bpc(p_rand)
    log(f"  step {step}: base {base_bpc:.4f} | hot-kill {hot_bpc:.4f} "
        f"(+{1000*(hot_bpc-base_bpc):.2f} mbpc) | rand-kill {rand_bpc:.4f} "
        f"(+{1000*(rand_bpc-base_bpc):.2f} mbpc) | killed {killed} slots")
    verdict = "KNOWLEDGE" if hot_bpc - base_bpc > rand_bpc - base_bpc else "no-knowledge-signal"
    log(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()