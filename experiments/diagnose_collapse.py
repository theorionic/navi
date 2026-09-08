"""Pool-collapse diagnostics on saved checkpoints.

Answers: does the Pool collapse? Three collapse modes checked per stage:
  1. Key collapse: subkeys converge -> queries can't separate slots.
     Metric: mean |cosine| between random subkey pairs (1.0 = collapsed),
     effective rank of the subkey table via SVD entropy.
  2. Value collapse: value rows die (vanish) or converge.
     Metric: dead-row fraction, row-norm mean, effective rank.
  3. Routing collapse: traffic concentrates on few slots.
     Metric: distinct slots touched by 32k real queries, top-100 share,
     Gini coefficient of slot traffic.
"""
import sys, os, pickle
sys.path.insert(0, "/kaggle/working")
import jax, jax.numpy as jnp
import numpy as np
import jax.tree_util as tu

CKPTS = [("S1-65k", "/kaggle/working/experiments/ckpt_C-S1.pkl", 64),
         ("S2-1M",  "/kaggle/working/experiments/ckpt_C-S2.pkl", 256),
         ("S3-16M", "/kaggle/working/experiments/ckpt_C-S3.pkl", 1024)]

# 500m run: newest rolling checkpoint (ckpt_500m_stepNNNNNN.pkl, params+opt)
import glob as _glob, re as _re
_c5 = sorted(f for f in _glob.glob("/kaggle/working/experiments/ckpt_500m_step*.pkl")
             if _re.match(r"ckpt_500m_step\d+\.pkl", os.path.basename(f)))
if _c5:
    CKPTS.append(("500m-@" + _re.search(r"(\d+)\.pkl", _c5[-1]).group(1), _c5[-1], 1024))

def eff_rank(x, cap=4096):
    s = np.linalg.svd(x[:: max(1, len(x) // cap)], compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

def key_stats(a):
    r = a.reshape(-1, a.shape[-1]).astype(np.float32)
    n = r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(n), (1500,))
    q = n[idx]
    cos = np.abs(q @ q.T)
    iu = np.triu_indices(len(idx), 1)
    return float(cos[iu].mean()), eff_rank(r)

def val_stats(v):
    r = v.reshape(-1, v.shape[-1]).astype(np.float32)
    vn = np.linalg.norm(r, axis=-1)
    return float((vn < 0.1 * vn.mean()).mean()), float(vn.mean()), eff_rank(r)

def gini(counts):
    c = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(c)
    if c.sum() == 0: return 0.0
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * c).sum()) / (n * c.sum()) - (n + 1) / n)

for tag, path, nonce in CKPTS:
    try:
        with open(path, "rb") as f:
            params = pickle.load(f)
        if isinstance(params, dict) and "params" in params:
            params = params["params"]
        flat = {}
        # keystr yields "['params']['block_0']['mem']['k1']"; normalize to
        # slash form "params/block_0/mem/k1" for uniform matching
        flat = {}
        for kp, v in tu.tree_flatten_with_path(params)[0]:
            ks = tu.keystr(kp)
            norm = (ks.replace("['", "/").replace("']", "")
                      .lstrip("/").replace("]", ""))
            flat[norm] = v
        print(f"\n===== {tag} =====")
        for pth in sorted(flat):
            if pth.endswith("/k1"):
                base = pth[:-2]  # keep trailing '/' so base+"k1" reassembles
                k1, k2, vals = flat[base+"k1"], flat[base+"k2"], flat[base+"values"]
                mc1, er1 = key_stats(np.asarray(k1))
                mc2, er2 = key_stats(np.asarray(k2))
                dead, vn, vr = val_stats(np.asarray(vals))
                print(f"{base}:")
                print(f"  keys   : mean|cos|={mc1:.3f}/{mc2:.3f} (collapse->1.0; init~0.0) "
                      f"eff_rank={er1:.0f}/{er2:.0f}")
                print(f"  values : dead_frac={dead:.4f} norm_mean={vn:.4f} eff_rank={vr:.0f}")
    except Exception as e:
        print(f"{tag}: FAILED {type(e).__name__}: {e}")