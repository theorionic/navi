"""Re-bucket zipf checkpoint accuracy by TRUE training exposure.

zipf_test.py bucketed by pair counts from a 2M-draw reference table, but
training made 32.7M draws: pairs unseen in the reference often had real
exposure (measured locally: "cold" eval triples average 217 true
exposures). This script replays the exact training sampler to build the
true (key,n1,n2) exposure table, loads ckpt_zipf.pkl, and evaluates
accuracy in true-exposure bins. Chance = 1/256 = 0.0039.
"""
import sys
import pickle
import time

sys.path.insert(0, "/kaggle/working/code")
import jax
import jax.numpy as jnp
import numpy as np

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi

T0 = time.time()
LB = 0.01
NN = 1024
N_KEYS = 16
VOCAB = 3 + N_KEYS + NN + 256
KEY0, NONCE0, VAL0 = 3, 3 + N_KEYS, 3 + N_KEYS + NN
CFG_MEM = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=16, n_classes=4,
                       score_temp=4.0, lb_weight=LB)
MODEL = ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2,
                    vocab_size=VOCAB)
mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))

_VAL = np.random.default_rng(1234).integers(0, 256, size=(N_KEYS, NN, NN), dtype=np.int16)
_zrng = np.random.default_rng(555)
_ranks = np.clip(_zrng.zipf(1.5, size=(2_000_000, 2)), 1, NN) - 1
_h1 = _ranks[:, 0].astype(np.int32)
_h2 = _ranks[:, 1].astype(np.int32)

print("[zipf-true] replaying true training exposures...", flush=True)
true_c = np.zeros((N_KEYS, NN, NN), dtype=np.int32)
for i in range(4000):
    r = np.random.default_rng(i)
    idx1 = r.integers(0, len(_h1), size=(512, 16))
    idx2 = r.integers(0, len(_h1), size=(512, 16))
    n1 = np.where(r.random((512, 16)) < 0.5, _h1[idx1], _h2[idx2])
    n2 = np.where(r.random((512, 16)) < 0.5, _h2[idx2], _h1[idx1])
    keys = r.integers(0, N_KEYS, size=512)
    np.add.at(true_c, (keys[:, None], n1, n2), 1)
print(f"[zipf-true] distinct facts seen: {(true_c > 0).sum():,} / {true_c.size:,}", flush=True)

with open("/kaggle/working/experiments/ckpt_zipf.pkl", "rb") as f:
    p = jax.device_put(pickle.load(f))
model = Navi(MODEL, CFG_MEM)

def fact_acc(lg, tg):
    return (lg[:, 2::4].argmax(-1) == tg[:, 2::4]).mean()

BINS = [(0, 1), (1, 2), (2, 4), (4, 16), (16, 64), (64, 10**9)]
res = {b: [] for b in BINS}
for i in range(16):
    r = np.random.default_rng(777 * 1000 + 4_000_000 + i)
    idx1 = r.integers(0, len(_h1), size=(512, 16))
    idx2 = r.integers(0, len(_h1), size=(512, 16))
    n1 = np.where(r.random((512, 16)) < 0.5, _h1[idx1], _h2[idx2])
    n2 = np.where(r.random((512, 16)) < 0.5, _h2[idx2], _h1[idx1])
    keys = r.integers(0, N_KEYS, size=512)
    val = _VAL[keys[:, None], n1, n2]
    tc = true_c[keys[:, None], n1, n2]
    for b in BINS:
        m = (tc >= b[0]) & (tc < b[1])
        if m.sum() < 8:
            continue
        ns = int(m.sum())
        pad = (-ns) % jax.device_count()
        seqs = np.full((ns + pad, 4), 2, dtype=np.int32)
        k_sel = np.broadcast_to(keys[:, None], m.shape)[m]
        seqs[:ns, 0] = k_sel + KEY0
        seqs[:ns, 1] = n1[m] + NONCE0
        seqs[:ns, 2] = n2[m] + NONCE0
        seqs[:ns, 3] = val[m] + VAL0
        if pad:
            seqs[ns:] = seqs[:pad]
        lg = model.apply(p, jax.device_put(seqs[:, :-1], BATCH), train=False)
        a = fact_acc(lg, jax.device_put(seqs[:, 1:], BATCH))
        res[b].append(float(a))

print("[zipf-true] accuracy by TRUE exposure (chance 0.0039):", flush=True)
for b in BINS:
    if res[b]:
        hi = "inf" if b[1] >= 10**8 else str(b[1])
        print(f"  exp {b[0]:>3}-{hi:>4}: {np.mean(res[b]):.4f} (draws={len(res[b])})", flush=True)
print(f"[zipf-true] DONE in {time.time()-T0:.0f}s")