"""Held-out generalization arm: does the pool GENERALIZE or only memorize?

Protocol (same airtight two-stage as proof_local.py):
  Stage 1: backbone-only on fact-free skeleton (no fact words anywhere).
  Stage 2: pool-only training on facts for TRAIN animals only.
Evidence:
  T1: generation for train vs held-out animals, base vs zero.
  T2: bpc on train-fact lines vs heldout-fact lines, base vs zero.
If held-out animals also improve (base < zero on their lines), the pool
generalizes; if not, it memorizes.
Env: HOLDOUT (comma list), S2_MEM_LR, S1_STEPS, S2_STEPS
"""
import os, sys, pickle
import numpy as np

os.environ.setdefault("NAVI_GEN", "0")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import jax
import jax.numpy as jnp
import optax
from navi.model import Navi
from navi.config import ModelConfig
from navi.pkm import MemoryConfig
from navi.train import init_params
import train500m_bpe as T
from tokenizers import Tokenizer

MEM_BLOCKS = (0,)
mem_cfg = MemoryConfig(c1=256, c2=256, cand_k=8, side_top=32, n_classes=4,
                       score_temp=4.0)
cfg_m = ModelConfig(d_model=256, n_layers=4, n_heads=8, memory_every=2,
                    vocab_size=T.VOCAB)
model = Navi(cfg_m, mem_cfg)
tok = Tokenizer.from_file(os.environ.get(
    "NAVI_TOK_PATH", os.path.join(ROOT, "experiments", "tokenizer_16k.json")))
BOS = 0
SEQ_S, BATCH_S = 128, 32
MEM_LR = float(os.environ.get("S2_MEM_LR", "1e-2"))
S2_TEMP = float(os.environ.get("S2_TEMP", "0.5"))
S1 = int(os.environ.get("S1_STEPS", "400"))
S2 = int(os.environ.get("S2_STEPS", "2500"))
HOLDOUT = [a.strip() for a in os.environ.get(
    "HOLDOUT", "owl,bear").split(",")]

# ---------------- fact set: split by ANIMAL ----------------
COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
ANIMALS = ["cat", "dog", "fox", "owl", "bear", "wolf"]
FOODS = ["fish", "berries", "honey", "seeds", "cheese", "eggs"]
rng = np.random.default_rng(0)
facts = [(a, f, COLORS[int(rng.integers(0, len(COLORS)))])
         for a in ANIMALS for f in FOODS]
train_facts = [t for t in facts if t[0] not in HOLDOUT]
heldout_facts = [t for t in facts if t[0] in HOLDOUT]
print(f"[gen] holdout animals: {HOLDOUT} "
      f"({len(train_facts)} train facts, {len(heldout_facts)} held-out facts)",
      flush=True)

FILLER = "the quick brown fox jumps over the lazy dog and then rests in the sun"
SKELETON = "The animal eats food. The animal is color."

def mk_text(a, f, c):
    return f"The {a} eats {f}. The {a} is {c}."

train_lines = [mk_text(*t) + " " + FILLER for t in train_facts]
heldout_lines = [mk_text(*t) + " " + FILLER for t in heldout_facts]
skel_lines = [SKELETON + " " + FILLER for _ in range(24)]

def mk_batch(lines, rngb, bs=BATCH_S):
    idx = rngb.integers(0, len(lines), size=bs)
    out = []
    for j in idx:
        e = tok.encode(lines[j]).ids[:SEQ_S - 1]
        out.append([BOS] + e + [0] * (SEQ_S - 1 - len(e)))
    return np.asarray(out, dtype=np.int32)

rngb = np.random.default_rng(5)
s1b = [mk_batch(skel_lines, rngb) for _ in range(S1)]
s2b = [mk_batch(train_lines, rngb) for _ in range(S2)]

# ---------------- two-stage training ----------------
p = init_params(model, SEQ_S, jax.random.PRNGKey(0))
tx1 = optax.multi_transform(
    {"core": optax.adamw(3e-4, weight_decay=0.01),
     "mem": optax.adamw(MEM_LR)},
    jax.tree_util.tree_map_with_path(
        lambda kp, x: "mem" if T.is_mem(kp) else "core", p))
o = tx1.init(p)

def seg_loss(p, batch):
    ids = jax.device_put(batch)
    tg = jnp.concatenate([ids[:, 1:],
                          jnp.full((ids.shape[0], 1), 0, jnp.int32)], 1)
    out = model.apply(p, ids, train=True, mem_temp=S2_TEMP)
    return optax.softmax_cross_entropy_with_integer_labels(out, tg).mean()

@jax.jit
def step1(p, o, b):
    g = jax.grad(seg_loss)(p, b)
    g = jax.tree_util.tree_map_with_path(
        lambda kp, x: jnp.zeros_like(x) if T.is_mem(kp) else x, g)
    u, o2 = tx1.update(g, o, p)
    return optax.apply_updates(p, u), o2

@jax.jit
def step2(p, o, b):
    g = jax.grad(seg_loss)(p, b)
    g = jax.tree_util.tree_map_with_path(
        lambda kp, x: x if T.is_mem(kp) else jnp.zeros_like(x), g)
    u, o2 = tx1.update(g, o, p)
    return optax.apply_updates(p, u), o2

print(f"[proof] stage1: backbone only, {S1} steps", flush=True)
for i in range(S1):
    p, o = step1(p, o, s1b[i])
print(f"[proof] stage1 loss {float(seg_loss(p, s1b[-1])):.4f}", flush=True)
print(f"[proof] stage2: pool only on TRAIN animals, {S2} steps", flush=True)
for i in range(S2):
    p, o = step2(p, o, s2b[i])
    if i % 500 == 0:
        print(f"[proof] s2 {i} loss {float(seg_loss(p, s2b[i])):.4f}", flush=True)
print(f"[proof] stage2 final loss {float(seg_loss(p, s2b[-1])):.4f}", flush=True)

# ---------------- ablations ----------------
pz = jax.tree_util.tree_map(lambda x: x, p)
for blk in MEM_BLOCKS:
    pz["params"][f"block_{blk}"]["mem"]["values"] = jnp.zeros_like(
        pz["params"][f"block_{blk}"]["mem"]["values"])

def bpc_lines(pv, lines, n=8):
    rnga = np.random.default_rng(31)
    tot = 0.0
    for _ in range(n):
        b = mk_batch(lines, rnga, bs=8)
        ids = jax.device_put(b)
        tg = jnp.concatenate([ids[:, 1:],
                              jnp.full((ids.shape[0], 1), 0, jnp.int32)], 1)
        out = model.apply(pv, ids, train=False)
        tot += float(optax.softmax_cross_entropy_with_integer_labels(
            out, tg).mean())
    return tot / n / np.log(2)

# ---- T2: train vs held-out bpc, base vs zero ----
print("==== T2: generalization bpc ====", flush=True)
for name, lines in [("TRAIN", train_lines), ("HELDOUT", heldout_lines)]:
    b_ = bpc_lines(p, lines)
    z_ = bpc_lines(pz, lines)
    print(f"  {name:8s} base={b_:.4f} zero={z_:.4f} "
          f"pool-gain={1000*(z_-b_):+.1f} mbpc", flush=True)

# ---- T1: generation, train vs heldout animals ----
FACT_WORDS = set(FOODS) | set(COLORS)
def gen(pv, pr_):
    ids = [BOS] + tok.encode(pr_, add_special_tokens=False).ids
    buf = np.zeros((1, SEQ_S), dtype=np.int32)
    buf[0, :len(ids)] = ids
    out = jax.device_get(T._gen_scan(model, pv, jnp.asarray(buf),
                       jnp.int32(len(ids) - 1), 10, 0.3, 7))
    return tok.decode([int(t) for t in out[:, 0]])

print("==== T1: generation base vs zero ====", flush=True)
for animal in ANIMALS:
    tag = "HELDOUT" if animal in HOLDOUT else "TRAIN"
    pr_ = f"The {animal} eats"
    b = gen(p, pr_); z = gen(pz, pr_)
    fb = sorted(w for w in FACT_WORDS if w in b)
    fz = sorted(w for w in FACT_WORDS if w in z)
    print(f"  [{animal}/{tag}] BASE {b!r} facts={fb}")
    print(f"  [{animal}/{tag}] ZERO {z!r} facts={fz}", flush=True)

with open(os.path.join(ROOT, "tmp", "gen_ckpt.pkl"), "wb") as f:
    pickle.dump({"params": jax.device_get(p), "holdout": HOLDOUT}, f)
print("[proof] DONE", flush=True)