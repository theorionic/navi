"""500M-class Pool model on FineWeb, streaming, with periodic text generation.

Model: d_model=512, 8 layers, Pool replacing every other FFN (layers 0,2,4,6),
c1=c2=512 -> 537M value params + ~19M backbone ~= 556M total. Byte-level
(260 vocab) so bpc stays comparable with the enwik8 runs.

Data: HuggingFace fineweb/sample-10BT streamed over the network into a 1GB
rolling byte buffer (the prefetch); batches are random windows off the buffer.
First ~4000 docs go to a held-out val buffer (24MB).

Generation: every GEN_EVERY steps, greedy-decode GEN_LEN bytes from fixed
prompts so training quality is visible in the log itself.

Env: NAVI_STEPS (default 20000), NAVI_GEN_EVERY (1000), NAVI_BS (256),
NAVI_SEQ (512), NAVI_RESUME=1 to resume from ckpt_500m.pkl.
"""
import sys
sys.path.insert(0, "/kaggle/working")
from functools import partial
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi
from navi.train import init_params
from fineweb_data import BOS, EOS, FineWebFeed

TAG = "500m"
STEPS = int(os.environ.get("NAVI_STEPS", "20000"))
BS = int(os.environ.get("NAVI_BS", "256"))
SEQ = int(os.environ.get("NAVI_SEQ", "512"))
GEN_EVERY = int(os.environ.get("NAVI_GEN_EVERY", "1000"))
GEN_LEN = 256
CKPT = os.path.expanduser("~/experiments/ckpt_500m.pkl")

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))


def shard_tree(tree):
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:  # slot tables: shard the slot axis across cores
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


def loss_fn(model, p, ids, tg):
    logits = model.apply(p, ids, train=True)
    return optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()


def is_mem(kp):
    ks = jax.tree_util.keystr(kp)
    return "values" in ks or "/k1" in ks or "/k2" in ks


def make_tx(params):
    # Lion: momentum-only optimizer state (1x params, not AdamW's 2x) --
    # the 537M value tables would OOM with replicated m+v on 8x16GB.
    core = optax.lion(learning_rate=3e-4, b1=0.9, b2=0.99, weight_decay=0.03)
    mem = optax.lion(learning_rate=3e-3, b1=0.9, b2=0.99, weight_decay=0.0)
    labels = jax.tree_util.tree_map_with_path(
        lambda kp, x: "mem" if is_mem(kp) else "core", params)
    return optax.multi_transform({"core": core, "mem": mem}, labels)


# ponytail: mark model as static since it's a Flax Module, not a JAX array
@partial(jax.jit, static_argnames=["model"])
def generate_step(model, params, ids):
    logits = model.apply(params, ids[:, -SEQ:], train=False)
    return logits[:, -1, :]
def generate(model, params, prompt_bytes: bytes, n_tokens: int, temp=0.8) -> str:
    """Greedy-ish sampling decode from byte-level model. Params are sharded;
    apply handles replication internally for inference."""
    ids = [BOS] + list(prompt_bytes)
    out = []
    for _ in range(n_tokens):
        logits = generate_step(model, params, jnp.array([ids[-SEQ:]], dtype=jnp.int32))
        logits = logits / temp
        nxt = int(jax.random.categorical(jax.random.PRNGKey(len(ids)), logits)[0])
        if nxt == EOS:
            break
        out.append(nxt)
        ids.append(nxt)
    return bytes(out).decode("utf-8", errors="replace")


PROMPTS = [b"The", b"Once upon a time", b"In 2026, the president of", b"Water is"]


def do_generation(model, params, step_i):
    print(f"[{TAG}] ---- GENERATION @ step {step_i} ----", flush=True)
    for pr in PROMPTS:
        txt = generate(model, params, pr, GEN_LEN)
        one = txt.replace("\n", " ")
        print(f"[{TAG}] GEN {pr.decode()!r} -> {one[:300]}", flush=True)
    print(f"[{TAG}] ---- END GEN ----", flush=True)


def val_loss(model, params, val_bytes, n_batches=8):
    @jax.jit
    def ev(p, ids, tg):
        logits = model.apply(p, ids, train=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()

    rng = np.random.default_rng(7)
    ces = []
    for _ in range(n_batches):
        offs = rng.integers(0, len(val_bytes) - SEQ - 2, size=128)
        idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
        win = val_bytes[idx]
        ces.append(float(ev(params, jax.device_put(win[:, :-1]),
                            jax.device_put(win[:, 1:]))))
    return float(np.mean(ces)) / np.log(2)


def main():
    print(f"== 500m: BS={BS} SEQ={SEQ} STEPS={STEPS} gen@{GEN_EVERY} "
          f"cores={jax.device_count()} ==", flush=True)
    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                           score_temp=4.0)
    cfg_m = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=2,
                        vocab_size=260)
    model = Navi(cfg_m, mem_cfg)
    p0 = init_params(model, SEQ, jax.random.PRNGKey(0))
    flat = jax.tree_util.tree_flatten_with_path(p0)[0]
    sz = sum(x.size for _, x in flat)
    msz = sum(x.size for k, x in flat if "mem" in jax.tree_util.keystr(k))
    print(f"[{TAG}] params {sz:,} (mem {msz:,})", flush=True)

    p = shard_tree(p0)
    tx = make_tx(p0)
    o = shard_tree(tx.init(p0))

    feed = FineWebFeed(n_val_docs=4000)
    feed.wait_ready(min_bytes=256 * 1024 * 1024)
    print(f"[{TAG}] feed ready: buf {len(feed.train_buf)//(1024*1024)}MB "
          f"val {feed.val.total//1024}KB", flush=True)
    rng = np.random.default_rng(0)

    @jax.jit
    def step(pp, oo, ids, tg):
        g = jax.grad(lambda q, a, t: loss_fn(model, q, a, t))(pp, ids, tg)
        g = jax.tree_util.tree_map_with_path(
            lambda kp, x: x * 10.0 if "mem" in jax.tree_util.keystr(kp) else x, g)
        u, oo2 = tx.update(g, oo, pp)
        return optax.apply_updates(pp, u), oo2

    ckpt_path = os.path.expanduser("~/experiments/ckpt_500m.pkl")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    start = 0
    if os.path.exists(ckpt_path) and os.environ.get("NAVI_RESUME") == "1":
        with open(ckpt_path, "rb") as f:
            st = pickle.load(f)
        p, o, start = st["params"], st["opt"], st["step"] + 1
        print(f"[{TAG}] RESUMED at step {start}", flush=True)

    t0 = time.time()
    losses = []
    for i in range(start, STEPS):
        win = feed.batch(rng, BS, SEQ)
        ids = jax.device_put(win[:, :-1], BATCH)
        tg = jax.device_put(win[:, 1:], BATCH)
        p, o = step(p, o, ids, tg)
        if i % 50 == 0 or i == STEPS - 1:
            l = float(loss_fn(model, p, ids, tg))
            losses.append(l)
            tps = (i - start + 1) * BS * SEQ / max(1e-9, time.time() - t0)
            print(f"[{TAG}] step {i:5d} loss {l:.4f} bpc {l/np.log(2):.4f} "
                  f"({tps/1e3:.0f}k tok/s) buf {len(feed.train_buf)//(1024*1024)}MB",
                  flush=True)
        if i % GEN_EVERY == 0 or i == STEPS - 1:
            do_generation(model, p, i)
        if i % 500 == 499 or i == STEPS - 1:
            with open(ckpt_path + ".tmp", "wb") as f:
                pickle.dump({"params": p, "opt": o, "rng": None, "step": i,
                             "losses": losses}, f)
            os.replace(ckpt_path + ".tmp", ckpt_path)
            print(f"[{TAG}] CKPT saved at step {i}", flush=True)

    vb = feed.val.array()
    print(f"[{TAG}] VAL bpc {val_loss(model, p, vb):.4f}", flush=True)
    print(f"[{TAG}] DONE", flush=True)


if __name__ == "__main__":
    main()