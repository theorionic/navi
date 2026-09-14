"""500M Pool model on FineWeb, 16k BPE vocab, resumable PhaseFeed stream.

Config (workhorse tier, from the 09-08 validation matrix + 09-13 sizing):
  d_model=512, 8 layers, Pool at layers 0/2/4/6 (every other FFN),
  c1=c2=512 x 4 classes = 1.048M slots/block -> 537M value params,
  backbone 28M (16k vocab: embed 16384x512 + 8 blocks + head tied? head
  untied), cand_k=8, side_top=64, score_temp=4.0, Lion 3e-4 core / 3e-3
  mem (mem grad scaled 10x in-step), wd=0.03 core / 0 values.

vs the 09-08 run: vocab 260 -> 16384 (kernel-trained BPE), data source
byte-stream -> PhaseFeed parquet stream (exact per-doc resume), ckpt
keep 3 -> 2 (disk).

Env: NAVI_STEPS (20000), NAVI_BS (256), NAVI_SEQ (512), NAVI_GEN_EVERY
(1000), NAVI_RESUME=1, NAVI_KEEP_CKPT (2).
"""
import sys
sys.path.insert(0, "/kaggle/working/code")
sys.path.insert(0, "/kaggle/working/code/tok")
from functools import partial
import gc
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
from grain_parquet_data import BOS, EOS, PhaseFeed

TOK_PATH = os.environ.get("NAVI_TOK_PATH", "/kaggle/working/tokenizer_16k.json")

TAG = "bpe500m"
STEPS = int(os.environ.get("NAVI_STEPS", "20000"))
BS = int(os.environ.get("NAVI_BS", "256"))
SEQ = int(os.environ.get("NAVI_SEQ", "512"))
GEN_EVERY = int(os.environ.get("NAVI_GEN_EVERY", "1000"))
CAND_K = int(os.environ.get("NAVI_CAND_K", "8"))
TEMP_END = float(os.environ.get("NAVI_TEMP_END", "4.0"))
CKPT_DIR = "/kaggle/working/experiments"
CKPT_PREFIX = "ckpt_bpe500m_step"
GEN_ON = os.environ.get("NAVI_GEN", "1") == "1"
GEN_LEN = 128
VOCAB = 16384

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))


def shard_tree(tree):
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores", None)))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


def reshard_tree(tree):
    """Re-place device arrays onto the mesh without cross-device copies."""
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores", None)))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


def load_ckpt_sharded(path):
    with open(path, "rb") as f:
        st = pickle.load(f)
    p = reshard_tree(st["params"])
    o = reshard_tree(st["opt"])
    return p, o, st["step"], st


def temp_at(step_i):
    f = min(1.0, step_i / 3000)
    return f  # single temp 1.0 in this run (baseline recipe, no anneal knobs)


def loss_fn(model, p, ids, tg, temp):
    out = model.apply(p, ids, train=True, mem_temp=temp)
    return optax.softmax_cross_entropy_with_integer_labels(out, tg).mean()


def is_mem(kp):
    ks = jax.tree_util.keystr(kp)
    return "values" in ks or "/k1" in ks or "/k2" in ks


def make_tx(params):
    core = optax.lion(learning_rate=3e-4, b1=0.9, b2=0.99, weight_decay=0.03)
    mem = optax.lion(learning_rate=3e-4, b1=0.9, b2=0.99, weight_decay=0.0)
    labels = jax.tree_util.tree_map_with_path(
        lambda kp, _: "mem" if is_mem(kp) else "core", params)
    return optax.multi_transform({"core": core, "mem": mem}, labels)


@partial(jax.jit, static_argnames=["model"])
def generate_step(model, params, ids, pos):
    logits = model.apply(params, ids, train=False)
    return logits[:, pos, :]


def generate(model, params, tok, prompt: str, n_tokens: int, temp=0.8, rng=None):
    if rng is None:
        rng = jax.random.PRNGKey(0)
    ids = [BOS] + tok.encode(prompt, add_special_tokens=False).ids
    out = []
    buf = np.zeros((1, SEQ), dtype=np.int32)
    for _ in range(n_tokens):
        buf.fill(0)
        if len(ids) <= SEQ:
            buf[0, :len(ids)] = ids
            pos = len(ids) - 1
        else:
            buf[0, :] = ids[-SEQ:]
            pos = SEQ - 1
        logits = generate_step(model, params, jnp.asarray(buf), jnp.int32(pos))
        logits = logits.at[0, EOS].set(-1e9)
        rng, subkey = jax.random.split(rng)
        if temp <= 0:
            nxt = int(jnp.argmax(logits[0]))
        else:
            nxt = int(jax.random.categorical(subkey, logits / temp)[0])
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out)


PROMPTS = ["The", "Once upon a time", "In 2026, the president of", "Water is"]


def do_generation(model, params, tok, step_i):
    print(f"[{TAG}] ---- GENERATION @ step {step_i} ----", flush=True)
    rng = jax.random.PRNGKey(step_i + 42)
    for pr in PROMPTS:
        rng, subkey = jax.random.split(rng)
        txt = generate(model, params, tok, pr, GEN_LEN, rng=subkey)
def val_loss(model, params, val_tokens, n_batches=8):
    @jax.jit
    def ev(p, ids, tg):
        logits = model.apply(p, ids, train=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()

    rng = np.random.default_rng(7)
    ces = []
    hi = len(val_tokens) - SEQ - 2
    for _ in range(n_batches):
        offs = rng.integers(0, hi, size=16)
        idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
        win = val_tokens[idx]
        ces.append(float(ev(params, jax.device_put(win[:, :-1]),
                            jax.device_put(win[:, 1:]))))
    return float(np.mean(ces)) / np.log(2)
    return float(np.mean(ces)) / np.log(2)

def main():
    print(f"== bpe500m: BS={BS} SEQ={SEQ} STEPS={STEPS} gen@{GEN_EVERY} "
          f"cores={jax.device_count()} ==", flush=True)
    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=CAND_K, side_top=64,
                           n_classes=4, score_temp=TEMP_END)
    cfg_m = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=2,
                        vocab_size=VOCAB)
    model = Navi(cfg_m, mem_cfg)
    p0 = init_params(model, SEQ, jax.random.PRNGKey(0))
    flat = jax.tree_util.tree_flatten_with_path(p0)[0]
    sz = sum(x.size for _, x in flat)
    msz = sum(x.size for k, x in flat if "mem" in jax.tree_util.keystr(k))
    print(f"[{TAG}] params {sz:,} (mem {msz:,})", flush=True)

    p = shard_tree(p0)
    tx = make_tx(p0)
    o = shard_tree(tx.init(p0))
    del p0, flat
    gc.collect()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    feed = PhaseFeed(buffer_mb=1024, val_docs=4000)
    feed.launch()
    if os.environ.get("NAVI_RESUME") != "1":
        # fresh run: ignore any stale state file
        st = feed.load_state()
        if st.get("global_step", 0) or st.get("docs_done", 0):
            feed.state = {"phase": 0, "shard_index": 0, "filename": None,
                          "docs_done": 0, "global_step": 0, "finished": False}
    else:
        feed.load_state()
    ready = feed.wait_ready(min_tokens=BS * SEQ * 8, timeout=900)
    print(f"[{TAG}] feed ready={ready}: buf {len(feed)//(1024*1024)}MB "
          f"val {feed.val_end//1024}KB", flush=True)
    rng = np.random.default_rng(0)
    val = feed.val[:feed.val_end].copy()
    print(f"[{TAG}] val buffer {len(val)/1024/1024:.1f}MB held out", flush=True)

    start = 0
    if os.environ.get("NAVI_RESUME") == "1":
        ckpts = sorted(f for f in os.listdir(CKPT_DIR)
                       if f.startswith(CKPT_PREFIX) and f.endswith(".pkl"))
        if ckpts:
            ckpt_path = os.path.join(CKPT_DIR, ckpts[-1])
            del p, o
            gc.collect()
            p, o, start, _ = load_ckpt_sharded(ckpt_path)
            print(f"[{TAG}] RESUMED from {ckpt_path} at step {start}", flush=True)

    def make_step(temp):
        @jax.jit
        def step(pp, oo, ids, tg):
            g = jax.grad(lambda q, a, t: loss_fn(model, q, a, t, temp))(pp, ids, tg)
            gm = jax.tree_util.tree_map_with_path(
                lambda kp, x: x if is_mem(kp) else jnp.zeros_like(x), g)
            gc_ = jax.tree_util.tree_map_with_path(
                lambda kp, x: jnp.zeros_like(x) if is_mem(kp) else x, g)
            g = jax.tree_util.tree_map_with_path(
                lambda kp, x: x * 10.0 if is_mem(kp) else x, g)
            u, oo2 = tx.update(g, oo, pp)
            nrm = (jnp.sqrt(jax.tree_util.tree_reduce(
                        lambda a, x: a + jnp.sum(x * x), gc_, jnp.float32(0.0))),
                   jnp.sqrt(jax.tree_util.tree_reduce(
                        lambda a, x: a + jnp.sum(x * x), gm, jnp.float32(0.0))))
            return optax.apply_updates(pp, u), oo2, nrm
        return step

    step = make_step(temp_at(0))

    keep = int(os.environ.get("NAVI_KEEP_CKPT", "2"))
    losses, gn_hist, val_hist = [], [], []
    t0 = time.time()
    for i in range(start, STEPS):
        win = feed.batch(rng, BS, SEQ)
        ids = jax.device_put(win[:, :-1], BATCH)
        tg = jax.device_put(win[:, 1:], BATCH)
        p, o, (gn_core, gn_mem) = step(p, o, ids, tg)
        if i % 50 == 0 or i == STEPS - 1:
            l = float(loss_fn(model, p, ids, tg, temp_at(i)))
            losses.append((i, l))
            tps = (i - start + 1) * BS * SEQ / max(1e-9, time.time() - t0)
            eta_s = (STEPS - i - 1) * BS * SEQ / max(1e-9, tps)
            print(f"[{TAG}] step {i:5d}/{STEPS} loss {l:.4f} "
                  f"bpc {l/np.log(2):.4f} gn(core) {gn_core:.3f} "
                  f"gn(mem) {gn_mem:.3f} ({tps/1e3:.0f}k tok/s "
                  f"eta {eta_s/3600:.1f}h) buf {len(feed)//(1024*1024)}MB",
                  flush=True)
        if i % 100 == 99:
            gn_hist.append((i, float(gn_core), float(gn_mem)))
        if i % GEN_EVERY == 0 or i == STEPS - 1:
            v = val_loss(model, p, val)
            val_hist.append((i, v))
            print(f"[{TAG}] VAL@{i} bpc {v:.4f} (held-out)", flush=True)
            if GEN_ON:
                do_generation(model, p, tok, i)
        if i % 500 == 499 or i == STEPS - 1:
            ck = os.path.join(CKPT_DIR, f"{CKPT_PREFIX}{i:06d}.pkl")
            with open(ck + ".tmp", "wb") as f:
                pickle.dump({"params": p, "opt": o, "step": i,
                             "losses": losses, "val_hist": val_hist,
                             "gn_hist": gn_hist}, f)
            os.replace(ck + ".tmp", ck)
            old = sorted(f for f in os.listdir(CKPT_DIR)
                         if f.startswith(CKPT_PREFIX) and f.endswith(".pkl"))
            for f in old[:-keep] if len(old) > keep else []:
                try:
                    os.remove(os.path.join(CKPT_DIR, f))
                except OSError:
                    pass
            print(f"[{TAG}] CKPT saved {ck} (keeping {min(len(old), keep)})",
                  flush=True)
            feed.note_step(i)  # data_state global_step tracks ckpt step

    v = val_loss(model, p, val)
    print(f"[{TAG}] VAL bpc {v:.4f}", flush=True)
    print(f"[{TAG}] VAL_HIST " + " ".join(f"{s}:{b:.4f}" for s, b in val_hist),
          flush=True)
    print(f"[{TAG}] DONE", flush=True)


import time  # noqa: E402

if __name__ == "__main__":
    main()