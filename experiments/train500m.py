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
sys.path.insert(0, "/kaggle/working/code")
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
from fineweb_data import BOS, EOS, FineWebFeed

TAG = "500m"
STEPS = int(os.environ.get("NAVI_STEPS", "20000"))
BS = int(os.environ.get("NAVI_BS", "256"))
SEQ = int(os.environ.get("NAVI_SEQ", "512"))
GEN_EVERY = int(os.environ.get("NAVI_GEN_EVERY", "1000"))
GEN_LEN = 256
# routing-fix levers: cand_k widens the gradient funnel; score_temp
# anneals 1.0 (flat, exploratory) -> NAVI_TEMP_END (sharp, exploit)
# over NAVI_TEMP_WARMUP steps. Baseline (master): cand_k=8, temp=4.0.
CAND_K = int(os.environ.get("NAVI_CAND_K", "8"))
TEMP_START = float(os.environ.get("NAVI_TEMP_START", "4.0"))
TEMP_END = float(os.environ.get("NAVI_TEMP_END", "4.0"))
TEMP_WARMUP = int(os.environ.get("NAVI_TEMP_WARMUP", "3000"))
LB_WEIGHT = float(os.environ.get("NAVI_LB", "0.0"))
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


def reshard_tree(tree):
    """Re-place arrays onto the mesh WITHOUT concatenating or copying data
    through host memory. jax.device_put with a sharding matching the current
    layout is a no-op; with a different one it moves just that array."""
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


def load_ckpt_sharded(path):
    """Pickle a checkpoint and put every array directly on its target
    shard. Loading yields host/numpy arrays; device_put distributes them,
    so HBM never holds a full unsharded replica of params+opt."""
    with open(path, "rb") as f:
        st = pickle.load(f)
    p = reshard_tree(st["params"])
    o = reshard_tree(st["opt"]) if st.get("opt") is not None else None
    return p, o, st["step"]


def grad_norm(g):
    """Global L2 norm of a gradient pytree (host float) - logging only."""
    sq, n = jax.tree_util.tree_reduce(
        lambda acc, x: (acc[0] + jnp.sum(x * x), acc[1] + x.size),
        g, (jnp.float32(0.0), 0))
    return float(jnp.sqrt(sq)), n


def temp_at(step_i):
    """Linear score_temp anneal TEMP_START -> TEMP_END over TEMP_WARMUP."""
    if TEMP_WARMUP <= 0 or TEMP_START == TEMP_END:
        return TEMP_END
    f = min(1.0, step_i / TEMP_WARMUP)
    return TEMP_START + f * (TEMP_END - TEMP_START)

def loss_fn(model, p, ids, tg, temp=None):
    # mem_temp must be POSITIONAL: flax 0.12 mis-traces dynamic kwargs
    # under grad ("NoneType is not iterable")
    if temp is None:
        logits, _aux, lb = model.apply(p, ids, True)
    else:
        logits, _aux, lb = model.apply(p, ids, True, temp)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()
    return ce + LB_WEIGHT * lb


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


# ponytail: mark model as static; keep input shape fixed to (1, SEQ) with pos to avoid recompiling every token
@partial(jax.jit, static_argnames=["model"])
def generate_step(model, params, ids, pos):
    logits = model.apply(params, ids, train=False)
    return logits[:, pos, :]


def generate(model, params, prompt_bytes: bytes, n_tokens: int, temp=0.8, rng=None) -> str:
    """Greedy-ish sampling decode from byte-level model. Params are sharded;
    apply handles replication internally for inference."""
    if rng is None:
        rng = jax.random.PRNGKey(0)
    ids = [BOS] + list(prompt_bytes)
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
        # ponytail: mask BOS and reserved tokens so only valid bytes (0-255) and EOS can be sampled
        logits = logits.at[0, BOS].set(-1e9)
        if logits.shape[-1] > 258:
            logits = logits.at[0, 258:].set(-1e9)
        rng, subkey = jax.random.split(rng)
        if temp <= 0:
            nxt = int(jnp.argmax(logits[0]))
        else:
            nxt = int(jax.random.categorical(subkey, logits / temp)[0])
        if nxt == EOS:
            break
        out.append(nxt)
        ids.append(nxt)
    return bytes(out).decode("utf-8", errors="replace")


PROMPTS = [b"The", b"Once upon a time", b"In 2026, the president of", b"Water is"]


def do_generation(model, params, step_i):
    print(f"[{TAG}] ---- GENERATION @ step {step_i} ----", flush=True)
    rng = jax.random.PRNGKey(step_i + 42)
    for pr in PROMPTS:
        rng, subkey = jax.random.split(rng)
        txt = generate(model, params, pr, GEN_LEN, rng=subkey)
        one = txt.replace("\r", " ").replace("\n", " ")
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
    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=CAND_K, side_top=64, n_classes=4,
                           score_temp=TEMP_END, lb_weight=LB_WEIGHT)
    print(f"[{TAG}] lb_weight {LB_WEIGHT}", flush=True)
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
    del p0, flat  # host-side init copies; not needed once tx is built
    gc.collect()

    feed = FineWebFeed(n_val_docs=4000)
    feed.wait_ready(min_bytes=256 * 1024 * 1024)
    print(f"[{TAG}] feed ready: buf {len(feed.train_buf)//(1024*1024)}MB "
          f"val {feed.val.total//1024}KB", flush=True)
    rng = np.random.default_rng(0)

    # Bucketed-constexpr temperature: temp is a closed Python float,
    # so each distinct bucket triggers one retrace (~7 total). This
    # sidesteps flax 0.12's dynamic-kwarg-under-grad breakage.
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

    _steps = {}  # bucket temp -> compiled step
    def step_for(i):
        t = round(temp_at(i) * 2) / 2  # 0.5-wide buckets -> ~7 compiles
        if t not in _steps:
            print(f"[{TAG}] temp bucket {t} (recompile)", flush=True)
            _steps[t] = make_step(t)
        return _steps[t]

    # checkpoints live in /kaggle/working (persists on notebook commit);
    # keep only the latest NAVI_KEEP_CKPT (default 3) to bound disk use
    ckpt_dir = "/kaggle/working/experiments"
    os.makedirs(ckpt_dir, exist_ok=True)
    keep = int(os.environ.get("NAVI_KEEP_CKPT", "3"))
    start = 0
    if os.environ.get("NAVI_RESUME") == "1":
        ckpts = sorted(f for f in os.listdir(ckpt_dir)
                       if f.startswith("ckpt_500m_step") and f.endswith(".pkl"))
        if ckpts:
            ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
            # must load AFTER freeing the init-time p0/o replicas? p0 is
            # still referenced by shard_tree outputs -- del before load
            del p, o
            gc.collect()
            p, o, start = load_ckpt_sharded(ckpt_path)
            print(f"[{TAG}] RESUMED from {ckpt_path} at step {start} "
                  f"(sharded onto mesh)", flush=True)
    t0 = time.time()
    losses = []
    gn_hist = []  # (step, core_gnorm, mem_gnorm) - trend between log points
    val_hist = []  # (step, val_bpc) - tracked at every gen point
    vb = feed.val.array()
    print(f"[{TAG}] val buffer {len(vb)/1024/1024:.1f}MB held out", flush=True)
    for i in range(start, STEPS):
        win = feed.batch(rng, BS, SEQ)
        ids = jax.device_put(win[:, :-1], BATCH)
        tg = jax.device_put(win[:, 1:], BATCH)
        p, o, (gn_core, gn_mem) = step_for(i)(p, o, ids, tg)
        if i % 50 == 0 or i == STEPS - 1:
            l = float(loss_fn(model, p, ids, tg, temp_at(i)))
            losses.append(l)
            tps = (i - start + 1) * BS * SEQ / max(1e-9, time.time() - t0)
            eta_s = (STEPS - i - 1) * BS * SEQ / max(1e-9, tps)
            print(f"[{TAG}] step {i:5d}/{STEPS} loss {l:.4f} "
                  f"bpc {l/np.log(2):.4f} gn(core) {gn_core:.3f} "
                  f"gn(mem) {gn_mem:.3f} ({tps/1e3:.0f}k tok/s "
                  f"eta {eta_s/3600:.1f}h) buf {len(feed.train_buf)//(1024*1024)}MB",
                  flush=True)
        if i % 100 == 99:  # grad-norm trend between log points
            gn_hist.append((i, float(gn_core), float(gn_mem)))
        if i % GEN_EVERY == 0 or i == STEPS - 1:
            v = val_loss(model, p, vb)
            val_hist.append((i, v))
            print(f"[{TAG}] VAL@{i} bpc {v:.4f} (held-out)", flush=True)
            do_generation(model, p, i)
        if i % 500 == 499 or i == STEPS - 1:
            ck = os.path.join(ckpt_dir, f"ckpt_500m_step{i:06d}.pkl")
            with open(ck + ".tmp", "wb") as f:
                pickle.dump({"params": p, "opt": o, "rng": None, "step": i,
                             "losses": losses, "val_hist": val_hist,
                             "gn_hist": gn_hist}, f)
            os.replace(ck + ".tmp", ck)  # atomic on same fs
            # rotate: newest NAVI_KEEP_CKPT survive
            old = sorted(f for f in os.listdir(ckpt_dir)
                         if f.startswith("ckpt_500m_step") and f.endswith(".pkl"))
            for f in old[:-keep] if len(old) > keep else []:
                try:
                    os.remove(os.path.join(ckpt_dir, f))
                except OSError:
                    pass
            print(f"[{TAG}] CKPT saved {ck} (keeping {min(len(old), keep)})", flush=True)

    v = val_loss(model, p, vb)
    print(f"[{TAG}] VAL bpc {v:.4f}", flush=True)
    print(f"[{TAG}] VAL_HIST " + " ".join(f"{s}:{b:.4f}" for s, b in val_hist), flush=True)
    print(f"[{TAG}] GN_SUMMARY last10 " + " ".join(
        f"{s}:{c:.3f}/{m:.3f}" for s, c, m in gn_hist[-10:]), flush=True)
    print(f"[{TAG}] DONE", flush=True)


if __name__ == "__main__":
    main()