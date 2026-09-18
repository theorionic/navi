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

import os as _os
_os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/kaggle/working/jax_cache")
_os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
_os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")

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
CKPT_DIR = os.environ.get("NAVI_CKPT_DIR", "/kaggle/working/experiments")
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


def _is_namedtuple(t):
    return isinstance(t, tuple) and hasattr(type(t), "_fields")


def _merge_opt_state(new_state, old_state):
    """Migrate a checkpointed optax state onto the CURRENT transform's
    state structure. Arrays (optimizer moments) come from the checkpoint;
    structurally-new leaves (e.g. the schedule `count` added when an LR
    schedule is introduced) start fresh from new_state. Handles optax's
    namedtuple/tuple/dict state nesting."""
    if isinstance(old_state, jax.Array):
        return old_state
    if _is_namedtuple(new_state):
        fields = {}
        for f in type(new_state)._fields:
            vn = getattr(new_state, f)
            vo = getattr(old_state, f, None)
            fields[f] = _merge_opt_state(vn, vo) if vo is not None else vn
        return type(new_state)(**fields)
    if isinstance(new_state, tuple):
        if isinstance(old_state, tuple) and len(old_state) == len(new_state):
            return tuple(_merge_opt_state(a, b)
                         for a, b in zip(new_state, old_state))
        if isinstance(old_state, tuple) and len(old_state) < len(new_state):
            head = [_merge_opt_state(a, b)
                    for a, b in zip(new_state[:len(old_state)], old_state)]
            return tuple(head) + tuple(new_state[len(old_state):])
        return new_state
    if isinstance(new_state, dict) and isinstance(old_state, dict):
        return {k: (_merge_opt_state(v, old_state[k]) if k in old_state else v)
                for k, v in new_state.items()}
    return new_state


def load_ckpt_sharded(path, new_opt_state=None):
    """Load params/opt; when new_opt_state (fresh tx.init() output of the
    CURRENT transform) is given, migrate the loaded state onto that
    structure - required when the tx structure changed across a restart
    (e.g. adding an LR schedule adds a count leaf)."""
    with open(path, "rb") as f:
        st = pickle.load(f)
    p = reshard_tree(st["params"])
    o = reshard_tree(st["opt"])
    if new_opt_state is not None:
        o = _merge_opt_state(new_opt_state, o)
        o = reshard_tree(o)
    return p, o, st["step"], st


def temp_at(step_i):
    # NAVI_TEMP_START: initial softmax temperature. Router grads scale with
    # temp, so 0.0 keeps the router frozen for the whole ramp; 0.5+ lets
    # routing learn from step 0. Ramps to 1.0 over NAVI_TEMP_RAMP steps.
    t0 = float(os.environ.get("NAVI_TEMP_START", "0.0"))
    ramp = float(os.environ.get("NAVI_TEMP_RAMP", "3000"))
    f = t0 + (1.0 - t0) * min(1.0, step_i / ramp)
    return f


def loss_fn(model, p, ids, tg, temp):
    out = model.apply(p, ids, train=True, mem_temp=temp)
    return optax.softmax_cross_entropy_with_integer_labels(out, tg).mean()

def is_mem(kp):
    # keystr uses bracket format: "['params']['block_0']['mem']['k1']".
    # The old "/k1" / "/k2" checks NEVER matched (dot-format assumption),
    # so k1/k2 were silently labeled "core" and trained with Lion+wd --
    # weight decay actively shrinking the router keys. Match 'k1'/'k2'
    # as quoted tokens so 'block_1' can't false-positive.
    ks = jax.tree_util.keystr(kp)
    return "values" in ks or "'k1'" in ks or "'k2'" in ks


LR = float(os.environ.get("NAVI_LR", "3e-4"))
WARMUP = int(os.environ.get("NAVI_WARMUP", "1000"))


def make_tx(params, total_steps):
    """Per-group LR policy (memory-layer structure):
    - backbone + pool keys/router: warmup -> cosine decay to ~5% of peak.
      Late-training key drift re-routes every read; decay locks routing in.
    - pool values: CONSTANT. Sparse top-k touches mean each slot sees ~80
      updates in 20k steps; decaying their LR freezes cold slots at stale
      values. Kept plastic to the end by design.
    """
    sched_core = optax.warmup_cosine_decay_schedule(
        init_value=LR * 0.05, peak_value=LR,
        warmup_steps=WARMUP, decay_steps=max(1, total_steps - WARMUP),
        end_value=LR * 0.05)
    core = optax.lion(learning_rate=sched_core, b1=0.9, b2=0.99,
                      weight_decay=0.03)
    # mem values get their OWN lr (default = LR). NAVI_MEM_LR=3e-3 is the
    # documented intent that was never wired: sign-based Lion ignores the
    # grad-scale hack, so value consolidation needs a real lr bump.
    mem_lr = float(os.environ.get("NAVI_MEM_LR", str(LR)))
    # ISSUE-01 deeper fix: Lion caps each value slot's effective step at
    # mem_lr no matter how many tokens read it in a step (grad summed over
    # batch*positions, then sign() collapses touch count into one vote).
    # Adam accumulates a per-slot momentum across touches (m/sqrt(v)
    # normalization), so a hot slot that is consistently read gets
    # lr-sized consolidation steps that COMPOUND - the "per-slot
    # accumulated momentum / bigger effective steps per touch" lever from
    # ISSUES.md. A/B'd against the Lion baseline (mem_lr=3e-4) on the
    # base-vs-zero ablation battery; NAVI_MEM_OPT=adam|lion selects.
    mem_opt = os.environ.get("NAVI_MEM_OPT", "adam")
    if mem_opt == "adam":
        mem = optax.adam(learning_rate=mem_lr, b1=0.9, b2=0.999)
    else:
        mem = optax.lion(learning_rate=mem_lr, b1=0.9, b2=0.99,
                         weight_decay=0.0)
    labels = jax.tree_util.tree_map_with_path(
        lambda kp, _: "mem" if is_mem(kp) else "core", params)
    return optax.multi_transform({"core": core, "mem": mem}, labels)


# ---- generation: fully jitted lax.scan decode, batched over prompts ----
# The old per-token eager loop (model-apply + logits.set + categorical +
# int() host sync, ~2.5k dispatch/sync round-trips per burst) livelocked the
# TPU driver: one driver thread pinned at 100% CPU, a device->host transfer
# that never completed. The scan compiles ONCE and runs the whole decode
# on-device; the only host sync is the final device_get of all samples.

PROMPTS = ["The", "Once upon a time", "In 2026, the president of", "Water is"]

@partial(jax.jit, static_argnames=("model", "n_steps"))
def _gen_scan(model, params, ids, pos0, n_steps, temp, seed=0):
    """ids: (G, SEQ) int32, prompts right-packed from column 0 (PAD=0 beyond).
    pos0: scalar, index of the last prompt token. Returns (n_steps, G) ids."""
    G = ids.shape[0]
    rows = jnp.arange(G)

    def body(carry, _):
        ids, pos, rng = carry
        logits = model.apply(params, ids, train=False)
        lg = logits[rows, pos]
        lg = lg.at[..., EOS].set(-1e9)
        rng, sk = jax.random.split(rng)
        nxt = jax.random.categorical(sk, lg / jnp.maximum(temp, 1e-6))
        ids = ids.at[rows, jnp.minimum(pos + 1, SEQ - 1)].set(nxt)
        return (ids, jnp.minimum(pos + 1, SEQ - 1), rng), nxt

    rng = jax.random.PRNGKey(seed)
    (ids, pos, rng), out = jax.lax.scan(body, (ids, pos0, rng), None,
                                        length=n_steps)
    return out


def generate(model, params, tok, prompts, n_tokens, temp=0.8, seed=0):
    """Right-packed prompts, read at pos, write sample at pos+1 - identical
    semantics to the old eager loop, minus the 2.5k host round-trips."""
    G = len(prompts)
    buf = np.zeros((G, SEQ), dtype=np.int32)
    pos0 = 0
    for g, pr in enumerate(prompts):
        ids = [BOS] + tok.encode(pr, add_special_tokens=False).ids
        ids = ids[:SEQ - n_tokens]          # keep room for the continuation
        buf[g, :len(ids)] = ids
        pos0 = max(pos0, len(ids) - 1)
    out = _gen_scan(model, params, jnp.asarray(buf), jnp.int32(pos0),
                    n_tokens, temp, seed)
    out = jax.device_get(out)               # the ONE device->host sync
    return [tok.decode([int(t) for t in out[:, g]]) for g in range(G)]


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


def pool_coverage(model_ra, params, val_tokens, n_batches=2):
    """Distinct-slot coverage per memory block over fresh val windows.

    Counts distinct (class, slot) reads the router actually selects, per
    memory block, as a fraction of the block's c1*c2*n_classes slot space.
    aux['mem_L'] is (b, l, classes*cand_k), CLASS-MAJOR (pkm.py packs
    slots.reshape(b, l, -1) from (b, l, classes, cand_k)).
    """
    n_slots = (int(os.environ.get("NAVI_C_POOL", "512")) ** 2) * 4
    rng = np.random.default_rng(11)
    hi = len(val_tokens) - SEQ - 2
    per_block = {}
    for _ in range(n_batches):
        offs = rng.integers(0, hi, size=16)
        idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
        win = val_tokens[idx]
        ids = jax.device_put(win[:, :-1])
        _, aux, _ = model_ra.apply(params, ids, train=False)
        for name, s in aux.items():
            arr = np.asarray(s)                     # (b, l, classes*cand_k)
            sets = per_block.setdefault(name, set())
            for cls in range(arr.shape[-1] // CAND_K):
                sl = arr[..., cls * CAND_K:(cls + 1) * CAND_K].reshape(-1)
                sets.update((cls, int(v)) for v in sl)
    names = sorted(per_block)
    return [len(per_block[n]) / n_slots for n in names]


def do_generation(model, params, tok, step_i):
    print(f"[{TAG}] ---- GENERATION @ step {step_i} ----", flush=True)
    texts = generate(model, params, tok, PROMPTS, GEN_LEN, temp=0.8,
                     seed=step_i + 42)
    for pr, txt in zip(PROMPTS, texts):
        one = txt.replace("\r", " ").replace("\n", " ")
        print(f"[{TAG}] GEN [{pr!r}] -> {one[:160]}", flush=True)

def main():
    print(f"== bpe500m: BS={BS} SEQ={SEQ} STEPS={STEPS} gen@{GEN_EVERY} "
          f"cores={jax.device_count()} ==", flush=True)
    D_MODEL = int(os.environ.get("NAVI_D_MODEL", "512"))
    N_LAYERS = int(os.environ.get("NAVI_N_LAYERS", "8"))
    N_HEADS = int(os.environ.get("NAVI_N_HEADS", "8"))
    C_POOL = int(os.environ.get("NAVI_C_POOL", "512"))
    SIDE_TOP = int(os.environ.get("NAVI_SIDE_TOP", "64"))
    mem_cfg = MemoryConfig(c1=C_POOL, c2=C_POOL, cand_k=CAND_K,
                           side_top=SIDE_TOP, n_classes=4,
                           score_temp=TEMP_END,
                           lb_weight=float(os.environ.get("NAVI_LB_WEIGHT", "0.0")),
                           lb_eps=float(os.environ.get("NAVI_LB_EPS", "0.0")))
    cfg_m = ModelConfig(d_model=D_MODEL, n_layers=N_LAYERS,
                        n_heads=N_HEADS, memory_every=2, vocab_size=VOCAB)
    model = Navi(cfg_m, mem_cfg)
    model_ra = Navi(cfg_m, mem_cfg, return_aux=True)  # coverage eval only
    p0 = init_params(model, SEQ, jax.random.PRNGKey(0))
    flat = jax.tree_util.tree_flatten_with_path(p0)[0]
    sz = sum(x.size for _, x in flat)
    msz = sum(x.size for k, x in flat if "mem" in jax.tree_util.keystr(k))
    print(f"[{TAG}] params {sz:,} (mem {msz:,})", flush=True)

    p = shard_tree(p0)
    tx = make_tx(p0, STEPS)
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
    # Pre-fill the data buffer to its FULL capacity (or 60s of producer
    # work, whichever first) before stepping. Previously readiness was
    # BS*SEQ*8 tokens (~65KB) - training started while the producer thread
    # was still downloading+tokenizing the first parquet shard, and its
    # GIL-heavy python work starved the training loop's dispatch path:
    # steps 0-300 ran at 5-7s instead of 213ms until the buffer filled.
    # Waiting for a full buffer moves that cost into startup, where it
    # overlaps the step compile, instead of into the first 300 steps.
    _prefill = min(feed.cap, int(os.environ.get("NAVI_PREFILL_MB", "512")) * 1024 * 1024)
    ready = feed.wait_ready(min_tokens=_prefill, timeout=900)
    print(f"[{TAG}] feed ready={ready}: buf {len(feed)//(1024*1024)}MB "
          f"(prefill target {_prefill//(1024*1024)}MB) "
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
            fresh_o = o
            del p
            gc.collect()
            p, o, start, _ = load_ckpt_sharded(ckpt_path, new_opt_state=fresh_o)
            del fresh_o   # drop the init-time opt-state copy: HBM is tight
            gc.collect()
            print(f"[{TAG}] RESUMED from {ckpt_path} at step {start}", flush=True)

    # temp is a TRACED argument, not a Python closure float: a closure float
    # changes identity whenever the schedule moves, and each distinct float
    # bakes a new constant into the HLO -> full recompile (O(minutes) for the
    # 500M graph) per temp value. As a traced scalar it rides along with the
    # batch and the step compiles exactly once.
    @jax.jit
    def step(pp, oo, ids, tg, temp):
        l, g = jax.value_and_grad(
            lambda q, a, t: loss_fn(model, q, a, t, temp))(pp, ids, tg)
        u, oo2 = tx.update(g, oo, pp)
        return optax.apply_updates(pp, u), oo2, l, g

    # Grad norms for the log line: the step returns the loss and grads;
    # norms are computed from the last grads ONLY at log steps (one host
    # sync per 50 steps, none between). The dead 10x mem grad-scale is gone:
    # per-group LRs now live solely in make_tx (NAVI_MEM_LR).
    @jax.jit
    def grad_norms(g):
        gm = jax.tree_util.tree_map_with_path(
            lambda kp, x: x if is_mem(kp) else jnp.zeros_like(x), g)
        gc_ = jax.tree_util.tree_map_with_path(
            lambda kp, x: jnp.zeros_like(x) if is_mem(kp) else x, g)
        return (
            jnp.sqrt(jax.tree_util.tree_reduce(
                lambda a, x: a + jnp.sum(x * x), gc_, jnp.float32(0.0))),
            jnp.sqrt(jax.tree_util.tree_reduce(
                lambda a, x: a + jnp.sum(x * x), gm, jnp.float32(0.0))))

    # [dbg:grads] NAVI_GRAD_DEBUG=1: per-parameter-group grad norms + param
    # norms for EVERY pool tensor (w_q/k1/k2/values/w_o per block) plus the
    # backbone (embed/head/attn/ff). This is the kill-shot diagnostic: it
    # shows exactly which link in the chain receives zero learning signal.
    DBG_GRADS = os.environ.get("NAVI_GRAD_DEBUG", "0") == "1"

    def _group_of(ks):
        if "'values'" in ks:
            return "values"
        if "'k1'" in ks or "'k2'" in ks:
            return "keys"
        if "'w_q'" in ks:
            return "w_q"
        if "'w_o'" in ks:
            return "w_o"
        return "core"

    @jax.jit
    def grad_debug(g, pp):
        groups = {"values": jnp.float32(0.0), "keys": jnp.float32(0.0),
                  "w_q": jnp.float32(0.0), "w_o": jnp.float32(0.0),
                  "core": jnp.float32(0.0)}
        for kp, x in jax.tree_util.tree_flatten_with_path(g)[0]:
            k = _group_of(jax.tree_util.keystr(kp))
            groups[k] = groups[k] + jnp.sum(x.astype(jnp.float32) ** 2)
        pnorms = dict(groups)
        for kp, x in jax.tree_util.tree_flatten_with_path(pp)[0]:
            k = _group_of(jax.tree_util.keystr(kp))
            pnorms[k] = pnorms[k] + jnp.sum(x.astype(jnp.float32) ** 2)
        return {k: jnp.sqrt(v) for k, v in groups.items()}, \
            {k: jnp.sqrt(v) for k, v in pnorms.items()}

    keep = int(os.environ.get("NAVI_KEEP_CKPT", "2"))
    losses, gn_hist, val_hist, cov_hist = [], [], [], []
    t0 = time.time()
    t_last = t0
    ema_ms = None       # EMA of per-step wall time (steady-state pace)
    g_last = None
    for i in range(start, STEPS):
        win = feed.batch(rng, BS, SEQ)
        ids = jax.device_put(win[:, :-1], BATCH)
        tg = jax.device_put(win[:, 1:], BATCH)
        p, o, l, g_last = step(p, o, ids, tg, temp_at(i))
        now = time.time()
        step_ms = (now - t_last) * 1000.0
        t_last = now
        ema_ms = step_ms if ema_ms is None else 0.9 * ema_ms + 0.1 * step_ms
        if i % 50 == 0 or i == STEPS - 1:
            gn_core, gn_mem = jax.device_get(grad_norms(g_last))
            losses.append((i, float(l)))
            el = now - t0
            tps = (i - start + 1) * BS * SEQ / max(1e-9, el)
            inst_tps = BS * SEQ / max(1e-9, ema_ms / 1000.0)
            eta_s = (STEPS - i - 1) * max(ema_ms, 1e-9) / 1000.0
            print(f"[{TAG}] step {i:5d}/{STEPS} loss {float(l):.4f} "
                  f"bpc {float(l)/np.log(2):.4f} gn(core) {gn_core:.3f} "
                  f"gn(mem) {gn_mem:.3f} "
                  f"step {ema_ms:.0f}ms inst {inst_tps/1e3:.0f}k tok/s "
                  f"avg {tps/1e3:.0f}k tok/s eta {eta_s/3600:.1f}h "
                  f"buf {len(feed)//(1024*1024)}MB "
                  f"temp {float(temp_at(i)):.3f}",
                  flush=True)
            if DBG_GRADS:
                gnorms, pnorms = jax.device_get(grad_debug(g_last, p))
                print(f"[{TAG}] [dbg:grads] step {i} " +
                      " ".join(f"{k}:g={gnorms[k]:.4e}/p={pnorms[k]:.4e}"
                               for k in ("values", "keys", "w_q", "w_o",
                                         "core")), flush=True)
        if i % 100 == 99:
            gn_hist.append((i, float(gn_core), float(gn_mem)))  # from log block
        if i % GEN_EVERY == 0 or i == STEPS - 1:
            v = val_loss(model, p, val)
            val_hist.append((i, v))
            print(f"[{TAG}] VAL@{i} bpc {v:.4f} (held-out)", flush=True)
            cov = pool_coverage(model_ra, p, val)
            cov_hist.append((i, cov))
            print(f"[{TAG}] COV@{i} " +
                  " ".join(f"b{bi}={c*100:.1f}%" for bi, c in enumerate(cov)),
                  flush=True)
            ck = os.path.join(CKPT_DIR, f"{CKPT_PREFIX}{i:06d}.pkl")
            os.makedirs(CKPT_DIR, exist_ok=True)
            with open(ck + ".tmp", "wb") as f:
                pickle.dump({"params": p, "opt": o, "step": i,
                             "losses": losses, "val_hist": val_hist,
                             "gn_hist": gn_hist, "cov_hist": cov_hist}, f)
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
    el = time.time() - t0
    print(f"[{TAG}] VAL final bpc {v:.4f} | total {el/3600:.2f}h "
          f"| steady pace {ema_ms:.0f}ms/step "
          f"| {STEPS - start} steps {BS*SEQ*max(1,STEPS-start)/max(1e-9,el)/1e3:.0f}k tok/s avg",
          flush=True)
    print(f"[{TAG}] VAL_HIST " + " ".join(f"{s}:{b:.4f}" for s, b in val_hist),
          flush=True)
    print(f"[{TAG}] DONE", flush=True)


import time  # noqa: E402

if __name__ == "__main__":
    main()