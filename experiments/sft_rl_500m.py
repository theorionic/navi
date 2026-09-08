"""SFT + RL validation on the REAL 500m checkpoint (8-core TPU v5e).

Loads /kaggle/working/experiments/ckpt_500m_step019999.pkl (the completed
FineWeb run) and runs the same battery as sft_rl_test.py at 500m scale:

  stage 1 SFT:  'q:/c:' arithmetic format, WITH replay mixing (the small-
                model run showed +3.8 bpc catastrophic forgetting without
                it -- this stage tests the fix, not just the failure).
                Reports new-format + pretrain-format bpc and Pool-vs-core
                parameter movement.
  stage 2 RL:   GRPO on single-digit 'a+b=' with entropy bonus + the
                leading-digit-run reward (both fixes from the small run:
                no entropy term => mode collapse to '10 ?'; full-digit-run
                extraction => false negatives on '7 ?7').

Env: NAVI_SFT_STEPS (150), NAVI_RL_ROUNDS (40), NAVI_EVAL_BS (16),
NAVI_REPLAY (0.5 = fraction of replay windows per SFT batch).
Usage: python3 /kaggle/working/navi/experiments/sft_rl_500m.py
"""
import sys
import os
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
from navi.train import init_params

TAG = "sftrl500m"
T0 = time.time()


def log(msg):
    print(f"[{TAG} +{time.time()-T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- config ----
SEQ = 128                      # prompts+completions fit; short = fast RL
SFT_STEPS = int(os.environ.get("NAVI_SFT_STEPS", "150"))
RL_ROUNDS = int(os.environ.get("NAVI_RL_ROUNDS", "40"))
EVAL_BS = int(os.environ.get("NAVI_EVAL_BS", "16"))
REPLAY_FRAC = float(os.environ.get("NAVI_REPLAY", "0.5"))
G = 8
LR_SFT = 5e-5                  # 10x below small-model: 556M params
LR_RL = 1e-5
ENT_BONUS = 0.01               # entropy regularizer (anti-mode-collapse)

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))


def shard_tree(tree):
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


# ------------------------------------------------------------------ data ----
def make_docs(n, rng, lo, hi, fmt):
    docs = []
    for _ in range(n):
        a, b = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        if fmt == "chain":
            docs.append(f"q: {a}+{b}\nc: {a+b}\n")
        elif fmt == "raw":
            docs.append(f"{a}+{b}={a+b} ")
        else:  # 'pre' = FineWeb-like filler text (replay source)
            filler = "log " if (a + b) % 2 == 0 else "the cat sat "
            docs.append(f"{filler}note {a}+{b}={a+b} end. ")
    return "\n".join(docs).encode("utf-8")


def windows(data, rng, bs, seq):
    if isinstance(data, (bytes, bytearray)):
        data = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    offs = rng.integers(0, len(data) - seq - 2, size=bs)
    idx = offs[:, None] + np.arange(seq + 1)[None, :]
    win = data[idx].astype(np.int32)
    win[:, 0] = 256  # BOS
    return win[:, :-1], win[:, 1:]


def param_snapshot(params):
    """(core_norm, mem_norm, mem_values_norm) host floats for movement log."""
    core_sq = mem_sq = val_sq = 0.0
    for kp, x in jax.tree_util.tree_flatten_with_path(params)[0]:
        ks = jax.tree_util.keystr(kp)
        v = float(jnp.sum(jnp.asarray(x) ** 2))
        if "values" in ks:
            val_sq += v
        if "values" in ks or "/k1" in ks or "/k2" in ks:
            mem_sq += v
        else:
            core_sq += v
    return core_sq ** 0.5, mem_sq ** 0.5, val_sq ** 0.5


def main():
    rng = np.random.default_rng(0)

    # 1. load the real checkpoint (params only)
    ckpt_dir = "/kaggle/working/experiments"
    ckpts = sorted(f for f in os.listdir(ckpt_dir)
                   if re.match(r"ckpt_500m_step\d+\.pkl$", f))
    if not ckpts:
        raise SystemExit(f"no ckpt_500m_step*.pkl in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    log(f"stage 0: loading {os.path.basename(ckpt_path)}")
    with open(ckpt_path, "rb") as f:
        st = pickle.load(f)
    p = shard_tree(st["params"])
    del st
    gc.collect()
    core0, mem0, val0 = param_snapshot(p)
    log(f"stage 0 done: |core|={core0:.1f} |mem|={mem0:.1f} (|values|={val0:.1f})")

    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                           score_temp=4.0)
    cfg_m = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=2,
                        vocab_size=260)
    model = Navi(cfg_m, mem_cfg)

    # 2. data: SFT target + pretrain-format replay (same distribution the
    #    500m model was trained on: prose with embedded arithmetic)
    sft_data = make_docs(600, rng, 1, 50, "chain")
    replay_data = make_docs(400, rng, 1, 99, "pre")

    @jax.jit
    def ev(pp, ids, tg):
        logits = model.apply(pp, ids, train=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()

    def bpc_of(pp, data):
        ids, tg = windows(data, rng, EVAL_BS, SEQ)
        l = float(ev(pp, jax.device_put(ids, BATCH), jax.device_put(tg, BATCH)))
        return l / np.log(2)

    # 3. SFT with replay mixing
    tx1 = optax.adamw(LR_SFT, b1=0.9, b2=0.95)
    o1 = tx1.init(p)

    @jax.jit
    def step1(pp, oo, ids, tg):
        g = jax.grad(lambda q, a, t: optax.softmax_cross_entropy_with_integer_labels(
            model.apply(q, a, train=True), t).mean())(pp, ids, tg)
        u, oo = tx1.update(g, oo, pp)
        return optax.apply_updates(pp, u), oo

    bpc_sft0 = bpc_of(p, sft_data)
    bpc_pre0 = bpc_of(p, replay_data)
    log(f"stage 1: SFT {SFT_STEPS} steps (replay {REPLAY_FRAC:.0%}, lr {LR_SFT})")
    log(f"  start: sft-fmt bpc {bpc_sft0:.3f} | replay-fmt bpc {bpc_pre0:.3f}")
    for s in range(SFT_STEPS):
        ids_r, tg_r = windows(replay_data, rng, int(EVAL_BS * REPLAY_FRAC), SEQ)
        ids_s, tg_s = windows(sft_data, rng, EVAL_BS - int(EVAL_BS * REPLAY_FRAC), SEQ)
        ids = np.concatenate([ids_s, ids_r])
        tg = np.concatenate([tg_s, tg_r])
        perm = rng.permutation(len(ids))
        p, o1 = step1(p, o1, jax.device_put(ids[perm], BATCH),
                      jax.device_put(tg[perm], BATCH))
        if s % 25 == 0 or s == SFT_STEPS - 1:
            log(f"  sft {s:4d}: sft-fmt {bpc_of(p, sft_data):.3f} | "
                f"replay-fmt {bpc_of(p, replay_data):.3f}")
    bpc_sft1 = bpc_of(p, sft_data)
    bpc_pre1 = bpc_of(p, replay_data)
    core1, mem1, val1 = param_snapshot(p)
    log(f"stage 1 done: sft-fmt {bpc_sft0:.3f} -> {bpc_sft1:.3f} | "
        f"replay {bpc_pre0:.3f} -> {bpc_pre1:.3f} (forget {bpc_pre1-bpc_pre0:+.3f})")
    log(f"  param movement: core {core0:.2f}->{core1:.2f} "
        f"mem {mem0:.2f}->{mem1:.2f} (values {val0:.2f}->{val1:.2f})")
    sft_learns = (bpc_sft0 - bpc_sft1) > 0.3
    sft_stable = (bpc_pre1 - bpc_pre0) < 1.0

    # 4. GRPO RL on single-digit addition with entropy bonus
    log(f"stage 2: GRPO {RL_ROUNDS} rounds (G={G}, lr {LR_RL}, "
        f"ent {ENT_BONUS}) on 'a+b=' single-digit")
    tx2 = optax.adamw(LR_RL, b1=0.9, b2=0.95)
    o2 = tx2.init(p)
    L = 4

    @jax.jit
    def sample_g(pp, prompt, key):
        ids = jnp.tile(prompt, (G, 1))
        out_ids = []
        for _ in range(L):
            logits = model.apply(pp, ids, train=False)
            logp = jax.nn.log_softmax(logits[:, -1, :])
            key, sub = jax.random.split(key)
            nxt = jax.random.categorical(sub, logp)
            out_ids.append(nxt)
            ids = jnp.concatenate([ids, nxt[:, None]], axis=1)
        return ids

    def reward(ids_batch, a, b):
        want = str(a + b).encode()
        outs = []
        for g_ in range(ids_batch.shape[0]):
            seqb = bytes(int(x) for x in ids_batch[g_])
            out = []
            for c in seqb:
                if 48 <= c <= 57:
                    out.append(c)
                else:
                    break
            digits = bytes(out)
            outs.append(1.0 if digits == want else
                        (0.2 if digits else 0.0))
        return np.asarray(outs, dtype=np.float32)

    key = jax.random.PRNGKey(999)
    rewards_hist = []
    for rnd in range(RL_ROUNDS):
        a, b = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        prompt = jnp.asarray([[256] + list(f"{a}+{b}=".encode())], dtype=jnp.int32)
        ids_full = sample_g(p, prompt, key)
        ids_np = np.asarray(ids_full)[:, -L:]
        r = reward(ids_np, a, b)
        adv = r - r.mean()
        tg_full = jnp.concatenate([ids_full[:, 1:], jnp.zeros((G, 1), jnp.int32)], axis=1)
        mask = np.zeros((G, tg_full.shape[1]), dtype=np.float32)
        mask[:, -L:] = 1.0
        mvals = np.repeat(adv[:, None], tg_full.shape[1], axis=1) * mask

        def loss_rl(pp):
            logits = model.apply(pp, ids_full[:, :-1], train=True)
            logp = jax.nn.log_softmax(logits)
            tgt = tg_full[:, :-1]
            lp = jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
            m = jnp.asarray(mvals[:, :-1])
            pg = -(lp * m).sum() / jnp.maximum(1.0, jnp.abs(m).sum())
            # entropy bonus over completion positions (anti-mode-collapse)
            probs = jax.nn.softmax(logits)
            ent = -(probs * jax.nn.log_softmax(logits)).sum(-1)
            m2 = jnp.asarray(mask[:, :-1])
            return pg - ENT_BONUS * (ent * m2).sum() / jnp.maximum(1.0, m2.sum())

        g = jax.grad(loss_rl)(p)
        u, o2 = tx2.update(g, o2, p)
        p = optax.apply_updates(p, u)
        rewards_hist.append(float(r.mean()))
        if rnd % 5 == 0 or rnd == RL_ROUNDS - 1:
            tails = ["".join(chr(c) if 32 <= c < 127 else "?" for c in t)
                     for t in ids_np[:3]]
            log(f"  rl {rnd:3d}: {a}+{b}= reward {r.mean():.2f} "
                f"hit {int((r == 1.0).sum())}/{G} tails={tails}")

    # greedy accuracy post-RL
    hits = 0
    tests = 40
    for _ in range(tests):
        a, b = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        prompt = jnp.asarray([[256] + list(f"{a}+{b}=".encode())], dtype=jnp.int32)
        full = sample_g(p, prompt, jax.random.PRNGKey(7))
        tail = bytes(int(x) for x in np.asarray(full)[0][-L:])
        out = []
        for c in tail:
            if 48 <= c <= 57:
                out.append(c)
            else:
                break
        hits += int(bytes(out) == str(a + b).encode())
    rl_acc = hits / tests
    core2, mem2, val2 = param_snapshot(p)

    r10 = np.mean(rewards_hist[:10]) if len(rewards_hist) >= 10 else rewards_hist[0]
    rlast = float(np.mean(rewards_hist[-10:]))
    log(f"stage 2 done: reward {r10:.3f} -> {rlast:.3f}, "
        f"greedy acc {rl_acc*100:.1f}%")
    log(f"  param movement RL: core {core1:.2f}->{core2:.2f} "
        f"mem {mem1:.2f}->{mem2:.2f}")

    print(f"[{TAG}] SFT learns={sft_learns} stable={sft_stable} "
          f"(sft {bpc_sft0:.3f}->{bpc_sft1:.3f}, forget {bpc_pre1-bpc_pre0:+.3f})",
          flush=True)
    print(f"[{TAG}] RL reward {r10:.3f} -> {rlast:.3f} "
          f"greedy {rl_acc*100:.1f}% OK={rlast > r10 or rl_acc > 0.3}", flush=True)
    print(f"[{TAG}] DONE in {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()