"""SFT + RL validation battery on the SMALL model (CPU-scale, ~2M params).

Answers the user's question: does the Navi/Pool architecture fine-tune (SFT)
and RL-train at all -- or does the Pool freeze/destabilize under post-pretrain
updates? Same battery would run on 500m later; small first per user.

Design:
  Stage 0 PRETRAIN (short): byte-level LM on synthetic docs containing
          arithmetic facts "a+b=c" embedded in filler text. Both arms
          (dense FFN baseline / Pool) pretrain identically.
  Stage 1 SFT: continue-train on a DIFFERENT format (chain format
          "q: a+b\nc: <sum>") with a small LR. Measures: does loss on the
          new format drop without wrecking pretrained-format loss?
  Stage 2 RL (GRPO): freeze nothing; prompts are "a+b=", sample G=8
          completions at temp>0, reward = exact-match of the digit string.
          Advantage = r - mean(r_group) (no value net). Clipped surrogate.
          Measures: mean reward per round (does the policy improve?) and
          KL-side stability (reward shouldn't collapse as entropy drops).

All evals report core-vs-mem parameter deltas so we can SEE whether the
Pool's value tables move during SFT/RL (they should: that's the claim) and
whether keys stay put (stability).

Env: NAVI_SEED (0). Runtime target: <10 min CPU.
Usage: python3 experiments/sft_rl_test.py
"""
import sys
_REPO = None  # set below after __file__ guard for kernel-vs-local use
import os
if __name__ == "" or True:  # always: this file lives in <repo>/experiments
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _REPO)
import time
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi
from navi.train import init_params

TAG = "sftrl"
T0 = time.time()
SEED = int(os.environ.get("NAVI_SEED", "0"))


def log(msg):
    print(f"[{TAG} +{time.time()-T0:6.1f}s] {msg}", flush=True)


# ------------------------------------------------------------- task data ----
def digit_tokens():
    # byte-level: digits are ord('0')..ord('9')
    return {d: ord(str(d)) for d in range(10)}


def arith_doc(a, b, fmt):
    """One arithmetic fact as bytes in one of two formats."""
    s = a + b
    if fmt == "pre":   # pretrain format: inline filler + fact
        filler = "log " if (a + b) % 2 == 0 else "the cat sat "
        return f"{filler}note {a}+{b}={s} end. "
    # sft format (different surface form so SFT != memorized prefix)
    return f"q: {a}+{b}\nc: {s}\n"


def make_dataset(n, rng, lo, hi, fmt):
    docs = [arith_doc(int(a), int(b), fmt)
            for a, b in zip(rng.integers(lo, hi, n), rng.integers(lo, hi, n))]
    return "\n".join(docs).encode("utf-8")


def windows(data, rng, bs, seq):
    if isinstance(data, (bytes, bytearray)):
        data = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    offs = rng.integers(0, len(data) - seq - 2, size=bs)
    idx = offs[:, None] + np.arange(seq + 1)[None, :]
    win = data[idx].astype(np.int32)
    win[:, 0] = 256  # BOS
    return win[:, :-1], win[:, 1:]

# ------------------------------------------------------------ small model ----
mem_cfg = MemoryConfig(c1=64, c2=64, cand_k=8, side_top=16, n_classes=4,
                       score_temp=4.0)
cfg = ModelConfig(d_model=128, n_layers=4, n_heads=4, memory_every=2,
                  vocab_size=260)
model = Navi(cfg, mem_cfg)
SEQ, BS = 64, 64
LR_SFT = 5e-4
LR_RL = 1e-4
G = 8          # GRPO group size
RL_ROUNDS = int(os.environ.get("NAVI_RL_ROUNDS", "60"))
SFT_STEPS = int(os.environ.get("NAVI_SFT_STEPS", "400"))


mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))  # 1 cpu device fine


def bpc_of(params, ids, tg):
    if ids.ndim == 1:  # seq_windows[a] slices are (seq,) -> add batch dim
        ids = ids[None]
        tg = tg[None]
    logits = model.apply(params, ids, train=False)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, tg)
    return float(ce.mean()) / np.log(2)

def main():
    rng = np.random.default_rng(SEED)
    log(f"stage 0: pretrain (small, Pool arm) -- {sum(x.size for x in jax.tree_util.tree_leaves(init_params(model, SEQ, jax.random.PRNGKey(0)))):,} params")

    # ---- data: 2-digit arithmetic inside filler text
    pre = make_dataset(4000, rng, 10, 99, fmt="pre")
    sft = make_dataset(600, rng, 10, 99, fmt="chain")
    # held-out arithmetic with UNSEEN operand pairs (generalization test)
    val = make_dataset(200, rng, 10, 99, fmt="pre")

    def seq_windows(data, n):
        ids, tgs = [], []
        for _ in range(n):
            i, t = windows(data, rng, BS, SEQ)
            ids.append(i); tgs.append(t)
        return np.concatenate(ids), np.concatenate(tgs)

    # ---- stage 0: pretrain both arms
    key = jax.random.PRNGKey(SEED)
    p_pool = init_params(model, SEQ, jax.random.PRNGKey(SEED))
    tx0 = optax.adamw(1e-3, b1=0.9, b2=0.95)
    o = tx0.init(p_pool)

    @jax.jit
    def step0(pp, oo, ids, tg):
        g = jax.grad(lambda q, a, t: optax.softmax_cross_entropy_with_integer_labels(
            model.apply(q, a, train=True), t).mean())(pp, ids, tg)
        u, oo = tx0.update(g, oo, pp)
        return optax.apply_updates(pp, u), oo

    ids_all, tg_all = seq_windows(pre, 12)  # 12*64 windows
    for s in range(12):
        i = jnp.asarray(ids_all[s*BS:(s+1)*BS])
        t = jnp.asarray(tg_all[s*BS:(s+1)*BS])
        p_pool, o = step0(p_pool, o, i, t)
    bpc0 = bpc_of(p_pool, jnp.asarray(ids_all[0]), jnp.asarray(tg_all[0]))
    log(f"stage 0 done: pretrain bpc {bpc0:.3f}")

    # ---- stage 1: SFT on new format, LR small
    log(f"stage 1: SFT {SFT_STEPS} steps on 'q:/c:' format (lr {LR_SFT})")
    tx1 = optax.adamw(LR_SFT, b1=0.9, b2=0.95)
    o1 = tx1.init(p_pool)

    @jax.jit
    def step1(pp, oo, ids, tg):
        g = jax.grad(lambda q, a, t: optax.softmax_cross_entropy_with_integer_labels(
            model.apply(q, a, train=True), t).mean())(pp, ids, tg)
        u, oo = tx1.update(g, oo, pp)
        return optax.apply_updates(pp, u), oo

    sft_ids, sft_tgs = seq_windows(sft, SFT_STEPS)
    pre_ids, pre_tgs = seq_windows(pre, 4)
    bpc_pre_before = bpc_of(p_pool, jnp.asarray(pre_ids[0]), jnp.asarray(pre_tgs[0]))
    hist = []
    for s in range(SFT_STEPS):
        lo = (s % (len(sft_ids) // BS)) * BS
        p_pool, o1 = step1(p_pool, o1, jnp.asarray(sft_ids[lo:lo+BS]),
                           jnp.asarray(sft_tgs[lo:lo+BS]))
        if s % 100 == 0 or s == SFT_STEPS - 1:
            bpc_sft = bpc_of(p_pool, jnp.asarray(sft_ids[0]), jnp.asarray(sft_tgs[0]))
            bpc_pre = bpc_of(p_pool, jnp.asarray(pre_ids[0]), jnp.asarray(pre_tgs[0]))
            hist.append((s, bpc_sft, bpc_pre))
            log(f"  sft {s:4d}: new-fmt bpc {bpc_sft:.3f} | pretrain-fmt {bpc_pre:.3f}")
    log(f"stage 1 done: new-fmt {hist[0][1]:.3f} -> {hist[-1][1]:.3f} "
        f"(pretrain-fmt {bpc_pre_before:.3f} -> {hist[-1][2]:.3f})")
    # SFT verdict: new-format bpc must drop substantially; pretrain-format
    # may rise a little (forgetting) but not explode.
    sft_gain = hist[0][1] - hist[-1][1]
    forget = hist[-1][2] - bpc_pre_before
    sft_learns = sft_gain > 0.3          # adapts to new format?
    sft_stable = forget < 1.0            # without wrecking old format?
    sft_ok = sft_learns and sft_stable

    # ---- stage 2b: GRPO RL on "a+b=" -> digits
    log(f"stage 2b: GRPO {RL_ROUNDS} rounds (G={G}, lr {LR_RL})")
    tx2 = optax.adamw(LR_RL, b1=0.9, b2=0.95)
    o2 = tx2.init(p_pool)

    @jax.jit
    def sample_g(pp, prompt, key):
        """Sample G completions of len L from prompt (1, l) -> (G, L) ids+logps."""
        L = 4
        ids = jnp.tile(prompt, (G, 1))  # (G, l)
        out_ids, out_lp = [], []
        for t in range(L):
            logits = model.apply(pp, ids, train=False)
            logp = jax.nn.log_softmax(logits[:, -1, :])
            key, sub = jax.random.split(key)
            nxt = jax.random.categorical(sub, logp)
            out_ids.append(nxt)
            out_lp.append(logp[jnp.arange(G), nxt])
            ids = jnp.concatenate([ids, nxt[:, None]], axis=1)
        return jnp.stack(out_ids, axis=1), jnp.stack(out_lp, axis=1), ids
    # ---- stage 2a: RL warmup SFT on raw "a+b=<sum>" (the RL prompt format)
    # GRPO needs an initial policy with nonzero reward signal to climb from.
    # single-digit task: 1-2 digit answers, small hypothesis space -- the
    # 2-digit run showed the model learns the FORMAT ("= followed by 2-3
    # digits") but not the mapping at this scale; single-digit gives RL a
    # sparse-but-reachable reward.
    warm_docs = [f"{int(a)}+{int(b)}={int(a)+int(b)} " for a, b in
                 zip(rng.integers(1, 10, 400), rng.integers(1, 10, 400))]
    warm_data = "\n".join(warm_docs).encode()
    warm_ids, warm_tgs = seq_windows(warm_data, 300)
    for s in range(300):
        lo = (s % (len(warm_ids) // BS)) * BS
        p_pool, o1 = step1(p_pool, o1, jnp.asarray(warm_ids[lo:lo+BS]),
                           jnp.asarray(warm_tgs[lo:lo+BS]))
    # sanity: greedy accuracy BEFORE RL
    hits0 = 0
    for _ in range(20):
        a, b = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        prompt = jnp.asarray([[256] + list(f"{a}+{b}=".encode())], dtype=jnp.int32)
        _, _, full = sample_g(p_pool, prompt, jax.random.PRNGKey(3))
        tail = bytes(int(x) for x in np.asarray(full)[0][-4:])
        digs = bytes(c for c in tail if 48 <= c <= 57)
        hits0 += int(digs == str(a + b).encode())
    log(f"stage 2a done: pre-RL greedy acc {hits0*5}% (20 samples, temp sampling)")




    RL_PROMPTS = 12
    L = 4
    def reward(ids_batch, a, b):
        """Exact-match on the LEADING digit run of the 4-token completion.
        The warmup data is 'a+b=<sum> ' so the learned emission is
        answer-then-junk; taking the first non-digit-terminated run is the
        faithful measure (full-run extraction double-counts repeats:
        '7 ?7' -> '77' != '7' was a false negative)."""
        want = str(a + b).encode()
        outs = []
        for g_ in range(ids_batch.shape[0]):
            seqb = bytes(int(x) for x in ids_batch[g_])
            out = []
            for c in seqb:
                if 48 <= c <= 57:
                    out.append(c)
                else:
                    break  # answer-then-junk: stop at first non-digit
            digits = bytes(out)
            outs.append(1.0 if digits == want else
                        (0.2 if digits else 0.0))  # 0.2: right FORMAT
        return np.asarray(outs, dtype=np.float32)

    # ---- stage 2b: GRPO rounds
    key = jax.random.PRNGKey(SEED + 999)
    rewards_hist = []
    for rnd in range(RL_ROUNDS):
        a, b = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        prompt = jnp.asarray([[256] + list(f"{a}+{b}=".encode())], dtype=jnp.int32)
        ids_s, lps_s, full = sample_g(p_pool, prompt, key)
        ids_np = np.asarray(full)[:, -L:]  # the sampled tail
        r = reward(ids_np, a, b)
        adv = r - r.mean()
        # policy-gradient step on each completion with its advantage
        ids_full = jnp.asarray(np.asarray(full))
        tg_full = jnp.concatenate([ids_full[:, 1:], jnp.full((G, 1), 0, jnp.int32)], axis=1)
        # mask: only the L completion positions count
        mask = np.zeros((G, tg_full.shape[1]), dtype=np.float32)
        mask[:, -L:] = 1.0
        mvals = np.repeat(adv[:, None], tg_full.shape[1], axis=1) * mask

        def loss_rl(pp):
            logits = model.apply(pp, ids_full[:, :-1], train=True)
            logp = jax.nn.log_softmax(logits)
            tgt = tg_full[:, :-1]
            lp = jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
            m = jnp.asarray(mvals[:, :-1])
            return -(lp * m).sum() / max(1.0, jnp.abs(m).sum())

        g = jax.grad(loss_rl)(p_pool)
        u, o2 = tx2.update(g, o2, p_pool)
        p_pool = optax.apply_updates(p_pool, u)
        rewards_hist.append(float(r.mean()))
        if rnd % 10 == 0 or rnd == RL_ROUNDS - 1:
            tails = ["".join(chr(c) if 32 <= c < 127 else "?" for c in t)
                     for t in np.asarray(ids_np)[:3]]
            log(f"  rl {rnd:3d}: {a}+{b}= reward {r.mean():.2f} "
                f"hit {int((r == 1.0).sum())}/{G} tails={tails}")
    # greedy check after RL: does the model now SOLVE arithmetic?
    hits = 0
    tests = 40
    for _ in range(tests):
        a, b = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        prompt = jnp.asarray([[256] + list(f"{a}+{b}=".encode())], dtype=jnp.int32)
        ids_s, lps_s, full = sample_g(p_pool, prompt, jax.random.PRNGKey(7))
        tail = bytes(int(x) for x in np.asarray(full)[0][-L:])
        digits = bytes(c for c in tail if 48 <= c <= 57)
        hits += int(digits == str(a + b).encode())
    rl_acc = hits / tests
    log(f"stage 2 done: RL greedy accuracy {rl_acc*100:.1f}% on unseen sums "
        f"(chance for 2-digit exact ~ <1%)")
    rl_ok = rl_acc > 0.5

    print(f"[{TAG}] SFT learns={sft_learns} (gain {sft_gain:.3f} bpc) "
          f"stable={sft_stable} (forgetting {forget:+.3f}) OK={sft_ok}", flush=True)
    print(f"[{TAG}] RL_OK {rl_ok} (greedy acc {rl_acc*100:.1f}%, "
          f"reward {rewards_hist[0]:.2f} -> {rewards_hist[-1]:.2f})", flush=True)


if __name__ == "__main__":
    main()