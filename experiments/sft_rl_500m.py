"""SFT + RL validation on the REAL 500m checkpoint (8-core TPU v5e).

Loads /kaggle/working/experiments/ckpt_500m_step019999.pkl (the completed
FineWeb run) and runs the same battery as sft_rl_test.py at 500m scale:

  stage 1 SFT:  'q:/c:' arithmetic format, WITH replay mixing (the small-
                model run showed +3.8 bpc catastrophic forgetting without
                it -- this stage tests the fix, not just the failure).
  stage 2 RL:   GRPO on single-digit 'a+b=' with entropy bonus + the
                leading-digit-run reward (both fixes from the small run).

FULLY ON-DEVICE EXECUTION (design contract):
  - Every stage is ONE jitted lax.scan: SFT = 1 compile for all 150 steps,
    RL = 1 compile for all 40 rounds, greedy eval = 1 compile for all 40
    tests. Nothing recompiles per step: all step-varying data (batches,
    prompts, targets) are fixed-shape device arrays passed as scan inputs.
  - No Python loops inside jitted code: generation/reward/update loops use
    lax.scan exclusively. Reward extraction (leading digit run, value
    parse, format credit) is pure tensor math on device.
  - Host syncs happen only at stage boundaries for logging (eval bpc,
    reward history, sample tails, param norms).

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
G = 8                          # GRPO group size
L = 4                          # completion length (tokens)
LR_SFT = 5e-5                  # 10x below small-model: 556M params
LR_RL = float(os.environ.get("NAVI_RL_LR", "1e-5"))
ENT_BONUS = 0.01               # entropy regularizer (anti-mode-collapse)
GREEDY_TESTS = 40

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))
BSTACK = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(None, "cores", None))


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
    #    500m model was trained on: prose with embedded arithmetic).
    #    ALL batches pre-generated on host once, shipped to device once,
    #    fixed shape (SFT_STEPS, BS, SEQ) -> scan input, no recompile.
    #    (windows() returns ids (bs, SEQ) and tg (bs, SEQ) -- the shifted
    #    pair -- so batches are SEQ wide, not SEQ-1.)
    sft_data = make_docs(600, rng, 1, 50, "chain")
    replay_data = make_docs(400, rng, 1, 99, "pre")
    eval_sft_ids, eval_sft_tg = windows(sft_data, rng, EVAL_BS, SEQ)
    eval_rep_ids, eval_rep_tg = windows(replay_data, rng, EVAL_BS, SEQ)
    all_ids = np.empty((SFT_STEPS, EVAL_BS, SEQ), dtype=np.int32)
    n_rep = int(EVAL_BS * REPLAY_FRAC)
    all_tg = np.empty((SFT_STEPS, EVAL_BS, SEQ), dtype=np.int32)
    for s in range(SFT_STEPS):
        ids_r, tg_r = windows(replay_data, rng, n_rep, SEQ)
        ids_s, tg_s = windows(sft_data, rng, EVAL_BS - n_rep, SEQ)
        ids = np.concatenate([ids_s, ids_r])
        tg = np.concatenate([tg_s, tg_r])
        perm = rng.permutation(len(ids))
        all_ids[s], all_tg[s] = ids[perm], tg[perm]

    @jax.jit
    def ev(pp, ids, tg):
        logits = model.apply(pp, ids, train=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()

    def bpc_of(pp, ids, tg):
        l = float(ev(pp, jax.device_put(ids, BATCH), jax.device_put(tg, BATCH)))
        return l / np.log(2)

    # 3. SFT stage: ONE compiled lax.scan over all steps.
    #    Returns per-step train loss (device array -> host once at end).
    tx1 = optax.adamw(LR_SFT, b1=0.9, b2=0.95)

    @jax.jit
    def sft_stage(pp, batches_ids, batches_tg):
        o = tx1.init(pp)

        def body(carry, batch):
            q, oo = carry
            ids, tg = batch

            def loss_fn(w):
                logits = model.apply(w, ids, train=True)
                return optax.softmax_cross_entropy_with_integer_labels(
                    logits, tg).mean()

            g = jax.grad(loss_fn)(q)
            u, oo2 = tx1.update(g, oo, q)
            q2 = optax.apply_updates(q, u)
            return (q2, oo2), loss_fn(q2)

        (pf, _), losses = jax.lax.scan(
            body, (pp, o), (batches_ids, batches_tg))
        return pf, losses

    bpc_sft0 = bpc_of(p, eval_sft_ids, eval_sft_tg)
    bpc_pre0 = bpc_of(p, eval_rep_ids, eval_rep_tg)
    log(f"stage 1: SFT {SFT_STEPS} steps x BS {EVAL_BS} "
        f"(replay {REPLAY_FRAC:.0%}, lr {LR_SFT}) -- single compile")
    log(f"  start: sft-fmt bpc {bpc_sft0:.3f} | replay-fmt bpc {bpc_pre0:.3f}")
    p, train_losses = sft_stage(
        p,
        jax.device_put(all_ids, BSTACK),
        jax.device_put(all_tg, BSTACK))
    tl = np.asarray(train_losses)
    if SFT_STEPS > 0:
        marks = sorted(set([0, SFT_STEPS // 4, SFT_STEPS // 2,
                            3 * SFT_STEPS // 4, SFT_STEPS - 1]))
        log("  train-loss curve: " +
            " ".join(f"@{m}:{tl[m]:.3f}" for m in marks))
    bpc_sft1 = bpc_of(p, eval_sft_ids, eval_sft_tg)
    bpc_pre1 = bpc_of(p, eval_rep_ids, eval_rep_tg)
    core1, mem1, val1 = param_snapshot(p)
    log(f"stage 1 done: sft-fmt {bpc_sft0:.3f} -> {bpc_sft1:.3f} | "
        f"replay {bpc_pre0:.3f} -> {bpc_pre1:.3f} (forget {bpc_pre1-bpc_pre0:+.3f})")
    log(f"  param movement: core {core0:.2f}->{core1:.2f} "
        f"mem {mem0:.2f}->{mem1:.2f} (values {val0:.2f}->{val1:.2f})")
    sft_learns = (bpc_sft0 - bpc_sft1) > 0.3
    sft_stable = (bpc_pre1 - bpc_pre0) < 1.0

    # 4. GRPO RL stage: ONE compiled lax.scan over all rounds.
    #    Per round (all on device): sample G completions (lax.scan over L),
    #    reward via tensor-math leading-digit-run parse, group-relative
    #    advantage, PG + entropy gradient, AdamW update.
    #    Outputs: per-round rewards (R, G) and tails (R, G, L) -> host once.
    # 3b. RL warmup: brief SFT on the RAW 'a+b=sum' format. The small-model
    #     battery proved GRPO needs the policy initialized near the task
    #     distribution: without this, samples never emit digits => zero
    #     reward variance => zero advantage => no learning signal.
    WARM_STEPS = int(os.environ.get("NAVI_WARM_STEPS", "120"))
    raw_data = make_docs(600, rng, 1, 9, "raw")
    eval_raw_ids, eval_raw_tg = windows(raw_data, rng, EVAL_BS, SEQ)
    all_wids = np.empty((WARM_STEPS, EVAL_BS, SEQ), dtype=np.int32)
    all_wtg = np.empty((WARM_STEPS, EVAL_BS, SEQ), dtype=np.int32)
    for s in range(WARM_STEPS):
        ids_r, tg_r = windows(replay_data, rng, n_rep, SEQ)
        ids_s, tg_s = windows(raw_data, rng, EVAL_BS - n_rep, SEQ)
        ids = np.concatenate([ids_s, ids_r])
        tg = np.concatenate([tg_s, tg_r])
        perm = rng.permutation(len(ids))
        all_wids[s], all_wtg[s] = ids[perm], tg[perm]

    log(f"stage 1b: RL-warmup SFT {WARM_STEPS} steps on raw 'a+b=sum' "
        f"(replay {REPLAY_FRAC:.0%}, lr {LR_SFT}) -- single compile")
    log(f"  start: raw-fmt bpc {bpc_of(p, eval_raw_ids, eval_raw_tg):.3f}")
    p, warm_losses = sft_stage(
        p,
        jax.device_put(all_wids, BSTACK),
        jax.device_put(all_wtg, BSTACK))
    wl = np.asarray(warm_losses)
    wmarks = sorted(set([0, WARM_STEPS // 4, WARM_STEPS // 2,
                         3 * WARM_STEPS // 4, WARM_STEPS - 1]))
    log("  warm-loss curve: " +
        " ".join(f"@{m}:{wl[m]:.3f}" for m in wmarks))

    # 4. GRPO RL stage: ONE compiled lax.scan over all rounds.
    tx2 = optax.adamw(LR_RL, b1=0.9, b2=0.95)

    @jax.jit
    def rl_stage(pp, prompts, targets, key):
        o = tx2.init(pp)

        def gen_completions(q, prompt, k):
            """G completions of L tokens; lax.scan over L (no python loop).
            Fixed-size buffer carry; generated tokens are scan OUTPUTS.
            Trailing buffer slots sit AFTER the current position, so the
            model's causal mask makes them invisible -> static shapes."""
            pf = jnp.tile(prompt, (G, 1))                    # (G, P)
            P = pf.shape[1]

            def gstep(carry, t):
                tokens, kk = carry                           # (G, L) fixed
                full = jnp.concatenate([pf, tokens], axis=1)
                logits = model.apply(q, full, train=False)
                logp = jax.nn.log_softmax(logits[:, P + t - 1, :])
                kk, sub = jax.random.split(kk)
                nxt = jax.random.categorical(sub, logp)      # (G,)
                tokens = tokens.at[:, t].set(nxt)
                return (tokens, kk), nxt

            (toks, _), _ = jax.lax.scan(
                gstep, (jnp.full((G, L), 32, jnp.int32), k),
                jnp.arange(L))
            return jnp.concatenate([pf, toks], axis=1)       # (G, P+L)

        def reward_of(tails, target):
            """Leading digit run -> value; 1.0 exact, 0.2 any-digits, else 0.
            Pure tensor math: first nondigit via argmax, place-value
            decode via powers of 10 (parity-verified vs python ref)."""
            d = tails - 48
            is_dig = (d >= 0) & (d <= 9)
            nondig = ~is_dig
            f = jnp.argmax(nondig, axis=-1)              # first nondigit
            f = jnp.where(nondig.any(-1), f, L)          # all-digits -> L
            pos = jnp.arange(L)
            in_run = pos[None] < f[:, None]
            dv = jnp.where(in_run, jnp.maximum(d, 0), 0)
            pw = f[:, None] - 1 - pos[None]
            scale = jnp.where(pw >= 0, 10.0 ** jnp.maximum(pw, 0), 0.0)
            val = (dv * scale).sum(-1)                   # (G,)
            return jnp.where(val == target, 1.0,
                             jnp.where(f > 0, 0.2, 0.0))

        def round_body(carry, prompt_target):
            """One GRPO round: sample G completions, tensor-math reward,
            group-relative advantage, PG + entropy gradient, AdamW."""
            q, oo, kk = carry
            prompt, target = prompt_target
            kk, ksub = jax.random.split(kk)
            ids_full = gen_completions(q, prompt, ksub)      # (G, P+L)
            tails = ids_full[:, -L:]
            r = reward_of(tails, target)
            adv = r - r.mean()
            tg_full = jnp.concatenate(
                [ids_full[:, 1:], jnp.zeros((G, 1), jnp.int32)], axis=1)
            mask = jnp.zeros((G, tg_full.shape[1]))
            mask = mask.at[:, -L:].set(1.0)
            mvals = adv[:, None] * mask
            def loss_full(w):
                logits = model.apply(w, ids_full[:, :-1], train=True)
                logp = jax.nn.log_softmax(logits)
                lp = jnp.take_along_axis(
                    logp, tg_full[:, :-1][..., None], axis=-1)[..., 0]
                m = mvals[:, :-1]
                pg = -(lp * m).sum() / jnp.maximum(1.0, jnp.abs(m).sum())
                ent = -(jax.nn.softmax(logits) * logp).sum(-1)
                m2 = mask[:, :-1]
                return pg - ENT_BONUS * (ent * m2).sum() / jnp.maximum(
                    1.0, m2.sum())

            g = jax.grad(loss_full)(q)
            u, oo2 = tx2.update(g, oo, q)
            q2 = optax.apply_updates(q, u)
            return (q2, oo2, kk), (r, tails)

        (pf, _, _), (rewards, tails) = jax.lax.scan(
            round_body, (pp, o, key), (prompts, targets))
        return pf, rewards, tails

    # prompts: RL ON THE FORMAT THE MODEL ACTUALLY LEARNED. TPU diag runs
    # (diag_rl, diag2) proved P(correct|'a+b=' prompts) <= 4% under every
    # warmup/temperature -- the model never learned raw-format arithmetic
    # (its pretrain was FineWeb prose; the small model succeeded only
    # because ITS pretrain was arithmetic text). But stage-1 SFT taught
    # 'q:/c:' to 1.42 bpc -- that mapping demonstrably exists. So GRPO
    # runs on few-shot 'q:/c:' prompts:
    #   'q: 3+4\nc: 7\nq: 2+5\nc: 7\nq: a+b\nc: '  -> sample '<sum>' etc.
    # Reward: leading digit run of the completion == sum.
    # Slot layout: 'q: 3+4\nc: 7\n' = 12 chars (single-digit sums only;
    # enforced by the ab filter below and ctx sampling).
    SLOT = 12                                        # 'q: A+B\nc: S\n'
    PROMPT_LEN = 1 + NCTX * SLOT + 10                # BOS + ctx + 'q: a+b\nc: '
    assert PROMPT_LEN == 47
    ab = rng.integers(1, 9, size=(RL_ROUNDS, 2))     # sums 2..17, mostly 1-digit
    ab = ab[(ab[:, 0] + ab[:, 1] <= 9)]              # keep single-digit sums
    while len(ab) < RL_ROUNDS:                       # top up
        more = rng.integers(1, 9, size=(RL_ROUNDS, 2))
        ab = np.concatenate([ab, more[(more[:, 0] + more[:, 1] <= 9)]])
    ab = ab[:RL_ROUNDS]

    def qc_slot(a, b):
        s = f"q: {a}+{b}\nc: {a+b}\n"
        assert len(s) == SLOT
        return list(s.encode())

    def make_prompts(seed):
        r2 = np.random.default_rng(seed)
        pr = []
        for i in range(RL_ROUNDS):
            toks = [256]
            for _ in range(NCTX):
                x, y = r2.integers(1, 9, 2)
                while x + y > 9: x, y = r2.integers(1, 9, 2)
                toks += qc_slot(int(x), int(y))
            a, b = ab[i]
            toks += list(f"q: {a}+{b}\nc: ".encode())
            pr.append(toks)
        return jnp.asarray(pr, dtype=jnp.int32)

    prompts = make_prompts(555)
    targets = jnp.asarray(ab[:, 0] + ab[:, 1], dtype=jnp.float32)
    log(f"stage 2: GRPO {RL_ROUNDS} rounds (G={G}, lr {LR_RL}, "
        f"ent {ENT_BONUS}) on 'q:/c:' ICL prompts (ctx {NCTX}) -- single compile")
    p, rewards, tails = rl_stage(p, prompts, targets, jax.random.PRNGKey(999))
    r_hist = np.asarray(rewards)          # (RL_ROUNDS, G)
    tails_np = np.asarray(tails)          # (RL_ROUNDS, G, L)

    def decode(t):
        return "".join(chr(c) if 32 <= c < 127 else "?" for c in t)

    for rnd in [0, RL_ROUNDS // 4, RL_ROUNDS // 2,
                3 * RL_ROUNDS // 4, RL_ROUNDS - 1]:
        log(f"  rl {rnd:3d}: reward {r_hist[rnd].mean():.2f} "
            f"hit {int((r_hist[rnd] == 1.0).sum())}/{G} "
            f"tails={[decode(t) for t in tails_np[rnd, :3]]}")

    # greedy accuracy post-RL: ONE compiled call, vmap over test prompts;
    # greedy decode is a scan over L with a FIXED token-buffer carry.
    ab_t = rng.integers(1, 9, size=(GREEDY_TESTS * 3, 2))
    ab_t = ab_t[(ab_t[:, 0] + ab_t[:, 1] <= 9)][:GREEDY_TESTS]

    @jax.jit
    def greedy_eval(pp, prompts_t, targets_t):
        def one(prompt, target):
            pf = prompt[None]                                # (1, P)
            P = pf.shape[1]

            def gstep(carry, t):
                tokens = carry                               # (1, L) fixed
                full = jnp.concatenate([pf, tokens], axis=1)
                logits = model.apply(pp, full, train=False)
                nxt = jnp.argmax(logits[0, P + t - 1, :])
                tokens = tokens.at[0, t].set(nxt)
                return tokens, nxt

            toks, _ = jax.lax.scan(
                gstep, jnp.full((1, L), 32, jnp.int32), jnp.arange(L))
            tails = toks                                     # (1, L)
            d = tails - 48
            is_dig = (d >= 0) & (d <= 9)
            nondig = ~is_dig
            f = jnp.argmax(nondig, axis=-1)
            f = jnp.where(nondig.any(-1), f, L)
            pos = jnp.arange(L)
            in_run = pos[None] < f[:, None]
            dv = jnp.where(in_run, jnp.maximum(d, 0), 0)
            pw = f[:, None] - 1 - pos[None]
            scale = jnp.where(pw >= 0, 10.0 ** jnp.maximum(pw, 0), 0.0)
            val = (dv * scale).sum(-1)
            return val[0] == target

        return jax.vmap(one)(prompts_t, targets_t)

    # greedy eval: SAME 'q:/c:' ICL distribution as training prompts
    pr_t = []
    r3 = np.random.default_rng(777)
    for i in range(GREEDY_TESTS):
        toks = [256]
        for _ in range(NCTX):
            x, y = r3.integers(1, 9, 2)
            while x + y > 9: x, y = r3.integers(1, 9, 2)
            toks += qc_slot(int(x), int(y))
        a, b = ab_t[i]
        toks += list(f"q: {a}+{b}\nc: ".encode())
        pr_t.append(toks)
    accs = greedy_eval(
        p,
        jnp.asarray(pr_t, dtype=jnp.int32),
        jnp.asarray((ab_t[:, 0] + ab_t[:, 1]).astype(np.float32)))
    rl_acc = float(np.mean(np.asarray(accs)))
    core2, mem2, val2 = param_snapshot(p)

    r10 = float(r_hist[:10].mean())
    rlast = float(r_hist[-10:].mean())
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