"""rl_validate_500m.py -- validate that RL works on the BIG model in its
useful regime.

Established by prior TPU runs:
  - P(correct)=chance     -> GRPO cannot climb (nothing to amplify)
  - P(correct)=0.863/100% -> no headroom; aggressive lr destroys the skill
This run tests the missing cell: PARTIAL competence. Instill pure arithmetic
until P(correct) lands in [P_LO, P_HI], snapshot, then from that ONE
snapshot run:
  arm A: GRPO lr 1e-5 (env NAVI_RL_LRS default "1e-5,3e-6" -> both arms)
  arm S: continued SFT control (equal segments, no RL)
Greedy accuracy on 80 fresh ICL tests before/after each arm decides.
Verdict: RL-works = any RL arm improves greedy acc by >= 10 points
without collapse, i.e. RL adds measurable skill beyond the instill point.

FULLY ON-DEVICE (same contract as instill_rl_500m): one compiled scan per
segment / RL stage; probes are one compiled vmap'd eval.
Usage: python3 /kaggle/working/navi/experiments/rl_validate_500m.py
"""
import sys
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi

TAG = "rlval"
T0 = time.time()


def log(msg):
    print(f"[{TAG} +{time.time()-T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- config ----
SEQ = 128
SEG = int(os.environ.get("NAVI_SEG", "250"))     # steps per instill segment
MAX_SEGS = 8
P_LO = float(os.environ.get("NAVI_P_LO", "0.25"))
P_HI = float(os.environ.get("NAVI_P_HI", "0.55"))
RL_ROUNDS = int(os.environ.get("NAVI_RL_ROUNDS", "300"))
RL_LRS = [float(x) for x in os.environ.get(
    "NAVI_RL_LRS", "1e-5,3e-6").split(",")]
ENT_BONUS = float(os.environ.get("NAVI_ENT", "0.01"))
LR_A = 3e-5
G = 8
L = 4
NCTX = 3
SLOT = 12                       # 'q: A+B\nc: S\n'
GREEDY_TESTS = 80

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
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
def make_arith_docs(rng, n, lo, hi):
    docs = []
    for _ in range(n):
        a, b = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        docs.append(f"q: {a}+{b}\nc: {a+b}\n")
    return "\n".join(docs).encode("utf-8")


def windows(data, rng, bs, seq):
    if isinstance(data, (bytes, bytearray)):
        data = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    offs = rng.integers(0, len(data) - seq - 2, size=bs)
    idx = offs[:, None] + np.arange(seq + 1)[None, :]
    win = data[idx].astype(np.int32)
    win[:, 0] = 256
    return win[:, :-1], win[:, 1:]


def qc_slot(a, b):
    s = f"q: {a}+{b}\nc: {a+b}\n"
    assert len(s) == SLOT
    return list(s.encode())


def make_prompt_batch(seed, n):
    r2 = np.random.default_rng(seed)
    pr, ab = [], []
    for _ in range(n):
        toks = [256]
        for _ in range(NCTX):
            x, y = r2.integers(1, 9, 2)
            while x + y > 9:
                x, y = r2.integers(1, 9, 2)
            toks += qc_slot(int(x), int(y))
        a, b = int(r2.integers(1, 9)), int(r2.integers(1, 9))
        while a + b > 9:
            a, b = int(r2.integers(1, 9)), int(r2.integers(1, 9))
        toks += list(f"q: {a}+{b}\nc: ".encode())
        pr.append(toks)
        ab.append(a + b)
    return jnp.asarray(pr, dtype=jnp.int32), jnp.asarray(ab, jnp.float32)


def decode(t):
    return "".join(chr(c) if 32 <= c < 127 else "?" for c in t)


def acc_of(tails, targets):
    d = tails - 48
    is_dig = (d >= 0) & (d <= 9)
    f = np.argmax(~is_dig, axis=-1)
    f = np.where((~is_dig).any(-1), f, tails.shape[-1])
    pos = np.arange(tails.shape[-1])
    in_run = pos[None] < f[:, None]
    dv = np.where(in_run, np.maximum(d, 0), 0)
    pw = f[:, None] - 1 - pos[None]
    sc = np.where(pw >= 0, 10.0 ** np.maximum(pw, 0), 0.0)
    val = (dv * sc).sum(-1)
    return float((val == targets).mean())


def load_model_and_ckpt():
    ckpt = sorted(
        f for f in os.listdir('/kaggle/working/experiments')
        if f.startswith('ckpt_500m_step') and f.endswith('.pkl'))[-1]
    with open(f'/kaggle/working/experiments/{ckpt}', 'rb') as f:
        p = pickle.load(f)['params']
    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64,
                           n_classes=4, score_temp=4.0)
    cfg_m = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=2,
                        vocab_size=260)
    model = Navi(cfg_m, mem_cfg)
    log(f"loaded {ckpt}")
    return model, p


# ------------------------------------------------------------------ main ----
def main():
    model, p = load_model_and_ckpt()
    rng = np.random.default_rng(0)

    # ------------------------------------ partial instill with probing ----
    arith = make_arith_docs(rng, 4000, 1, 9)
    tx = optax.adamw(LR_A, b1=0.9, b2=0.95)

    @jax.jit
    def instill_segment(pp, bids, btg):
        o = tx.init(pp)

        def body(c, batch):
            q, oo = c
            ids, tg = batch
            lf = lambda w: optax.softmax_cross_entropy_with_integer_labels(
                model.apply(w, ids, train=True), tg).mean()
            g = jax.grad(lf)(q)
            u, oo2 = tx.update(g, oo, q)
            q2 = optax.apply_updates(q, u)
            return (q2, oo2), lf(q2)

        (pf, _), losses = jax.lax.scan(body, (pp, o), (bids, btg))
        return pf, losses

    probe_prompts, probe_targets = make_prompt_batch(2027, 64)

    @jax.jit
    def probe(pp):
        logits = jax.vmap(lambda pr: model.apply(pp, pr[None], train=False)[0])(
            probe_prompts)
        logp = jax.nn.log_softmax(logits[:, -1, :])
        return jnp.exp(logp[jnp.arange(64), (48 + probe_targets).astype(jnp.int32)])

    BS = 8
    partial_params, partial_p, partial_step = None, -1.0, -1
    fallback_params, fallback_p, fallback_step = None, -1.0, -1
    segs_run = 0
    for s in range(MAX_SEGS):
        ids_l, tgs_l = [], []
        for _ in range(SEG):
            i, t = windows(arith, rng, BS, SEQ)
            ids_l.append(i)
            tgs_l.append(t)
        p, losses = instill_segment(
            p,
            jax.device_put(np.stack(ids_l), BSTACK),
            jax.device_put(np.stack(tgs_l), BSTACK))
        pc = float(jnp.mean(probe(p)))
        step = (s + 1) * SEG
        log(f"  instill seg {s+1} (step {step}): loss {float(losses[0]):.2f}"
            f"->{float(losses[-1]):.2f}  P(correct) {pc:.3f}")
        segs_run = s + 1
        if P_LO <= pc <= P_HI:
            partial_params, partial_p, partial_step = p, pc, step
            break
        if fallback_params is None or abs(pc - 0.4) < abs(fallback_p - 0.4):
            fallback_params, fallback_p, fallback_step = p, pc, step
    if partial_params is None:
        partial_params, partial_p, partial_step = \
            fallback_params, fallback_p, fallback_step
        log(f"no segment landed in [{P_LO},{P_HI}]; using closest "
            f"P={partial_p:.3f}@{partial_step}")
    log(f"partial snapshot: P(correct) {partial_p:.3f} @step {partial_step}")
    pp_host = jax.device_get(partial_params)     # host copy for all arms

    # ------------------------------------------------------- test prompts --
    ab_t = rng.integers(1, 9, size=(GREEDY_TESTS * 3, 2))
    ab_t = ab_t[(ab_t[:, 0] + ab_t[:, 1] <= 9)][:GREEDY_TESTS]

    def make_test_prompts(seed):
        r2 = np.random.default_rng(seed)
        pr, tg = [], []
        for i in range(len(ab_t)):
            toks = [256]
            for _ in range(NCTX):
                x, y = r2.integers(1, 9, 2)
                while x + y > 9:
                    x, y = r2.integers(1, 9, 2)
                toks += qc_slot(int(x), int(y))
            a, b = int(ab_t[i, 0]), int(ab_t[i, 1])
            toks += list(f"q: {a}+{b}\nc: ".encode())
            pr.append(toks)
            tg.append(a + b)
        return jnp.asarray(pr, jnp.int32), jnp.asarray(tg, jnp.float32)

    test_prompts, test_targets = make_test_prompts(777)
    tt_np = np.asarray(test_targets)

    @jax.jit
    def greedy_eval(pp):
        def one(pr):
            toks = jnp.full((L,), 32, jnp.int32)
            P = pr.shape[0]

            def gstep(carry, t):
                tokens, = carry
                full = jnp.concatenate([pr[None], tokens[None]], axis=1)
                logits = model.apply(pp, full, train=False)[0]
                nxt = jnp.argmax(logits[P + t - 1])
                tokens = tokens.at[t].set(nxt)
                return (tokens,), nxt

            (toks,), _ = jax.lax.scan(gstep, (toks,), jnp.arange(L))
            return toks

        return jax.vmap(one)(test_prompts)

    acc_before = acc_of(np.asarray(greedy_eval(partial_params)), tt_np)
    log(f"BASELINE (partial): greedy acc {acc_before:.1%}")

    # ------------------------------------------------------------- arms ----
    rl_prompts, rl_targets = make_prompt_batch(555, RL_ROUNDS)

    def make_rl_stage(lr):
        tx2 = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr, b1=0.9, b2=0.95))

        @jax.jit
        def rl_stage(pp, prompts, targets, key):
            o = tx2.init(pp)

            def gen_completions(q, prompt, k):
                pf = jnp.tile(prompt, (G, 1))
                P = pf.shape[1]

                def gstep(carry, t):
                    tokens, kk = carry
                    full = jnp.concatenate([pf, tokens], axis=1)
                    logits = model.apply(q, full, train=False)
                    logp = jax.nn.log_softmax(logits[:, P + t - 1, :])
                    kk, sub = jax.random.split(kk)
                    nxt = jax.random.categorical(sub, logp)
                    tokens = tokens.at[:, t].set(nxt)
                    return (tokens, kk), nxt

                (toks, _), _ = jax.lax.scan(
                    gstep, (jnp.full((G, L), 32, jnp.int32), k),
                    jnp.arange(L))
                return jnp.concatenate([pf, toks], axis=1)

            def reward_of(tails, target):
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
                ndig = jnp.where(
                    val > 0,
                    jnp.floor(jnp.log10(jnp.maximum(val, 1.0))) + 1.0, 1.0)
                first_dig_val = jnp.floor(val / 10.0 ** (ndig - 1.0))
                tgt_pow = jnp.where(target >= 10, 1.0, 0.0)
                tgt_d0 = jnp.floor(target / 10.0 ** tgt_pow)
                tgt_d1 = jnp.mod(target, 10.0)
                first_ok = (f > 0) & (first_dig_val == tgt_d0)
                prefix2_ok = (target >= 10) & (f >= 2) & \
                             (dv[:, 0] == tgt_d0) & (dv[:, 1] == tgt_d1)
                exact = val == target
                base = jnp.where(f > 0, 0.2, 0.0)
                return base + 0.3 * first_ok + 0.2 * prefix2_ok + 0.3 * exact

            def round_body(carry, prompt_target):
                q, oo, kk = carry
                prompt, target = prompt_target
                kk, ksub = jax.random.split(kk)
                ids_full = gen_completions(q, prompt, ksub)
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

        return rl_stage

    results = {}
    for lr in [float(x) for x in os.environ.get(
            "NAVI_RL_LRS", "1e-5,3e-6").split(",")]:
        p_arm = shard_tree(pp_host)
        log(f"arm RL lr={lr}: GRPO {RL_ROUNDS} rounds (G={G}, "
            f"ent {ENT_BONUS}) -- single compile")
        p_arm, rewards, tails = make_rl_stage(lr)(
            p_arm, rl_prompts, rl_targets, jax.random.PRNGKey(999))
        r_hist = np.asarray(rewards)
        q0, q1 = RL_ROUNDS // 4, 3 * RL_ROUNDS // 4
        log(f"  reward {r_hist[:q0].mean():.2f} -> {r_hist[q1:].mean():.2f} "
            f"(last-round hits {int((r_hist[-1] == 1.0).sum())}/{G})")
        acc_arm = acc_of(np.asarray(greedy_eval(p_arm)), tt_np)
        log(f"arm RL lr={lr}: greedy acc {acc_arm:.1%}")
        results[f"rl_{lr:g}"] = (r_hist[:q0].mean(), r_hist[q1:].mean(),
                                 acc_arm)

    # SFT control: equal instill budget, from the SAME partial snapshot
    p_ctl = shard_tree(pp_host)
    n_ctl = 4 * SEG
    log(f"arm SFT control: {n_ctl} more instill steps from the same snapshot")
    for s in range(4):
        ids_l, tgs_l = [], []
        for _ in range(SEG):
            i, t = windows(arith, rng, BS, SEQ)
            ids_l.append(i)
            tgs_l.append(t)
        p_ctl, _ = instill_segment(
            p_ctl,
            jax.device_put(np.stack(ids_l), BSTACK),
            jax.device_put(np.stack(tgs_l), BSTACK))
    acc_ctl = acc_of(np.asarray(greedy_eval(p_ctl)), tt_np)
    log(f"arm SFT control: greedy acc {acc_ctl:.1%}")
    results["sft_control"] = acc_ctl

    # ---------------------------------------------------------- verdict ----
    rl_bests = {k: v[2] for k, v in results.items() if k.startswith("rl_")}
    best_rl = max(rl_bests.values())
    works = best_rl >= acc_before + 0.10
    log(f"VERDICT: RL {'WORKS' if works else 'NO GAIN'} on the big model "
        f"(partial greedy {acc_before:.1%}; best RL arm {best_rl:.1%}; "
        f"SFT control {acc_ctl:.1%})")
    with open('/kaggle/working/rl_validate_result.txt', 'w') as f:
        f.write(f"partial P(correct): {partial_p:.3f} @step {partial_step}\n"
                f"greedy baseline: {acc_before:.3f}\n"
                + "".join(f"{k}: reward {v[0]:.3f}->{v[1]:.3f}, "
                          f"greedy {v[2]:.3f}\n"
                          for k, v in results.items() if k.startswith("rl_"))
                + f"sft_control greedy: {acc_ctl:.3f}\n"
                f"VERDICT: {'WORKS' if works else 'NO-GAIN'}\n")
    log("saved /kaggle/working/rl_validate_result.txt")


if __name__ == "__main__":
    main()