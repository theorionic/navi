"""instill_rl_500m.py -- THE decisive experiment.

Question (user, 2026-09): take the ALREADY-PRETRAINED 500m checkpoint,
continue-train it on arithmetic examples so the skill EXISTS at partial
strength ("instill"), then run GRPO on top ("sharpen"). Does RL sharpen
a partially-instilled skill? Verify with before/after P(correct).

Stage A (instill): continue-train the PRETRAINED ckpt on arithmetic text
  ONLY (no replay -- this is deliberate: replay protects old knowledge
  but DILUTES the arithmetic signal; diag4's 1000-step weighted run with
  50% replay moved P(correct) DOWN). 3000 steps x BS 8 windows of pure
  'q: a+b\\nc: s\\n' docs (lo=1,hi=9; single-digit sums, matching the RL
  target distribution exactly). Probe P(correct|ICL prompt) every 500
  steps -- the instill curve is the primary scientific output.
Stage B (amplify): GRPO from the BEST probe checkpoint, shaped reward,
  grad clip, ent 0.05, lr 3e-5. Before/after greedy accuracy decides.

FULLY ON-DEVICE (same contract as sft_rl_500m): every stage one compiled
lax.scan; probes are one compiled vmap'd eval per checkpoint.
Env: NAVI_A_STEPS (3000) NAVI_B_ROUNDS (400) NAVI_LR_A (3e-5)
Usage: python3 /kaggle/working/navi/experiments/instill_rl_500m.py
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

TAG = "instillrl"
T0 = time.time()


def log(msg):
    print(f"[{TAG} +{time.time()-T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- config ----
SEQ = 128
A_STEPS = int(os.environ.get("NAVI_A_STEPS", "3000"))
B_ROUNDS = int(os.environ.get("NAVI_B_ROUNDS", "400"))
LR_A = float(os.environ.get("NAVI_LR_A", "3e-5"))
G = 8
L = 4
ENT_BONUS = 0.05
NCTX = 3                       # ICL ctx for probe AND RL prompts
SLOT = 12                      # 'q: A+B\nc: S\n'
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
    """BOS + NCTX solved ICL slots + 'q: a+b\\nc: ' query (len 47)."""
    r2 = np.random.default_rng(seed)
    pr = []
    ab = []
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

# ------------------------------------------------------------------ main ----
def main():
    model, p = load_model_and_ckpt()
    rng = np.random.default_rng(0)

    # ------------------------------------------------ stage A: instill ----
    # pure-arithmetic windows; NO replay (deliberate: replay diluted the
    # signal in diag4; old-knowledge protection is NOT the goal here --
    # maximizing P(correct) is).
    arith = make_arith_docs(rng, 4000, 1, 9)
    tx = optax.adamw(LR_A, b1=0.9, b2=0.95)

    PROBE_EVERY = 500
    n_chk = A_STEPS // PROBE_EVERY                # probe segments
    BS = 8
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

    # probe: P(correct first token) over 64 fresh ICL prompts, vmap'd
    probe_prompts, probe_targets = make_prompt_batch(2027, 64)

    @jax.jit
    def probe(pp):
        logits = jax.vmap(lambda pr: model.apply(pp, pr[None], train=False)[0])(
            probe_prompts)
        logp = jax.nn.log_softmax(logits[:, -1, :])
        return jnp.exp(logp[jnp.arange(64), probe_targets.astype(jnp.int32)])

    p_corr_hist = []
    best_p, best_p_params, best_seg = -1.0, None, -1

    # pre-build all segment batches on host (BS=8 x SEQ=128 x A_STEPS)
    seg_ids, seg_tgs = [], []
    for s in range(n_chk):
        ids_l, tgs_l = [], []
        for _ in range(PROBE_EVERY):
            i, t = windows(arith, rng, BS, SEQ)
            ids_l.append(i)
            tgs_l.append(t)
        seg_ids.append(np.stack(ids_l))
        seg_tgs.append(np.stack(tgs_l))
    log(f"stage A: {A_STEPS} steps pure arith (BS {BS}, lr {LR_A}), "
        f"{n_chk} segments, probing every {PROBE_EVERY}")

    for s in range(n_chk):
        p, losses = instill_segment(
            p,
            jax.device_put(seg_ids[s], BSTACK),
            jax.device_put(seg_tgs[s], BSTACK))
        pc = float(jnp.mean(probe(p)))
        p_corr_hist.append(pc)
        if pc > best_p:
            best_p = pc
            best_seg = (s + 1) * PROBE_EEPS if False else (s + 1) * PROBE_EVERY
            best_p_params = p
        l0 = float(losses[0])
        l1 = float(losses[-1])
        log(f"  A seg {s+1}/{n_chk} (step {(s+1)*PROBE_EVERY}): "
            f"loss {l0:.2f}->{l1:.2f}  P(correct) {pc:.3f}")

    # restore best-probe checkpoint for RL
    p = best_p_params
    log(f"stage A done: best P(correct) {best_p:.3f} @step {best_seg}; "
        f"restored for RL")

    # ------------------------------------------------ stage B: GRPO -------
    LR_RL = 3e-5
    tx2 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LR_RL, b1=0.9, b2=0.95))

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
            ndig = jnp.where(val > 0,
                             jnp.floor(jnp.log10(jnp.maximum(val, 1.0))) + 1.0,
                             1.0)
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

        tails = jax.vmap(one)(test_prompts)
        return tails

    tails_before = np.asarray(greedy_eval(p))
    acc_before = acc_of(tails_before, np.asarray(test_targets))
    log(f"BEFORE RL: greedy acc {acc_before:.1%} "
        f"tails {[decode(t) for t in tails_before[:3]]}")

    p, rewards, tails = rl_stage(p, rl_prompts, rl_targets,
                                 jax.random.PRNGKey(999))
    r_hist = np.asarray(rewards)
    tails_np = np.asarray(tails)
    for rnd in [0, B_ROUNDS // 4, B_ROUNDS // 2, 3 * B_ROUNDS // 4,
                B_ROUNDS - 1]:
        log(f"  rl {rnd:3d}: reward {r_hist[rnd].mean():.2f} "
            f"hit {int((r_hist[rnd] == 1.0).sum())}/{G} "
            f"tails={[decode(t) for t in tails_np[rnd, :3]]}")

    tails_after = np.asarray(greedy_eval(p))
    acc_after = acc_of(tails_after, np.asarray(test_targets))
    log(f"AFTER RL:  greedy acc {acc_after:.1%} "
        f"tails {[decode(t) for t in tails_after[:3]]}")

    ok = acc_after > acc_before + 0.05
    log(f"VERDICT: RL sharpening {'WORKS' if ok else 'DOES NOT WORK'} "
        f"(greedy {acc_before:.1%} -> {acc_after:.1%})")
    with open('/kaggle/working/instill_rl_result.txt', 'w') as f:
        f.write(f"instill P(correct) curve: {p_corr_hist}\n"
                f"best P: {best_p:.3f} @step {best_seg}\n"
                f"greedy acc before RL: {acc_before:.3f}\n"
                f"greedy acc after RL:  {acc_after:.3f}\n"
                f"reward first/last: {r_hist[0].mean():.3f} -> "
                f"{r_hist[-1].mean():.3f}\n"
                f"VERDICT: {'WORKS' if ok else 'DOES-NOT-WORK'}\n")
    log("saved /kaggle/working/instill_rl_result.txt")


if __name__ == "__main__":
    main()