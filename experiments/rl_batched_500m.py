"""rl_batched_500m.py -- definitive RL-at-scale test on the 556m Navi model.

Prior validated results (all real TPU v5e-8):
  chance-start  : GRPO G=8 cannot climb (nothing to amplify)
  partial 55%   : GRPO G=8 no gain (collapse at lr 1e-5, flat at 3e-6)
  saturation    : no headroom; hard push destroys
  SFT control   : 55% -> 88.8% with equal budget
Diagnosis: advantage from G=8 samples on ONE prompt is too noisy per
update at 556M. Fix under test: MORE SAMPLES PER UPDATE.

Three arms compared from the same fresh partial snapshot:
  arm B-GRPO : prompt-batched GRPO -- PB=16 prompts x G=16 completions
               per AdamW step (256 completions/step; 16x more samples
               per gradient than single-prompt G=8), group-relative
               advantage per prompt, on-policy (1 update per batch).
  arm RWB    : REINFORCE with batch baseline -- PB=16 prompts x 1 sample,
               adv = r - mean(r_batch); the cheap dense-batch estimator.
  arm SFT-C  : continued supervised control, equal instill budget.
Greedy accuracy on 80 fresh ICL tests decides each arm.

Env: NAVI_PARTIAL_STEPS (1000) NAVI_ROUNDS (150) NAVI_PB (16) NAVI_G (16)
     NAVI_RL_LR (1e-5) NAVI_ENT (0.005)
Usage: python3 /kaggle/working/navi/experiments/rl_batched_500m.py
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

TAG = "rlbatch"
T0 = time.time()


def log(msg):
    print(f"[{TAG} +{time.time()-T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- config ----
SEQ = 128
PARTIAL_STEPS = int(os.environ.get("NAVI_PARTIAL_STEPS", "1000"))
ROUNDS = int(os.environ.get("NAVI_ROUNDS", "150"))
PB = int(os.environ.get("NAVI_PB", "16"))       # prompts per update
G = int(os.environ.get("NAVI_G", "16"))         # completions per prompt
ENT_BONUS = float(os.environ.get("NAVI_ENT", "0.005"))
LR_RL = float(os.environ.get("NAVI_RL_LR", "1e-5"))
LR_A = 3e-5
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

    # -------------------------------- partial instill (same protocol) ----
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
    for s in range(PARTIAL_STEPS // 250):
        ids_l, tgs_l = [], []
        for _ in range(250):
            i, t = windows(arith, rng, BS, SEQ)
            ids_l.append(i)
            tgs_l.append(t)
        p, _ = instill_segment(
            p,
            jax.device_put(np.stack(ids_l), BSTACK),
            jax.device_put(np.stack(tgs_l), BSTACK))
        log(f"  instill {(s+1)*250}: P(correct) {float(jnp.mean(probe(p))):.3f}")
    pp_host = jax.device_get(p)

    # ------------------------------------------------------- eval setup ---
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

    acc_before = acc_of(np.asarray(greedy_eval(pp_host)), tt_np)
    log(f"BASELINE: greedy acc {acc_before:.1%}")

    # ------------------------------------------------------ shared reward --
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

    rl_prompts, rl_targets = make_prompt_batch(555, ROUNDS * PB)

    # ================================================ arm B-GRPO =========
    tx_b = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LR_RL, b1=0.9, b2=0.95))

    # vmap over PB prompts; each generates G completions (scanned gstep);
    # group-relative advantage; one on-policy AdamW step per round.
    @jax.jit
    def bgrpo_round(pp, oo, prompts_r, targets_r, kk):
        def gen_one(pr, k1):
            pf = jnp.tile(pr, (G, 1))
            Pp = pf.shape[1]

            def gstep(carry, t):
                tokens, k2 = carry
                full = jnp.concatenate([pf, tokens], axis=1)
                logits = model.apply(pp, full, train=False)
                logp = jax.nn.log_softmax(logits[:, Pp + t - 1, :])
                k2, sub = jax.random.split(k2)
                nxt = jax.random.categorical(sub, logp)
                tokens = tokens.at[:, t].set(nxt)
                return (tokens, k2), nxt

            (toks, _), _ = jax.lax.scan(
                gstep, (jnp.full((G, L), 32, jnp.int32), k1),
                jnp.arange(L))
            return jnp.concatenate([pf, toks], axis=1)

        keys = jax.random.split(kk, PB)
        ids_all = jax.vmap(gen_one)(prompts_r, keys)       # (PB, G, P+L)
        tails = ids_all[:, :, -L:]
        r = jax.vmap(reward_of)(tails, targets_r)          # (PB, G)
        adv = r - r.mean(axis=1, keepdims=True)            # group-relative
        P_ = prompts_r.shape[1]
        ids_flat = ids_all.reshape((PB * G, P_ + L))
        adv_flat = adv.reshape(PB * G)

        T = ids_flat.shape[1] - 1
        region = (jnp.arange(T) >= T - L).astype(jnp.float32)

        def loss_fn(w):
            logits = model.apply(w, ids_flat[:, :-1], train=True)
            logp = jax.nn.log_softmax(logits)
            lp = jnp.take_along_axis(
                logp, ids_flat[:, 1:][..., None], axis=-1)[..., 0]
            m = region[None, :] * adv_flat[:, None]
            pg = -(lp * m).sum() / jnp.maximum(1.0, jnp.abs(m).sum())
            ent = -(jax.nn.softmax(logits) * logp).sum(-1)   # (N, T)
            ent_term = (ent * region[None, :]).sum() / jnp.maximum(
                1.0, region.sum() * PB * G)
            return pg - ENT_BONUS * ent_term

        g = jax.grad(loss_fn)(pp)
        u, oo2 = tx_b.update(g, oo, pp)
        return optax.apply_updates(pp, u), oo2, r

    log(f"arm B-GRPO: {ROUNDS} rounds x PB={PB} x G={G} "
        f"(= {ROUNDS * PB * G} completions)")
    pp_b = shard_tree(pp_host)
    oo_b = tx_b.init(pp_b)
    kk = jax.random.PRNGKey(123)
    r_hist_b = []
    for rnd in range(ROUNDS):
        kk, ksub = jax.random.split(kk)
        pp_b, oo_b, r_b = bgrpo_round(
            pp_b, oo_b, rl_prompts[rnd * PB:(rnd + 1) * PB],
            rl_targets[rnd * PB:(rnd + 1) * PB], ksub)
        r_hist_b.append(float(r_b.mean()))
        if rnd % 25 == 0 or rnd == ROUNDS - 1:
            log(f"  b-grpo {rnd:3d}: reward {float(r_b.mean()):.2f}")
    acc_b = acc_of(np.asarray(greedy_eval(pp_b)), tt_np)
    log(f"arm B-GRPO: greedy acc {acc_b:.1%} "
        f"tails {[decode(t) for t in np.asarray(greedy_eval(pp_b))[:3]]}")

    # ================================================== arm RWB ===========
    # REINFORCE with batch baseline: 1 sample per prompt, adv = r - mean.
    @jax.jit
    def rwb_round(pp, oo, prompts_r, targets_r, kk):
        def gen_one(pr, k1):
            pf = jnp.tile(pr, (1, 1))
            Pp = pf.shape[1]

            def gstep(carry, t):
                tokens, k2 = carry
                full = jnp.concatenate([pf, tokens], axis=1)
                logits = model.apply(pp, full, train=False)
                logp = jax.nn.log_softmax(logits[:, Pp + t - 1, :])
                k2, sub = jax.random.split(k2)
                nxt = jax.random.categorical(sub, logp)
                tokens = tokens.at[:, t].set(nxt)
                return (tokens, k2), nxt

            (toks, _), _ = jax.lax.scan(
                gstep, (jnp.full((1, L), 32, jnp.int32), k1),
                jnp.arange(L))
            return jnp.concatenate([pf, toks], axis=1)     # (1, P+L)

        keys = jax.random.split(kk, PB)
        ids_all = jax.vmap(gen_one)(prompts_r, keys)       # (PB, 1, P+L)
        tails = ids_all[:, -1, -L:]                        # (PB, L)
        r = reward_of(tails, targets_r)                    # (PB,)
        adv = r - r.mean()                                  # batch baseline
        P_ = prompts_r.shape[1]
        ids_flat = ids_all[:, 0, :]                        # (PB, P+L)

        T = ids_flat.shape[1] - 1
        region = (jnp.arange(T) >= T - L).astype(jnp.float32)

        def loss_fn(w):
            logits = model.apply(w, ids_flat[:, :-1], train=True)
            logp = jax.nn.log_softmax(logits)
            lp = jnp.take_along_axis(
                logp, ids_flat[:, 1:][..., None], axis=-1)[..., 0]
            m = region[None, :] * adv[:, None]
            pg = -(lp * m).sum() / jnp.maximum(1.0, jnp.abs(m).sum())
            ent = -(jax.nn.softmax(logits) * logp).sum(-1)
            ent_term = (ent * region[None, :]).sum() / jnp.maximum(
                1.0, region.sum() * PB)
            return pg - ENT_BONUS * ent_term

        g = jax.grad(loss_fn)(pp)
        u, oo2 = tx_rwb.update(g, oo, pp)
        return optax.apply_updates(pp, u), oo2, r.mean()

    tx_rwb = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LR_RL, b1=0.9, b2=0.95))

    log(f"arm RWB: {ROUNDS} rounds x PB={PB} x 1 sample")
    pp_r = shard_tree(pp_host)
    oo_r = tx_rwb.init(pp_r)
    kk = jax.random.PRNGKey(1234)
    r_hist_r = []
    for rnd in range(ROUNDS):
        kk, ksub = jax.random.split(kk)
        pp_r, oo_r, rm = rwb_round(
            pp_r, oo_r, rl_prompts[rnd * PB:(rnd + 1) * PB],
            rl_targets[rnd * PB:(rnd + 1) * PB], ksub)
        r_hist_r.append(float(rm))
        if rnd % 25 == 0 or rnd == ROUNDS - 1:
            log(f"  rwb    {rnd:3d}: reward {rm:.2f}")
    acc_r = acc_of(np.asarray(greedy_eval(pp_r)), tt_np)
    log(f"arm RWB: greedy acc {acc_r:.1%}")

    # ================================================ arm SFT-C ==========
    p_ctl = shard_tree(pp_host)
    n_seg = PARTIAL_STEPS // 250
    log(f"arm SFT-C: {n_seg * 250} more supervised steps (same budget)")
    for s in range(n_seg):
        ids_l, tgs_l = [], []
        for _ in range(250):
            i, t = windows(arith, rng, BS, SEQ)
            ids_l.append(i)
            tgs_l.append(t)
        p_ctl, _ = instill_segment(
            p_ctl,
            jax.device_put(np.stack(ids_l), BSTACK),
            jax.device_put(np.stack(tgs_l), BSTACK))
    acc_ctl = acc_of(np.asarray(greedy_eval(p_ctl)), tt_np)
    log(f"arm SFT-C: greedy acc {acc_ctl:.1%}")

    # ---------------------------------------------------------- verdict ---
    log(f"VERDICT TABLE: baseline {acc_before:.1%} | B-GRPO {acc_b:.1%} | "
        f"RWB {acc_r:.1%} | SFT-C {acc_ctl:.1%}")
    with open('/kaggle/working/rl_batched_result.txt', 'w') as f:
        f.write(f"baseline greedy: {acc_before:.3f}\n"
                f"b_grpo greedy: {acc_b:.3f}\n"
                f"rwb greedy: {acc_r:.3f}\n"
                f"sft_control greedy: {acc_ctl:.3f}\n")
    log("saved /kaggle/working/rl_batched_result.txt")


if __name__ == "__main__":
    main()