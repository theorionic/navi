"""rl_genpolish_500m.py -- RL polish of byte-level generation quality.

Setup: the FineWeb-pretrained 500m (ckpt_500m_step019999.pkl) models text
well under teacher forcing (val ~1.5-2 bpc) but its SAMPLED generations
degrade: one bad byte derails context, and byte-level vocab has no slack
-- output is "correct in some places, wrong in most". This is the classic
exposure-bias / error-compounding gap: training loss never sees the
model's own samples.

Fix under test: REINFORCE ON GENERATED SEQUENCES with an ON-DEVICE
text-quality reward. Competence exists (the model was pretrained on this
data), samples are mostly-plausible sometimes -- exactly the regime where
policy gradient has signal. The reward measures the user's complaint
directly, per generated sample:

  r = 0.4*ascii_ok + 0.3*word_ok + 0.2*norepeat + 0.1*not_eos_immediate
    ascii_ok : fraction of bytes in printable ASCII (text-quality proxy)
    word_ok  : fraction of space-delimited tokens that are dictionary
               words (compact on-device word set: 2048 common words
               hashed into a bloom-style table of 8192 bits)
    norepeat : 1 - (fraction of repeated 4-grams in the sample)
               (anti-loop / anti-degeneration)
    eos_pen  : 1 if the sample ran full length without immediate EOS

All computable with jnp on (N, L) byte arrays -- fully on-device, part of
the compiled scan. REINFORCE with batch baseline; one on-policy AdamW
step per batch of sampled continuations; grad clip; entropy floor.

Arms:
  arm RL  : REINFORCE polish, NAVI_POLISH_ROUNDS rounds
  arm REF : no-update control (same eval, frozen params)
Verification: before/after (1) the SAME reward metric on fresh greedy
samples, (2) held-out teacher-forced bpc (must not degrade), (3) sample
tails printed for qualitative check.

Env: NAVI_POLISH_ROUNDS (200) NAVI_GEN_L (64) NAVI_PROMPTS (32)
     NAVI_LR (2e-6) NAVI_SAMPLES (16)
Usage: python3 /kaggle/working/navi/experiments/rl_genpolish_500m.py
"""
import os
import pickle
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fineweb_data import BOS, EOS, FineWebFeed
from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi

SEQ = int(os.environ.get("NAVI_SEQ", "160"))       # generation context
GEN_L = int(os.environ.get("NAVI_GEN_L", "48"))     # sampled continuation len
N_PROMPTS = int(os.environ.get("NAVI_PROMPTS", "16"))
N_SAMPLES = int(os.environ.get("NAVI_SAMPLES", "8"))   # per prompt
LR = float(os.environ.get("NAVI_LR", "2e-6"))

TAG = "genpolish"
T0 = time.time()


def log(msg):
    print(f"[{TAG} +{time.time()-T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- config ----
ROUNDS = int(os.environ.get("NAVI_POLISH_ROUNDS", "200"))
ENT_FLOOR = 0.0                # optional entropy bonus (0 = off)
EVAL_TESTS = 24

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))
PSTACK = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(None, "cores", None))


def shard_tree(tree):
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


# ------------------------------------------------------------- reward -----
# Bloom-style word table: ~500 common English words -> 16384-bit table.

_VOCAB = """
the be to of and a in that have it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us is are was were been has had said when where why may more man woman child world school state family student group country problem hand part place case week company system program question government number night point home room mother area money story fact month lot right study book eye job word business issue side kind head service friend father power hour game end line law car city community name president team minute idea kid body information parent face others level office door health person art war history party result change morning reason research girl guy moment air teacher force education foot boy age policy process music market sense nation plan college interest death experience effect class control care field development role effort rate heart drug show leader light voice wife police mind price report decision son view relationship town road arm difference value building action model season society tax director position player record paper space ground form event official matter center couple site project activity star table need court oil situation cost industry figure street image phone data picture practice piece land product doctor wall patient worker news test movie north love support technology south board white black red blue green yellow tree sun moon star sky sea land road bridge farm hill river lake mountain valley forest wood fire light earth wind rain snow ice smoke steam
today yesterday tomorrow morning afternoon evening together better must future said told asked thought felt knew wanted needed started continued stopped finished ended kept left went came got made put took gave found became seemed looked turned followed showed held brought carried spoke stood sat ran moved lived considered decided expected remembered realized appeared arrived created happened remained served stayed waited walked watched
"""


def build_word_table():
    words = sorted(set(w.lower() for w in _VOCAB.split() if w.isalpha()))
    tab = np.zeros(16384, dtype=bool)
    for w in words:
        h = 0
        for c in w.encode():
            h = (h * 131 + c) & 0x3FFF
        tab[h] = True
    return tab, len(words)


WORD_TAB, N_WORDS = build_word_table()
WORD_TAB = jnp.asarray(WORD_TAB)


def _hashbuf(low, is_alpha):
    """Per-position rolling word hash, reset at non-alpha. hb[:, j] is the
    hash of the current word prefix ending at j (0 outside words)."""
    def body(carry, x):
        h = carry
        a, c = x
        h = jnp.where(a, (h * 131 + c) & 0x3FFF, 0)
        return h, h
    # scan over sequence axis: is_alpha (N, L), low (N, L)
    _, hb = jax.lax.scan(
        body, jnp.zeros((low.shape[0],), jnp.int32),
        (is_alpha.transpose(1, 0), low.transpose(1, 0)))
    return hb.transpose(1, 0)                    # (N, L) int32


def word_ok_and_rep(tails):
    """(word_ok, word_rep_pen) each (N,). word_ok = fraction of word starts
    whose FULL word hits the table (hash read at the word's END position).
    word_rep_pen = 1 - min(3 * frac(adjacent equal words), 1)."""
    low = jnp.where((tails >= 65) & (tails <= 90), tails + 32, tails)
    is_alpha = (low >= 97) & (low <= 122)
    hb = _hashbuf(low, is_alpha)
    hit = WORD_TAB[hb] & is_alpha
    nxt_alpha = jnp.concatenate(
        [is_alpha[:, 1:], jnp.zeros((tails.shape[0], 1), bool)], axis=1)
    ends = is_alpha & ~nxt_alpha
    starts = is_alpha & ~jnp.concatenate(
        [jnp.zeros((tails.shape[0], 1), bool), is_alpha[:, :-1]], axis=1)
    # propagate each run's end-hit back to the run start: reverse scan
    def rbody(carry, x):
        e, h = x
        cur = jnp.where(e, h, carry)
        return cur, cur
    _, hit_end = jax.lax.scan(
        rbody, jnp.zeros((tails.shape[0],), bool),
        (ends.transpose(1, 0), hit.transpose(1, 0)),
        reverse=True)
    hit_end = hit_end.transpose(1, 0)
    tok_ok = starts & hit_end
    n_starts = starts.sum(-1)
    ok_starts = tok_ok.sum(-1)
    w_ok = jnp.where(n_starts > 0, ok_starts / jnp.maximum(n_starts, 1), 1.0)
    # adjacent equal-word repeats: compare end hashes of consecutive words
    end_h = hb * ends                          # hash at word ends only
    def rep_body(carry, x):
        prev_h, cnt, eq = carry
        eh, e = x
        new_eq = e & (prev_h >= 0) & (eh == prev_h)
        return (jnp.where(e, eh, prev_h), cnt + e, eq + new_eq), None
    (last_h, n_words, n_eq), _ = jax.lax.scan(
        rep_body, (jnp.full((tails.shape[0],), -1, jnp.int32),
                   jnp.zeros((tails.shape[0],), jnp.float32),
                   jnp.zeros((tails.shape[0],), jnp.float32)),
        (end_h.transpose(1, 0), ends.transpose(1, 0)))
    rep = jnp.where(n_words > 1, n_eq / jnp.maximum(n_words - 1, 1), 0.0)
    rep_pen = 1.0 - jnp.minimum(rep * 3.0, 1.0)
    return w_ok, rep_pen


def norepeat4(tails):
    """1 - min(3 * max lagged 4-gram self-similarity, 1). Catches loops
    with period 4/8/12 that adjacent-gram checks miss."""
    n = tails.shape[1] - 3
    gn = jnp.stack([tails[:, i:i + n] for i in range(4)], axis=2)
    rep = jnp.zeros((tails.shape[0],), jnp.float32)
    for lag in (4, 8, 12):
        if n > lag:
            eq = (gn[:, :-lag] == gn[:, lag:]).all(-1).mean(-1)
            rep = jnp.maximum(rep, eq)
    return 1.0 - jnp.minimum(rep * 3.0, 1.0)


def quality_reward(tails):
    """tails: (N, L) int32 generated bytes. Returns (N,) reward in [0,1].
    Verified discrimination (numpy twin, /tmp/reward_check.py + eval):
      good prose 1.0/0.88/0.90, junk 0.32, nonsense 0.31, loop 0.55."""
    low = jnp.where((tails >= 65) & (tails <= 90), tails + 32, tails)
    letter_ok = (((low >= 97) & (low <= 122)) | (low == 32) |
                 (low == 10)).mean(axis=-1)
    w_ok, rep_pen = word_ok_and_rep(tails)
    nr = jnp.minimum(norepeat4(tails), rep_pen)
    return 0.15 * letter_ok + 0.40 * w_ok + \
        0.45 * nr * (0.3 + 0.7 * w_ok)



# ------------------------------------------------------------- main --------
def main():
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

    feed = FineWebFeed(n_val_docs=4000)
    feed.wait_ready(min_bytes=64 * 1024 * 1024)
    rng = np.random.default_rng(0)

    # real FineWeb prompts: windows from the val buffer, cut at a space;
    # FIXED shape (40 wide). Chunk is RIGHT-ALIGNED: EOS left-padding is
    # in-distribution (pretrain stream is ...EOS BOS bytes...), and
    # sampling always starts at the last window byte.
    val_bytes = np.asarray(feed.val.array(), dtype=np.uint8)
    FIXPL = 40                                  # [BOS] + <=39 chunk bytes
    def make_prompts(n, seed):
        r2 = np.random.default_rng(seed)
        arr = np.full((n, FIXPL), EOS, dtype=np.int32)
        for i in range(n):
            off = r2.integers(0, len(val_bytes) - SEQ - GEN_L - 4)
            chunk = bytes(val_bytes[off:off + FIXPL - 1])
            sp = chunk.rfind(b' ')
            if sp > 8:
                chunk = chunk[:sp]
            ids = [BOS] + list(chunk)
            arr[i, FIXPL - len(ids):] = ids
        return jnp.asarray(arr, jnp.int32)

    # ------------------------------- eval: greedy samples + reward -------
    eval_prompts = make_prompts(EVAL_TESTS, 777)

    @jax.jit
    def sample_prompt(pp, prompt, key):
        """(N_SAMPLES, SEQ) buffers for ONE prompt; generated bytes are
        scan outputs starting right after the fixed-length window."""
        def gstep(carry, t):
            buf, kk = carry
            logits = model.apply(pp, buf, train=False)
            logits = logits.at[:, :, BOS].set(-1e9)
            logits = logits.at[:, :, 258:].set(-1e9)
            kk, sub = jax.random.split(kk)
            nxt = jax.random.categorical(sub, logits[:, t, :])  # (S,)
            buf = buf.at[:, t + 1].set(nxt)
            return (buf, kk), nxt

        # right-aligned prompt: last byte at FIXPL-1; scan reads it and
        # writes generated bytes at [FIXPL .. FIXPL+GEN_L-1]
        buf0 = jnp.zeros((N_SAMPLES, SEQ), jnp.int32)
        buf0 = buf0.at[:, :FIXPL].set(jnp.tile(prompt, (N_SAMPLES, 1)))
        (buf, _), out = jax.lax.scan(
            gstep, (buf0, key),
            jnp.arange(FIXPL - 1, FIXPL - 1 + GEN_L))
        return buf, out                        # (S, SEQ), (GEN_L, S)

    @jax.jit
    def reward_outs(outs):
        """outs (GEN_L, S) -> (S,) quality rewards."""
        return quality_reward(outs.T)

    def sample_eval(pp, prompts, key):
        """Sequential over prompts (memory); returns outs (P, S, GEN_L)."""
        all_outs = []
        for i in range(prompts.shape[0]):
            _, out = sample_prompt(pp, prompts[i], key)
            all_outs.append(np.asarray(out.T))          # (S, GEN_L)
        return np.stack(all_outs)                       # (P, S, GEN_L)


    # quick pre-check: sample from pretrained model, measure reward
    kk = jax.random.PRNGKey(42)
    pp = shard_tree(p)
    outs_np = sample_eval(pp, eval_prompts, kk)     # (P, S, GEN_L)
    r0 = np.asarray([quality_reward(
        jnp.asarray(o.reshape(-1, GEN_L))).mean() for o in outs_np])
    log(f"BEFORE: mean quality reward {r0.mean():.3f} "
        f"(ascii/word/loop mix; tails below)")
    ob = outs_np                                    # (P, S, GEN_L)
    for i in range(3):
        log(f"  pre sample: {bytes(np.clip(ob[i, 0], 0, 255)).decode('utf-8', 'replace')!r}")

    # ------------------------------- REINFORCE polish --------------------
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LR, b1=0.9, b2=0.95))


    def run_polish(pp, oo):
        hist = []
        for rnd in range(ROUNDS):
            pr_arr = make_prompts(N_PROMPTS, 1000 + rnd)
            kk2, sub = jax.random.split(kk)
            grads = None
            r_mean_acc = 0.0
            C = 4                                       # prompts per chunk
            n_chunks = N_PROMPTS // C
            for c in range(N_PROMPTS // C):
                sl = slice(c * C, (c + 1) * C)
                kk2, sub = jax.random.split(kk2)
                g_c, rm_c = polish_chunk_jit(pp, pr_arr[sl], sub)
                grads = g_c if grads is None else jax.tree_util.tree_map(
                    jnp.add, grads, g_c)
                r_mean_acc += float(rm_c)
            rm = r_mean_acc / (N_PROMPTS // C)
            u, oo = tx.update(grads, oo, pp)
            pp = optax.apply_updates(pp, u)
            hist.append(rm)
            if rnd % 20 == 0 or rnd == ROUNDS - 1:
                log(f"  polish {rnd:3d}: batch reward {rm:.3f}")
        return pp, oo, hist

    @jax.jit
    def polish_chunk_jit(pp, prompts, kk):
        def one_prompt(pr, k1):
            def gstep(carry, t):
                buf, k2 = carry
                logits = model.apply(pp, buf, train=False)
                logits = logits.at[:, :, BOS].set(-1e9)
                logits = logits.at[:, :, 258:].set(-1e9)
                k2, sub = jax.random.split(k2)
                nxt = jax.random.categorical(sub, logits[:, t, :])
                buf = buf.at[:, t + 1].set(nxt)
                return (buf, k2), nxt

            # right-aligned fixed-length prompt; sampling starts at its
            # last byte (FIXPL-1), generated bytes land at [FIXPL ..]
            buf0 = jnp.zeros((N_SAMPLES, SEQ), jnp.int32)
            buf0 = buf0.at[:, :FIXPL].set(jnp.tile(pr, (N_SAMPLES, 1)))
            (buf, _), out = jax.lax.scan(
                gstep, (buf0, k1),
                jnp.arange(FIXPL - 1, FIXPL - 1 + GEN_L))
            return buf, out                              # (S,SEQ),(GEN_L,S)

        keys = jax.random.split(kk, prompts.shape[0])
        bufs, outs = jax.vmap(one_prompt)(prompts, keys)     # (C,S,SEQ)
        r = quality_reward(jnp.transpose(outs, (0, 2, 1)).reshape(
            -1, GEN_L))
        adv = r - r.mean()
        bufs_f = bufs.reshape((-1, SEQ))
        adv_f = adv.reshape(-1)

        def loss_fn(w):
            logits = model.apply(w, bufs_f[:, :-1], train=True)
            logp = jax.nn.log_softmax(logits)
            lp = jnp.take_along_axis(
                logp, bufs_f[:, 1:][..., None], axis=-1)[..., 0]
            T = bufs_f.shape[1] - 1
            # policy-gradient only on generated bytes [FIXPL .. ]
            mask = (jnp.arange(T) >= FIXPL - 1).astype(lp.dtype)
            pg = -(lp * adv_f[:, None] * mask[None, :]).sum() / \
                jnp.maximum(1.0, jnp.abs(adv_f).sum() * mask.sum())
            return pg

        g = jax.grad(loss_fn)(pp)
        return g, r.mean()

    log(f"arm RL: {ROUNDS} rounds x {N_PROMPTS} prompts x {N_SAMPLES} "
        f"samples (GEN_L={GEN_L})")
    oo = tx.init(pp)
    pp, oo, hist = run_polish(pp, oo)

    # ------------------------------- post eval ---------------------------
    outs_np2 = sample_eval(pp, eval_prompts, kk)    # (P, S, GEN_L)
    r1 = np.asarray([quality_reward(
        jnp.asarray(o.reshape(-1, GEN_L))).mean() for o in outs_np2])
    log(f"AFTER: mean quality reward {r1.mean():.3f} (before {r0.mean():.3f})")
    ob2 = outs_np2
    for i in range(3):
        log(f"  post sample: {bytes(np.clip(ob2[i, 0], 0, 255)).decode('utf-8', 'replace')!r}")

    # teacher-forced val bpc must not degrade
    vb = jnp.asarray(np.asarray(val_bytes[:SEQ * 8], dtype=np.uint8).reshape(8, SEQ), jnp.int32)
    @jax.jit
    def val_bpc(pp):
        logits = model.apply(pp, vb[:, :-1], train=False)
        ce = optax.softmax_cross_entropy_with_integer_labels(
            logits, vb[:, 1:])
        return ce.mean() / jnp.log(2)

    bpc_before = float(val_bpc(shard_tree(p)))
    bpc_after = float(val_bpc(pp))
    log(f"teacher-forced val bpc: {bpc_before:.4f} -> {bpc_after:.4f}")

    ok = r1.mean() > r0.mean() + 0.02 and bpc_after < bpc_before + 0.05
    log(f"VERDICT: RL polish {'WORKS' if ok else 'NO GAIN'} "
        f"(reward {r0.mean():.3f}->{r1.mean():.3f}, "
        f"bpc {bpc_before:.3f}->{bpc_after:.3f})")
    with open('/kaggle/working/rl_genpolish_result.txt', 'w') as f:
        f.write(f"reward before: {r0.mean():.4f}\nreward after: {r1.mean():.4f}\n"
                f"bpc before: {bpc_before:.4f}\nbpc after: {bpc_after:.4f}\n"
                f"VERDICT: {'WORKS' if ok else 'NO-GAIN'}\n")
    with open('/kaggle/working/genpolish_params.pkl', 'wb') as f:
        pickle.dump({'params': jax.device_get(pp), 'step': 'polished'}, f)
    log("saved /kaggle/working/genpolish_params.pkl")


if __name__ == "__main__":
    main()