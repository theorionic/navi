"""Product-key memory: the Pool mechanism, hardware-independent.

Per token (per class): score the query against two small subkey tables,
take the top-side candidates from each, form only their candidate sub-grid,
pick the top-k product pairs, gather those rows of the value table,
softmax-weight, sum. The two-sided filter is what keeps this viable at
Pool sizes where scoring the full c1 x c2 grid (let alone 1B slots) would
be the dominant cost.

Value rows are the byte bulk -> the tier that gets offloaded to RAM/disk in
the shipping design. Subkey tables stay resident next to the Reasoner.
"""

import os

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Array

from navi.config import MemoryConfig

# NAVI_MEM_DEBUG=1 -> stage-by-stage jax.debug.print trace of the Pool read
# path (query/router scores/top-k/values/weights/readout). Zero cost when
# off: the flag is read at import time and the prints sit behind `if`.
_MEM_DEBUG = os.environ.get("NAVI_MEM_DEBUG", "0") == "1"


def _dbg(tag: str, name: str, *vals) -> None:
    jax.debug.print("[dbg:" + tag + "] " + name + " "
                    + " ".join(["{:.6f}"] * len(vals)), *vals)


class ProductKeyMemory(nn.Module):
    """Trainable key -> value memory addressed by product keys.

    per_class_dim is the subkey/query width and per-slot value width;
    n_classes independent lookups run per token and are projected to
    value_dim by w_o.
    """

    cfg: MemoryConfig
    per_class_dim: int
    value_dim: int

    def setup(self) -> None:
        c = self.cfg
        self.w_q = nn.Dense(c.n_classes * max(c.c3, 2) * self.per_class_dim, use_bias=False, name="w_q")
        self.w_o = nn.Dense(self.value_dim, use_bias=False, name="w_o")
        self.k1 = self.param(
            "k1", jax.nn.initializers.normal(0.02), (c.n_classes, c.c1, self.per_class_dim)
        )
        self.k2 = self.param(
            "k2", jax.nn.initializers.normal(0.02), (c.n_classes, c.c2, self.per_class_dim)
        )
        # E2: third factor table; values index = (i1*c2 + i2)*c3 + i3.
        if c.c3 >= 2:
            self.k3 = self.param(
                "k3", jax.nn.initializers.normal(0.02),
                (c.n_classes, c.c3, self.per_class_dim))
        # E6: learned injection head. LayerNorm first: the head reads the
        # residual stream, whose scale drifts across training; unnormalized
        # input makes the head track drift instead of usage.
        if c.learned_inject:
            self.ln_inj = nn.LayerNorm()
            self.inj_in = nn.Dense(c.inject_hidden, use_bias=False)
            self.inj_out = nn.Dense(
                c.n_classes * c.c1 * c.c2, use_bias=False,
                kernel_init=jax.nn.initializers.normal(0.01))
        # the Pool: values[class, slot, dim] -- the offload tier at scale.
        # bf16 storage: half the HBM for params + Lion state + grads (the
        # 8-block run23 config needs it: fp32 x3 = 12.9GB > 9.5GB/core).
        # Values are a lookup table accumulated over tens of thousands of
        # touches - bf16 (8 mantissa bits) is quantization-insensitive here.
        # Cast to fp32 after the gather; softmax/weighting stay fp32.
        self.values = self.param(
            "values",
            jax.nn.initializers.normal(0.02, dtype=jnp.bfloat16),
            (c.n_classes, c.c1 * c.c2 * max(c.c3, 1), self.per_class_dim),
        )

    def _dense_path(self, x, q1, q2, s1, s2, temp):
        print("[trace] PKM._dense_path (chunked scan) compiling", flush=True)
        """Dense read: softmax over the FULL c1*c2 grid per token/class.

        Size-agnostic by construction: values are combined in fixed-size
        token chunks (cfg.dense_chunk), so the transient per chunk is
        chunk * c1*c2 * D -- independent of batch, sequence, pool size and
        backbone. Any pool fits any host by turning dense_chunk down.

        Functional form identical to the sparse read (softmax(temp*scores)
        weighted value sum over slots) -> params transfer to the sparse
        two-sided path with no export step. All keys/values receive
        gradient every step: no selection, no zero-gradient death.
        """
        c = self.cfg
        b, l, _ = x.shape
        t = c.score_temp if temp is None else temp
        n_slots = c.c1 * c.c2
        D = self.per_class_dim

        bl = b * l
        s1f = s1.reshape(bl, c.n_classes, c.c1)
        s2f = s2.reshape(bl, c.n_classes, c.c2)
        vals = self.values.astype(jnp.float32)         # (C, n_slots, D)

        def chunk_read(carry, w_s1s2):
            # w_s1s2: (chunk, 2, C, side) -> grid (chunk, C, n_slots)
            s1c, s2c = w_s1s2
            grid = s1c[..., :, None] + s2c[..., None, :]
            w = jax.nn.softmax(t * grid, axis=-1)
            w = w.reshape(w.shape[0], c.n_classes, n_slots)
            h = jnp.einsum("bln,cnd->bcd", w, vals)
            return carry, h

        # Chunked scan over tokens: transient per chunk is
        # chunk * C * n_slots fp32 (dense_chunk=8, c=512: 33MB) -
        # independent of batch/seq/pool. Remat keeps backward's live
        # set to one chunk; softmax recomputes from s1/s2 (cheap).
        chunk = max(1, c.dense_chunk)
        pad = (-bl) % chunk
        s1p = jnp.pad(s1f, ((0, pad), (0, 0), (0, 0)))
        s2p = jnp.pad(s2f, ((0, pad), (0, 0), (0, 0)))
        n_chunks = (bl + pad) // chunk
        s1c = s1p.reshape(n_chunks, chunk, c.n_classes, c.c1)
        s2c = s2p.reshape(n_chunks, chunk, c.n_classes, c.c2)
        _, hp = jax.lax.scan(
            jax.checkpoint(chunk_read), None, (s1c, s2c))
        h = hp.reshape(bl + pad, c.n_classes, D)[:bl]
        h = h.reshape(b, l, c.n_classes * D)
        # lb: identical to sparse -- side entropy over full tables so the
        # aux loss means the same thing in both modes.
        lb = jnp.float32(0.0)
        if c.lb_weight > 0.0:
            p1 = jax.nn.softmax(s1, axis=-1)
            p2 = jax.nn.softmax(s2, axis=-1)
            e1 = -(p1 * jnp.log(p1 + 1e-9)).sum(axis=-1).mean()
            e2 = -(p2 * jnp.log(p2 + 1e-9)).sum(axis=-1).mean()
            lb = -(e1 + e2)
        # aux slots: grid argmax per class, chunked like the read so the
        # full (bl, C, n_slots) tensor never materializes.
        def chunk_am(carry, w_s1s2):
            s1c, s2c = w_s1s2
            grid = (s1c[..., :, None] + s2c[..., None, :]).reshape(
                s1c.shape[0], c.n_classes, n_slots)
            return carry, grid.argmax(axis=-1)

        _, amp = jax.lax.scan(chunk_am, None, (s1c, s2c))
        am = amp.reshape(bl + pad, c.n_classes)[:bl].reshape(
            b, l, c.n_classes)
        aux = {"slots": am[..., None],
               "scores": jnp.ones((b, l, c.n_classes)), "lb": lb}
        return self.w_o(h), aux

    def __call__(self, x, train=False, ctx_ids=None, temp=None,
                 dense=False, visit_off=None, eps_t=None,
                 usage_state=None):
        """ctx_ids: (b, l) token IDs of this block's input context. Required
        when cfg.hash_slots -- slot candidates are hash(ctx_ids) mod slots,
        so each (key,n1,n2) context maps to fixed value rows on first touch
        and the write gradient lands in exactly those rows. No keys involved.
        dense=True routes to the full-grid chunked read (see _dense_path).
        usage_state: E9 (usage, n_touched) pair -- on-device visit EMA for
        selection offsets. Updated here and returned via aux so the trainer
        threads it back next step (device-resident; no host sync).
        """
        c = self.cfg
        b, l, _ = x.shape
        if _MEM_DEBUG:
            # [dbg:in] residual-stream stats entering the memory layer
            _dbg("in", "x_mean x_std x_sqmean",
                 x.mean(), x.std(), jnp.sqrt((x * x).mean()))
        if c.hash_slots:
            assert ctx_ids is not None, "hash_slots requires ctx_ids"
            return self._hash_path(x, ctx_ids)
        if c.c3 >= 2:
            return self._read3(x, temp, dense)
        if c.learned_inject:
            return self._read_inject(x, temp, dense)
        qf = self.w_q(x).reshape(b, l, c.n_classes, -1, self.per_class_dim)
        q1, q2 = qf[..., 0, :], qf[..., 1, :]
        s1 = jnp.einsum("blcd,ckd->blck", q1, self.k1)
        s2 = jnp.einsum("blcd,ckd->blck", q2, self.k2)
        if dense:
            return self._dense_path(x, q1, q2, s1, s2, temp)
        print("[trace] PKM sparse two-sided top-k path compiling",
              flush=True)
        # E9 selection offsets: usage tables (C, c1)/(C, c2) touch-fraction
        # EMA carried across steps on-device. rel in [-1, 1]; beta=0 ->
        # identity. Applied to SELECTION scores only (sel1/sel2): the
        # readout softmax below is recomputed from RAW s1/s2 so the read
        # functional form is untouched -- this is a race bias, not a
        # readout distortion.
        sel1, sel2 = s1, s2
        if usage_state is not None and c.balance_beta > 0.0:
            u1s, u2s = usage_state                       # (C,c1) (C,c2)
            r1 = jnp.clip(u1s / (u1s.mean() + 1e-9) - 1.0, -1.0, 1.0)
            r2 = jnp.clip(u2s.mean(axis=0) / (u2s.mean() + 1e-9) - 1.0,
                          -1.0, 1.0)
            sel1 = s1 - c.balance_beta * r1[None, None]
            sel2 = s2 - c.balance_beta * jnp.reshape(r2, (1, 1, 1, -1))
        if _MEM_DEBUG:
            # [dbg:score] router health: score spread vs key norm scale.
            # std(s) ~ q_std * k_std * sqrt(per_class_dim); if the query
            # norm collapses (dead w_q) or keys explode, this shows it.
            _dbg("score", "s1_mean s1_std q1_std k1_norm",
                 s1.mean(), s1.std(), q1.std(),
                 jnp.sqrt((self.k1 * self.k1).mean()))
        # two-sided filter; side_top candidates per side, then the product
        # grid. With side_top=64 and c1=c2=512 the grid covers only 1.56%
        # of slots - measured routing collapse sits exactly at that ceiling
        # (run22f: 0.9-3.5% alive). Side_top=128 raises reach to 6.2%, but
        # the (side,side) grid must be built per class (lax.map over the
        # class axis) to keep the transient at today's 8.6GB peak: classes
        # are independent planes, so chunking is EXACT, not approximate.
        i1 = jax.lax.top_k(sel1, c.side_top)[1]  # (b, l, classes, side)
        i2 = jax.lax.top_k(sel2, c.side_top)[1]
        if c.explore_beta > 0.0 and visit_off is not None:
            # E4: host-maintained per-subkey visit EMA folded in as a
            # static offset BEFORE top-k. Cold subkeys (visit ~ 0) get
            # +beta, hot ones 0 -- unselected keys re-enter the race.
            # Offsets are constants w.r.t. this trace: no grad flows to
            # them; the EMA update happens on the host each step.
            o1, o2 = visit_off
            s1 = s1 + jnp.reshape(o1, (1, 1, 1, -1))
            s2 = s2 + jnp.reshape(o2, (1, 1, 1, -1))
            i1 = jax.lax.top_k(s1, c.side_top)[1]
            i2 = jax.lax.top_k(s2, c.side_top)[1]
        g1 = jnp.take_along_axis(sel1, i1, axis=-1)
        g2 = jnp.take_along_axis(sel2, i2, axis=-1)

        # Two-level top-k over the side grid: for any row i, the top-cand_k
        # entries are g1[i] + top-cand_k(g2) (additive grid => per-row order
        # is g2's order). The global top-k must therefore lie in the union
        # of per-row top-(cand_k) sets, i.e. the (side, cand_k) sub-grid.
        # This is EXACT (proof: if j* not in top-cand_k(g2), then k entries
        # in the same row i* beat grid[i*, j*], so (i*,j*) can't be global
        # top-k) and shrinks the transient from (side x side) to
        # (side x cand_k): side_top=128 -> 1024 pairs, 16x smaller than
        # 16384, keeping the HBM transient at ~0.27GB per replica instead
        # of 4.3GB. side_top can now scale to 256+ with no memory cliff.
        g2k = min(c.cand_k, c.side_top)
        g2_top = jax.lax.top_k(g2, g2k)[1]          # (b, l, classes, g2k)
        g2_vals = jnp.take_along_axis(g2, g2_top, axis=-1)

        # Batched class grid: classes are independent planes, and the
        # top-k over each (side, g2k) sub-grid can run on ALL classes at
        # once by folding the class axis into the batch. The old
        # lax.map(class_grid, ...) unrolled 4 sequential TopK ops per
        # block (16 per step fwd+bwd); this does ONE batched top_k.
        # (b, l, classes, side, g2k) built via broadcasting -- no per-class loop.
        sub = g1[..., :, None] + g2_vals[..., None, :]  # (b,l,C,side,g2k)
        sub_b = sub.transpose(2, 0, 1, 3, 4)            # (C,b,l,side,g2k)
        side = sub_b.shape[-2]
        flat = sub_b.reshape(c.n_classes, b, l, side * g2k)
        k = min(c.cand_k, side * g2k)
        f_idx = jax.lax.top_k(flat, k)[1]               # (C,b,l,k)
        scores = jnp.take_along_axis(flat, f_idx, -1)
        # (classes, b, l, k) -> (b, l, classes, k)
        f_idx = jnp.moveaxis(f_idx, 0, 2)
        scores = jnp.moveaxis(scores, 0, 2)
        r, col = f_idx // g2k, f_idx % g2k
        pi1 = jnp.take_along_axis(i1, r, axis=-1)
        pi2 = jnp.take_along_axis(i2, g2_top, axis=-1)
        pi2 = jnp.take_along_axis(pi2, col, axis=-1)
        slots = pi1 * c.c2 + pi2
        if c.hybrid_hash and ctx_ids is not None:
            # E3: append hash-derived candidates so every read touches
            # router-independent slots. Hash rows are scored by the same
            # q.k product against the FLATTENED key grid (fair competition
            # in the softmax) and get a weight floor hybrid_eps so their
            # value rows always receive gradient.
            hh = ctx_ids.astype(jnp.uint32)
            hh = hh + jnp.uint32(0x9E3779B9)
            hh = (hh ^ (hh >> jnp.uint32(16))) * jnp.uint32(0x85EBCA6B)
            hh = hh ^ (hh >> jnp.uint32(13))
            hh = (hh * jnp.uint32(0xC2B2AE35)) + jnp.uint32(l)
            hh = hh ^ (hh >> jnp.uint32(16))
            hslot = (hh % jnp.uint32(c.c1 * c.c2)).astype(jnp.int32)  # (b,l)
            if c.usage_hash and visit_off is not None:
                # visit_off is the (c1*c2,) normalized visit table directly
                # (trainer passes visits_t; the 3-tuple explore mode is not
                # used in the 500M trainer).
                # E7: re-hash the token hash with each candidate slot's
                # usage band, then keep probing (deterministic jitter)
                # until a low-visit slot is found. visit_off[2] is the
                # host-maintained per-slot visit table (c1*c2,), normalized
                # 0..1. Cold slots (< median) accept on first probe; hot
                # ones rotate the hash forward. Fully deterministic per
                # (token, table-state) -- no RNG, no new params.
                visits = visit_off                       # (c1*c2,) fp32 0..1
                n_slots = c.c1 * c.c2
                # 8 probes: slot_j = (hslot + j * stride) % n, pick the
                # coldest of the probes per position.
                stride = jnp.uint32(2654435761 % n_slots).astype(jnp.int32)
                probes = (hslot[..., None]
                          + jnp.arange(8, dtype=jnp.int32) * stride) % n_slots
                pv = visits[probes]                      # (b,l,8)
                pick = jnp.argmin(pv, axis=-1)           # (b,l)
                hslot = jnp.take_along_axis(
                    probes, pick[..., None], axis=-1)[..., 0]
            cls = jnp.arange(c.n_classes)
            hslots = (hslot[None] + (cls[:, None, None] * 7919)) % (c.c1 * c.c2)
            hslots = hslots.astype(jnp.int32)[..., None]          # (C,b,l,1)
            hslots = jnp.transpose(hslots, (1, 2, 0, 3))          # (b,l,C,1)
            # score = s1[i1] + s2[i2] via the already-computed side tables
            hi1 = hslots // c.c2
            hi2 = hslots % c.c2
            hs = (jnp.take_along_axis(s1, hi1, axis=-1)
                  + jnp.take_along_axis(s2, hi2, axis=-1))        # (b,l,C,1)
            slots = jnp.concatenate([slots, hslots], axis=-1)     # (b,l,C,k+1)
            scores = jnp.concatenate([scores, hs], axis=-1)
        v = self.values[jnp.arange(c.n_classes)[None, None, :, None], slots]
        v = v.astype(jnp.float32)  # bf16 storage; math in fp32
        if _MEM_DEBUG:
            # [dbg:gather] value-table health at the point of read
            _dbg("gather", "v_mean v_std", v.mean(), v.std())
        t = c.score_temp if temp is None else temp
        w = jax.nn.softmax(t * scores, axis=-1)
        if _MEM_DEBUG:
            # [dbg:w] softmax temperature effect: with t ramping from ~0,
            # w -> uniform (max-min -> 0) and value-gradient concentration
            # dies; this line quantifies exactly that.
            _dbg("w", "w_mean w_min w_max", w.mean(), w.min(), w.max())
        if c.lb_eps > 0.0:
            w = w * (1.0 - c.lb_eps) + c.lb_eps / w.shape[-1]
        if c.hybrid_hash:
            # E3 floor (fixed): renormalize WITHOUT the hash row, then
            # blend: w = (1-eps) * softmax(router rows) + eps * 1 on the
            # hash row. Mass is exactly 1; hash row always gets eps grad;
            # no double-count => no NaN blowup at high temp.
            # E8: eps_t (host scalar) scales the floor per batch --
            # drift gate multiplies exploration when topics shift.
            eps = c.hybrid_eps if eps_t is None else eps_t
            eps = jnp.minimum(eps, 0.5)
            w_r = jax.nn.softmax(t * scores[..., :-1], axis=-1)
            w = jnp.concatenate([w_r * (1.0 - eps),
                                 jnp.full_like(w_r[..., :1], eps)],
                                axis=-1)
        h = (w[..., None] * v).sum(axis=-2)
        if _MEM_DEBUG:
            # [dbg:out] the actual memory contribution magnitude
            _dbg("h", "h_mean h_std", h.mean(), h.std())
        h = h.reshape(b, l, c.n_classes * self.per_class_dim)
        lb = jnp.float32(0.0)
        if c.lb_weight > 0.0:
            # Balance via ROUTER ENTROPY over the side-score tables.
            # The switch n_slots * sum(f_i * p_i) form is vacuous for
            # sparse top-k (f_i over the gathered subset is always 1/k,
            # so the term can't tell balanced from collapsed). Instead:
            # maximize entropy of softmax(s1/s2) per class -- gradients
            # flow through ALL side keys, spreading candidate mass off
            # the hot subkeys. Loss = mean negative entropy, O(1) scale.
            p1 = jax.nn.softmax(s1, axis=-1)  # (b, l, classes, c1)
            p2 = jax.nn.softmax(s2, axis=-1)
            e1 = -(p1 * jnp.log(p1 + 1e-9)).sum(axis=-1).mean()
            e2 = -(p2 * jnp.log(p2 + 1e-9)).sum(axis=-1).mean()
            lb = -(e1 + e2)
        # E9 usage EMA update (device-side, gradient-free): mark the
        # c1-side/c2-side subkeys actually selected this step. per-class
        # tables (C, c1)/(C, c2): i1 is (b, l, C, side) -> one-hot count
        # per class. EMA decays old counts; fraction normalized by tokens.
        if usage_state is not None and c.balance_beta > 0.0:
            u1s, u2s = usage_state
            n_tok = b * l
            hits1 = jax.nn.one_hot(i1, c.c1, dtype=jnp.float32).sum(
                axis=(0, 1, 3)) / max(1, n_tok)        # (C, c1)
            hits2 = jax.nn.one_hot(i2, c.c2, dtype=jnp.float32).sum(
                axis=(0, 1, 3)) / max(1, n_tok)        # (C, c2)
            d = c.balance_ema
            u1n = u1s * d + hits1 * (1.0 - d)
            u2n = u2s * d + hits2 * (1.0 - d)
            usage_new = (u1n, u2n)
        else:
            usage_new = usage_state
        return self.w_o(h), {"slots": slots.reshape(b, l, -1),
                             "scores": scores.reshape(b, l, -1), "lb": lb,
                             "usage": usage_new}

    def _read_inject(self, x, temp, dense):
        """E6: learned injection -- a small head scores the FLATTENED slot
        grid per token per class from the residual stream; top inject_k
        become appended candidates (same shape/mechanics as E3's hash
        rows: scored by the router's side tables, eps-floored in the
        softmax). Content-aware injection: the head can learn to
        nominate semantically-relevant cold slots, unlike uniform hash.

        Head params: Dense(d->inject_hidden) + Dense(hidden->c1*c2 per
        class). Top-k over (c1*c2) per class is ONE batched top_k after
        folding classes into batch. The injected rows are appended AFTER
        the router's own cand_k picks and compete in the softmax with
        side-table scores (fair competition) + eps floor.
        """
        print("[trace] PKM learned-inject read compiling", flush=True)
        c = self.cfg
        b, l, _ = x.shape
        h = self.ln_inj(x)
        hin = nn.relu(self.inj_in(h))                    # (b,l,inject_hidden)
        inj_scores = self.inj_out(hin)                   # (b,l,C*c1*c2)
        inj_scores = inj_scores.reshape(b, l, c.n_classes, c.c1 * c.c2)
        k = min(c.inject_k, c.c1 * c.c2)
        _, islots = jax.lax.top_k(inj_scores, k)         # (b,l,C,k)

        # router's own candidates (identical to the sparse path)
        qf = self.w_q(x).reshape(b, l, c.n_classes, -1, self.per_class_dim)
        q1, q2 = qf[..., 0, :], qf[..., 1, :]
        s1 = jnp.einsum("blcd,ckd->blck", q1, self.k1)
        s2 = jnp.einsum("blcd,ckd->blck", q2, self.k2)
        i1 = jax.lax.top_k(s1, c.side_top)[1]
        i2 = jax.lax.top_k(s2, c.side_top)[1]
        g1 = jnp.take_along_axis(s1, i1, axis=-1)
        g2 = jnp.take_along_axis(s2, i2, axis=-1)
        g2k = min(c.cand_k, c.side_top)
        g2_top = jax.lax.top_k(g2, g2k)[1]
        g2_vals = jnp.take_along_axis(g2, g2_top, axis=-1)
        sub = g1[..., :, None] + g2_vals[..., None, :]
        sub_b = sub.transpose(2, 0, 1, 3, 4)
        side = sub_b.shape[-2]
        flat = sub_b.reshape(c.n_classes, b, l, side * g2k)
        kk = min(c.cand_k, side * g2k)
        f_idx = jax.lax.top_k(flat, kk)[1]
        rscores = jnp.take_along_axis(flat, f_idx, -1)
        f_idx = jnp.moveaxis(f_idx, 0, 2)
        rscores = jnp.moveaxis(rscores, 0, 2)
        r, col = f_idx // g2k, f_idx % g2k
        pi1 = jnp.take_along_axis(i1, r, axis=-1)
        pi2 = jnp.take_along_axis(i2, g2_top, axis=-1)
        pi2 = jnp.take_along_axis(pi2, col, axis=-1)
        rslots = pi1 * c.c2 + pi2                        # (b,l,C,kk)

        # inject: side-table scores for the head-nominated slots
        hi1 = islots // c.c2
        hi2 = islots % c.c2
        hsc = (jnp.take_along_axis(s1, hi1, axis=-1)
               + jnp.take_along_axis(s2, hi2, axis=-1))  # (b,l,C,k)
        slots = jnp.concatenate([rslots, islots], axis=-1)
        scores = jnp.concatenate([rscores, hsc], axis=-1)

        v = self.values[jnp.arange(c.n_classes)[None, None, :, None], slots]
        v = v.astype(jnp.float32)
        t = c.score_temp if temp is None else temp
        # renormalized blend (same fixed scheme as E3): router rows get
        # (1-eps) mass split by softmax, injected rows share eps.
        w_r = jax.nn.softmax(t * scores[..., :-c.inject_k], axis=-1)
        w_i = jnp.full_like(w_r[..., :1], c.hybrid_eps / c.inject_k)
        w_i = w_i.repeat(c.inject_k, axis=-1)
        w = jnp.concatenate([w_r * (1.0 - c.hybrid_eps), w_i], axis=-1)
        out = (w[..., None] * v).sum(axis=-2)
        out = out.reshape(b, l, c.n_classes * self.per_class_dim)
        lb = jnp.float32(0.0)
        if c.lb_weight > 0.0:
            p1 = jax.nn.softmax(s1, axis=-1)
            p2 = jax.nn.softmax(s2, axis=-1)
            e1 = -(p1 * jnp.log(p1 + 1e-9)).sum(axis=-1).mean()
            e2 = -(p2 * jnp.log(p2 + 1e-9)).sum(axis=-1).mean()
            lb = -(e1 + e2)
        aux = {"slots": slots.reshape(b, l, -1),
               "scores": scores.reshape(b, l, -1), "lb": lb}
        return self.w_o(out), aux

    def _read3(self, x, temp, dense):
        """E2: 3-factor product-key read (c1 x c2 x c3 per class).

        Conditional 3-stage top-k, exact for additive grids by the same
        exchange argument as the two-level proof: for any (i1,i2) the
        top-cand3 entries of s1[i1]+s2[i2]+s3 must lie in the top-cand3
        of s3, and iteratively for the outer stages. Transient is
        (side, cand2, cand3) per class-plane, not c1*c2*c3.
        """
        print("[trace] PKM 3-factor read compiling", flush=True)
        c = self.cfg
        b, l, _ = x.shape
        D = self.per_class_dim
        qf = self.w_q(x).reshape(b, l, c.n_classes, -1, D)
        q1, q2, q3 = qf[..., 0, :], qf[..., 1, :], qf[..., 2, :]
        s1 = jnp.einsum("blcd,ckd->blck", q1, self.k1)
        s2 = jnp.einsum("blcd,ckd->blck", q2, self.k2)
        s3 = jnp.einsum("blcd,ckd->blck", q3, self.k3)

        i1 = jax.lax.top_k(s1, c.side_top)[1]
        g1 = jnp.take_along_axis(s1, i1, axis=-1)
        i2 = jax.lax.top_k(s2, min(c.cand2, c.c2))[1]
        g2 = jnp.take_along_axis(s2, i2, axis=-1)
        i3 = jax.lax.top_k(s3, min(c.cand3, c.c3))[1]
        g3 = jnp.take_along_axis(s3, i3, axis=-1)

        grid = g1[..., :, None, None] + g2[..., None, :, None] + g3[..., None, None, :]
        side, s2k, s3k = grid.shape[3], grid.shape[4], grid.shape[5]
        flat = grid.transpose(2, 0, 1, 3, 4, 5).reshape(
            c.n_classes, b, l, side * s2k * s3k)
        k = min(c.cand_k, side * s2k * s3k)
        f_idx = jax.lax.top_k(flat, k)[1]
        scores = jnp.take_along_axis(flat, f_idx, -1)
        f_idx = jnp.moveaxis(f_idx, 0, 2)                    # (b,l,C,k)
        scores = jnp.moveaxis(scores, 0, 2)
        r1 = f_idx // (s2k * s3k)
        rem = f_idx % (s2k * s3k)
        r2, r3 = rem // s3k, rem % s3k
        pi1 = jnp.take_along_axis(i1, r1, axis=-1)
        pi2 = jnp.take_along_axis(i2, r2, axis=-1)
        pi3 = jnp.take_along_axis(i3, r3, axis=-1)
        slots = (pi1 * c.c2 + pi2) * c.c3 + pi3
        v = self.values[jnp.arange(c.n_classes)[None, None, :, None], slots]
        v = v.astype(jnp.float32)
        t = c.score_temp if temp is None else temp
        w = jax.nn.softmax(t * scores, axis=-1)
        h = (w[..., None] * v).sum(axis=-2)
        h = h.reshape(b, l, c.n_classes * D)
        lb = jnp.float32(0.0)
        if c.lb_weight > 0.0:
            p1 = jax.nn.softmax(s1, axis=-1)
            p2 = jax.nn.softmax(s2, axis=-1)
            e1 = -(p1 * jnp.log(p1 + 1e-9)).sum(axis=-1).mean()
            e2 = -(p2 * jnp.log(p2 + 1e-9)).sum(axis=-1).mean()
            lb = -(e1 + e2)
        aux = {"slots": slots.reshape(b, l, -1),
               "scores": scores.reshape(b, l, -1), "lb": lb}
        return self.w_o(h), aux

    def _hash_path(self, x, ctx_ids):
        """Deterministic placement: slot = mix(token ids) mod (c1*c2).

        mix() = splitmix64-style integer mixing, stable in JAX (int32-safe:
        masks to 32 bits at each step to avoid TPU int64 overhead/overflow).
        Gather hash_k rows per class, uniform weights, project with w_o —
        the reasoner still learns WHAT to read; the WHERE is structural.
        """
        c = self.cfg
        b, l, _ = x.shape
        n_slots = c.c1 * c.c2
        h = ctx_ids.astype(jnp.uint32)
        # splitmix64 finalizer on 32-bit lanes: xorshift-multiply chains
        h = h + jnp.uint32(0x9E3779B9)
        h = (h ^ (h >> jnp.uint32(16))) * jnp.uint32(0x85EBCA6B)
        h = h ^ (h >> jnp.uint32(13))
        h = (h * jnp.uint32(0xC2B2AE35)) + jnp.uint32(ctx_ids.shape[1])
        h = h ^ (h >> jnp.uint32(16))
        slot = (h % jnp.uint32(n_slots)).astype(jnp.int32)  # (b, l)
        cls = jnp.arange(c.n_classes)
        # hash_k distinct-ish slots per position: offset by class*prime so
        # classes read different planes, plus small deterministic jitter
        slots = slot[None] + (cls[:, None] * jnp.uint32(2654435761 % n_slots)).astype(jnp.int32)[..., None]  # (C, b, l)
        slots = slots % n_slots
        v = self.values[cls[:, None, None], slots]  # (C, b, l, D)
        hsum = v.sum(axis=0)  # (b, l, D) uniform combine — w_o learns the readout
        out = x + self.w_o(hsum)
        aux = {"slots": jnp.transpose(slots, (1, 2, 0)).reshape(b, l, -1),
               "scores": jnp.ones((b, l, c.n_classes)), "lb": jnp.float32(0.0)}
        return out, aux

    def num_slots(self) -> int:
        return self.cfg.n_classes * self.cfg.c1 * self.cfg.c2


def exact_topk_slots(
    q1: Array, q2: Array, k1: Array, k2: Array, cfg: MemoryConfig
) -> Array:
    """Brute-force top-k slot ids by scoring the full c1 x c2 grid.

    Reference for tests: the two-sided candidate filter must recover these
    slots with high recall while scoring ~2*side_top instead of c1 + c2 +
    c1*c2 vectors.
    """
    s1 = jnp.einsum("cd,ckd->ck", q1, k1)  # (classes, c1)
    s2 = jnp.einsum("cd,ckd->ck", q2, k2)
    grid = s1[..., :, None] + s2[..., None, :]
    grid = grid.reshape(grid.shape[0], -1)
    return jax.lax.top_k(grid, cfg.cand_k)[1]