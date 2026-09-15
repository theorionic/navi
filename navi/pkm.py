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

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Array

from navi.config import MemoryConfig


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
        self.w_q = nn.Dense(c.n_classes * 2 * self.per_class_dim, use_bias=False, name="w_q")
        self.w_o = nn.Dense(self.value_dim, use_bias=False, name="w_o")
        self.k1 = self.param(
            "k1", jax.nn.initializers.normal(0.02), (c.n_classes, c.c1, self.per_class_dim)
        )
        self.k2 = self.param(
            "k2", jax.nn.initializers.normal(0.02), (c.n_classes, c.c2, self.per_class_dim)
        )
        # the Pool: values[class, slot, dim] -- the offload tier at scale.
        # bf16 storage: half the HBM for params + Lion state + grads (the
        # 8-block run23 config needs it: fp32 x3 = 12.9GB > 9.5GB/core).
        # Values are a lookup table accumulated over tens of thousands of
        # touches - bf16 (8 mantissa bits) is quantization-insensitive here.
        # Cast to fp32 after the gather; softmax/weighting stay fp32.
        self.values = self.param(
            "values",
            jax.nn.initializers.normal(0.02, dtype=jnp.bfloat16),
            (c.n_classes, c.c1 * c.c2, self.per_class_dim),
        )

    def __call__(self, x, train=False, ctx_ids=None, temp=None):
        """ctx_ids: (b, l) token IDs of this block's input context. Required
        when cfg.hash_slots -- slot candidates are hash(ctx_ids) mod slots,
        so each (key,n1,n2) context maps to fixed value rows on first touch
        and the write gradient lands in exactly those rows. No keys involved.
        """
        c = self.cfg
        b, l, _ = x.shape
        if c.hash_slots:
            assert ctx_ids is not None, "hash_slots requires ctx_ids"
            return self._hash_path(x, ctx_ids)
        q = self.w_q(x).reshape(b, l, c.n_classes, 2, self.per_class_dim)
        q1, q2 = q[..., 0, :], q[..., 1, :]
        s1 = jnp.einsum("blcd,ckd->blck", q1, self.k1)
        s2 = jnp.einsum("blcd,ckd->blck", q2, self.k2)
        # two-sided filter; side_top candidates per side, then the product
        # grid. With side_top=64 and c1=c2=512 the grid covers only 1.56%
        # of slots - measured routing collapse sits exactly at that ceiling
        # (run22f: 0.9-3.5% alive). Side_top=128 raises reach to 6.2%, but
        # the (side,side) grid must be built per class (lax.map over the
        # class axis) to keep the transient at today's 8.6GB peak: classes
        # are independent planes, so chunking is EXACT, not approximate.
        i1 = jax.lax.top_k(s1, c.side_top)[1]  # (b, l, classes, side)
        i2 = jax.lax.top_k(s2, c.side_top)[1]
        g1 = jnp.take_along_axis(s1, i1, axis=-1)
        g2 = jnp.take_along_axis(s2, i2, axis=-1)

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

        def class_grid(gs):
            g1c, g2c = gs  # each (b, l, side) / (b, l, g2k) - class stripped
            # build (side, g2k) grid, top over side*g2k pairs
            sub = g1c[..., :, None] + g2c[..., None, :]  # (b,l,side,g2k)
            side = sub.shape[-2]
            flat = sub.reshape(b, l, side * g2k)
            k = min(c.cand_k, side * g2k)
            f_idx = jax.lax.top_k(flat, k)[1]
            scores = jnp.take_along_axis(flat, f_idx, -1)
            return f_idx, scores

        # map over the class axis (classes are independent planes: exact)
        g1c = jnp.moveaxis(g1, 2, 0)
        g2c = jnp.moveaxis(g2_vals, 2, 0)
        f_idx, scores = jax.lax.map(class_grid, (g1c, g2c))
        # (classes, b, l, k) -> (b, l, classes, k)
        f_idx = jnp.moveaxis(f_idx, 0, 2)
        scores = jnp.moveaxis(scores, 0, 2)
        side = c.side_top
        r, col = f_idx // g2k, f_idx % g2k
        pi1 = jnp.take_along_axis(i1, r, axis=-1)
        pi2 = jnp.take_along_axis(i2, g2_top, axis=-1)
        pi2 = jnp.take_along_axis(pi2, col, axis=-1)
        slots = pi1 * c.c2 + pi2
        v = self.values[jnp.arange(c.n_classes)[None, None, :, None], slots]
        v = v.astype(jnp.float32)  # bf16 storage; math in fp32
        t = c.score_temp if temp is None else temp
        w = jax.nn.softmax(t * scores, axis=-1)
        if c.lb_eps > 0.0:
            w = w * (1.0 - c.lb_eps) + c.lb_eps / w.shape[-1]
        h = (w[..., None] * v).sum(axis=-2)
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
        return self.w_o(h), {"slots": slots.reshape(b, l, -1),
                             "scores": scores.reshape(b, l, -1), "lb": lb}

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