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
        # the Pool: values[class, slot, dim] -- the offload tier at scale
        self.values = self.param(
            "values",
            jax.nn.initializers.normal(0.02),
            (c.n_classes, c.c1 * c.c2, self.per_class_dim),
        )

    def __call__(self, x, train=False):
        c = self.cfg
        b, l, _ = x.shape
        q = self.w_q(x).reshape(b, l, c.n_classes, 2, self.per_class_dim)
        q1, q2 = q[..., 0, :], q[..., 1, :]
        s1 = jnp.einsum("blcd,ckd->blck", q1, self.k1)
        s2 = jnp.einsum("blcd,ckd->blck", q2, self.k2)
        # two-sided filter; with side_top=16 the side^2 grid is ~134MB at
        # b=512 - fully vectorized, no chunk loop, no scan.
        i1 = jnp.top_k(s1, c.side_top, axis=-1)[1]
        i2 = jnp.top_k(s2, c.side_top, axis=-1)[1]
        g1 = jnp.take_along_axis(s1, i1, axis=-1)
        g2 = jnp.take_along_axis(s2, i2, axis=-1)
        sub = g1[..., :, None] + g2[..., None, :]  # (b, l, classes, side, side)
        side = sub.shape[-2]
        flat = sub.reshape(b, l, c.n_classes, side * side)
        f_idx = jnp.top_k(flat, c.cand_k, axis=-1)[1]
        r, col = f_idx // side, f_idx % side
        pi1 = jnp.take_along_axis(i1, r, axis=-1)
        pi2 = jnp.take_along_axis(i2, col, axis=-1)
        slots = pi1 * c.c2 + pi2
        scores = jnp.take_along_axis(flat, f_idx, -1)
        v = self.values[jnp.arange(c.n_classes)[None, None, :, None], slots]
        w = jax.nn.softmax(c.score_temp * scores, axis=-1)
        h = (w[..., None] * v).sum(axis=-2)
        h = h.reshape(b, l, c.n_classes * self.per_class_dim)
        return self.w_o(h), {"slots": slots.reshape(b, l, -1), "scores": scores.reshape(b, l, -1)}

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
    return jnp.top_k(grid, cfg.cand_k)[1]