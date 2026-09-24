"""Pallas TPU kernel for the Product-Key Memory read path.

Fuses the PKM candidate chain into one VMEM-resident kernel:
  q·k1ᵀ / q·k2ᵀ (MXU dot) -> two-sided top-side_top filter ->
  (side x g2k) additive sub-grid -> EXACT top-cand_k via bitonic merge
  network (cheap Mosaic ops only: compare/select) -> value gather ->
  softmax-weighted sum -> (h, slots, scores).

Exactness contract (verified against jax.lax.top_k in tests):
  - selected VALUES are identical to the reference path,
  - index tie-break is deterministic (higher score, then lower index);
    lax.top_k's tie order is implementation-defined, so tests assert
    value equality + descending order, not index equality on ties.

Fallback: on non-TPU backends (CPU/CI) or when NAVI_PALLAS=0, callers use
the pure-JAX reference path in navi/pkm.py. This module exposes the TPU
kernel and a dispatcher.

Design notes (docs.jax.dev/en/latest/pallas/tpu/details):
  - compare/select/max are cheap (vector unit); we avoid /, %, exp in the
    kernel body where possible (softmax stays outside, on the fp32 scores).
  - VMEM blocks: last dims multiple of 8/128 where required.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

try:
    import jax.experimental.pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    _HAS_PALLAS_TPU = True
except ImportError:  # CPU-only dev box
    _HAS_PALLAS_TPU = False

from navi.config import MemoryConfig


def pallas_enabled() -> bool:
    """Kernel dispatch gate: TPU backend + env opt-in."""
    return (
        _HAS_PALLAS_TPU
        and jax.default_backend() == "tpu"
        and os.environ.get("NAVI_PALLAS", "1") == "1"
    )


# ---------------------------------------------------------------------------
# Host-side reference implementations (CPU-testable, also used as fallback)
# ---------------------------------------------------------------------------

def lex_topk(scores: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """Exact top-k, values descending, tie -> lower index. N power of 2.

    Built only from compare/select — the same op budget Mosaic allows inside
    kernels. Used both as the CPU reference and, in tiled form, inside the
    Pallas kernel body.
    """
    n = scores.shape[-1]
    ar = jnp.arange(n, dtype=jnp.int32)
    s = scores
    i = jnp.broadcast_to(ar.astype(jnp.float32), s.shape)

    def lex_before(s1, i1, s2, i2):
        return (s1 > s2) | ((s1 == s2) & (i1 < i2))

    for p in range(int(np.log2(n))):
        kk = 1 << (p + 1)
        dir_up = (ar & kk) == 0
        for q in range(p, -1, -1):
            half = 1 << q
            jx = ar ^ half
            lo = (ar & half) == 0
            sj, ij = s[..., jx], i[..., jx]
            pb = lex_before(sj, ij, s, i)
            cond = jnp.where(dir_up, jnp.where(lo, pb, ~pb),
                                    jnp.where(lo, ~pb, pb))
            s = jnp.where(cond, sj, s)
            i = jnp.where(cond, ij, i)
    return s[..., :k], i[..., :k].astype(jnp.int32)


# ---------------------------------------------------------------------------
# Pallas TPU kernel: fused sub-grid top-k + value gather + weighted sum.
#
# Grid over (batch*seq) rows. Per program instance:
#   - in:  g1 (side), g2v (g2k), i1 (side), i2_top (g2k), values plane (c1*c2, D)
#   - out: h (D), slots (cand_k), scores (cand_k)
# The (side x g2k) sub-grid (<= 256x8) is built in registers, top-cand_k via
# the bitonic network above (Mosaic-cheap ops), then the cand_k value rows are
# gathered and softmax-weighted OUTSIDE (scores returned in fp32; the small
# softmax runs on the vector unit fine, but keeping it outside preserves
# exact parity with the reference path for the same score values).
# ---------------------------------------------------------------------------

def _fused_read_kernel(g1_ref, g2v_ref,
                       slots_ref, scores_ref,
                       side: int, g2k: int, cand_k: int,
                       c2: int):
    """One program = PB=8 positions of one class plane (batched rows).

    Computes EXACT top-cand_k over the (side, g2k) additive grid via an
    unrolled bitonic merge network, VMEM-resident. Value gather + softmax +
    weighted sum happen OUTSIDE the kernel (see pkm_read_fused): Mosaic's
    ref-gather lowering can't handle gather over a large (slots, D) table,
    but XLA lowers that fine as a plain lax gather on small (P, k) indices.
    """
    g1 = g1_ref[:].astype(jnp.float32)              # (PB, side)
    g2v = g2v_ref[:].astype(jnp.float32)            # (PB, g2k)
    sub = g1[:, :, None] + g2v[:, None, :]          # (PB, side, g2k)
    flat = sub.reshape(g1.shape[0], side * g2k)     # (PB, side*g2k)
    # pad to next pow2 >= side*g2k with -inf scores, index sentinel.
    # NOTE: pad is a compile-time constant; jnp.full((0,)) is illegal in
    # Mosaic, so the pow2 padding always materializes >=1 element via max(2,..)
    npad = 1 << int(np.ceil(np.log2(max(2, side * g2k))))
    pad = npad - side * g2k
    flat = jnp.pad(flat, [(0, 0), (0, pad)], constant_values=-jnp.inf)
    idx0 = jnp.broadcast_to(
        jnp.arange(side * g2k, dtype=jnp.int32).astype(jnp.float32),
        (g1.shape[0], side * g2k))
    ip = jnp.pad(idx0, [(0, 0), (0, pad)],
                 constant_values=float(jnp.iinfo(jnp.int32).max))
    k = min(cand_k, side * g2k)

    # Iterative argmax extraction instead of a bitonic sort network.
    # Why: Mosaic lowers take_along_axis over a 512-wide row to
    # tpu.dynamic_gather across MULTIPLE source vregs -> "Not implemented:
    # Multiple source vregs along gather dimension". Full-row permutations
    # are unimplementable; reductions (max over axis) and static lane
    # selects (s[:, :k]) lower fine. Each round: argmax over the row,
    # add -inf at the winner's lane via a one-hot mask (broadcast compare,
    # no gather), repeat k times. Cost: k * (2 reductions + selects) vs
    # log^2(npad) permutations; for k=8 << npad=512 this is far cheaper
    # AND compileable. Tie-break: lower flat index wins (jnp.max on the
    # index where scores equal via the (s == max) & (i < imax) trick).
    ar = jnp.arange(npad, dtype=jnp.int32)[None, :]  # (1, npad)
    s, i = flat, ip
    outs_s, outs_i = [], []
    for _ in range(k):
        smax = jnp.max(s, axis=-1, keepdims=True)            # (PB,1)
        cand = (s == smax)
        # tie-break to lowest index: among max lanes keep smallest i
        imin_mask = jnp.min(jnp.where(cand, i, jnp.float32(1e9)),
                            axis=-1, keepdims=True)
        pick = cand & (i == imin_mask)
        pk = jnp.sum(pick.astype(jnp.float32) * i, axis=-1)  # (PB,)
        sk = jnp.sum(pick.astype(jnp.float32) * s, axis=-1)  # (PB,)
        outs_s.append(sk)
        outs_i.append(pk)
        # kill the winner lane. Two Mosaic quirks force this shape:
        # (1) jnp.where(pick, -inf, s) mislowers (inf handling) -> use a
        #     finite sentinel far below any real score;
        # (2) the boolean pick mask itself can misbroadcast -> kill by
        #     extracted index equality, a value compare on i.
        s = jnp.where(i == pk[:, None], -1e30, s)
    top_scores = jnp.stack(outs_s, axis=-1)                  # (PB, k)
    top_f = jnp.stack(outs_i, axis=-1).astype(jnp.int32)     # (PB, k)
    # f_idx encoding identical to reference: r * g2k + col. The side indices
    # (i1/i2t) stay in host memory; decode flat index here, host does the
    # i1/i2t lookup and slot composition in JAX (cheap (P,k) gathers).
    r, col = top_f // g2k, top_f % g2k
    slots = r * g2k + col                                 # (PB, k) sub-grid ids
    slots_ref[...] = slots
    scores_ref[...] = top_scores


def pkm_read_fused(g1, g2v, i1, i2t, values_plane, c2: int, cand_k: int):
    """Host wrapper. Shapes:
      g1: (P, side) fp32        — side scores per position (P = b*l*C)
      g2v: (P, g2k) fp32
      i1: (P, side) int32       — side row indices into the c1 axis
      i2t: (P, g2k) int32       — g2k col indices into the c2 axis
      values_plane: (c1*c2, D)  — bf16 value table for ONE class
    Returns h (P, D) fp32, slots (P, cand_k) int32, scores (P, cand_k) fp32.

    NOTE: current production path is the batched-JAX read in navi/pkm.py
    (single fused lax.top_k over classes). This kernel is the TPU
    specialization: same contract, one VMEM-resident pass, no HBM transient
    for the (side, g2k) grid.
    """
    P, side = g1.shape
    _, g2k = g2v.shape
    dim = values_plane.shape[-1]
    k = min(cand_k, side * g2k)
    # Mosaic requires P-axis blocks of >=8 (divisible by 8) unless the block
    # equals the full array dim. Chunk positions by PB=8; pad the last chunk.
    PB = 8
    Ppad = ((P + PB - 1) // PB) * PB
    pad_n = Ppad - P
    if pad_n:
        g1 = jnp.pad(g1, ((0, pad_n), (0, 0)))
        g2v = jnp.pad(g2v, ((0, pad_n), (0, 0)))
        i1 = jnp.pad(i1, ((0, pad_n), (0, 0)))
        i2t = jnp.pad(i2t, ((0, pad_n), (0, 0)))

    def kernel(g1_ref, g2v_ref, i1_ref, i2t_ref, slots_ref, scores_ref):
        _fused_read_kernel(g1_ref, g2v_ref,
                           slots_ref, scores_ref,
                           side, g2k, cand_k, c2)

    # NOTE: on TPU the Mosaic backend compiles this quickly; on CPU the
    # unrolled bitonic network takes minutes to compile through XLA — use
    # interpret=True only for single-program smoke tests, and the host mirror
    # (lex_topk + same gather/softmax chain, verified 0-diff vs lax.top_k in
    # tests/test_cpu_equiv.py) for CPU CI.
    result = pl.pallas_call(
        kernel,
        grid=(Ppad // PB,),
        in_specs=[
            pl.BlockSpec((PB, side), lambda p: (p, 0)),
            pl.BlockSpec((PB, g2k), lambda p: (p, 0)),
            pl.BlockSpec((PB, side), lambda p: (p, 0)),
            pl.BlockSpec((PB, g2k), lambda p: (p, 0)),
        ],
        out_specs=[
            pl.BlockSpec((PB, k), lambda p: (p, 0)),
            pl.BlockSpec((PB, k), lambda p: (p, 0)),
        ],
        out_shape=(
            jax.ShapeDtypeStruct((Ppad, k), jnp.int32),
            jax.ShapeDtypeStruct((Ppad, k), jnp.float32),
        ),
    )(g1, g2v, i1, i2t)
    f_idx_pad, scores_pad = result
    f_idx = f_idx_pad[:P]                             # (P, k) flat sub-grid ids
    scores_out = scores_pad[:P]
    # Host-side decode: flat (row, col) -> i1/i2t lookup -> plane slot ids.
    r = f_idx // g2k
    col = f_idx % g2k
    rows = jnp.arange(P, dtype=jnp.int32)[:, None]    # (P, 1)
    pi1 = i1[rows, r]                                 # (P, k) row-wise gather
    pi2 = i2t[rows, col]                              # (P, k)
    slots_out = pi1 * c2 + pi2                        # (P, k)
    # Value gather + softmax + weighted sum in plain JAX (XLA fuses the
    # (P, k)->(P, D) gather fine; only the top-k chain needed the kernel).
    v = values_plane[slots_out].astype(jnp.float32)   # (P, k, D)
    w = jax.nn.softmax(scores_out, axis=-1)[..., None]  # (P, k, 1)
    h_out = (w * v).sum(axis=1)                       # (P, D)
    return h_out, slots_out, scores_out