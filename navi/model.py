"""The Reasoner: dense transformer backbone with Pool layers interleaved.

Block layout follows Meta's memory-layer recipe: standard transformer
blocks where every other FFN is replaced by a ProductKeyMemory read, added
into the residual stream. At GPU scale those value gathers are the only
tensors that ever need to leave the resident device -> exactly the seam
where the RAM/disk tier plugs in.
"""

import os

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Array

from navi.config import MemoryConfig, ModelConfig
from navi.pkm import ProductKeyMemory


def _identity(cls):
    return cls


class Block(nn.Module):
    cfg: ModelConfig
    use_memory: bool
    mem_cfg: MemoryConfig
    return_aux: bool = False  # True -> (x, slot_ids) from memory blocks

    def setup(self) -> None:
        self.ln1 = nn.LayerNorm()
        self.use_fused_attn = os.environ.get("NAVI_FLASH", "1") == "1"
        if self.use_fused_attn:
            # Fused path: single DenseGeneral produces head-split qkv in
            # one GEMM; attention runs through jax.nn.dot_product_attention,
            # which XLA lowers to the fused TPU FlashAttention kernel on
            # v5e (no materialized (l,l) softmax, no remat of it).
            # Param tree is identical to nn.SelfAttention(num_heads,
            # use_bias=False): qkv_features = 3*d_model in one kernel
            # named 'qkv', out_features = d_model named 'out'. Old ckpts
            # with 'attn' subtrees do NOT map -- NAVI_FLASH=0 restores
            # the legacy path (set in env at resume time).
            self.qkv = nn.DenseGeneral(
                features=(3, self.cfg.n_heads, self.cfg.d_model
                          // self.cfg.n_heads),
                axis=-1, name="qkv")
            self.proj = nn.Dense(self.cfg.d_model, use_bias=False, name="proj")
        else:
            self.attn = nn.SelfAttention(
                num_heads=self.cfg.n_heads,
                use_bias=False,
                deterministic=True,
                name="attn",
            )
        self.ln2 = nn.LayerNorm()
        if self.use_memory:
            self.mem = ProductKeyMemory(
                cfg=self.mem_cfg,
                per_class_dim=self.cfg.d_model // self.mem_cfg.n_classes,
                value_dim=self.cfg.d_model,
                name="mem",
            )
        else:
            self.ff = nn.Dense(4 * self.cfg.d_model, name="ff_in")
            self.ff_out = nn.Dense(self.cfg.d_model, name="ff_out")

    def __call__(self, x: Array, train: bool = False, ctx_ids: Array | None = None,
                 mem_temp=None, dense: bool = False,
                 visit_off=None, eps_t=None,
                 usage_state=None) -> Array | tuple[Array, Array]:
        # resolve None on the host side: PKM must never see a None it
        # branches on under trace (traced arrays break `is None` checks)
        if mem_temp is None:
            mem_temp = self.mem_cfg.score_temp
        if self.use_fused_attn:
            qkv = self.qkv(self.ln1(x))            # (b, l, 3, H, dh)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            a = jax.nn.dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(a.reshape(x.shape[0], x.shape[1], -1))
        else:
            mask = nn.attention.make_causal_mask(
                jnp.ones((x.shape[1],), dtype=jnp.bool_))
            x = x + self.attn(self.ln1(x), mask=mask)
        h = self.ln2(x)
        if self.use_memory:
            out, m = self.mem(h, train, ctx_ids=ctx_ids, temp=mem_temp,
                              dense=dense, visit_off=visit_off, eps_t=eps_t,
                              usage_state=usage_state)
            out = out.reshape(x.shape)
            if self.return_aux:
                slots = m["slots"].reshape(x.shape[0], x.shape[1], -1)
                return x + out, (slots, m["lb"], m.get("usage"))
            return x + out
        return x + self.ff_out(nn.relu(self.ff(h)))


class Navi(nn.Module):
    cfg: ModelConfig
    mem_cfg: MemoryConfig
    return_aux: bool = False  # True -> (logits, per-memory-layer slot ids)

    def setup(self) -> None:
        self.embed = nn.Embed(self.cfg.vocab_size, self.cfg.d_model, name="embed")
        # remat per block: 16 materialized causal-attention softmaxes at
        # BS=256/SEQ=512 cost ~4.3GB HBM for backward; recompute trades
        # ~30% step time for fitting. NAVI_REMAT=0 skips it for small
        # models whose activations fit trivially.
        BlockCls = Block if os.environ.get("NAVI_REMAT", "1") == "1" else _identity(Block)
        self.blocks = [
            BlockCls(
                cfg=self.cfg,
                use_memory=(
                    self.cfg.memory_every > 0
                    and i % self.cfg.memory_every == self.cfg.memory_from_layer
                ),
                mem_cfg=self.mem_cfg,
                return_aux=self.return_aux,
                name=f"block_{i}",
            )
            for i in range(self.cfg.n_layers)
        ]
        self.ln_f = nn.LayerNorm()
        self.head = nn.Dense(self.cfg.vocab_size, use_bias=False, name="head")

    def __call__(self, ids: Array, train: bool = False, mem_temp=None,
                 dense: bool = False,
                 visit_off=None, eps_t=None,
                 usage_states=None) -> Array | tuple[Array, dict[str, Array]]:
        if mem_temp is None:
            mem_temp = self.mem_cfg.score_temp
        x = self.embed(ids) * jnp.sqrt(float(self.cfg.d_model))
        x = x + _sin_pe(ids.shape[1], self.cfg.d_model)
        aux: dict[str, Array] = {}
        lb_total = jnp.float32(0.0)
        mi = 0
        for i, block in enumerate(self.blocks):
            if self.return_aux and block.use_memory:
                us = (usage_states[mi] if usage_states is not None else None)
                x, (slots, lb, usage_new) = block(
                    x, train, ctx_ids=ids, mem_temp=mem_temp,
                    dense=dense, visit_off=visit_off, eps_t=eps_t,
                    usage_state=us)
                aux[f"mem_{i}"] = slots
                aux[f"usage_{i}"] = usage_new
                lb_total = lb_total + lb
                mi += 1
            else:
                x = block(x, train, ctx_ids=ids, mem_temp=mem_temp,
                          dense=dense, visit_off=visit_off, eps_t=eps_t)
        logits = self.head(self.ln_f(x))
        return (logits, aux, lb_total) if self.return_aux else logits


def _sin_pe(t: int, d: int) -> Array:
    # sinusoidal positional encoding; keeps v0 parameter-minimal and portable
    pos = jnp.arange(t)[:, None]
    div = jnp.exp(jnp.arange(0, d, 2) * (-jnp.log(10000.0) / d))
    pe = jnp.zeros((t, d)).at[:, 0::2].set(jnp.sin(pos * div)).at[:, 1::2].set(jnp.cos(pos * div))
    return pe[None, :, :]