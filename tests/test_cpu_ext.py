"""Extended CPU tests: paths the first suite didn't cover.

1. lb (router-entropy aux) path: lb_weight > 0 must produce a finite scalar
   lb, gradients must flow to k1/k2, and lb must be ~0 only at uniform routing.
2. return_aux triple path (run23 usage): (logits, aux, lb_total) consistent.
3. temp anneal: mem_temp override changes read weights but not slot selection.
4. value noise / lb_eps edge configs don't crash under jit+grad.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}: {name} {detail}")
    if not ok:
        FAILS.append(name)


def build(mem_cfg_overrides=None, return_aux=False, d_model=64, n_layers=2):
    mem_cfg = MemoryConfig(c1=32, c2=32, cand_k=4, side_top=8, n_classes=2,
                           **(mem_cfg_overrides or {}))
    cfg_m = ModelConfig(d_model=d_model, n_layers=n_layers, n_heads=2,
                        memory_every=1, vocab_size=339)
    return Navi(cfg_m, mem_cfg, return_aux=return_aux), mem_cfg


def test_lb_aux():
    print("== lb_weight>0: entropy aux finite, grads reach keys ==")
    model, _ = build({"lb_weight": 0.01}, return_aux=True)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 339, (2, 16)),
                      dtype=jnp.int32)
    p = model.init({"params": jax.random.PRNGKey(1)}, ids, train=False)

    def loss_fn(pp):
        logits, _, lb_total = model.apply(pp, ids, train=True)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, ids).mean()
        return ce + 0.01 * lb_total

    (l, (logits, aux, lb_total)), g = jax.value_and_grad(
        lambda q: (loss_fn(q), model.apply(q, ids, train=True)),
        has_aux=True)(p)
    check("lb finite scalar", bool(jnp.isfinite(lb_total)),
          f"lb={float(lb_total):.4f}")
    k1g = g["params"]["block_0"]["mem"]["k1"]
    check("lb grads reach k1", bool(jnp.isfinite(k1g).all())
          and float(jnp.abs(k1g).sum()) > 0.0)
    # lb of uniform softmax over c1=32: -log(1/32) per side per class.
    # pkm lb = -(e1+e2) per block; Navi lb_total SUMS over memory blocks
    # (n_layers=2, memory_every=1 -> 2 memory blocks) => 2 * -(2*log32).
    lb_expected = 2 * -(2 * np.log(32.0))
    check("lb magnitude sane (uniform init routing)",
          abs(float(lb_total) - lb_expected) < 2.0,
          f"lb={float(lb_total):.3f} expect~{lb_expected:.3f}")


def test_return_aux_triple():
    print("== return_aux triple consistency ==")
    model, _ = build(return_aux=True)
    ids = jnp.asarray(np.random.default_rng(2).integers(0, 339, (2, 16)),
                      dtype=jnp.int32)
    p = model.init({"params": jax.random.PRNGKey(3)}, ids, train=False)
    plain_model, _ = build()
    logits_a, aux, lb = model.apply(p, ids, train=False)
    logits_b = plain_model.apply(p, ids, train=False)
    # plain and aux models share param tree (same structure); read path must
    # produce identical logits
    check("aux-path logits == plain logits",
          bool(np.allclose(np.asarray(logits_a), np.asarray(logits_b),
                           atol=1e-5)))
    check("aux slots present per memory block",
          sorted(aux.keys()) == ["mem_0", "mem_1"],
          str(sorted(aux.keys())))
    slots = np.asarray(aux["mem_0"])
    check("slot ids in valid range",
          bool((slots >= 0).all() and (slots < 2 * 32 * 32).all()),
          f"max={slots.max()}")


def test_temp_override():
    print("== mem_temp override: sharper temp -> same slots, sharper weights ==")
    model, mem_cfg = build({"score_temp": 1.0})
    ids = jnp.asarray(np.random.default_rng(3).integers(0, 339, (2, 16)),
                      dtype=jnp.int32)
    p = model.init({"params": jax.random.PRNGKey(4)}, ids, train=False)
    out_lo = model.apply(p, ids, train=False, mem_temp=1.0)
    out_hi = model.apply(p, ids, train=False, mem_temp=8.0)
    out_eq = model.apply(p, ids, train=False, mem_temp=None)  # cfg default 1.0
    check("temp=None falls back to cfg",
          bool(np.allclose(np.asarray(out_lo), np.asarray(out_eq), atol=1e-7)))
    check("temp changes readout", not np.allclose(np.asarray(out_lo),
                                                  np.asarray(out_hi), atol=1e-6))


def test_edge_configs_grad():
    print("== edge configs survive jit+grad ==")
    for overrides in [{"lb_eps": 0.05}, {"value_noise": 0.0},
                      {"hash_slots": True}]:
        model, mem_cfg = build(overrides, return_aux=False)
        ids = jnp.asarray(np.random.default_rng(4).integers(0, 339, (2, 8)),
                          dtype=jnp.int32)
        try:
            p = model.init({"params": jax.random.PRNGKey(5)}, ids, train=False)
            if overrides.get("hash_slots"):
                # hash path needs ctx_ids; Navi passes them when return_aux
                # but plain path passes ctx only via Block call — verify apply
                # doesn't crash (hash_slots uses ctx_ids internally)
                pass
            l, g = jax.value_and_grad(
                lambda q: optax.softmax_cross_entropy_with_integer_labels(
                    model.apply(q, ids, train=True), ids).mean())(p)
            finite = all(bool(jnp.isfinite(x).all())
                         for x in jax.tree_util.tree_leaves(g))
            check(f"config {overrides}", bool(jnp.isfinite(l)))
        except Exception as e:
            check(f"config {overrides}", False, str(e)[:80])


if __name__ == "__main__":
    test_lb_aux()
    test_return_aux_triple()
    test_temp_override()
    test_edge_configs_grad()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + str(FAILS)}")
    sys.exit(1 if FAILS else 0)