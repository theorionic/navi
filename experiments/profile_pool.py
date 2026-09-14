"""TPU step-time decomposition for the 500M Pool config: ablations + HLO census.

Same model/sharding as train500m.py (mesh over 8 cores: batch sharded,
values slot-sharded, everything else replicated). Reports:
  1. full fwd+bwd step time at BS 256/128/64 (scaling exponent)
  2. top HLO ops of the compiled step (census)
  3. ablations at BS 256, each a fresh compile of the same backbone:
       B no Pool        (memory_every=0 -> dense FFN everywhere)
       C no lb aux      (lb_weight=0)
       D values frozen  (stop_gradient on values -> scatter-add grad gone)
  4. standalone PKM layer fwd+bwd micro-time at BS 32 (token-linear op).

Writes /kaggle/working/profile.json. Repo untouched.
"""
import json
import re
import sys
import time

sys.path.insert(0, "/kaggle/working/code")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi
from navi.train import init_params

BS, SEQ = 256, 512
CAND_K, SIDE_TOP, LB = 16, 64, 0.01
OUT = "/kaggle/working/profile.json"
RES = {}

mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))


def shard(tree):
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)


def mk(lb=LB, mem=True):
    mc = MemoryConfig(c1=512, c2=512, cand_k=CAND_K, side_top=SIDE_TOP,
                      n_classes=4, score_temp=4.0, lb_weight=lb)
    mcfg = ModelConfig(d_model=512, n_layers=8, n_heads=8,
                       memory_every=2 if mem else 0, vocab_size=260)
    return Navi(mcfg, mc, return_aux=lb > 0)


def timeit(fn, iters=12, warmup=5):
    for _ in range(warmup):
        fn()
    jax.block_until_ready(fn())
    t0 = time.time()
    for _ in range(iters):
        fn()
    jax.block_until_ready(fn())
    return (time.time() - t0) / iters


def step_fn(lb, freeze_values=False):
    model = mk(lb=lb)

    def loss(p, ids, tg):
        if freeze_values:
            p = jax.tree_util.tree_map_with_path(
                lambda kp, x: jax.lax.stop_gradient(x) if "values" in jax.tree_util.keystr(kp) else x,
                p)
        out = model.apply(p, ids, True, 4.0)
        if lb > 0:
            logits, _aux, lbt = out
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, tg).mean() + lb * lbt
        return optax.softmax_cross_entropy_with_integer_labels(out, tg).mean()

    @jax.jit
    def st(p, ids, tg):
        return jax.grad(loss)(p, ids, tg)

    return model, st


rng = np.random.default_rng(0)

# ---------- 1. full step at three batch sizes ----------
model, st = step_fn(LB)
p0 = shard(init_params(model, SEQ, jax.random.PRNGKey(0)))
for bs in (256, 128, 64):
    i_ = jax.device_put(np.asarray(rng.integers(0, 260, (bs, SEQ)), np.int32), BATCH)
    t_ = jax.device_put(np.asarray(rng.integers(0, 260, (bs, SEQ)), np.int32), BATCH)
    t = timeit(lambda: st(p0, i_, t_))
    RES[f"full_bs{bs}"] = t
    print(f"[prof] full fwd+bwd bs={bs}: {t*1e3:.1f} ms", flush=True)

# ---------- 2. HLO census ----------
try:
    exe = st.lower(p0, jax.device_put(np.zeros((BS, SEQ), np.int32), BATCH),
                   jax.device_put(np.zeros((BS, SEQ), np.int32), BATCH)).compile()
    hlo = exe.as_text()
    open("/kaggle/working/step.hlo", "w").write(hlo)
    census = {}
    for m in re.finditer(r"custom_call[^=]*target=([a-zA-Z0-9_.:-]+)", hlo):
        census["cc:" + m.group(1)] = census.get("cc:" + m.group(1), 0) + 1
    for m in re.finditer(r"=\s+([a-z0-9-]+)\s*\(", hlo):
        census[m.group(1)] = census.get(m.group(1), 0) + 1
    RES["hlo_census_top"] = sorted(census.items(), key=lambda kv: -kv[1])[:30]
    print("[prof] hlo top:", RES["hlo_census_top"][:15], flush=True)
except Exception as e:
    RES["hlo_error"] = repr(e)
    print("[prof] hlo census failed:", repr(e), flush=True)

# ---------- 3. ablations ----------
# B: no Pool
model_b, st_b = step_fn.__wrapped__ if False else (None, None)  # placeholder
mc0 = MemoryConfig(c1=512, c2=512, cand_k=CAND_K, side_top=SIDE_TOP, n_classes=4,
                   score_temp=4.0, lb_weight=0.0)
mcfg0 = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=0, vocab_size=260)
model_b = Navi(mcfg0, mc0)


@jax.jit
def st_b(p, ids, tg):
    def l(q, a, b):
        out = model_b.apply(q, a, True, 4.0)
        return optax.softmax_cross_entropy_with_integer_labels(out, b).mean()
    return jax.grad(l)(p, ids, tg)


p_b = shard(init_params(model_b, SEQ, jax.random.PRNGKey(0)))
RES["abl_no_pool"] = timeit(lambda: st_b(p_b, jax.device_put(np.zeros((BS, SEQ), np.int32), BATCH),
                                            jax.device_put(np.zeros((BS, SEQ), np.int32), BATCH)))
print(f"[prof] abl B no-pool: {RES['abl_no_pool']*1e3:.1f} ms", flush=True)

# C: Pool, no lb
model_c, st_c = step_fn(lb=0.0)
p_c = shard(init_params(model_c, SEQ, jax.random.PRNGKey(0)))
RES["abl_no_lb"] = timeit(lambda: st_c(p_c, jax.device_put(np.zeros((BS, SEQ), np.int32), BATCH),
                                       jax.device_put(np.zeros((BS, SEQ), np.int32), BATCH)))
print(f"[prof] abl C no-lb: {RES['abl_no_lb']*1e3:.1f} ms", flush=True)

# D: values frozen (stop_gradient) -> scatter-add grad eliminated
model_d, st_d = step_fn(LB, freeze_values=True)
RES["abl_values_frozen"] = timeit(lambda: st_d(p0, jax.device_put(np.zeros((BS, SEQ), np.int32), BATCH),
                                               jax.device_put(np.zeros((BS, SEQ), np.int32), BATCH)))
print(f"[prof] abl D values-frozen: {RES['abl_values_frozen']*1e3:.1f} ms", flush=True)

# ---------- 4. standalone PKM micro (bs=32 to fit one core) ----------
from navi.pkm import ProductKeyMemory  # noqa: E402

mc = MemoryConfig(c1=512, c2=512, cand_k=CAND_K, side_top=SIDE_TOP, n_classes=4,
                  score_temp=4.0, lb_weight=0.0)
pkm = ProductKeyMemory(mc, per_class_dim=128, value_dim=128)
x32 = jnp.asarray(rng.normal(size=(32, SEQ, 512)), dtype=jnp.float32)
pk = pkm.init(jax.random.PRNGKey(1), x32)


@jax.jit
def pkm_step(q, x):
    def l(qq):
        return pkm.apply(qq, x)[0].sum()
    return jax.grad(l)(q, x)


t = timeit(lambda: pkm_step(pk, x32))
RES["pkm_layer_fwd_bwd_bs32"] = t
print(f"[prof] pkm layer fwd+bwd bs32: {t*1e3:.1f} ms (x4 layers)", flush=True)

json.dump(RES, open(OUT, "w"), indent=1, default=float)
print("[prof] SUMMARY " + json.dumps(RES, default=float), flush=True)