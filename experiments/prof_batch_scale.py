import sys, os, time; sys.path.insert(0, '/kaggle/working')
os.environ['JAX_PLATFORMS'] = ''
# Measure WHERE the 29s/it goes: profile one step with host-side timers around
# (a) fwd gather, (b) full step, (c) scatter in backward. Compare batch 512 vs 64:
# if time scales with batch -> gather-bound; if constant -> scatter-bound (values
# width) -> sharding is the fix.
import jax, jax.numpy as jnp, optax
import navi.data as D
from navi.config import MemoryConfig, ModelConfig, TrainConfig
from navi.model import Navi
from navi.train import init_params
import sweep

for BS in (64, 512):
    cfg = ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2, vocab_size=D.VOCAB)
    model = Navi(cfg, sweep.CFG_MEM)
    cfg_t = TrainConfig(total_steps=3, batch_size=BS, seq_len=64, warmup_steps=1, log_every=1, mem_lr_mult=10.0)
    params = init_params(model, 64, jax.random.fold_in(jax.random.PRNGKey(0), 1))
    tx = sweep.make_tx(cfg_t, params)
    o = tx.init(params)

    @jax.jit
    def step(p, o, b):
        g = jax.grad(lambda pp, bb: sweep.loss_fn(model, pp, bb))(p, b)
        u, o2 = tx.update(g, o, p)
        return optax.apply_updates(p, u), o2

    b = jax.device_put(D.sample_batch(jax.random.PRNGKey(1), BS, 64))
    t0 = time.time(); p2, o2 = step(params, o, b); jax.block_until_ready(p2)
    t_compile = time.time() - t0
    t0 = time.time()
    for _ in range(3):
        p2, o2 = step(p2, o2, b)
    jax.block_until_ready(p2)
    per_it = (time.time() - t0) / 3
    print(f'BS={BS}: compile {t_compile:.1f}s  per-it {per_it:.3f}s', flush=True)
