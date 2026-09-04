"""Training loop + the A/B experiment: identical backbone, FFN vs Pool.

At matched parameter budget and data, does replacing every other FFN with
a product-key memory store facts better? Both arms see identical data and
schedule.
"""

import jax
import jax.numpy as jnp
import optax

from navi.config import TrainConfig
from navi.data import fact_recall_acc, sample_batch
from navi.model import Navi


def loss_fn(params, model, ids):
    logits = model.apply(params, ids[:, :-1], train=True)
    targets = ids[:, 1:]
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def _is_mem(kp) -> bool:
    ks = jax.tree_util.keystr(kp)
    return "values" in ks or "/k1" in ks or "/k2" in ks


def make_tx(cfg, params_like):
    """AdamW with warmup-cosine; Pool params (keys+values) train
    mem_lr_mult x faster via optax.multi_transform (Meta's memory recipe)."""
    core = optax.adamw(
        optax.schedules.warmup_cosine_decay_schedule(
            init_value=cfg.lr * 0.02, peak_value=cfg.lr,
            decay_steps=cfg.total_steps, warmup_steps=cfg.warmup_steps,
        ),
        b1=0.9, b2=0.95, weight_decay=cfg.weight_decay,
    )
    if cfg.mem_lr_mult == 1.0:
        return core
    mem = optax.adamw(
        optax.schedules.warmup_cosine_decay_schedule(
            init_value=cfg.lr * 0.02, peak_value=cfg.lr * cfg.mem_lr_mult,
            decay_steps=cfg.total_steps, warmup_steps=cfg.warmup_steps,
        ),
        b1=0.9, b2=0.95, weight_decay=cfg.weight_decay,
    )
    labels = jax.tree_util.tree_map_with_path(
        lambda kp, x: "mem" if _is_mem(kp) else "core", params_like
    )
    return optax.multi_transform({"core": core, "mem": mem}, labels)


def make_train_step(model, tx):
    @jax.jit
    def step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, model, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return step


def make_eval_step(model):
    @jax.jit
    def step(params, batch):
        logits = model.apply(params, batch[:, :-1], train=False)
        return fact_recall_acc(logits, batch[:, 1:])

    return step


def init_params(model, seq_len, key):
    ids = jnp.zeros((1, seq_len), dtype=jnp.int32)
    return model.init({"params": key}, ids, train=False)


def run(cfg_t, use_memory):
    tag = "pool" if use_memory else "ffn"
    cfg_m = ModelConfig(memory_every=2 if use_memory else 0)
    model = Navi(cfg_m, MemoryConfig())
    rng = jax.random.PRNGKey(cfg_t.seed)
    params = init_params(model, cfg_t.seq_len, jax.random.fold_in(rng, 1))

    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    total = sum(p.size for _, p in flat)
    mem_params = sum(
        p.size for k, p in flat if "mem" in jax.tree_util.keystr(k)
    )

    tx = make_tx(cfg_t, params)
    opt_state = tx.init(params)
    step = make_train_step(model, tx)
    eval_step = make_eval_step(model)

    for step_idx in range(cfg_t.total_steps):
        rng, data_rng = jax.random.split(rng)
        batch = jax.device_put(sample_batch(data_rng, cfg_t.batch_size, cfg_t.seq_len))
        params, opt_state, loss = step(params, opt_state, batch)
        if step_idx % cfg_t.log_every == 0 or step_idx == cfg_t.total_steps - 1:
            print(f"[{tag}] step {step_idx:5d}  loss {float(loss):.4f}")

    accs = []
    for i in range(8):
        rng, eval_rng = jax.random.split(rng)
        batch = jax.device_put(sample_batch(eval_rng, 4 * cfg_t.batch_size, cfg_t.seq_len))
        accs.append(float(eval_step(params, batch)))
    acc = sum(accs) / len(accs)
    print(f"[{tag}] fact-recall acc: {acc:.4f}  params: {total:,} (mem: {mem_params:,})")
    return {"acc": acc, "params": total, "mem_params": mem_params}


def main():
    print("== A/B: FFN baseline vs Pool (product-key memory), identical backbone ==")
    ffn = run(TrainConfig(), use_memory=False)
    pool = run(TrainConfig(), use_memory=True)
    print("")
    print("== summary ==")
    print("FFN : acc %.4f  params %d" % (ffn["acc"], ffn["params"]))
    print("Pool: acc %.4f  params %d" % (pool["acc"], pool["params"]))


from navi.config import MemoryConfig, ModelConfig  # noqa: E402

if __name__ == "__main__":
    main()