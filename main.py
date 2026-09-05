"""Navi CLI.

Usage:
  python main.py test                # unit checks (fast, CPU)
  python main.py train [--steps N]   # FFN-vs-Pool A/B comparison
  python main.py roundtrip [steps]   # train briefly, save, restore, verify
"""

import shutil
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

from navi.config import MemoryConfig, ModelConfig, TrainConfig
from navi.data import sample_batch
from navi.model import Navi
from navi.pkm import exact_topk_slots
from navi.train import (
    init_params,
    loss_fn,
    make_eval_step,
    make_train_step,
    make_tx,
)


def candidate_recall():
    """Two-sided top-k filter must recover the brute-force top-k slots."""
    cfg = MemoryConfig()
    pd = 16
    rng = jax.random.PRNGKey(0)
    recalls = []
    for _ in range(20):
        rng, qk, kk = jax.random.split(rng, 3)
        q1 = jax.random.normal(qk, (cfg.n_classes, pd))
        q2 = jax.random.normal(qk, (cfg.n_classes, pd))
        k1 = jax.random.normal(kk, (cfg.n_classes, cfg.c1, pd))
        k2 = jax.random.normal(kk, (cfg.n_classes, cfg.c2, pd))
        exact = exact_topk_slots(q1, q2, k1, k2, cfg)
        s1 = jnp.einsum("cd,ckd->ck", q1, k1)
        s2 = jnp.einsum("cd,ckd->ck", q2, k2)
        i1 = jax.lax.top_k(s1, cfg.side_top)[1]
        i2 = jax.lax.top_k(s2, cfg.side_top)[1]
        sub = jnp.take_along_axis(s1, i1, -1)[..., :, None] + jnp.take_along_axis(s2, i2, -1)[..., None, :]
        side = sub.shape[-2]
        got = jax.lax.top_k(sub.reshape(cfg.n_classes, side * side), cfg.cand_k)[1]
        r, c = got // side, got % side
        got_slots = jnp.take_along_axis(i1, r, -1) * cfg.c2 + jnp.take_along_axis(i2, c, -1)
        recalls.append(float(jnp.isin(got_slots, exact).mean()))
    return sum(recalls) / len(recalls)


def cmd_test():
    failures = 0

    rec = candidate_recall()
    if rec < 0.95:
        print("FAIL recall: %.3f < 0.95" % rec)
        failures += 1
    else:
        print("ok  recall of exact top-k via two-sided filter: %.3f" % rec)

    # forward pass + aux returns
    model = Navi(ModelConfig(), MemoryConfig())
    model_aux = Navi(ModelConfig(), MemoryConfig(), return_aux=True)
    ids = jnp.zeros((2, 16), dtype=jnp.int32)
    params = model.init({"params": jax.random.PRNGKey(0)}, ids, train=False)
    logits, aux = model_aux.apply(params, ids, train=False)
    if logits.shape != (2, 16, ModelConfig().vocab_size):
        print("FAIL bad logits shape", logits.shape)
        failures += 1
    n_mem = sum(1 for k in aux if k.startswith("mem_"))
    if n_mem != 3:
        print("FAIL aux: expected 3 memory layers, got", n_mem)
        failures += 1
    else:
        print("ok  forward pass, aux from %d memory layers" % n_mem)

    # gradient reaches Pool values
    ids64 = jnp.zeros((4, 64), dtype=jnp.int32)
    loss, grads = jax.value_and_grad(loss_fn)(params, model, ids64)
    vgrad = grads["params"]["block_0"]["mem"]["values"]
    gsum = float(jnp.abs(vgrad).sum())
    if gsum <= 0:
        print("FAIL no gradient reached Pool values")
        failures += 1
    else:
        print("ok  gradients reach Pool values (|g| = %.4f)" % gsum)

    # checkpoint roundtrip
    tmp = tempfile.mkdtemp()
    try:
        from navi.checkpoint import restore_params, save_params

        save_params(tmp + "/ckpt", params)
        restored = restore_params(tmp + "/ckpt", params)
        same = jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.array_equal,
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(restored),
            )
        )
        if not same:
            print("FAIL checkpoint roundtrip mismatch")
            failures += 1
        else:
            print("ok  orbax checkpoint roundtrip identical")
    finally:
        shutil.rmtree(tmp)

    print("test:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


def cmd_train(steps, batch=None, seq=None):
    cfg = TrainConfig()
    if steps:
        cfg = cfg.replace(total_steps=steps)
    if batch:
        cfg = cfg.replace(batch_size=batch)
    if seq:
        cfg = cfg.replace(seq_len=seq)
    cfg = cfg.replace(log_every=max(1, cfg.total_steps // 8))
    print("== A/B: FFN baseline vs Pool (product-key memory), identical backbone ==")
    print(cfg)
    from navi.train import run

    ffn = run(cfg, use_memory=False)
    pool = run(cfg, use_memory=True)
    print("")
    print("== summary ==")
    print("FFN : acc %.4f  params %d" % (ffn["acc"], ffn["params"]))
    print("Pool: acc %.4f  params %d" % (pool["acc"], pool["params"]))
    return 0


def cmd_roundtrip(steps):
    """Train briefly, save, restore, verify byte-identical params."""
    cfg = TrainConfig(total_steps=steps, log_every=max(1, steps // 2))
    model = Navi(ModelConfig(), MemoryConfig())
    rng = jax.random.PRNGKey(cfg.seed)
    params = init_params(model, cfg.seq_len, jax.random.fold_in(rng, 1))
    tx = make_tx(cfg)
    opt_state = tx.init(params)
    step = make_train_step(model, tx)
    loss = jnp.float32(0.0)
    for step_idx in range(cfg.total_steps):
        rng, d = jax.random.split(rng)
        batch = jax.device_put(sample_batch(d, cfg.batch_size, cfg.seq_len))
        params, opt_state, loss = step(params, opt_state, batch)
    print("trained %d steps, final loss %.4f" % (cfg.total_steps, float(loss)))

    import orbax.checkpoint as ocp

    tmp = Path(tempfile.mkdtemp())
    try:
        ckpt = ocp.PyTreeCheckpointer()
        ckpt.save(tmp / "params", params, force=True)
        restored = ckpt.restore(tmp / "params", item=params)
        ckpt.close()
        same = jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.array_equal,
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(restored),
            )
        )
        print("roundtrip:", "OK" if same else "MISMATCH")
        return 0 if same else 1
    finally:
        shutil.rmtree(tmp)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        return cmd_test()
    if cmd == "train":
        steps = None
        batch = None
        seq = None
        if "--steps" in sys.argv:
            steps = int(sys.argv[sys.argv.index("--steps") + 1])
        if "--batch" in sys.argv:
            batch = int(sys.argv[sys.argv.index("--batch") + 1])
        if "--seq" in sys.argv:
            seq = int(sys.argv[sys.argv.index("--seq") + 1])
        return cmd_train(steps, batch, seq)
    if cmd == "roundtrip":
        return cmd_roundtrip(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())