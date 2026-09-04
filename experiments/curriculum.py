"""Staged fact-space curriculum for the Pool: does recall survive expansion?

The hypothesis under test: 16.78M-fact recall fails from gradient starvation
(uniform sampling gives each fact's slots too few sharpened-key updates), not
from capacity or retrieval mechanics (anchor: 65k facts -> 100%).

Protocol (stage = train -> eval -> warm-start next stage from final params):
  S1  nonce=64   65k facts    4000 steps  (~500 exp/fact; expect ~100%)
  S2  nonce=256  1.05M facts  4000 steps  (~31 exp/fact)  warm from S1
  S3  nonce=1024 16.8M facts  4000 steps  (~2 exp/fact)   warm from S2
Each stage's fresh-eval draws from ITS OWN fact space, so "fresh" after
expansion measures recall over facts never seen in ANY prior stage.
If recall holds through expansion, the Pool grows knowledge incrementally —
the property the architecture exists for. If S2 collapses despite warm-start
from a converged S1, expansion itself breaks key sharpening and the router
needs rework (cand_k, score_temp, or key-lr schedule).

Usage: NAVI_NONCE is set per stage inside the script; params are pickled to
/kaggle/working/experiments/ckpt_<stage>.pkl between stages.
"""
import sys, time, os, pickle
sys.path.insert(0, "/kaggle/working")
import jax, jax.numpy as jnp
import optax
import sweep, navi.data as D
from dataclasses import replace as _dc_replace
from navi.config import ModelConfig, TrainConfig
from navi.model import Navi
from navi.train import init_params

sweep.CFG_MEM = _dc_replace(sweep.CFG_MEM, side_top=16)

NC = jax.local_device_count()
mesh = jax.sharding.Mesh(jax.local_devices(), ("cores",))
REPL = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
BATCH = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("cores"))

def shard_tree(tree):
    def place(kp, x):
        ks = jax.tree_util.keystr(kp)
        if "values" in ks:
            return jax.device_put(x, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "cores")))
        return jax.device_put(x, REPL)
    return jax.tree_util.tree_map_with_path(place, tree)

PBS = 512
EVAL_PBS = 512
N_EVAL = 16

def run_stage(tag, steps, mult, warm_from=None):
    """One curriculum stage over the CURRENT NAVI_NONCE fact space."""
    model = Navi(sweep.MODEL, sweep.CFG_MEM)
    cfg_t = TrainConfig(total_steps=steps, batch_size=PBS, seq_len=sweep.SEQ,
                              warmup_steps=min(200, steps // 10), log_every=500,
                              mem_lr_mult=mult)
    # Vocab grows across stages (nonce 64->256->1024): embed/head can't
    # transfer, so they re-init fresh. Pool keys/values and all matching-shape
    # params carry over below.
    params = init_params(model, sweep.SEQ,
                         jax.random.fold_in(jax.random.PRNGKey(cfg_t.seed), 1))
    if warm_from and os.path.exists(warm_from):
        with open(warm_from, "rb") as f:
            old = pickle.load(f)
        # pickle holds the nested tree; keystr gives slash-paths -> flatten
        old_flat = {}
        jax.tree_util.tree_map_with_path(
            lambda kp, x: old_flat.__setitem__(jax.tree_util.keystr(kp), x), old)
        transferred, skipped = [], []
        def carry(kp, new_x):
            ks = jax.tree_util.keystr(kp)
            old_x = old_flat.get(ks)
            if old_x is not None and old_x.shape == new_x.shape:
                transferred.append(ks)
                return old_x
            skipped.append(ks)
            return new_x
        params = jax.tree_util.tree_map_with_path(
            lambda kp, x: carry(kp, x), params)
        print(f"[C-{tag}] warm: {len(transferred)} tensors carried, "
              f"{len(skipped)} re-inited (vocab-dependent)", flush=True)
        # Pool keys/values MUST carry -- that is the curriculum's whole point
        assert not any("values" in k or "/k1" in k or "/k2" in k for k in skipped), \
            f"Pool params lost in warm-start: {skipped}"

    p = shard_tree(params)
    tx = sweep.make_tx(cfg_t, params)
    o = shard_tree(tx.init(params))

    @jax.jit
    def step(p, o, b):
        g = jax.grad(lambda pp, bb: sweep.loss_fn(model, pp, bb))(p, b)
        u, o2 = tx.update(g, o, p)
        return optax.apply_updates(p, u), o2

    @jax.jit
    def loss_scalar(p, b):
        return sweep.loss_fn(model, p, b)

    @jax.jit
    def acc(pp, b):
        lg = model.apply(pp, b[:, :-1], train=False)
        return D.fact_recall_acc(lg, b[:, 1:])

    def ev(pp, seed_base):
        s = 0.0
        for i in range(N_EVAL):
            r = jax.random.fold_in(jax.random.PRNGKey(9), seed_base + i)
            b = D.sample_batch(r, EVAL_PBS, sweep.SEQ)
            s += float(acc(pp, jax.device_put(b, BATCH)))
        return s / N_EVAL

    t0 = time.time()
    for i in range(steps):
        rng = jax.random.fold_in(jax.random.PRNGKey(9), i)
        b = jax.device_put(D.sample_batch(rng, PBS, sweep.SEQ), BATCH)
        p, o = step(p, o, b)
        if i % 250 == 0 or i == steps - 1:
            l = float(loss_scalar(p, b))
            print(f"[{tag}] step {i:5d} loss {l:.4f} ({(time.time()-t0)/(i+1):.3f}s/it)", flush=True)

    # eval: fresh draws from this stage's fact space (seed_base 4_000_000 is
    # outside every training range), plus sabotage attribution
    fr = ev(p, 4_000_000 + os.getpid())
    print(f"[{tag}] EVAL fresh {fr:.4f}", flush=True)
    zeroed = sweep.zero_pool(p)
    print(f"[{tag}] EVAL zero-fresh {ev(zeroed, 4_000_000 + os.getpid()):.4f}", flush=True)

    with open(f"/kaggle/working/experiments/ckpt_{tag}.pkl", "wb") as f:
        pickle.dump(jax.device_get(p), f)
    print(f"== {tag} DONE ==", flush=True)
    return fr

if __name__ == "__main__":
    # VOCAB depends on NAVI_NONCE; stages run as separate processes via the
    # chain script. sweep.MODEL must be set before run_stage builds Navi.
    sweep.MODEL = ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2,
                              vocab_size=D.VOCAB)
    stage = sys.argv[1]
    warm = {"S2": "/kaggle/working/experiments/ckpt_C-S1.pkl",
            "S3": "/kaggle/working/experiments/ckpt_C-S2.pkl"}.get(stage)
    mult = float(os.environ.get("NAVI_MEM_MULT", "1.0"))
    run_stage(f"C-{stage}", int(os.environ.get("NAVI_STEPS", "4000")),
              mult, warm)