import sys, time, os
sys.path.insert(0, "/kaggle/working")
import jax, jax.numpy as jnp
import sweep, navi.data as D
from dataclasses import replace as _dc_replace
from navi.config import ModelConfig
from navi.model import Navi
from navi.train import init_params

sweep.CFG_MEM = _dc_replace(sweep.CFG_MEM, side_top=16)
sweep.CFG_MEM_T1 = _dc_replace(sweep.CFG_MEM_T1, side_top=16)
sweep.CFG_MEM_BIG = _dc_replace(sweep.CFG_MEM_BIG, side_top=16)
sweep.CFG_MEM_MID = _dc_replace(sweep.CFG_MEM_BIG, c1=1024, c2=1024, side_top=16)

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

PBS = int(os.environ.get("NAVI_PBS", "512"))   # GLOBAL batch (sharded auto)
EVAL_PBS = 512

def shard_arm(key, cfg_m, memc, mult, sabotages, steps, tag):
    sweep.CFG_MEM = memc
    model = Navi(cfg_m, memc)
    cfg_t = sweep.TrainConfig(total_steps=steps, batch_size=PBS, seq_len=sweep.SEQ,
                              warmup_steps=min(200, steps // 10), log_every=500,
                              mem_lr_mult=mult)
    params = init_params(model, sweep.SEQ, jax.random.fold_in(jax.random.PRNGKey(cfg_t.seed), 1))
    p = shard_tree(params)
    tx = sweep.make_tx(cfg_t, params)
    o = shard_tree(tx.init(params))

    @jax.jit
    def step(p, o, b):
        g = jax.grad(lambda pp, bb: sweep.loss_fn(model, pp, bb))(p, b)
        g = jax.lax.psum(g, "cores") if False else g
        u, o2 = tx.update(g, o, p)
        return optax_apply(p, u), o2

    import optax
    def optax_apply(p, u):
        return optax.apply_updates(p, u)

    @jax.jit
    def loss_scalar(p, b):
        return sweep.loss_fn(model, p, b)

    t0 = time.time()
    for i in range(steps):
        rng = jax.random.fold_in(jax.random.PRNGKey(9), i)
        b = jax.device_put(D.sample_batch(rng, PBS, sweep.SEQ), BATCH)
        p, o = step(p, o, b)
        if i % 250 == 0 or i == steps - 1:
            l = float(loss_scalar(p, b))
            print(f"[{tag}] step {i:5d} loss {l:.4f} ({(time.time()-t0)/(i+1):.3f}s/it)", flush=True)
    params = p

    @jax.jit
    def acc(pp, b):
        lg = model.apply(pp, b[:, :-1], train=False)
        return D.fact_recall_acc(lg, b[:, 1:])

    # seen: replay the exact final training batches (memorization metric);
    # fresh: new uniform draws from the same fixed fact table
    def ev(pp, seen):
        s = 0.0
        for i in range(sweep.N_EVAL):
            step_idx = steps - sweep.N_EVAL + i if seen else 4000000 + i
            r = jax.random.fold_in(jax.random.PRNGKey(9), step_idx)
            b = D.sample_batch(r, EVAL_PBS, sweep.SEQ)
            s += float(acc(pp, jax.device_put(b, BATCH)))
        return s / sweep.N_EVAL

    for s in sabotages:
        pp = params
        if s == "zero": pp = sweep.zero_pool(params)
        elif s == "shuffle": pp = sweep.shuffle_pool(params, jax.random.PRNGKey(4242))
        print(f"[{tag}] EVAL {s:10s} seen {ev(pp, True):.4f}  fresh {ev(pp, False):.4f}", flush=True)
    print(f"== {tag} DONE ==", flush=True)

if __name__ == "__main__":
    key = sys.argv[1]
    STEPS = int(os.environ.get("NAVI_STEPS", "4000"))
    MODEL = ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2, vocab_size=D.VOCAB)
    arms = {
        "P4":   (MODEL, sweep.CFG_MEM, 10.0, ("none", "zero", "shuffle")),
        "P4L1": (MODEL, sweep.CFG_MEM_T1, 1.0, ("none",)),
        "P4BIG": (MODEL, sweep.CFG_MEM_BIG, 10.0, ("none", "zero", "shuffle")),
        "P4MID": (MODEL, sweep.CFG_MEM_MID, 10.0, ("none", "zero", "shuffle")),
        "H4":   (MODEL, sweep.CFG_HASH, 1.0, ("none", "zero", "shuffle")),
        "H4MID": (MODEL, sweep.CFG_HASH_MID, 1.0, ("none", "zero", "shuffle")),
    }
    if key == "D4":
        from navi.config import ModelConfig as MC
        dense = MC(d_model=256, n_layers=8, n_heads=4, memory_every=0, vocab_size=D.VOCAB)
        arms["D4"] = (dense, sweep.CFG_MEM, 1.0, ("none",))
    cfg_m, memc, mult, sab = arms[key]
    shard_arm(key, cfg_m, memc, mult, sab, STEPS, f"S9:{key}")
